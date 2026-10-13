#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只对已有 generations.jsonl 评分；不加载待测生成模型，也不打开图片。

示例：
    python script/evaluate_rag_metrics.py --generations A/generations.jsonl B/generations.jsonl
    python script/evaluate_rag_metrics.py --config script/evaluate_rag_metrics.json

评分依赖和答案解析复用 script/lib；此脚本不导入推理脚本。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.lib.answer_metrics import question_type_for_sample, score_answer
from script.lib.atomic_factscore import AtomicFactScorer
from script.lib.evaluation_statistics import paired_summary, summarize_values
from script.lib.reference_judge import ReferenceJudge
from script.lib.token_budget import FACT_LENGTH_BUDGET_DEFAULTS, normalize_length_budget
from script.lib.text_metrics import compute_bleu4, compute_rouge_l

LOGGER = logging.getLogger("rag_eval_metrics")
DEFAULT_CONFIG = Path(__file__).with_name("evaluate_rag_metrics.json")
QUESTION_TYPES = ("open_ended", "closed_ended", "single_choice", "multiple_choice")
TYPE_LABELS = {"open_ended": "开放式问题", "closed_ended": "封闭式问题",
               "single_choice": "单选题", "multiple_choice": "多选题"}
METRICS = ("answer_accuracy", "choice_f1", "bleu4", "rouge_l", "semantic_score")
PRIMA_METRICS = ("factscore", "answer_accuracy", "explanation_rouge_l")
METRIC_LABELS = {"answer_accuracy": "Accuracy", "choice_f1": "Choice F1", "bleu4": "BLEU-4",
                 "rouge_l": "ROUGE-L", "semantic_score": "Semantic score",
                 "factscore": "FActScore", "explanation_rouge_l": "Explanation ROUGE-L"}
MODEL_PATHS = {
    "Qwen3-VL-2B-Instruct": PROJECT_ROOT / "Qwen3-VL-2B-Instruct",
    "DeepSeek-Model": PROJECT_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B",
}
EVAL_DEFAULTS = {
    "judgeModel": "Qwen3-VL-2B-Instruct", "judgeModelPath": None,
    "judgeScope": "open_ended", "judgeMaxNewTokens": 256, "judgeRetries": 1,
    "bootstrapSamples": 1000,
    "prima": {
        "enabled": True, "separateQuestionTypes": True, "questionTypes": list(QUESTION_TYPES),
        "factScore": True, "explanationRougeL": True, "factScoreMaxNewTokens": 1024,
        "factScoreBatchSize": 8, "factScoreLengthBudget": FACT_LENGTH_BUDGET_DEFAULTS,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number} JSON 解析失败：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} 必须为 JSON 对象。")
            rows.append(row)
    if not rows:
        raise ValueError(f"generations 文件为空：{path}")
    return rows


def _resolve_generations(value: str | Path, base: Path) -> list[Path]:
    path = Path(value).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if path.is_file():
        if path.name.casefold() != "generations.jsonl":
            raise ValueError(f"输入文件应为 generations.jsonl：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"找不到 generations 文件或目录：{path}")
    direct = path / "generations.jsonl"
    if direct.is_file():
        return [direct]
    found = sorted(path.rglob("generations.jsonl"))
    if not found:
        raise FileNotFoundError(f"目录中没有 generations.jsonl：{path}")
    return found


def _safe_label(value: str) -> str:
    # Windows 会把带扩展名的设备名也视为保留名，例如 CON.txt。
    device_stem = value.split(".", 1)[0].rstrip(" .").upper() if isinstance(value, str) else ""
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value in {".", ".."} or value.endswith(".")
            or not re.fullmatch(r"[\w .\-\u4e00-\u9fff]{1,100}", value)
            or device_stem in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)(?:[1-9]|[¹²³])", device_stem, re.I)):
        raise ValueError(f"标签不能安全用作目录名：{value!r}")
    return value


def _label_directory(root: Path, label: str) -> Path:
    """解析并确认标签目录始终位于本次评分输出根目录内。"""
    root = root.resolve()
    directory = (root / _safe_label(label)).resolve()
    if directory == root or root not in directory.parents:
        raise ValueError(f"评分标签目录越出输出根目录：{directory}")
    return directory


def _read_sidecar_evaluation(path: Path) -> dict | None:
    sidecar = path.parent / "summary.json"
    if not sidecar.is_file():
        return None
    try:
        summary = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取 generations 对应的 summary.json：{sidecar}: {exc}") from exc
    value = summary.get("evaluation") if isinstance(summary, dict) else None
    return value if isinstance(value, dict) else None


def _normalize_evaluation(raw: dict | None, config_dir: Path) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("evaluation 必须为 JSON 对象。")
    unknown = set(raw) - set(EVAL_DEFAULTS)
    if unknown:
        raise ValueError(f"evaluation 出现未知配置项：{sorted(unknown)}")
    result = {**EVAL_DEFAULTS, **raw}
    prima_raw = raw.get("prima", {})
    if not isinstance(prima_raw, dict) or set(prima_raw) - (set(EVAL_DEFAULTS["prima"]) | {"announceQuestionType"}):
        raise ValueError(f"evaluation.prima 配置无效；可选字段为 {list(EVAL_DEFAULTS['prima'])}")
    prima = {**EVAL_DEFAULTS["prima"], **prima_raw}
    for field in ("enabled", "separateQuestionTypes", "factScore", "explanationRougeL"):
        if type(prima[field]) is not bool:
            raise ValueError(f"evaluation.prima.{field} 必须为布尔值。")
    types = prima["questionTypes"]
    if (not isinstance(types, list) or not types or len(set(types)) != len(types)
            or any(item not in QUESTION_TYPES for item in types)):
        raise ValueError(f"evaluation.prima.questionTypes 必须是有效题型列表：{QUESTION_TYPES}")
    for field, low, high in (("factScoreMaxNewTokens", 128, 8192), ("factScoreBatchSize", 1, 32)):
        if type(prima[field]) is not int or not low <= prima[field] <= high:
            raise ValueError(f"evaluation.prima.{field} 范围为 {low}～{high}。")
    budget = prima["factScoreLengthBudget"]
    if not isinstance(budget, dict):
        raise ValueError("evaluation.prima.factScoreLengthBudget 必须为对象。")
    prima["factScoreLengthBudget"] = normalize_length_budget(
        budget, "evaluation.prima.factScoreLengthBudget", defaults=FACT_LENGTH_BUDGET_DEFAULTS)
    result["prima"] = prima
    if result["judgeModel"] not in MODEL_PATHS:
        raise ValueError(f"judgeModel 仅支持：{', '.join(MODEL_PATHS)}")
    if result["judgeScope"] not in {"none", "open_ended", "all"}:
        raise ValueError("judgeScope 可选 none、open_ended、all。")
    for field, low, high in (("judgeMaxNewTokens", 32, 2048), ("judgeRetries", 0, 1),
                             ("bootstrapSamples", 100, 10000)):
        if type(result[field]) is not int or not low <= result[field] <= high:
            raise ValueError(f"evaluation.{field} 范围为 {low}～{high}。")
    judge_path = result["judgeModelPath"]
    if judge_path is None:
        result["judgeModelPath"] = str(MODEL_PATHS[result["judgeModel"]].resolve())
    elif isinstance(judge_path, str) and judge_path.strip():
        candidate = Path(judge_path).expanduser()
        result["judgeModelPath"] = str((config_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    else:
        raise ValueError("evaluation.judgeModelPath 必须为路径字符串或 null。")
    return result


class _TextJudgeModel:
    """按需加载固定裁判模型，仅提供文本接口；绝不接收图片。"""

    def __init__(self, model_type: str, model_path: str, *, max_input_tokens: int = 16384):
        import torch

        self.torch = torch
        self.model_type, self.model_path = model_type, model_path
        self.max_input_tokens = max_input_tokens
        self.model = self.tokenizer = self.processor = None
        self.last_generation_info = {}
        self.system_prompt = "你是严格的医学事实核验助手。只依据给定的参考资料评分，不使用外部知识。"
        try:
            from script.lib.prompt_config import prompt
            self.system_prompt = prompt("evaluation.fact_system", default=self.system_prompt)
        except (FileNotFoundError, KeyError):
            pass

    def _load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        device = "cuda" if self.torch.cuda.is_available() else "cpu"
        dtype = (self.torch.bfloat16 if device == "cuda" and self.torch.cuda.is_bf16_supported()
                 else self.torch.float16 if device == "cuda" else self.torch.float32)
        if self.model_type == "DeepSeek-Model":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, local_files_only=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True).to(device).eval()
        else:
            try:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path, dtype=dtype, local_files_only=True)
            except (ImportError, ValueError):
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path, dtype=dtype, local_files_only=True)
            self.model = self.model.to(device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self.tokenizer = self.processor.tokenizer
        self.device = device

    def count_tokens(self, text: str) -> int:
        self._load()
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def text(self, prompt_text: str, max_new_tokens: int = 256) -> str:
        self._load()
        if self.model_type == "DeepSeek-Model":
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt_text}]
            rendered = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": ""}],
                tokenize=False, add_generation_prompt=False)
            eos = self.tokenizer.eos_token
            if eos and rendered.endswith(eos):
                rendered = rendered[:-len(eos)]
            inputs = self.tokenizer(rendered, return_tensors="pt", truncation=False, add_special_tokens=False)
            decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
        else:
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt_text}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[rendered], images=None, videos=None, return_tensors="pt")
            decode = lambda ids: self.processor.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        input_length = int(inputs["input_ids"].shape[1])
        if input_length > self.max_input_tokens:
            raise ValueError(f"裁判输入 {input_length} tokens 超过 max_input_tokens={self.max_input_tokens}。")
        inputs = inputs.to(self.device)
        self.model.config.use_cache = True
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         num_beams=1, pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id)
        new_ids = output[0, input_length:]
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = set(eos_ids) if isinstance(eos_ids, (list, tuple, set)) else {eos_ids}
        ended = bool(new_ids.numel() and int(new_ids[-1]) in eos_ids)
        self.last_generation_info = {"input_tokens": input_length, "output_tokens": int(new_ids.shape[0]),
                                     "hit_token_limit": int(new_ids.shape[0]) >= max_new_tokens and not ended,
                                     "max_new_tokens": max_new_tokens,
                                     "elapsed_seconds": time.perf_counter() - started}
        return decode(new_ids).strip()

    def close(self):
        self.model = self.tokenizer = self.processor = None
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _metric_names(evaluation: dict) -> tuple[str, ...]:
    return PRIMA_METRICS if evaluation["prima"]["enabled"] else METRICS


def _metric_applies(row: dict, metric: str) -> bool:
    kind = row["answer_details"]["question_type"]
    if metric == "answer_accuracy":
        return kind in {"single_choice", "multiple_choice", "closed_ended"}
    if metric == "choice_f1":
        return kind in {"single_choice", "multiple_choice"}
    if metric == "factscore":
        return row["judge_required"] and row["judge_metric"] == "factscore"
    if metric == "semantic_score":
        return row["judge_required"] and row["judge_metric"] == "semantic_score"
    if metric == "explanation_rouge_l":
        return row["explanation_enabled"]
    return True


def _summarize(rows: list[dict], evaluation: dict, seed: int) -> dict:
    result = {}
    prima = evaluation["prima"]
    for metric in _metric_names(evaluation):
        selected = [row for row in rows if _metric_applies(row, metric)]
        result[metric] = summarize_values([row.get(metric) for row in selected], seed=seed,
                                          bootstrap_samples=evaluation["bootstrapSamples"],
                                          binary=metric == "answer_accuracy")
    return result


def _update_summary(summary: dict, rows: list[dict], seed: int) -> None:
    evaluation = summary["evaluation"]
    summary["metrics"] = _summarize(rows, evaluation, seed)
    types = QUESTION_TYPES if evaluation["prima"]["enabled"] else sorted(
        {row["answer_details"]["question_type"] for row in rows})
    summary["by_question_type"] = {
        kind: {"num_samples": sum(row["answer_details"]["question_type"] == kind for row in rows),
               "metrics": _summarize([row for row in rows if row["answer_details"]["question_type"] == kind],
                                     evaluation, seed)}
        for kind in types
    }
    summary["judge_failures"] = sum(
        (row.get("factscore_details") if row["judge_metric"] == "factscore" else row.get("semantic_details", {})).get("status") == "judge_failed"
        for row in rows)
    summary["judge_required"] = sum(row["judge_required"] for row in rows)
    summary["possible_truncations"] = sum(bool(row.get("generation_info", {}).get("hit_token_limit")) for row in rows)
    summary["review_count"] = sum(_needs_review(row) for row in rows)
    summary["scoring_status"] = "needs_review" if summary["judge_failures"] else "complete"


def _needs_review(row: dict) -> bool:
    details = row.get("factscore_details") if row.get("judge_metric") == "factscore" else row.get("semantic_details", {})
    return bool(details.get("status") == "judge_failed"
                or row.get("answer_details", {}).get("parse_status") in {"prediction_unparseable", "invalid_reference"}
                or row.get("generation_info", {}).get("hit_token_limit")
                or (row.get("explanation_enabled") and row["answer_details"].get("explanation_status") != "ok"))


def _load_dataset_fallback(dataset_path: str | None, paths: list[Path], records: list[dict]) -> list[dict]:
    needs_answer_fields = any("answer" not in row or ("options" not in row and "Options: " not in row.get("question", ""))
                             for row in records)
    if not needs_answer_fields:
        return records
    if dataset_path is None:
        # Old generations may have the serialized answer in reference; score_answer handles JSON text.
        return records
    source = Path(dataset_path).expanduser().resolve()
    identifiers = list(dict.fromkeys(str(row["id"]) for row in records))
    from script.lib.evaluation_data import _load_jsonl, _load_selected_mira
    if source.is_dir():
        source_records = _load_selected_mira(source, identifiers)
    else:
        by_id = {str(row["id"]): row for row in _load_jsonl(source)}
        missing = [identifier for identifier in identifiers if identifier not in by_id]
        if missing:
            raise ValueError(f"回源数据缺少 {len(missing)} 个 ID，例如：{missing[:5]}")
        source_records = [by_id[identifier] for identifier in identifiers]
    by_id = {str(row["id"]): row for row in source_records}
    return [{**by_id.get(str(row["id"]), {}), **row,
             **{key: by_id[str(row["id"])][key] for key in ("answer", "options", "question_type")
                if key not in row and key in by_id.get(str(row["id"]), {})}}
            for row in records]


def _make_rows(records: list[dict], evaluation: dict, label: str) -> list[dict]:
    prima = evaluation["prima"]
    rows = []
    seen = set()
    for source in records:
        if "id" not in source or str(source["id"]).strip() == "":
            raise ValueError(f"{label}: generations 记录缺少 id。")
        key = str(source["id"]).strip()
        if key in seen:
            raise ValueError(f"{label}: generations 包含重复 id：{key}")
        seen.add(key)
        question = source.get("question")
        prediction = source.get("prediction")
        reference = source.get("reference", source.get("answer"))
        if not isinstance(question, str) or not question.strip() or not isinstance(prediction, str):
            raise ValueError(f"{label}/{key}: question 与 prediction 必须为文本。")
        if reference is None:
            raise ValueError(f"{label}/{key}: 缺少 reference/answer。")
        sample = {**source, "reference": reference}
        sample.setdefault("answer", reference)
        answers = score_answer(sample, prediction)
        if prima["enabled"]:
            judge_metric = "factscore"
            judge_required = prima["factScore"] and answers["question_type"] == "open_ended" and "open_ended" in prima["questionTypes"]
        else:
            judge_metric = "semantic_score"
            judge_required = evaluation["judgeScope"] == "all" or (
                evaluation["judgeScope"] == "open_ended" and answers["question_type"] == "open_ended")
        explanation_enabled = prima["enabled"] and prima["explanationRougeL"]
        explanation_score = None
        if explanation_enabled and answers["explanation_eligible"]:
            explanation_score = compute_rouge_l(answers["prediction_explanation"], answers["reference_explanation"])
        normalized_prediction = answers["normalized_prediction"]
        normalized_reference = answers["normalized_reference"]
        row = {**sample, "prediction": prediction, "question_type": answers["question_type"],
               "normalized_prediction": normalized_prediction, "normalized_reference": normalized_reference,
               "answer_details": answers, "answer_accuracy": answers["answer_accuracy"],
               "choice_f1": answers["choice_f1"], "bleu4": compute_bleu4(normalized_prediction, normalized_reference),
               "rouge_l": compute_rouge_l(normalized_prediction, normalized_reference),
               "semantic_score": None, "factscore": None, "explanation_rouge_l": explanation_score,
               "explanation_enabled": explanation_enabled, "judge_required": judge_required,
               "judge_metric": judge_metric,
               "semantic_details": {"score": None, "status": "pending" if judge_required and judge_metric == "semantic_score" else "not_applicable"},
               "factscore_details": {"score": None, "status": "pending" if judge_required and judge_metric == "factscore" else "not_applicable"},
               "retrieved_evidence": source.get("retrieved_evidence", []),
               "retrieved_count": len(source.get("retrieved_evidence", [])),
               "config_name": source.get("config_name"),
               "generation_fingerprint": source.get("generation_fingerprint"),
               "generation_from_cache": False, "generation_info": source.get("generation_info", {})}
        rows.append(row)
    return rows


def _judge_rows(rows: list[dict], evaluation: dict, llm, cache_dir: Path, identity: dict) -> None:
    prima = evaluation["prima"]
    scorer = None
    if prima["enabled"] and prima["factScore"]:
        scorer = AtomicFactScorer(llm, cache_dir, identity,
                                  max_new_tokens=prima["factScoreMaxNewTokens"],
                                  batch_size=prima["factScoreBatchSize"], retries=evaluation["judgeRetries"],
                                  length_budget=prima["factScoreLengthBudget"])
    elif not prima["enabled"] and evaluation["judgeScope"] != "none":
        scorer = ReferenceJudge(llm, cache_dir, identity,
                                max_new_tokens=evaluation["judgeMaxNewTokens"], retries=evaluation["judgeRetries"])
    if scorer is None:
        return
    required = [row for row in rows if row["judge_required"]]
    for row in tqdm(required, desc="FActScore" if prima["enabled"] else "语义裁判"):
        answers = row["answer_details"]
        sample = {**row, "evaluation_reference": answers["semantic_reference"]}
        details = scorer.score(sample, answers["semantic_prediction"])
        if prima["enabled"]:
            row["factscore_details"] = details
            row["factscore"] = details["score"]
        else:
            row["semantic_details"] = details
            row["semantic_score"] = details["score"]


def _paired_metrics(after: dict, before: dict, evaluation: dict, seed: int) -> dict:
    if after.keys() != before.keys():
        raise ValueError("配对差值要求两组 generations 的 ID 集合完全一致。")
    result = {}
    for metric in _metric_names(evaluation):
        keys = [key for key in sorted(after) if _metric_applies(after[key], metric) and _metric_applies(before[key], metric)]
        result[metric] = paired_summary([after[key][metric] for key in keys], [before[key][metric] for key in keys],
                                        seed=seed, bootstrap_samples=evaluation["bootstrapSamples"])
    return result


def _summary(label: str, generation_path: Path, source_records: list[dict], rows: list[dict], evaluation: dict,
             seed: int, elapsed: float, sidecar: dict | None = None) -> dict:
    test_fingerprint = _fingerprint([{key: row.get(key) for key in
                                      ("id", "question", "reference", "answer", "options", "images", "question_type")}
                                     for row in source_records])
    sidecar = sidecar or {}
    return {"name": label, "config_name": rows[0].get("config_name"), "generations_path": str(generation_path),
            "num_samples": len(rows), "test_fingerprint": test_fingerprint,
            "dataset_fingerprint": sidecar.get("dataset_fingerprint", test_fingerprint),
            "generation_fingerprint": rows[0].get("generation_fingerprint"),
            "evaluation": evaluation, "metric_scale": "0-1", "std_ddof": 0,
            "aggregation": "macro average over test samples", "protocol": "prima" if evaluation["prima"]["enabled"] else "reference_semantic",
            "model": sidecar.get("model"), "model_path": sidecar.get("model_path"),
            "lora_path": sidecar.get("lora_path"), "use_rag": sidecar.get("use_rag"),
            "top_k": sidecar.get("top_k"), "chroma_db_dir": sidecar.get("chroma_db_dir"),
            "chroma_collection": sidecar.get("chroma_collection"),
            "chroma_id_prefix": sidecar.get("chroma_id_prefix"),
            "scoring_version": sidecar.get("scoring_version"),
            "bleu_smoothing": sidecar.get("bleu_smoothing"),
            "generation_settings": sidecar.get("generation_settings"), "seed": seed,
            "generation_seconds_excluding_load": sum(float(row.get("generation_info", {}).get("elapsed_seconds", 0) or 0) for row in rows),
            "generation_input_mode": sidecar.get("generation_input_mode", "from_generations"), "scoring_seconds": elapsed}


def _save_one(root: Path, label: str, generation_path: Path, records: list[dict], evaluation: dict,
              seed: int, judge, judge_identity: dict | None, cache_dir: Path,
              sidecar: dict | None = None) -> dict:
    started = time.perf_counter()
    records = _load_dataset_fallback(None, [generation_path], records)
    rows = _make_rows(records, evaluation, label)
    if any(row["judge_required"] for row in rows):
        if judge is None:
            raise RuntimeError("当前配置需要裁判，但裁判模型未初始化。")
        _judge_rows(rows, evaluation, judge, cache_dir, judge_identity or {})
    summary = _summary(label, generation_path, records, rows, evaluation, seed,
                       time.perf_counter() - started, sidecar)
    _update_summary(summary, rows, seed)
    directory = _label_directory(root, label)
    _write_jsonl(directory / "predictions.jsonl", rows)
    _write_json(directory / "summary.json", summary)
    review = [row for row in rows if _needs_review(row)]
    _write_json(directory / "review_needed.json", review)
    prima = evaluation["prima"]
    if prima["enabled"] and prima["separateQuestionTypes"]:
        for kind in QUESTION_TYPES:
            group = [row for row in rows if row["answer_details"]["question_type"] == kind]
            group_dir = directory / "by_question_type" / kind
            group_summary = {**summary, "question_type": kind, "num_samples": len(group),
                             "included": kind in prima["questionTypes"]}
            _update_summary(group_summary, group, seed)
            _write_jsonl(group_dir / "predictions.jsonl", group)
            _write_json(group_dir / "summary.json", group_summary)
            _write_json(group_dir / "review_needed.json", [row for row in group if _needs_review(row)])
    return summary


def _summary_display(metrics: dict) -> str:
    value = metrics.get("mean")
    return "—" if value is None else f"{value * 100:.2f} ± {metrics['std'] * 100:.2f} (n={metrics['n']})"


def _write_reports(root: Path, summaries: list[dict], by_label: dict[str, dict], evaluation: dict, seed: int) -> dict:
    paired = []
    names = list(by_label)
    # 默认报告 LoRA 与其原版的同题差值；仅在两组 ID 集合完全相同时计算。
    summary_by_name = {summary["name"]: summary for summary in summaries}
    for fine_name in names:
        fine_summary = summary_by_name[fine_name]
        fine_lora = bool(fine_summary.get("lora_path") or re.search(r"lora|微调", fine_name, re.I))
        if not fine_lora:
            continue
        base_candidates = [name for name in names if name != fine_name and not (
            summary_by_name[name].get("lora_path") or re.search(r"lora|微调", name, re.I))]
        compatible_candidates = []
        for base_name in base_candidates:
            base_summary = summary_by_name[base_name]
            compatible_fields = ["model_path", "use_rag", "top_k", "test_fingerprint", "dataset_fingerprint",
                                 "evaluation", "scoring_version", "bleu_smoothing", "generation_input_mode",
                                 "generation_settings"]
            compatible_fields.append("model")
            if any(fine_summary.get(field) != base_summary.get(field) for field in compatible_fields):
                continue
            if fine_summary.get("use_rag") and any(
                    fine_summary.get(field) != base_summary.get(field)
                    for field in ("chroma_db_dir", "chroma_collection", "chroma_id_prefix")):
                continue
            compatible_candidates.append(base_name)
        if not fine_summary.get("model") and len(compatible_candidates) > 1:
            # 未提供 sidecar 模型身份时，不从多个候选原版中猜测配对关系。
            compatible_candidates = []
        for base_name in compatible_candidates:
            base_summary = summary_by_name[base_name]
            after, before = by_label[fine_name], by_label[base_name]
            after_map = {str(row["id"]): row for row in after}
            before_map = {str(row["id"]): row for row in before}
            if after_map.keys() != before_map.keys():
                LOGGER.warning("未计算 %s 相对 %s 的配对差值：ID 集合不一致（%d vs %d）。",
                               fine_name, base_name, len(after_map), len(before_map))
                continue
            delta = {"name": fine_name, "baseline": base_name, "num_samples": len(after_map),
                     "metrics": _paired_metrics(after_map, before_map, evaluation, seed)}
            if evaluation["prima"]["enabled"]:
                delta["by_question_type"] = {
                    kind: _paired_metrics(
                        {key: row for key, row in after_map.items() if row["answer_details"]["question_type"] == kind},
                        {key: row for key, row in before_map.items() if row["answer_details"]["question_type"] == kind},
                        evaluation, seed)
                    for kind in QUESTION_TYPES}
            paired.append(delta)
    same_data = len({summary["test_fingerprint"] for summary in summaries}) <= 1
    result = {"schema_version": "rag-metrics-v1", "status": "complete", "protocol": "prima" if evaluation["prima"]["enabled"] else "reference_semantic",
              "same_test_data": same_data, "summaries": summaries, "paired_deltas": paired,
              "note": "多选题按所选选项集合精确匹配；缺失评分记 null 并列入人工复核，不以零分替代。"}
    question_rows = []
    lines = ["# MIRA 生成结果评分", "", "分数范围为 0–100%；±为有效样本总体标准差，n 为有效评分数。", "",
             "| 评估名称 | 题型 | 题数 | FActScore (%) | Accuracy (%) | 解释 ROUGE-L (%) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for summary in summaries:
        groups = summary.get("by_question_type", {}) if evaluation["prima"]["separateQuestionTypes"] else {"all": summary}
        if not groups:
            groups = {"all": summary}
        for kind, group in groups.items():
            metrics = group["metrics"]
            row = {"name": summary["name"], "question_type": kind, "num_samples": group["num_samples"],
                   "metrics": metrics}
            question_rows.append(row)
            lines.append("| " + " | ".join([summary["name"], TYPE_LABELS.get(kind, "合并"), str(group["num_samples"]),
                                                   _summary_display(metrics.get("factscore", {})),
                                                   _summary_display(metrics.get("answer_accuracy", {})),
                                                   _summary_display(metrics.get("explanation_rouge_l", {}))]) + " |")
    if paired:
        lines.extend(["", "配对比较（只比较同 ID 且 ID 集合一致的 LoRA/原版结果）：", ""])
        for delta in paired:
            lines.append(f"- {delta['name']} 相对 {delta['baseline']}：" + "; ".join(
                f"{METRIC_LABELS[key]} {value['mean'] * 100:+.2f}（n={value['n']}）"
                for key, value in delta["metrics"].items() if value["mean"] is not None))
    if not same_data:
        lines.extend(["", "输入间测试 ID 或内容不同；不得直接横向比较总体均值。"])
    lines.extend(["", result["note"], ""])
    result["question_type_comparison"] = question_rows
    _write_json(root / "suite_comparison.json", result)
    _write_json(root / "comparison.json", result)
    _write_json(root / "question_type_comparison.json", {"rows": question_rows, "paired_deltas": paired})
    (root / "suite_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _load_config(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    tasks = [raw] if isinstance(raw, dict) else raw
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("评估配置必须为对象或非空对象数组。")
    return tasks


def _run_task(task: dict, config_path: Path, cli_generations: list[str] | None,
              cli_output_dir: str | None, cli_resume: bool | None,
              cli_dataset_path: str | None = None) -> int:
    config_dir = config_path.parent
    generation_values = cli_generations if cli_generations is not None else task.get("generations")
    if not isinstance(generation_values, list) or not generation_values:
        raise ValueError("请通过 JSON generations 或 --generations 提供至少一个文件/目录。")
    path_labels = task.get("labels", {})
    if not isinstance(path_labels, dict):
        raise ValueError("labels 必须是从路径到标签的 JSON 对象。")
    entries, labels_seen = [], set()
    for value in generation_values:
        paths = _resolve_generations(value, config_dir)
        for path in paths:
            records = _read_jsonl(path)
            config_names = {row.get("config_name") for row in records if row.get("config_name") is not None}
            if len(config_names) > 1:
                raise ValueError(f"同一 generations 文件含多个 config_name：{path}: {sorted(config_names)}")
            fallback_name = next(iter(config_names), path.parent.name)
            label = path_labels.get(str(value), path_labels.get(str(path), fallback_name))
            label = _safe_label(label)
            if label.casefold() in labels_seen:
                raise ValueError(f"输入标签重复（Windows 不区分大小写）：{label}")
            labels_seen.add(label.casefold())
            entries.append((label, path, records))
    sidecars = [value for _, path, _ in entries if (value := _read_sidecar_evaluation(path)) is not None]
    selected_eval = task.get("evaluation")
    if selected_eval is None and sidecars:
        selected_eval = sidecars[0]
    evaluation = _normalize_evaluation(selected_eval, config_dir)
    # 输入来自相同实验协议时评价配置应一致；从 sidecar 推断时同样逐一核对。
    normalized_sidecars = [_normalize_evaluation(value, config_dir) for value in sidecars]
    if any(_canonical(value) != _canonical(evaluation) for value in normalized_sidecars):
        raise ValueError("所有 generations 对应 summary.json 的 evaluation 设置必须一致，且与评估配置相同。")
    seed = task.get("seed", 42)
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise ValueError("seed 必须是 [0, 2**32) 内的整数。")
    output_value = cli_output_dir or task.get("outputDir", "../output/eval_results_metrics")
    output = Path(output_value).expanduser()
    output = (config_dir / output).resolve() if not output.is_absolute() else output.resolve()
    if len(_load_config(config_path)) > 1 and not cli_output_dir:
        task_name = _safe_label(task.get("name", f"task_{int(time.time())}"))
        output = output / task_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "eval.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")], force=True)
    resume = cli_resume if cli_resume is not None else task.get("resume", True)
    if type(resume) is not bool:
        raise ValueError("resume 必须为布尔值。")
    cache_dir = output / "judge_cache"
    if not resume and cache_dir.exists():
        LOGGER.info("resume=false：本轮不删除既有 judge_cache，成功裁判缓存仍可安全复用。")
    needs_judge = (evaluation["prima"]["enabled"] and evaluation["prima"]["factScore"]
                   or not evaluation["prima"]["enabled"] and evaluation["judgeScope"] != "none")
    judge = None
    judge_identity = None
    summaries, by_label = [], {}
    try:
        if needs_judge and any(
                (evaluation["prima"]["enabled"] and evaluation["prima"]["factScore"]
                 and any(question_type_for_sample(row) == "open_ended" for row in records))
                or (not evaluation["prima"]["enabled"] and evaluation["judgeScope"] != "none")
                for _, _, records in entries):
            judge = _TextJudgeModel(evaluation["judgeModel"], evaluation["judgeModelPath"])
            judge_identity = {"model_type": evaluation["judgeModel"], "model_path": evaluation["judgeModelPath"],
                              "revision": "local", "software": {name: importlib.metadata.version(name)
                              for name in ("transformers", "torch") if _has_package(name)}}
        _write_json(output / "run_config.json", {"name": task.get("name"), "generations": [str(path) for _, path, _ in entries],
                                                   "labels": [label for label, _, _ in entries], "evaluation": evaluation,
                                                   "seed": seed, "resume": resume, "status": "running"})
        for label, path, records in entries:
            # JSONL 通常自带 answer/options；若 datasetPath 给出则仅缺字段时回源。
            dataset_path = cli_dataset_path if cli_dataset_path is not None else task.get("datasetPath")
            if dataset_path is not None:
                candidate = Path(dataset_path).expanduser()
                dataset_path = str((config_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
                records = _load_dataset_fallback(dataset_path, [path], records)
            sidecar = json.loads((path.parent / "summary.json").read_text(encoding="utf-8-sig")) \
                if (path.parent / "summary.json").is_file() else {}
            summary = _save_one(output, label, path, records, evaluation, seed, judge,
                                judge_identity, cache_dir, sidecar)
            summaries.append(summary)
            by_label[label] = _read_jsonl(output / label / "predictions.jsonl")
        report = _write_reports(output, summaries, by_label, evaluation, seed)
        manifest = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        manifest.update(status="needs_review" if any(s["judge_failures"] for s in summaries) else "complete",
                        report_status="complete", summaries_count=len(summaries))
        _write_json(output / "run_config.json", manifest)
        print(f"评估完成：{output}")
        return 2 if any(summary["judge_failures"] for summary in summaries) else 0
    finally:
        if judge is not None:
            judge.close()
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler):
                logging.getLogger().removeHandler(handler)
                handler.close()


def _has_package(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只评分已有 generations.jsonl，不运行推理。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="评估 JSON 配置路径。")
    parser.add_argument("--generations", nargs="+", help="generations.jsonl 文件或包含该文件的目录。")
    parser.add_argument("--output_dir", help="覆盖 JSON 中 outputDir。")
    parser.add_argument("--dataset_path", help="缺少答案/选项元数据时回源数据集的路径。")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None,
                        help="复用输出目录下的 judge_cache。")
    parser.add_argument("--no_resume", dest="resume", action="store_false", help="不复用本轮评分结果。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        tasks = _load_config(config_path)
        statuses = [_run_task(task, config_path, args.generations, args.output_dir, args.resume,
                              args.dataset_path) for task in tasks]
        return max(statuses, default=0)
    except Exception as exc:
        print(f"配置或评估失败：{exc}", file=sys.stderr)
        LOGGER.exception("评估失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
