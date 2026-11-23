# -*- coding: utf-8 -*-
"""A/B 对比裁判路径：evaluate_rag_metrics._TextJudgeModel vs inference_rag.QwenGenerator。

对同一批真实开放题构造与 atomic_factscore 完全相同的事实拆分提示词，
分别用两条管线调用本地裁判 Qwen3-VL-2B-Instruct，对比：
  1) 渲染出的聊天模板结构（开头/结尾）
  2) 输入 token 数
  3) 裁判原始输出与生成统计

输出写入 --out 指定文件（默认与本脚本同目录 judge_ab_result.txt），
同时打印到 stdout。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = str(PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
GENERATIONS = (PROJECT_ROOT / "output" / "Inference" / "evaluation_prima_adaptive2"
               / "Qwen_lora" / "no_rag" / "generations.jsonl")
MAX_NEW_TOKENS = 1024


class Tee:
    def __init__(self, path: Path):
        self.stream = open(path, "w", encoding="utf-8")

    def write(self, text: str):
        sys.stdout.write(text)
        self.stream.write(text)

    def flush(self):
        sys.stdout.flush()
        self.stream.flush()


def load_samples(num: int) -> list[tuple[str, str, str, str]]:
    from script.lib.atomic_factscore import EXTRACT_PROMPT, _canonical
    from script.lib.answer_metrics import question_type_for_sample

    samples: list[tuple[str, str, str, str]] = []
    with GENERATIONS.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if len(samples) >= num:
                break
            row = json.loads(line)
            if question_type_for_sample(row) != "open_ended":
                continue
            question = (row.get("question") or "").strip()
            prediction = (row.get("prediction") or "").strip()
            if not question or not prediction:
                continue
            payload = {"question": question, "prediction": prediction}
            prompt = EXTRACT_PROMPT + _canonical(payload)
            samples.append((str(row["id"]), prompt, question, prediction))
    return samples


def run_path_a(samples, log):
    from script.evaluate_rag_metrics import _TextJudgeModel

    log("\n========== PATH A: _TextJudgeModel (evaluate_rag_metrics.py) ==========")
    judge = _TextJudgeModel("Qwen3-VL-2B-Instruct", MODEL_PATH, max_input_tokens=16384)
    log(f"PATH A system_prompt: {judge.system_prompt!r}")
    try:
        for rid, prompt, question, prediction in samples:
            messages = [{"role": "system", "content": judge.system_prompt},
                        {"role": "user", "content": prompt}]
            try:
                rendered = judge.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                log(f"\n[{rid}] rendered head: {rendered[:160]!r}")
                log(f"[{rid}] rendered tail: {rendered[-120:]!r}")
            except Exception as exc:
                log(f"[{rid}] render failed: {exc!r}")
            started = time.perf_counter()
            try:
                raw = judge.text(prompt, max_new_tokens=MAX_NEW_TOKENS)
                info = dict(judge.last_generation_info)
                log(f"[{rid}] A raw={raw[:600]!r}")
                log(f"[{rid}] A info={info} elapsed={time.perf_counter() - started:.1f}s")
            except Exception as exc:
                log(f"[{rid}] A exception: {type(exc).__name__}: {exc}")
    finally:
        judge.close()


def run_path_b(samples, log):
    from script import inference_rag

    log("\n========== PATH B: QwenGenerator (inference_rag.py) ==========")
    log(f"PATH B FACT_SYSTEM: {inference_rag.FACT_SYSTEM!r}")
    args = argparse.Namespace(
        model_path=MODEL_PATH, model_revision="main", attn_implementation="sdpa",
        dtype="auto", device="auto", seed=42, max_input_tokens=16384,
        lora_path=None, text_only=False, min_pixels=4096, max_pixels=262144,
        model="Qwen3-VL-2B-Instruct",
    )
    judge = inference_rag.QwenGenerator(args)
    try:
        for rid, prompt, question, prediction in samples:
            messages = [{"role": "system",
                         "content": [{"type": "text", "text": inference_rag.FACT_SYSTEM}]},
                        {"role": "user",
                         "content": [{"type": "text", "text": prompt}]}]
            try:
                rendered = judge.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                log(f"\n[{rid}] rendered head: {rendered[:160]!r}")
                log(f"[{rid}] rendered tail: {rendered[-120:]!r}")
            except Exception as exc:
                log(f"[{rid}] render failed: {exc!r}")
            started = time.perf_counter()
            try:
                raw = judge.text(prompt, max_new_tokens=MAX_NEW_TOKENS)
                info = dict(judge.last_generation_info)
                log(f"[{rid}] B raw={raw[:600]!r}")
                log(f"[{rid}] B info={info} elapsed={time.perf_counter() - started:.1f}s")
            except Exception as exc:
                log(f"[{rid}] B exception: {type(exc).__name__}: {exc}")
    finally:
        judge.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3, help="测试的开放题数量")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(__file__).with_name("judge_ab_result.txt")
    tee = Tee(out_path)
    log = lambda text="": tee.write(text + "\n")

    import torch
    import transformers
    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    log(f"transformers={transformers.__version__}")
    log(f"model_path={MODEL_PATH}")
    log(f"out={out_path}")

    samples = load_samples(args.num)
    log(f"loaded {len(samples)} open-ended samples:")
    for rid, prompt, question, prediction in samples:
        log(f"  - {rid}")
        log(f"      question : {question[:100]!r}")
        log(f"      prediction: {prediction[:100]!r}")

    run_path_a(samples, log)
    run_path_b(samples, log)
    log("\nDONE")


if __name__ == "__main__":
    main()
