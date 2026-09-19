"""按固定测试编号或官方测试划分准备评估数据，不加载模型或图片。"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import random
import re

from .mira_eval_data import load_mira_dataset

from inferenceValid.embed_mira_chroma import image_path, iter_samples, json_text, parse_image_paths


LOGGER = logging.getLogger("rag_eval")
MIRA_ID = re.compile(
    r"mira:(train|validation|test):(0|[1-9][0-9]*):"
    r"(open_ended|closed_ended|single_choice|multiple_choice):(0|[1-9][0-9]*)"
)


def _id_key(value) -> str:
    """和评估缓存约定一致：整数 1 与字符串 '1' 是同一个编号。"""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("数据编号必须是非空字符串或整数，不能是布尔值。")
    key = str(value).strip()
    if not key:
        raise ValueError("数据编号不能为空。")
    if key.startswith("mira:") and MIRA_ID.fullmatch(key) is None:
        raise ValueError(f"MIRA 数据编号格式不正确：{key}")
    return key


def _validate_ids(values, field: str, *, allow_empty: bool = False) -> list:
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{field} 必须是{'可为空的' if allow_empty else '非空'}编号数组。")
    seen = set()
    for value in values:
        key = _id_key(value)
        if key in seen:
            raise ValueError(f"{field} 包含重复编号：{key}")
        seen.add(key)
    return values


def load_test_ids(path: str | Path) -> list:
    """支持 {test_ids: [...], train_ids: [...]} 或纯编号数组，拒绝训练/测试重叠。"""
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        values = _validate_ids(data.get("test_ids"), "test_ids")
        training = _validate_ids(data.get("train_ids", []), "train_ids", allow_empty=True)
        overlap = {_id_key(value) for value in values} & {_id_key(value) for value in training}
        if overlap:
            raise ValueError(f"datasetFilter 的 train_ids 与 test_ids 重叠：{sorted(overlap)[:5]}")
    else:
        values = _validate_ids(data, "datasetFilter")
    return list(values)


def _record(data_root: Path, identifier: str, question, answer, options, paths,
            image_cache: dict[str, str]) -> dict:
    """仅把问题、选项及图片送给模型；答案单独作为参考，不混入图片说明。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"测试问答 {identifier} 缺少有效 question；请修复数据或筛选列表。")
    if answer in (None, "", {}, []) or (isinstance(answer, str) and not answer.strip()):
        raise ValueError(f"测试问答 {identifier} 缺少有效 answer；请修复数据或筛选列表。")
    question = question.strip()
    if options:
        question += "\nOptions: " + json_text(options)
    images = []
    for name in paths:
        if name not in image_cache:
            image_cache[name] = os.path.abspath(image_path(data_root, name))
        images.append(image_cache[name])
    return {"id": identifier, "question": question, "images": images,
            "reference": answer if isinstance(answer, str) else json_text(answer)}


def _load_selected_mira(data_root: Path, identifiers: list) -> list[dict]:
    """CSV 流式扫描，只解码指定行；保持筛选文件顺序，不展开整个训练集。"""
    targets = {}
    for value in identifiers:
        key = _id_key(value)
        match = MIRA_ID.fullmatch(key)
        if match is None:
            raise ValueError(f"MIRA 数据目录要求完整的 mira:... 编号：{key}")
        split, row_index, category, qa_index = match.groups()
        targets.setdefault(split, {}).setdefault(int(row_index), []).append(
            (key, category, int(qa_index)))
    found, image_cache = {}, {}
    csv.field_size_limit(64 * 1024 * 1024)
    for split, rows in targets.items():
        source = data_root / f"{split}.csv"
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not {"image_path", "vqa_json"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{source} 缺少 image_path 或 vqa_json 列。")
            remaining = set(rows)
            for row_index, row in enumerate(reader):
                if row_index not in remaining:
                    continue
                try:
                    if None in row or row.get("vqa_json") is None:
                        raise ValueError("CSV 字段数量不正确。")
                    questions = json.loads(row["vqa_json"])
                    if not isinstance(questions, dict):
                        raise ValueError("vqa_json 必须为对象。")
                    for key, category, qa_index in rows[row_index]:
                        category_items = questions.get(category)
                        if not isinstance(category_items, list) or qa_index >= len(category_items):
                            raise ValueError(f"筛选编号不存在：{key}")
                        qa = category_items[qa_index]
                        if not isinstance(qa, dict):
                            raise ValueError(f"问答不是对象：{key}")
                        paths = parse_image_paths(qa.get("image_paths", qa.get("image_path", row["image_path"])))
                        found[key] = _record(data_root, key, qa.get("question"), qa.get("answer"),
                                             qa.get("options"), paths, image_cache)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{source.name} 数据行索引 {row_index}：{exc}") from exc
                remaining.remove(row_index)
                if not remaining:
                    break
    missing = [_id_key(value) for value in identifiers if _id_key(value) not in found]
    if missing:
        raise ValueError(f"datasetFilter 中有 {len(missing)} 个编号无法在数据集中找到：{missing[:5]}")
    return [found[_id_key(value)] for value in identifiers]


def _load_official_mira_test(data_root: Path) -> list[dict]:
    """datasetFilter=null 时严格使用官方 test.csv，不从 train.csv 重新随机切分。"""
    rows, image_cache = [], {}
    for sample in iter_samples(data_root, "test"):
        rows.append(_record(data_root, sample.id, sample.question, sample.answer,
                            sample.options, sample.images, image_cache))
    if not rows:
        raise ValueError("MIRA 官方 test.csv 中没有问答。")
    return rows


def _load_jsonl(path: Path) -> list[dict]:
    """JSONL 使用 id/question/images/reference 字段，支持无图片的纯文本数据。"""
    rows, seen, image_cache = [], set(), {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("每行必须是 JSON 对象。")
                missing = {"id", "question", "images", "reference"} - row.keys()
                if missing:
                    raise ValueError(f"缺少字段：{sorted(missing)}")
                key = _id_key(row["id"])
                if key in seen:
                    raise ValueError(f"重复数据编号：{key}")
                seen.add(key)
                for field in ("question", "reference"):
                    if not isinstance(row[field], str) or not row[field].strip():
                        raise ValueError(f"{field} 必须为非空文本。")
                if not isinstance(row["images"], list):
                    raise ValueError("images 必须是图片路径数组。")
                images = []
                for raw in row["images"]:
                    if not isinstance(raw, str) or not raw.strip():
                        raise ValueError("图片路径必须为非空字符串。")
                    if raw not in image_cache:
                        image = Path(raw).expanduser()
                        image_cache[raw] = str((image if image.is_absolute() else path.parent / image).resolve())
                    images.append(image_cache[raw])
                rows.append({"id": row["id"], "question": row["question"],
                             "images": images, "reference": row["reference"]})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError("JSONL 数据集为空。")
    return rows


def _image_keys(paths) -> set[str]:
    return {os.path.normcase(os.path.realpath(path)) for path in paths}


def prepare_evaluation_data(args, configs) -> tuple[list[dict], list[dict], dict]:
    """准备四组评估共享的数据；只有启用 RAG 时才读取知识库。"""
    data_path = Path(args.dataset_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集：{data_path}")
    limit = getattr(args, "N", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--N 必须为正整数；不指定时评估所选测试集的全部问答。")
    knowledge_limit = getattr(args, "knowledge_size", None)
    if knowledge_limit is not None and (type(knowledge_limit) is not int or knowledge_limit < 1):
        raise ValueError("--knowledge_size 必须为正整数。")
    filter_path = getattr(args, "dataset_filter", None)
    needs_knowledge = any(config["use_rag"] for config in configs)
    original_jsonl = None
    if filter_path is not None:
        reserved_ids = load_test_ids(filter_path)
        selected_ids = reserved_ids if limit is None else reserved_ids[:limit]
        if data_path.is_dir():
            test = _load_selected_mira(data_path, selected_ids)
        else:
            original_jsonl = _load_jsonl(data_path)
            by_id = {_id_key(row["id"]): row for row in original_jsonl}
            missing = [_id_key(value) for value in reserved_ids if _id_key(value) not in by_id]
            if missing:
                raise ValueError(f"datasetFilter 中编号不在 JSONL 数据集中：{missing[:5]}")
            test = [by_id[_id_key(value)] for value in selected_ids]
        selection, ids_source = "filter", str(Path(filter_path).expanduser().resolve())
    elif data_path.is_dir():
        test = _load_official_mira_test(data_path)
        reserved_ids = [row["id"] for row in test]
        selection, ids_source = "official_test", str(data_path / "test.csv")
    else:
        original_jsonl = _load_jsonl(data_path)
        test = original_jsonl
        reserved_ids = [row["id"] for row in test]
        selection, ids_source = "jsonl_all", str(data_path)
    requested_count = len(reserved_ids)
    loaded_ids = {_id_key(row["id"]) for row in test}
    if limit is not None:
        if limit > requested_count:
            LOGGER.warning("--N=%d 大于固定测试集的 %d 条，使用全部测试问答。", limit, requested_count)
        test = test[:limit]

    # 限量调试不会把其余指定测试题移进知识库，以防出现测试答案泄漏。
    reserved_keys = {_id_key(value) for value in reserved_ids}
    knowledge, excluded_images, candidate_count = [], 0, 0
    if original_jsonl is not None:
        loaded_ids.update(_id_key(row["id"]) for row in original_jsonl)
    if needs_knowledge:
        knowledge_source = (load_mira_dataset(data_path, getattr(args, "source_splits", ("train",)))
                            if data_path.is_dir() else original_jsonl)
        loaded_ids.update(_id_key(row["id"]) for row in knowledge_source)
        test_images = _image_keys(image for row in test for image in row["images"]) \
            if getattr(args, "exclude_shared_images", False) else set()
        for row in knowledge_source:
            if _id_key(row["id"]) in reserved_keys:
                continue
            if test_images and _image_keys(row["images"]) & test_images:
                excluded_images += 1
                continue
            knowledge.append(row)
        candidate_count = len(knowledge)
        if knowledge_limit is not None and knowledge_limit < candidate_count:
            indices = sorted(random.Random(args.seed).sample(range(candidate_count), knowledge_limit))
            knowledge = [knowledge[index] for index in indices]
    metadata = {
        "test_selection": selection, "requested_test_count": requested_count,
        "reserved_test_ids": list(reserved_ids),
        "test_ids_source": ids_source, "loaded_samples": len(loaded_ids),
        "knowledge_candidates": candidate_count, "excluded_shared_image_samples": excluded_images,
        "excluded_by_knowledge_limit": candidate_count - len(knowledge),
    }
    LOGGER.info("测试数据来源=%s，固定测试集=%d 条，本次评估=%d 条，知识库=%d 条。",
                selection, requested_count, len(test), len(knowledge))
    return test, knowledge, metadata
