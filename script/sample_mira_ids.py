"""检查 MIRA 关键词/向量覆盖情况，并生成训练集和测试集问答编号。

先修改本脚本同目录的 sample_mira_ids.json，再直接运行，无须输入抽样参数：
    python script/sample_mira_ids.py
或在 script 目录中运行：
    python sample_mira_ids.py

JSON 参数说明（路径可为绝对路径，相对路径以 JSON 所在目录为基准）：
    train_count / test_count：训练集 / 测试集问答总数。
    train_ratios / test_ratios：各题型的相对比例，四个键均须提供；允许填 0，
        程序会自动归一化，0.25/0.25/0.25/0.25 与 25/25/25/25 等效。
        按最大余数法换算为整数配额，保证四类配额之和等于总数。旧的
        train_counts / test_counts 整数配额方式仍兼容。
    data_root：MIRA 数据目录。
    keywords_file：关键词文件；设为 null 时不做关键词过滤，所有问答都视为关键词命中。
    db_dir：已有 Chroma 目录；null 使用数据目录旁的 MIRA-chroma。
    collection：Chroma 集合名。
    splits：抽样来源列表，支持 train、validation、test；默认只从 train.csv 抽样。
    seed：随机种子，相同数据和参数下可复现抽样。
    exclude_shared_images：true 时额外排除与测试题共用图片的训练题。
    output：生成的 ID 清单，默认项目 output/mira_split_ids.json，不能覆盖输入配置。
    check_config：true 时只显示解析后的参数，不扫描数据，也不写入抽样结果。
    _说明：可选的中文说明文本，不参与抽样。

默认配置以脚本位置定位，与运行时工作目录无关。为兼容旧调用仍保留命令行选项，
显式命令行参数优先于 JSON；正常使用只需编辑 JSON。重复运行会更新 output 指定的清单，
如需保留已有训练/测试划分，请在 JSON 中为 output 设置不同文件名。

数量以一个问答对为单位。先选测试题，再选训练题，优先级依次为：
关键词与向量均匹配、仅有向量、仅匹配关键词、两者均不满足。
会核查所有已有 train/validation/test CSV，但只从 splits 指定的来源抽样。
指定关键词文件时，匹配问题、选项、答案和额外问答文本，不匹配共用图片标题；
忽略大小写，使用词边界并允许短语内不同空白，括号中的缩写也可匹配。每道题只
统计一次。keywords_file 为 null 时跳过关键词文件读取，所有问答均算作匹配。
只使用标准库，并只读访问 Chroma 元数据；向量覆盖表示存在对应编号，不代表向量质量。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "inferenceValid"))
from embed_mira_chroma import DEFAULT_DATA_ROOT, Sample, image_path, iter_samples


LOGGER = logging.getLogger("sample_mira_ids")
SPLITS = ("train", "validation", "test")
DEFAULT_COLLECTION = "mira_qwen3_vl_embedding"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_suffix(".json")
KEYWORDS_FILENAME = "cardiovascular_ai_keywords2.txt"
PRIORITY_ORDER = ("keyword_and_embedded", "embedded_only", "keyword_only", "neither")
QUESTION_TYPES = ("open_ended", "closed_ended", "single_choice", "multiple_choice")
QUESTION_TYPE_LABELS = {"single_choice": "单选题", "multiple_choice": "多选题",
                        "open_ended": "开放式问题", "closed_ended": "封闭式问题"}
T = TypeVar("T")


def load_keywords(path: Path) -> list[str]:
    """Read one keyword per UTF-8 line, allowing a BOM, blanks and # comments."""
    keywords, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        term = " ".join(line.split())
        if not term or term.startswith("#") or term.casefold() in seen:
            continue
        keywords.append(term)
        seen.add(term.casefold())
    if not keywords:
        raise ValueError(f"No keywords found in {path}")
    return keywords


def compile_keywords(keywords: list[str]) -> re.Pattern[str]:
    aliases = set()
    for term in keywords:
        abbreviation = re.fullmatch(r"(.+?)\s*\(([A-Za-z][A-Za-z0-9]*)\)", term)
        alternatives = abbreviation.groups() if abbreviation else (term,)
        aliases.update(" ".join(alias.split()).casefold() for alias in alternatives if alias.strip())
    if not aliases:
        raise ValueError("At least one nonempty keyword is required.")
    phrases = [r"\s+".join(re.escape(word) for word in alias.split()) for alias in sorted(aliases)]
    return re.compile(r"(?<!\w)(?:" + "|".join(phrases) + r")(?!\w)", re.IGNORECASE)


def text_values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from text_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from text_values(nested)


def matches_keywords(sample: Sample, pattern: re.Pattern[str]) -> bool:
    fields = (sample.question, sample.options, sample.answer, sample.extra)
    return any(pattern.search(text) is not None for text in text_values(fields))


def load_embedded_ids(db_dir: Path, collection: str) -> set[str]:
    """Read a snapshot of Chroma's stored IDs without initializing its indexes."""
    database = Path(db_dir).expanduser().resolve() / "chroma.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"Existing Chroma database not found: {database}")
    LOGGER.info("Reading stored IDs from %s / %s (read-only)", database, collection)
    try:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            collections = connection.execute(
                "SELECT c.id FROM collections c JOIN databases d ON c.database_id = d.id "
                "WHERE c.name = ? AND d.name = 'default_database' AND d.tenant_id = 'default_tenant'",
                (collection,),
            ).fetchall()
            if len(collections) != 1:
                raise ValueError(f"Chroma collection {collection!r} not found or ambiguous in {db_dir}")
            segments = connection.execute(
                "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA' "
                "AND type = 'urn:chroma:segment/metadata/sqlite'", (collections[0][0],),
            ).fetchall()
            if len(segments) != 1:
                raise ValueError(f"Expected one Chroma SQLite metadata segment for {collection!r}.")
            # Chroma 1.x stores record IDs in the metadata segment's covering index.
            rows = connection.execute("SELECT embedding_id FROM embeddings WHERE segment_id = ?",
                                      (segments[0][0],))
            embedded_ids: set[str] = set()
            for (sample_id,) in rows:
                if not isinstance(sample_id, str) or not sample_id or sample_id in embedded_ids:
                    raise ValueError("Chroma contains invalid or duplicate stored IDs.")
                embedded_ids.add(sample_id)
    except sqlite3.DatabaseError as error:
        raise ValueError(
            f"Cannot read Chroma SQLite metadata: {error}. "
            "This reader expects the local Chroma 1.x database schema.") from error
    LOGGER.info("Loaded %d stored Chroma IDs", len(embedded_ids))
    return embedded_ids


def priority_sample(items: Iterable[T], count: int, rng: random.Random,
                    priority: Callable[[T], int]) -> tuple[list[T], int, list[int]]:
    """Uniformly sample each priority tier using bounded reservoirs."""
    reservoirs: list[list[T]] = [[] for _ in PRIORITY_ORDER]
    counts = [0] * len(PRIORITY_ORDER)
    for item in items:
        tier = priority(item)
        counts[tier] += 1
        selected = reservoirs[tier]
        if len(selected) < count:
            selected.append(item)
        else:
            position = rng.randrange(counts[tier])
            if position < count:
                selected[position] = item
    for selected in reservoirs:
        rng.shuffle(selected)
    selected = [item for reservoir in reservoirs for item in reservoir][:count]
    rng.shuffle(selected)
    return selected, sum(counts), counts


def priority_sample_by_type(
    items: Iterable[tuple[str, list[str], str]], quotas: dict[str, int],
    rng: random.Random, priority: Callable[[tuple[str, list[str], str]], int],
) -> tuple[list[tuple[str, list[str], str]], dict[str, int], dict[str, list[int]]]:
    """一次扫描按题型维护四级优先级蓄水池，不把某一题型补成另一题型。"""
    reservoirs = {category: [[] for _ in PRIORITY_ORDER] for category in QUESTION_TYPES}
    counts = {category: 0 for category in QUESTION_TYPES}
    priority_counts = {category: [0] * len(PRIORITY_ORDER) for category in QUESTION_TYPES}
    for item in items:
        _, _, category = item
        if category not in quotas:
            raise ValueError(f"未知题型：{category}")
        counts[category] += 1
        tier = priority(item)
        priority_counts[category][tier] += 1
        quota = quotas[category]
        if quota <= 0:
            continue
        selected = reservoirs[category][tier]
        if len(selected) < quota:
            selected.append(item)
        else:
            position = rng.randrange(priority_counts[category][tier])
            if position < quota:
                selected[position] = item
    result = []
    for category in QUESTION_TYPES:
        # 层内随机，按优先级依次取足；不能先混洗四层而破坏关键词/向量优先级。
        for reservoir in reservoirs[category]:
            rng.shuffle(reservoir)
        chosen = [item for reservoir in reservoirs[category] for item in reservoir]
        result.extend(chosen[:quotas[category]])
    rng.shuffle(result)
    return result, counts, priority_counts


def iter_records(data_root: Path, splits: list[str]) -> Iterator[tuple[str, list[str], str]]:
    for split in splits:
        for sample in iter_samples(data_root, split):
            yield sample.id, sample.images, sample.category


def image_keys(data_root: Path, names: list[str]) -> set[str]:
    return {os.path.normcase(os.path.abspath(image_path(data_root, name))) for name in names}


def sample_dataset(data_root: Path, train_count: int | None = None, test_count: int | None = None, *,
                   splits: list[str] | None = None, seed: int = 42,
                   exclude_shared_images: bool = False, db_dir: Path | None = None,
                   collection: str = DEFAULT_COLLECTION, keywords_file: Path | None = None,
                   all_samples_match_keywords: bool = False,
                   train_counts: dict[str, int] | None = None,
                   test_counts: dict[str, int] | None = None) -> dict:
    """Audit all splits and prefer keyword-matching QAs with stored embeddings."""
    if (train_counts is None) != (test_counts is None):
        raise ValueError("train_counts 和 test_counts 必须同时提供。")
    category_mode = train_counts is not None
    if category_mode:
        train_counts = validate_type_counts(train_counts, "train_counts")
        test_counts = validate_type_counts(test_counts, "test_counts")
        train_count, test_count = sum(train_counts.values()), sum(test_counts.values())
    elif (type(train_count) is not int or type(test_count) is not int
          or train_count <= 0 or test_count <= 0):
        raise ValueError("train-count and test-count must be positive integers.")
    splits = ["train"] if splits is None else list(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("splits must contain unique values from train, validation, test.")
    data_root = Path(data_root).expanduser().resolve()
    for split in splits:
        source = data_root / f"{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {source}")

    if all_samples_match_keywords:
        if keywords_file is not None:
            raise ValueError("全量关键词命中模式不能同时指定 keywords_file。")
        keywords = []
        keyword_pattern = None
        LOGGER.info("keywords_file 为 null：所有问答均视为关键词命中。")
    else:
        keywords_file = (Path(keywords_file) if keywords_file is not None
                         else data_root.parent / "MIRA_myConfig" / KEYWORDS_FILENAME).expanduser().resolve()
        keywords = load_keywords(keywords_file)
        keyword_pattern = compile_keywords(keywords)
        LOGGER.info("Loaded %d keywords from %s", len(keywords), keywords_file)
    db_dir = (Path(db_dir) if db_dir is not None else data_root.parent / "MIRA-chroma").expanduser().resolve()
    embedded_ids = load_embedded_ids(db_dir, collection)
    coverage_splits = [split for split in SPLITS if (data_root / f"{split}.csv").is_file()]
    split_counts = {}
    keyword_split_counts = {}
    keyword_ids: set[str] = set()
    question_type_counts = {split: {category: 0 for category in QUESTION_TYPES}
                            for split in coverage_splits}
    sampling_pool_type_counts = {category: 0 for category in QUESTION_TYPES}

    def sample_priority(sample_id: str) -> int:
        if sample_id in embedded_ids:
            return 0 if sample_id in keyword_ids else 1
        return 2 if sample_id in keyword_ids else 3

    def audited_records() -> Iterator[tuple[str, list[str], str]]:
        for split in coverage_splits:
            LOGGER.info("Checking all QA IDs in %s.csv", split)
            counts = {"total_samples": 0, "embedded_samples": 0, "missing_samples": 0}
            keyword_counts = {"matched_samples": 0, "matched_embedded_samples": 0}
            for sample in iter_samples(data_root, split):
                embedded = sample.id in embedded_ids
                matched = all_samples_match_keywords or matches_keywords(sample, keyword_pattern)
                counts["total_samples"] += 1
                counts["embedded_samples"] += embedded
                keyword_counts["matched_samples"] += matched
                keyword_counts["matched_embedded_samples"] += matched and embedded
                question_type_counts[split][sample.category] += 1
                if split in splits:
                    sampling_pool_type_counts[sample.category] += 1
                    if matched:
                        keyword_ids.add(sample.id)
                    yield sample.id, sample.images, sample.category
            counts["missing_samples"] = counts["total_samples"] - counts["embedded_samples"]
            split_counts[split] = counts
            keyword_split_counts[split] = keyword_counts

    rng = random.Random(seed)
    LOGGER.info("Sampling %d test QAs from %s (%s)", test_count, data_root, ", ".join(splits))
    if category_mode:
        tests, type_candidates, type_priority_counts = priority_sample_by_type(
            audited_records(), test_counts, rng, lambda item: sample_priority(item[0]))
        total = sum(type_candidates.values())
        priority_counts = [sum(type_priority_counts[c][i] for c in QUESTION_TYPES)
                           for i in range(len(PRIORITY_ORDER))]
    else:
        tests, total, priority_counts = priority_sample(
            audited_records(), test_count, rng, lambda item: sample_priority(item[0]))
        type_candidates = sampling_pool_type_counts.copy()
    embedded_samples = priority_counts[0] + priority_counts[1]
    dataset_total = sum(counts["total_samples"] for counts in split_counts.values())
    dataset_embedded = sum(counts["embedded_samples"] for counts in split_counts.values())
    coverage = {
        "total_samples": dataset_total,
        "embedded_samples": dataset_embedded,
        "missing_samples": dataset_total - dataset_embedded,
        "all_embedded": dataset_total > 0 and dataset_total == dataset_embedded,
        "collection_count": len(embedded_ids),
        "unmatched_chroma_ids": len(embedded_ids) - dataset_embedded,
        "splits": split_counts,
    }
    keyword_coverage = {
        "keyword_count": len(keywords),
        "matched_samples": sum(counts["matched_samples"] for counts in keyword_split_counts.values()),
        "matched_embedded_samples": sum(counts["matched_embedded_samples"]
                                        for counts in keyword_split_counts.values()),
        "splits": keyword_split_counts,
    }
    print(f"Dataset groups (all existing CSV splits): {dataset_total:,}", flush=True)
    print(f"Groups with stored embeddings: {dataset_embedded:,}", flush=True)
    print(f"Groups without embeddings: {coverage['missing_samples']:,}", flush=True)
    print(f"All dataset groups embedded: {'yes' if coverage['all_embedded'] else 'no'}", flush=True)
    print(f"Groups matching at least one keyword: {keyword_coverage['matched_samples']:,}", flush=True)
    print(f"Keyword-matching groups with stored embeddings: "
          f"{keyword_coverage['matched_embedded_samples']:,}", flush=True)
    for split, counts in split_counts.items():
        keyword_counts = keyword_split_counts[split]
        print(f"  {split}: total={counts['total_samples']:,}, "
              f"embedded={counts['embedded_samples']:,}, missing={counts['missing_samples']:,}, "
              f"keyword={keyword_counts['matched_samples']:,}, "
              f"keyword+embedded={keyword_counts['matched_embedded_samples']:,}", flush=True)
    print(f"Chroma IDs not matched to this dataset: {coverage['unmatched_chroma_ids']:,}", flush=True)
    print(f"Sampling pool ({', '.join(splits)}): {total:,}; embedded={embedded_samples:,}; "
          f"keyword={len(keyword_ids):,}; keyword+embedded={priority_counts[0]:,}", flush=True)
    print("Sampling priority: " + " > ".join(PRIORITY_ORDER), flush=True)
    dataset_type_counts = {category: sum(question_type_counts[split][category] for split in coverage_splits)
                           for category in QUESTION_TYPES}
    print("题型统计（按问答对计数；全数据包含所有现存 CSV，抽样池只包含 splits 指定来源）：", flush=True)
    for category, label in QUESTION_TYPE_LABELS.items():
        print(f"  {label} ({category}): 全数据={dataset_type_counts[category]:,}，"
              f"抽样池={sampling_pool_type_counts[category]:,}", flush=True)
    if category_mode:
        for category, label in QUESTION_TYPE_LABELS.items():
            print(f"  计划抽取{label}：训练={train_counts[category]:,}，测试={test_counts[category]:,}", flush=True)
        shortages = [f"{QUESTION_TYPE_LABELS[c]} ({c}): 需要训练 {train_counts[c]} + "
                     f"测试 {test_counts[c]}，可用 {type_candidates[c]}"
                     for c in QUESTION_TYPES if type_candidates[c] < train_counts[c] + test_counts[c]]
        if shortages:
            raise ValueError("抽样池题型数量不足：" + "；".join(shortages))
    if total < train_count + test_count:
        raise ValueError(
            f"Requested {train_count} train + {test_count} test QAs, "
            f"but the selected source splits contain only {total} QAs.")
    test_ids = [sample_id for sample_id, _, _ in tests]
    test_id_set = set(test_ids)
    test_images = set()
    if exclude_shared_images:
        for _, names, _ in tests:
            test_images.update(image_keys(data_root, names))

    # A second streaming pass avoids retaining the million-record population.
    def eligible_train_ids() -> Iterator[tuple[str, list[str], str]]:
        for sample_id, names, category in iter_records(data_root, splits):
            if sample_id in test_id_set:
                continue
            if test_images and image_keys(data_root, names) & test_images:
                continue
            yield sample_id, names, category

    LOGGER.info("Sampling %d training QAs from the remaining population", train_count)
    if category_mode:
        train_records, train_type_candidates, _ = priority_sample_by_type(
            eligible_train_ids(), train_counts, rng, lambda item: sample_priority(item[0]))
        train_candidates = sum(train_type_candidates.values())
        train_shortages = [f"{QUESTION_TYPE_LABELS[c]} ({c}): 需要 {train_counts[c]}，"
                           f"剩余可用 {train_type_candidates[c]}" for c in QUESTION_TYPES
                           if train_type_candidates[c] < train_counts[c]]
        if train_shortages:
            raise ValueError("训练集题型数量不足（排除测试题/共图后）：" + "；".join(train_shortages))
    else:
        train_records, train_candidates, _ = priority_sample(
            eligible_train_ids(), train_count, rng, lambda item: sample_priority(item[0]))
        if train_candidates < train_count:
            raise ValueError(
                f"Requested {train_count} train QAs, but only {train_candidates} remain "
                "after excluding test IDs and shared images. Reduce the requested counts.")
    train_ids = [sample_id for sample_id, _, _ in train_records]
    selected_embedding_counts = {
        "train": sum(sample_id in embedded_ids for sample_id in train_ids),
        "test": sum(sample_id in embedded_ids for sample_id in test_ids),
    }
    selected_keyword_counts = {
        label: sum(sample_id in keyword_ids for sample_id in ids)
        for label, ids in (("train", train_ids), ("test", test_ids))
    }
    selected_keyword_embedding_counts = {
        label: sum(sample_id in keyword_ids and sample_id in embedded_ids for sample_id in ids)
        for label, ids in (("train", train_ids), ("test", test_ids))
    }
    selected_type_counts = {
        label: {category: sum(sample_id.split(":")[3] == category for sample_id in ids)
                for category in QUESTION_TYPES}
        for label, ids in (("train", train_ids), ("test", test_ids))
    }
    for label, ids in (("train", train_ids), ("test", test_ids)):
        matched = selected_embedding_counts[label]
        print(f"Selected {label}: total={len(ids):,}, embedded={matched:,}, "
              f"unembedded fallback={len(ids) - matched:,}, keyword={selected_keyword_counts[label]:,}, "
              f"keyword+embedded={selected_keyword_embedding_counts[label]:,}, "
              f"outside-top-priority={len(ids) - selected_keyword_embedding_counts[label]:,}", flush=True)
        print(f"  {'训练集' if label == 'train' else '测试集'}实际题型数量：" + "，".join(
            f"{name}={selected_type_counts[label][category]:,}"
            for category, name in QUESTION_TYPE_LABELS.items()), flush=True)
    return {
        "data_root": str(data_root),
        "db_dir": str(db_dir),
        "chroma_collection": collection,
        "keywords_file": str(keywords_file) if keywords_file is not None else None,
        "keywords": keywords,
        "keyword_matching": ("all_samples" if all_samples_match_keywords
                             else "case_insensitive_whole_phrase_or_abbreviation"),
        "all_samples_match_keywords": all_samples_match_keywords,
        "keyword_text_fields": ["question", "options", "answer", "extra"],
        "sampling_priority": list(PRIORITY_ORDER),
        "source_splits": splits,
        "seed": seed,
        "sample_unit": "question_answer",
        "id_format": "mira:{source_split}:{row_index_0based}:{category}:{qa_index_0based}",
        "exclude_shared_images": exclude_shared_images,
        "total_samples": total,
        "embedded_samples": embedded_samples,
        "keyword_samples": len(keyword_ids),
        "keyword_embedded_samples": priority_counts[0],
        "embedding_coverage": coverage,
        "keyword_coverage": keyword_coverage,
        "question_type_counts": question_type_counts,
        "dataset_question_type_counts": dataset_type_counts,
        "sampling_pool_question_type_counts": sampling_pool_type_counts,
        "selected_question_type_counts": selected_type_counts,
        "requested_question_type_counts": ({"train": train_counts, "test": test_counts}
                                             if category_mode else None),
        "selected_embedding_counts": selected_embedding_counts,
        "selected_keyword_counts": selected_keyword_counts,
        "selected_keyword_embedding_counts": selected_keyword_embedding_counts,
        "train_candidates": train_candidates,
        "excluded_shared_image_samples": total - test_count - train_candidates,
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def validate_type_counts(value: dict[str, int], name: str) -> dict[str, int]:
    """验证四类题型配额；0 表示该集合不抽取此题型。"""
    if not isinstance(value, dict) or set(value) != set(QUESTION_TYPES):
        raise ValueError(f"{name} 必须包含且仅包含四个题型键：{', '.join(QUESTION_TYPES)}。")
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise ValueError(f"{name} 中的数量必须是非负整数。")
    if sum(value.values()) <= 0:
        raise ValueError(f"{name} 的总数量必须大于 0。")
    return {category: value[category] for category in QUESTION_TYPES}


def validate_type_ratios(value: dict[str, int | float], name: str) -> dict[str, float]:
    """验证四类题型的相对权重，并归一化为总和 1 的比例。"""
    if not isinstance(value, dict) or set(value) != set(QUESTION_TYPES):
        raise ValueError(f"{name} 必须包含且仅包含四个题型键：{', '.join(QUESTION_TYPES)}。")
    ratios = {}
    for category in QUESTION_TYPES:
        ratio = value[category]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError(f"{name}.{category} 必须是有限的非负数字。")
        try:
            numeric_ratio = float(ratio)
        except (OverflowError, ValueError):
            raise ValueError(f"{name}.{category} 必须是有限的非负数字。") from None
        if not math.isfinite(numeric_ratio) or numeric_ratio < 0:
            raise ValueError(f"{name}.{category} 必须是有限的非负数字。")
        ratios[category] = numeric_ratio
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError(f"{name} 至少有一个题型比例必须大于 0。")
    return {category: ratios[category] / total for category in QUESTION_TYPES}


def allocate_type_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """用最大余数法分配整数配额；余数相同时按 QUESTION_TYPES 顺序打破平局。"""
    exact = {category: total * ratios[category] for category in QUESTION_TYPES}
    counts = {category: math.floor(exact[category]) for category in QUESTION_TYPES}
    remainder = total - sum(counts.values())
    order = sorted(QUESTION_TYPES,
                   key=lambda category: (-(exact[category] - counts[category]),
                                         QUESTION_TYPES.index(category)))
    for category in order[:remainder]:
        counts[category] += 1
    return counts


def _validated_config(payload: dict, config_path: Path) -> argparse.Namespace:
    """严格检查 JSON 类型，避免字符串 false 或布尔数量被误当作有效参数。"""
    defaults = {
        "data_root": str(DEFAULT_DATA_ROOT), "keywords_file": None, "db_dir": None,
        "collection": DEFAULT_COLLECTION, "splits": ["train"], "seed": 42,
        "exclude_shared_images": False,
        "train_counts": None, "test_counts": None,
        "train_ratios": None, "test_ratios": None,
        "output": str(PROJECT_ROOT / "output" / "mira_split_ids.json"), "check_config": False,
    }
    if not isinstance(payload, dict):
        raise ValueError("抽样配置必须是 JSON 对象。")
    unknown = set(payload) - set(defaults) - {"train_count", "test_count", "_说明"}
    if unknown:
        raise ValueError(f"抽样配置包含未知字段：{sorted(unknown)}")
    if "_说明" in payload and not isinstance(payload["_说明"], str):
        raise ValueError("_说明 必须是中文说明字符串。")
    keywords_explicitly_disabled = "keywords_file" in payload and payload["keywords_file"] is None
    settings = defaults | {key: value for key, value in payload.items() if key != "_说明"}
    counts_mode = settings["train_counts"] is not None or settings["test_counts"] is not None
    ratios_mode = settings["train_ratios"] is not None or settings["test_ratios"] is not None
    if counts_mode and ratios_mode:
        raise ValueError("train_counts/test_counts 与 train_ratios/test_ratios 不能同时配置。")
    if counts_mode:
        if (settings["train_counts"] is None) != (settings["test_counts"] is None):
            raise ValueError("train_counts 和 test_counts 必须同时设置。")
        if "train_count" in payload or "test_count" in payload:
            raise ValueError("train_counts/test_counts 与 train_count/test_count 不能同时配置。")
        settings["train_counts"] = validate_type_counts(settings["train_counts"], "train_counts")
        settings["test_counts"] = validate_type_counts(settings["test_counts"], "test_counts")
        settings["train_count"] = sum(settings["train_counts"].values())
        settings["test_count"] = sum(settings["test_counts"].values())
        settings["train_ratios"] = settings["test_ratios"] = None
        settings["normalized_train_ratios"] = settings["normalized_test_ratios"] = None
    elif ratios_mode:
        if (settings["train_ratios"] is None) != (settings["test_ratios"] is None):
            raise ValueError("train_ratios 和 test_ratios 必须同时设置。")
        for key in ("train_count", "test_count"):
            value = settings.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} 必须设置为正整数。")
        settings["normalized_train_ratios"] = validate_type_ratios(
            settings["train_ratios"], "train_ratios")
        settings["normalized_test_ratios"] = validate_type_ratios(
            settings["test_ratios"], "test_ratios")
        settings["train_counts"] = allocate_type_counts(
            settings["train_count"], settings["normalized_train_ratios"])
        settings["test_counts"] = allocate_type_counts(
            settings["test_count"], settings["normalized_test_ratios"])
    else:
        settings["normalized_train_ratios"] = settings["normalized_test_ratios"] = None
        for key in ("train_count", "test_count"):
            value = settings.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} 必须设置为正整数。")
    if type(settings["seed"]) is not int:
        raise ValueError("seed 必须为整数。")
    for key in ("exclude_shared_images", "check_config"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} 必须为 JSON true 或 false，不能带引号。")
    splits = settings["splits"]
    if (not isinstance(splits, list) or not splits
            or any(not isinstance(split, str) or split not in SPLITS for split in splits)
            or len(splits) != len(set(splits))):
        raise ValueError("splits 必须是非空且不重复的列表，可选 train、validation、test。")
    if not isinstance(settings["collection"], str) or not settings["collection"].strip():
        raise ValueError("collection 必须为非空字符串。")
    for key in ("data_root", "keywords_file", "db_dir", "output"):
        value = settings[key]
        if value is None and key in {"keywords_file", "db_dir"}:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须为非空路径字符串。")
        path = Path(value).expanduser()
        settings[key] = (path if path.is_absolute() else config_path.parent / path).resolve()
    data_root = settings["data_root"]
    if settings["db_dir"] is None:
        settings["db_dir"] = data_root.parent / "MIRA-chroma"
    if settings["keywords_file"] is None and not keywords_explicitly_disabled:
        settings["keywords_file"] = data_root.parent / "MIRA_myConfig" / KEYWORDS_FILENAME
    output = settings["output"]
    if output.suffix.lower() != ".json":
        raise ValueError("output 必须使用 .json 扩展名。")
    protected = {config_path, DEFAULT_CONFIG_PATH.resolve()}
    if settings["keywords_file"] is not None:
        protected.add(settings["keywords_file"])
    if output in protected:
        raise ValueError("output 不能覆盖抽样配置文件或关键词文件，请指定独立的 ID 清单路径。")
    return argparse.Namespace(config=config_path, **settings)


def load_config(path: Path) -> argparse.Namespace:
    """读取同名 JSON；相对路径统一以该 JSON 的目录为基准。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"抽样配置文件不存在：{path}")
    return _validated_config(json.loads(path.read_text(encoding="utf-8-sig")), path)


def resolve_run_config(args: argparse.Namespace) -> argparse.Namespace:
    """常规使用全部来自 JSON；仅显式指定的旧命令行参数可以覆盖配置。"""
    path = args.config.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"抽样配置文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("抽样配置必须是 JSON 对象。")
    # 先验证原始配置，避免命令行覆盖掩盖 JSON 中的类型错误或危险输出路径。
    base_settings = _validated_config(payload, path)
    # 显式 --train-count/--test-count 覆盖总数；比例模式会据此重算配额。
    if args.train_count is not None or args.test_count is not None:
        # 对旧的整数配额模式，保留兼容行为：命令行数量会切回仅控制总数的抽样。
        if base_settings.train_ratios is None and base_settings.train_counts is not None:
            payload.pop("train_counts", None)
            payload.pop("test_counts", None)
            payload["train_count"] = base_settings.train_count
            payload["test_count"] = base_settings.test_count
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        # 兼容旧命令行的路径语义：显式 CLI 相对路径以当前工作目录为基准。
        payload[key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    return _validated_config(payload, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="配置文件；默认读取脚本同目录的 sample_mira_ids.json。")
    parser.add_argument("--check-config", action="store_true", default=None,
                        help="仅检查配置，不进行抽样；也可在 JSON 设置 check_config=true。")
    parser.add_argument("--train-count", type=int, help="兼容旧调用：覆盖 JSON 的 train_count。")
    parser.add_argument("--test-count", type=int, help="兼容旧调用：覆盖 JSON 的 test_count。")
    parser.add_argument("--data-root", type=Path,
                        help="兼容旧调用：覆盖 JSON 的 data_root。")
    parser.add_argument("--keywords-file", type=Path,
                        help="兼容旧调用：覆盖 JSON 的 keywords_file。")
    parser.add_argument("--db-dir", type=Path,
                        help="兼容旧调用：覆盖 JSON 的 db_dir。")
    parser.add_argument("--collection", help="兼容旧调用：覆盖 JSON 的 collection。")
    parser.add_argument("--splits", nargs="+", choices=SPLITS,
                        help="兼容旧调用：覆盖 JSON 的 splits。")
    parser.add_argument("--seed", type=int, help="兼容旧调用：覆盖 JSON 的 seed。")
    parser.add_argument("--exclude-shared-images", action=argparse.BooleanOptionalAction, default=None,
                        help="兼容旧调用：覆盖 JSON 的 exclude_shared_images。")
    parser.add_argument("--output", type=Path, help="兼容旧调用：覆盖 JSON 的 output。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        args = resolve_run_config(args)
        output = args.output
        print(f"读取抽样配置：{args.config}", flush=True)
        if args.check_config:
            print(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str))
            print("配置检查通过；未扫描数据、未写入抽样结果。", flush=True)
            return 0
        result = sample_dataset(args.data_root, getattr(args, "train_count", 0),
                                getattr(args, "test_count", 0),
                                splits=args.splits, seed=args.seed,
                                exclude_shared_images=args.exclude_shared_images,
                                db_dir=args.db_dir, collection=args.collection,
                                keywords_file=args.keywords_file,
                                all_samples_match_keywords=args.keywords_file is None,
                                train_counts=args.train_counts, test_counts=args.test_counts)
        if args.train_ratios is not None:
            result["requested_question_type_ratios"] = {
                "train": args.train_ratios,
                "test": args.test_ratios,
            }
            result["normalized_question_type_ratios"] = {
                "train": args.normalized_train_ratios,
                "test": args.normalized_test_ratios,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"已保存 {result['train_count']} 个训练编号和 {result['test_count']} 个测试编号：{output}")
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
