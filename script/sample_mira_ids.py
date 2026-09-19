"""检查 MIRA 关键词/向量覆盖情况，并生成训练集和测试集问答编号。

先修改本脚本同目录的 sample_mira_ids.json，再直接运行，无须输入抽样参数：
    python script/sample_mira_ids.py
或在 script 目录中运行：
    python sample_mira_ids.py

JSON 参数说明（路径可为绝对路径，相对路径以 JSON 所在目录为基准）：
    train_count / test_count：训练集 / 测试集的问答数量，均为正整数。
    data_root：MIRA 数据目录。
    keywords_file：关键词文件；null 使用数据目录旁的 MIRA_myConfig 中的默认文件。
    db_dir：已有 Chroma 目录；null 使用数据目录旁的 MIRA-chroma。
    collection：Chroma 集合名。
    splits：抽样来源列表，支持 train、validation、test；默认只从 train.csv 抽样。
    seed：随机种子，相同数据和参数下可复现抽样。
    exclude_shared_images：true 时额外排除与测试题共用图片的训练题。
    output：生成的 ID 清单，默认 mira_split_ids.json，不能覆盖输入配置。
    check_config：true 时只显示解析后的参数，不扫描数据，也不写入抽样结果。
    _说明：可选的中文说明文本，不参与抽样。

默认配置以脚本位置定位，与运行时工作目录无关。为兼容旧调用仍保留命令行选项，
显式命令行参数优先于 JSON；正常使用只需编辑 JSON。重复运行会更新 output 指定的清单，
如需保留已有训练/测试划分，请在 JSON 中为 output 设置不同文件名。

数量以一个问答对为单位。先选测试题，再选训练题，优先级依次为：
关键词与向量均匹配、仅有向量、仅匹配关键词、两者均不满足。
会核查所有已有 train/validation/test CSV，但只从 splits 指定的来源抽样。
关键词匹配问题、选项、答案和额外问答文本，不匹配共用图片标题；忽略大小写，
使用词边界并允许短语内不同空白，括号中的缩写也可匹配。每道题只统计一次。
只使用标准库，并只读访问 Chroma 元数据；向量覆盖表示存在对应编号，不代表向量质量。
"""

from __future__ import annotations

import argparse
import json
import logging
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


def iter_records(data_root: Path, splits: list[str]) -> Iterator[tuple[str, list[str]]]:
    for split in splits:
        for sample in iter_samples(data_root, split):
            yield sample.id, sample.images


def image_keys(data_root: Path, names: list[str]) -> set[str]:
    return {os.path.normcase(os.path.abspath(image_path(data_root, name))) for name in names}


def sample_dataset(data_root: Path, train_count: int, test_count: int, *,
                   splits: list[str] | None = None, seed: int = 42,
                   exclude_shared_images: bool = False, db_dir: Path | None = None,
                   collection: str = DEFAULT_COLLECTION, keywords_file: Path | None = None) -> dict:
    """Audit all splits and prefer keyword-matching QAs with stored embeddings."""
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train-count and test-count must be positive integers.")
    splits = ["train"] if splits is None else list(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - set(SPLITS):
        raise ValueError("splits must contain unique values from train, validation, test.")
    data_root = Path(data_root).expanduser().resolve()
    for split in splits:
        source = data_root / f"{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Dataset CSV not found: {source}")

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

    def sample_priority(sample_id: str) -> int:
        if sample_id in embedded_ids:
            return 0 if sample_id in keyword_ids else 1
        return 2 if sample_id in keyword_ids else 3

    def audited_records() -> Iterator[tuple[str, list[str]]]:
        for split in coverage_splits:
            LOGGER.info("Checking all QA IDs in %s.csv", split)
            counts = {"total_samples": 0, "embedded_samples": 0, "missing_samples": 0}
            keyword_counts = {"matched_samples": 0, "matched_embedded_samples": 0}
            for sample in iter_samples(data_root, split):
                embedded = sample.id in embedded_ids
                matched = matches_keywords(sample, keyword_pattern)
                counts["total_samples"] += 1
                counts["embedded_samples"] += embedded
                keyword_counts["matched_samples"] += matched
                keyword_counts["matched_embedded_samples"] += matched and embedded
                if split in splits:
                    if matched:
                        keyword_ids.add(sample.id)
                    yield sample.id, sample.images
            counts["missing_samples"] = counts["total_samples"] - counts["embedded_samples"]
            split_counts[split] = counts
            keyword_split_counts[split] = keyword_counts

    rng = random.Random(seed)
    LOGGER.info("Sampling %d test QAs from %s (%s)", test_count, data_root, ", ".join(splits))
    tests, total, priority_counts = priority_sample(
        audited_records(), test_count, rng, lambda item: sample_priority(item[0]))
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
    if total < train_count + test_count:
        raise ValueError(
            f"Requested {train_count} train + {test_count} test QAs, "
            f"but the selected source splits contain only {total} QAs.")
    test_ids = [sample_id for sample_id, _ in tests]
    test_id_set = set(test_ids)
    test_images = set()
    if exclude_shared_images:
        for _, names in tests:
            test_images.update(image_keys(data_root, names))

    # A second streaming pass avoids retaining the million-record population.
    def eligible_train_ids() -> Iterator[str]:
        for sample_id, names in iter_records(data_root, splits):
            if sample_id in test_id_set:
                continue
            if test_images and image_keys(data_root, names) & test_images:
                continue
            yield sample_id

    LOGGER.info("Sampling %d training QAs from the remaining population", train_count)
    train_ids, train_candidates, _ = priority_sample(
        eligible_train_ids(), train_count, rng, sample_priority)
    if train_candidates < train_count:
        raise ValueError(
            f"Requested {train_count} train QAs, but only {train_candidates} remain "
            "after excluding test IDs and shared images. Reduce the requested counts.")
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
    for label, ids in (("train", train_ids), ("test", test_ids)):
        matched = selected_embedding_counts[label]
        print(f"Selected {label}: total={len(ids):,}, embedded={matched:,}, "
              f"unembedded fallback={len(ids) - matched:,}, keyword={selected_keyword_counts[label]:,}, "
              f"keyword+embedded={selected_keyword_embedding_counts[label]:,}, "
              f"outside-top-priority={len(ids) - selected_keyword_embedding_counts[label]:,}", flush=True)
    return {
        "data_root": str(data_root),
        "db_dir": str(db_dir),
        "chroma_collection": collection,
        "keywords_file": str(keywords_file),
        "keywords": keywords,
        "keyword_matching": "case_insensitive_whole_phrase_or_abbreviation",
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


def _validated_config(payload: dict, config_path: Path) -> argparse.Namespace:
    """严格检查 JSON 类型，避免字符串 false 或布尔数量被误当作有效参数。"""
    defaults = {
        "data_root": str(DEFAULT_DATA_ROOT), "keywords_file": None, "db_dir": None,
        "collection": DEFAULT_COLLECTION, "splits": ["train"], "seed": 42,
        "exclude_shared_images": False, "output": "mira_split_ids.json", "check_config": False,
    }
    if not isinstance(payload, dict):
        raise ValueError("抽样配置必须是 JSON 对象。")
    unknown = set(payload) - set(defaults) - {"train_count", "test_count", "_说明"}
    if unknown:
        raise ValueError(f"抽样配置包含未知字段：{sorted(unknown)}")
    if "_说明" in payload and not isinstance(payload["_说明"], str):
        raise ValueError("_说明 必须是中文说明字符串。")
    settings = defaults | {key: value for key, value in payload.items() if key != "_说明"}
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
    if settings["keywords_file"] is None:
        settings["keywords_file"] = data_root.parent / "MIRA_myConfig" / KEYWORDS_FILENAME
    output = settings["output"]
    if output.suffix.lower() != ".json":
        raise ValueError("output 必须使用 .json 扩展名。")
    protected = {config_path, DEFAULT_CONFIG_PATH.resolve(), settings["keywords_file"]}
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
    _validated_config(payload, path)
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
        result = sample_dataset(args.data_root, args.train_count, args.test_count,
                                splits=args.splits, seed=args.seed,
                                exclude_shared_images=args.exclude_shared_images,
                                db_dir=args.db_dir, collection=args.collection, keywords_file=args.keywords_file)
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
