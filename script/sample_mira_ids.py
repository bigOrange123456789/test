"""Audit MIRA keyword/embedding coverage and sample training/test QA IDs.

Example:
    python script/sample_mira_ids.py --train-count 5000 --test-count 500

All existing train/validation/test CSVs are checked against the Chroma IDs.
The default sampling pool is train.csv; use --splits to select other sources.
Test QAs are selected first, then training QAs. Priority: keyword + embedding,
embedding only, keyword only, neither. Counts refer to individual QA pairs.
Keywords match question/options/answer/extra QA text, not shared captions.
Matching ignores case, uses word boundaries and flexible phrase whitespace,
and treats a term's parenthesized abbreviation as an alternative. Each QA
counts once, even if several keywords match. No stemming or synonym expansion.
Uses only the Python standard library and reads Chroma SQLite metadata in
read-only mode. Coverage means a matching stored ID, not vector quality.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-count", type=int, required=True, help="Number of training QAs.")
    parser.add_argument("--test-count", type=int, required=True, help="Number of test QAs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"MIRA CSV directory (default: {DEFAULT_DATA_ROOT}).")
    parser.add_argument("--keywords-file", type=Path,
                        help=f"UTF-8 keyword file (default: MIRA_myConfig/{KEYWORDS_FILENAME} beside data-root).")
    parser.add_argument("--db-dir", type=Path,
                        help="Existing Chroma directory (default: MIRA-chroma beside data-root).")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Existing Chroma collection (default: {DEFAULT_COLLECTION}).")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train"],
                        help="Source CSV splits to sample from (default: train).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--exclude-shared-images", action="store_true",
                        help="Exclude training QAs sharing any image path with the test set.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("mira_split_ids.json"),
                        help="Output JSON path (default: script/mira_split_ids.json).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = args.output.expanduser().resolve()
        if output.suffix.lower() != ".json":
            raise ValueError("--output must have a .json extension.")
        result = sample_dataset(args.data_root, args.train_count, args.test_count,
                                splits=args.splits, seed=args.seed,
                                exclude_shared_images=args.exclude_shared_images,
                                db_dir=args.db_dir, collection=args.collection, keywords_file=args.keywords_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Saved {result['train_count']} train IDs and {result['test_count']} test IDs: {output}")
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
