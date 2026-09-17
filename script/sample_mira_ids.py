"""Audit MIRA embedding coverage and sample disjoint training/test QA IDs.

Example:
    python script/sample_mira_ids.py --train-count 5000 --test-count 500

All existing train/validation/test CSVs are checked against the Chroma IDs.
The default sampling pool is train.csv; use --splits to select other sources.
Test QAs are selected first, then training QAs. Each set prefers embedded QAs,
falling back to unembedded QAs when needed. Counts refer to individual QAs.
Uses only the Python standard library and reads Chroma SQLite metadata in
read-only mode. Coverage means a matching stored ID, not vector quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "inferenceValid"))
from embed_mira_chroma import DEFAULT_DATA_ROOT, image_path, iter_samples


LOGGER = logging.getLogger("sample_mira_ids")
SPLITS = ("train", "validation", "test")
DEFAULT_COLLECTION = "mira_qwen3_vl_embedding"
T = TypeVar("T")


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
                    is_embedded: Callable[[T], bool]) -> tuple[list[T], int, int]:
    """Uniformly sample each priority tier using two bounded reservoirs."""
    reservoirs: list[list[T]] = [[], []]
    counts = [0, 0]
    for item in items:
        tier = 0 if is_embedded(item) else 1
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
    selected = (reservoirs[0] + reservoirs[1])[:count]
    rng.shuffle(selected)
    return selected, sum(counts), counts[0]


def iter_records(data_root: Path, splits: list[str]) -> Iterator[tuple[str, list[str]]]:
    for split in splits:
        for sample in iter_samples(data_root, split):
            yield sample.id, sample.images


def image_keys(data_root: Path, names: list[str]) -> set[str]:
    return {os.path.normcase(os.path.abspath(image_path(data_root, name))) for name in names}


def sample_dataset(data_root: Path, train_count: int, test_count: int, *,
                   splits: list[str] | None = None, seed: int = 42,
                   exclude_shared_images: bool = False, db_dir: Path | None = None,
                   collection: str = DEFAULT_COLLECTION) -> dict:
    """Audit all source splits and prefer embedded QAs within the sampling pool."""
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

    db_dir = (Path(db_dir) if db_dir is not None else data_root.parent / "MIRA-chroma").expanduser().resolve()
    embedded_ids = load_embedded_ids(db_dir, collection)
    coverage_splits = [split for split in SPLITS if (data_root / f"{split}.csv").is_file()]
    split_counts = {}

    def audited_records() -> Iterator[tuple[str, list[str]]]:
        for split in coverage_splits:
            LOGGER.info("Checking all QA IDs in %s.csv", split)
            counts = {"total_samples": 0, "embedded_samples": 0, "missing_samples": 0}
            for sample_id, names in iter_records(data_root, [split]):
                counts["total_samples"] += 1
                counts["embedded_samples"] += sample_id in embedded_ids
                if split in splits:
                    yield sample_id, names
            counts["missing_samples"] = counts["total_samples"] - counts["embedded_samples"]
            split_counts[split] = counts

    rng = random.Random(seed)
    LOGGER.info("Sampling %d test QAs from %s (%s)", test_count, data_root, ", ".join(splits))
    tests, total, embedded_samples = priority_sample(
        audited_records(), test_count, rng, lambda item: item[0] in embedded_ids)
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
    print(f"Dataset groups (all existing CSV splits): {dataset_total:,}", flush=True)
    print(f"Groups with stored embeddings: {dataset_embedded:,}", flush=True)
    print(f"Groups without embeddings: {coverage['missing_samples']:,}", flush=True)
    print(f"All dataset groups embedded: {'yes' if coverage['all_embedded'] else 'no'}", flush=True)
    for split, counts in split_counts.items():
        print(f"  {split}: total={counts['total_samples']:,}, "
              f"embedded={counts['embedded_samples']:,}, missing={counts['missing_samples']:,}", flush=True)
    print(f"Chroma IDs not matched to this dataset: {coverage['unmatched_chroma_ids']:,}", flush=True)
    print(f"Sampling pool ({', '.join(splits)}): {total:,}; embedded={embedded_samples:,}", flush=True)
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
        eligible_train_ids(), train_count, rng, lambda sample_id: sample_id in embedded_ids)
    if train_candidates < train_count:
        raise ValueError(
            f"Requested {train_count} train QAs, but only {train_candidates} remain "
            "after excluding test IDs and shared images. Reduce the requested counts.")
    selected_embedding_counts = {
        "train": sum(sample_id in embedded_ids for sample_id in train_ids),
        "test": sum(sample_id in embedded_ids for sample_id in test_ids),
    }
    for label, ids in (("train", train_ids), ("test", test_ids)):
        matched = selected_embedding_counts[label]
        print(f"Selected {label}: total={len(ids):,}, embedded={matched:,}, "
              f"unembedded fallback={len(ids) - matched:,}", flush=True)
    return {
        "data_root": str(data_root),
        "db_dir": str(db_dir),
        "chroma_collection": collection,
        "source_splits": splits,
        "seed": seed,
        "sample_unit": "question_answer",
        "id_format": "mira:{source_split}:{row_index_0based}:{category}:{qa_index_0based}",
        "exclude_shared_images": exclude_shared_images,
        "total_samples": total,
        "embedded_samples": embedded_samples,
        "embedding_coverage": coverage,
        "selected_embedding_counts": selected_embedding_counts,
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
                                db_dir=args.db_dir, collection=args.collection)
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
