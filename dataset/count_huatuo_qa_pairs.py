# -*- coding: utf-8 -*-
"""
Count usable Huatuo QA pairs from local JSONL files.

Default "usable" rule:
- question text is not empty
- answer text is not empty
- answer is not only a URL

Usage:
    python dataset/count_huatuo_qa_pairs.py
    python dataset/count_huatuo_qa_pairs.py dataset/huatuo_samples
    python dataset/count_huatuo_qa_pairs.py --include-url-answers
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUESTION_KEYS = ("question", "questions", "query", "prompt", "instruction", "input")
ANSWER_KEYS = ("answer", "answers", "response", "output", "completion", "target")
URL_ONLY_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


@dataclass
class FileStats:
    path: Path
    total_lines: int = 0
    valid_json_lines: int = 0
    usable_pairs: int = 0
    invalid_json: int = 0
    missing_question: int = 0
    missing_answer: int = 0
    url_only_answer: int = 0


def flatten_to_text(value: Any) -> str:
    """Flatten nested dataset values into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [flatten_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = [flatten_to_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    return str(value).strip()


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = flatten_to_text(row.get(key))
        if text:
            return text
    return ""


def is_url_only(text: str) -> bool:
    return bool(URL_ONLY_RE.match(text.strip()))


def count_file(path: Path, include_url_answers: bool) -> FileStats:
    stats = FileStats(path=path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stats.total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.invalid_json += 1
                continue

            if not isinstance(row, dict):
                stats.invalid_json += 1
                continue

            stats.valid_json_lines += 1
            question = first_text(row, QUESTION_KEYS)
            answer = first_text(row, ANSWER_KEYS)

            if not question:
                stats.missing_question += 1
                continue
            if not answer:
                stats.missing_answer += 1
                continue
            if is_url_only(answer) and not include_url_answers:
                stats.url_only_answer += 1
                continue

            stats.usable_pairs += 1

    return stats


def print_stats(stats_list: list[FileStats], root: Path) -> None:
    print("Huatuo QA pair statistics")
    print(f"Data path: {root.resolve()}")
    print()
    print(
        f"{'file':56} {'lines':>7} {'json':>7} {'usable':>8} "
        f"{'no_q':>6} {'no_a':>6} {'url_a':>7} {'bad_json':>8}"
    )
    print("-" * 113)

    total = FileStats(path=root)
    for stats in stats_list:
        rel_path = stats.path.relative_to(root) if stats.path.is_relative_to(root) else stats.path
        print(
            f"{str(rel_path):56.56} "
            f"{stats.total_lines:7d} "
            f"{stats.valid_json_lines:7d} "
            f"{stats.usable_pairs:8d} "
            f"{stats.missing_question:6d} "
            f"{stats.missing_answer:6d} "
            f"{stats.url_only_answer:7d} "
            f"{stats.invalid_json:8d}"
        )

        total.total_lines += stats.total_lines
        total.valid_json_lines += stats.valid_json_lines
        total.usable_pairs += stats.usable_pairs
        total.missing_question += stats.missing_question
        total.missing_answer += stats.missing_answer
        total.url_only_answer += stats.url_only_answer
        total.invalid_json += stats.invalid_json

    print("-" * 113)
    print(
        f"{'TOTAL':56} "
        f"{total.total_lines:7d} "
        f"{total.valid_json_lines:7d} "
        f"{total.usable_pairs:8d} "
        f"{total.missing_question:6d} "
        f"{total.missing_answer:6d} "
        f"{total.url_only_answer:7d} "
        f"{total.invalid_json:8d}"
    )
    print()
    print(f"Usable QA pairs: {total.usable_pairs}")


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).resolve().parent / "huatuo_samples"
    parser = argparse.ArgumentParser(
        description="Count usable QA pairs in local Huatuo JSONL files."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=default_data_dir,
        help="Directory containing Huatuo .jsonl files.",
    )
    parser.add_argument(
        "--include-url-answers",
        action="store_true",
        help="Count answer values that are only URLs as usable pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    if not data_dir.exists():
        raise SystemExit(f"Data path does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"Data path is not a directory: {data_dir}")

    jsonl_files = sorted(data_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found under: {data_dir}")

    stats_list = [count_file(path, args.include_url_answers) for path in jsonl_files]
    print_stats(stats_list, data_dir)


if __name__ == "__main__":
    main()
