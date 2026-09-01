# -*- coding: utf-8 -*-
"""
Extract usable Huatuo QA pairs for DeepSeek LoRA fine-tuning.

Default output:
    dataset/huatuo_deepseek_finetune/all_usable.jsonl
    dataset/huatuo_deepseek_finetune/train.jsonl
    dataset/huatuo_deepseek_finetune/test.jsonl
    dataset/huatuo_deepseek_finetune/summary.json

Usage:
    python dataset/extract_huatuo_finetune_data.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from count_huatuo_qa_pairs import ANSWER_KEYS, QUESTION_KEYS, first_text, is_url_only


DEFAULT_SYSTEM_PROMPT = (
    "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_source_line"] = line_no
                rows.append(row)
    return rows


def build_example(
    source_root: Path,
    source_file: Path,
    source_line: int,
    question: str,
    answer: str,
    system_prompt: str,
) -> dict[str, Any]:
    relative_source = source_file.relative_to(source_root).as_posix()
    example_id = f"{source_file.stem}:{source_line}"

    return {
        "id": example_id,
        "source_file": relative_source,
        "source_line": source_line,
        "question": question,
        "answer": answer,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def collect_usable_pairs(
    source_dir: Path,
    include_url_answers: bool,
    system_prompt: str,
) -> list[dict[str, Any]]:
    examples = []

    for source_file in sorted(source_dir.rglob("*.jsonl")):
        for row in read_jsonl(source_file):
            question = first_text(row, QUESTION_KEYS)
            answer = first_text(row, ANSWER_KEYS)

            if not question or not answer:
                continue
            if is_url_only(answer) and not include_url_answers:
                continue

            examples.append(
                build_example(
                    source_root=source_dir,
                    source_file=source_file,
                    source_line=row["_source_line"],
                    question=question,
                    answer=answer,
                    system_prompt=system_prompt,
                )
            )

    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    source_dir: Path,
    output_dir: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    include_url_answers: bool,
) -> None:
    summary = {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_file": "train.jsonl",
        "test_file": "test.jsonl",
        "all_usable_file": "all_usable.jsonl",
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "total_count": len(train_rows) + len(test_rows),
        "seed": seed,
        "include_url_answers": include_url_answers,
        "train_ids": [row["id"] for row in train_rows],
        "test_ids": [row["id"] for row in test_rows],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract Huatuo QA data into train/test JSONL files for DeepSeek LoRA."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=script_dir / "huatuo_samples",
        help="Directory containing Huatuo .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "huatuo_deepseek_finetune",
        help="Directory where extracted train/test files will be written.",
    )
    parser.add_argument("--train-size", type=int, default=34)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-url-answers",
        action="store_true",
        help="Keep QA rows whose answer is only a URL.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt stored in each chat-format training example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    required_count = args.train_size + args.test_size

    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    if args.train_size < 1:
        raise SystemExit("--train-size must be at least 1")
    if args.test_size < 1:
        raise SystemExit("--test-size must be at least 1")

    examples = collect_usable_pairs(
        source_dir=source_dir,
        include_url_answers=args.include_url_answers,
        system_prompt=args.system_prompt,
    )

    if len(examples) < required_count:
        raise SystemExit(
            f"Only found {len(examples)} usable pairs, but "
            f"{required_count} are required."
        )

    random.Random(args.seed).shuffle(examples)
    selected = examples[:required_count]
    train_rows = selected[: args.train_size]
    test_rows = selected[args.train_size : required_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_usable.jsonl", selected)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_summary(
        path=output_dir / "summary.json",
        source_dir=source_dir,
        output_dir=output_dir,
        train_rows=train_rows,
        test_rows=test_rows,
        seed=args.seed,
        include_url_answers=args.include_url_answers,
    )

    print(f"Extracted usable pairs: {len(selected)}")
    print(f"Train pairs: {len(train_rows)} -> {output_dir / 'train.jsonl'}")
    print(f"Test pairs: {len(test_rows)} -> {output_dir / 'test.jsonl'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
