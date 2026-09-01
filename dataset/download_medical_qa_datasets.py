# -*- coding: utf-8 -*-
"""
Download and normalize public medical QA/SFT datasets for DeepSeek LoRA.

The output format matches DeepSeek-Model/finetune_huatuo_lora.py:
each JSONL row contains question, answer, and messages fields.

Examples:
    python dataset/download_medical_qa_datasets.py --list-sources
    python dataset/download_medical_qa_datasets.py --source medical_zh --limit 5000 --test-size 200
    python dataset/download_medical_qa_datasets.py --source huatuo_sharegpt --limit 3000
    python dataset/download_medical_qa_datasets.py --source medical_o1_zh --limit 3000 --include-cot

If huggingface.co is slow or unavailable in China, add:
    --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_SYSTEM_PROMPT = (
    "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


MARKDOWN_LINK_RE = re.compile(r"^\s*\[[^\]]+\]\((https?://[^)\s]+)\)\s*$")


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().strip("\"'")
    markdown_match = MARKDOWN_LINK_RE.match(endpoint)
    if markdown_match:
        endpoint = markdown_match.group(1)
    if endpoint.startswith("<") and endpoint.endswith(">"):
        endpoint = endpoint[1:-1].strip()
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class SourcePreset:
    key: str
    description: str
    loader: str
    dataset: str | None = None
    config: str | None = None
    split: str = "train"
    data_files: tuple[str, ...] = ()
    format_hint: str = "auto"


SOURCE_PRESETS: dict[str, SourcePreset] = {
    "medical_zh": SourcePreset(
        key="medical_zh",
        description=(
            "shibing624/medical 中文指令微调数据，Alpaca 格式；"
            "问题在 instruction/input，答案在 output。"
        ),
        loader="json_urls",
        dataset="shibing624/medical",
        data_files=("finetune/train_zh_0.json",),
        format_hint="alpaca",
    ),
    "huatuo_sharegpt": SourcePreset(
        key="huatuo_sharegpt",
        description=(
            "shibing624/huatuo_medical_qa_sharegpt，多轮中文医疗对话，"
            "ShareGPT conversations 格式。"
        ),
        loader="hf_dataset",
        dataset="shibing624/huatuo_medical_qa_sharegpt",
        split="train",
        format_hint="sharegpt",
    ),
    "huatuogpt_sft": SourcePreset(
        key="huatuogpt_sft",
        description=(
            "FreedomIntelligence/HuatuoGPT-sft-data-v1，HuatuoGPT SFT 阶段"
            "中文医疗指令/对话数据。"
        ),
        loader="hf_dataset",
        dataset="FreedomIntelligence/HuatuoGPT-sft-data-v1",
        split="train",
        format_hint="auto",
    ),
    "medical_o1_zh": SourcePreset(
        key="medical_o1_zh",
        description=(
            "FreedomIntelligence/medical-o1-reasoning-SFT 中文混合医学推理数据；"
            "字段通常为 Question/Complex_CoT/Response。"
        ),
        loader="hf_dataset",
        dataset="FreedomIntelligence/medical-o1-reasoning-SFT",
        config="zh_mix",
        split="train",
        format_hint="medical_o1",
    ),
    "medical_o1_en": SourcePreset(
        key="medical_o1_en",
        description=(
            "FreedomIntelligence/medical-o1-reasoning-SFT 英文医学推理数据；"
            "适合验证推理格式迁移。"
        ),
        loader="hf_dataset",
        dataset="FreedomIntelligence/medical-o1-reasoning-SFT",
        config="en",
        split="train",
        format_hint="medical_o1",
    ),
}


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "患者": "user",
    "病人": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "doctor": "assistant",
    "医生": "assistant",
    "system": "system",
}


def flatten_to_text(value: Any) -> str:
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


def build_file_url(endpoint: str, dataset: str, file_path: str) -> str:
    endpoint = normalize_endpoint(endpoint)
    return f"{endpoint}/datasets/{dataset}/resolve/main/{file_path}"


def load_dataset_rows(source: SourcePreset, endpoint: str, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with:\n"
            "  pip install -r dataset/requirements-download.txt"
        ) from exc

    endpoint = normalize_endpoint(endpoint)
    if endpoint and endpoint != "https://huggingface.co":
        os.environ["HF_ENDPOINT"] = endpoint

    if source.loader == "json_urls":
        if source.dataset is None:
            raise ValueError(f"{source.key} is missing dataset id")
        urls = [build_file_url(endpoint, source.dataset, file_path) for file_path in source.data_files]
        data_files = {"train": urls[0] if len(urls) == 1 else urls}
        return load_dataset("json", data_files=data_files, split="train", streaming=streaming)

    if source.loader == "hf_dataset":
        kwargs: dict[str, Any] = {"split": source.split, "streaming": streaming}
        if source.config:
            return load_dataset(source.dataset, source.config, **kwargs)
        return load_dataset(source.dataset, **kwargs)

    raise ValueError(f"Unsupported loader: {source.loader}")


def role_from_message(message: dict[str, Any]) -> str:
    raw_role = message.get("role", message.get("from", ""))
    return ROLE_MAP.get(str(raw_role).strip().lower(), str(raw_role).strip().lower())


def content_from_message(message: dict[str, Any]) -> str:
    return first_text(message, ("content", "value", "text"))


def normalize_conversations(
    row: dict[str, Any],
    source_key: str,
    row_index: int,
    system_prompt: str,
) -> dict[str, Any] | None:
    conversations = row.get("conversations", row.get("messages"))
    if not isinstance(conversations, list):
        return None

    messages = []
    has_system = False
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = role_from_message(message)
        content = content_from_message(message)
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if role == "system":
            has_system = True
        messages.append({"role": role, "content": content})

    if not messages:
        return None
    if not has_system and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 3:
        return None

    first_user = next((item["content"] for item in messages if item["role"] == "user"), "")
    last_answer = messages[-1]["content"]
    return build_output_row(source_key, row_index, first_user, last_answer, messages)


def normalize_medical_o1(
    row: dict[str, Any],
    source_key: str,
    row_index: int,
    system_prompt: str,
    include_cot: bool,
) -> dict[str, Any] | None:
    question = first_text(row, ("Question", "question", "prompt"))
    response = first_text(row, ("Response", "response", "answer", "output"))
    cot = first_text(row, ("Complex_CoT", "complex_cot", "cot", "reasoning"))
    if not question or not response:
        return None

    answer = response
    if include_cot and cot:
        answer = f"<think>\n{cot}\n</think>\n{response}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return build_output_row(source_key, row_index, question, answer, messages)


def normalize_alpaca(
    row: dict[str, Any],
    source_key: str,
    row_index: int,
    system_prompt: str,
) -> dict[str, Any] | None:
    instruction = first_text(row, ("instruction", "prompt", "question", "query"))
    input_text = first_text(row, ("input", "context"))
    answer = first_text(row, ("output", "answer", "answers", "response"))

    if input_text and instruction:
        question = f"{instruction}\n\n{input_text}"
    else:
        question = instruction or input_text
    if not question or not answer:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return build_output_row(source_key, row_index, question, answer, messages)


def normalize_auto(
    row: dict[str, Any],
    source_key: str,
    row_index: int,
    system_prompt: str,
    include_cot: bool,
) -> dict[str, Any] | None:
    conversation_row = normalize_conversations(row, source_key, row_index, system_prompt)
    if conversation_row:
        return conversation_row

    medical_o1_row = normalize_medical_o1(
        row, source_key, row_index, system_prompt, include_cot
    )
    if medical_o1_row:
        return medical_o1_row

    return normalize_alpaca(row, source_key, row_index, system_prompt)


def normalize_row(
    row: dict[str, Any],
    source: SourcePreset,
    row_index: int,
    system_prompt: str,
    include_cot: bool,
) -> dict[str, Any] | None:
    if source.format_hint == "sharegpt":
        return normalize_conversations(row, source.key, row_index, system_prompt)
    if source.format_hint == "medical_o1":
        return normalize_medical_o1(row, source.key, row_index, system_prompt, include_cot)
    if source.format_hint == "alpaca":
        return normalize_alpaca(row, source.key, row_index, system_prompt)
    return normalize_auto(row, source.key, row_index, system_prompt, include_cot)


def build_output_row(
    source_key: str,
    row_index: int,
    question: str,
    answer: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "id": f"{source_key}:{row_index}",
        "dataset": source_key,
        "question": question,
        "answer": answer,
        "messages": messages,
    }


def collect_rows(
    source: SourcePreset,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = load_dataset_rows(source, endpoint=args.endpoint, streaming=not args.no_streaming)
    normalized_rows = []
    seen = 0
    skipped = 0

    for row in rows:
        seen += 1
        if not isinstance(row, dict):
            skipped += 1
            continue

        normalized = normalize_row(
            row=row,
            source=source,
            row_index=seen,
            system_prompt=args.system_prompt,
            include_cot=args.include_cot,
        )
        if normalized is None:
            skipped += 1
            continue

        normalized_rows.append(normalized)
        if args.limit > 0 and len(normalized_rows) >= args.limit:
            break

    print(
        f"{source.key}: collected {len(normalized_rows)} normalized rows "
        f"(seen {seen}, skipped {skipped})"
    )
    return normalized_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: list[dict[str, Any]],
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if test_size < 1:
        raise ValueError("--test-size must be at least 1")
    if len(rows) <= test_size:
        raise ValueError(f"Need more than {test_size} rows, got {len(rows)}")
    return rows[test_size:], rows[:test_size]


def write_summary(
    output_dir: Path,
    source_keys: list[str],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    summary = {
        "sources": source_keys,
        "output_dir": str(output_dir.resolve()),
        "train_file": "train.jsonl",
        "test_file": "test.jsonl",
        "all_file": "all.jsonl",
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "total_count": len(train_rows) + len(test_rows),
        "limit_per_source": args.limit,
        "seed": args.seed,
        "endpoint": args.endpoint,
        "include_cot": args.include_cot,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def selected_sources(source_arg: str) -> list[SourcePreset]:
    if source_arg == "all":
        return list(SOURCE_PRESETS.values())
    keys = [item.strip() for item in source_arg.split(",") if item.strip()]
    unknown = [key for key in keys if key not in SOURCE_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown source(s): {', '.join(unknown)}")
    return [SOURCE_PRESETS[key] for key in keys]


def list_sources() -> None:
    print("Available medical QA/SFT sources:")
    for key, source in SOURCE_PRESETS.items():
        dataset = source.dataset or ", ".join(source.data_files)
        config = f", config={source.config}" if source.config else ""
        print(f"  - {key}: {dataset}{config}")
        print(f"    {source.description}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download public medical QA datasets and convert them to chat JSONL."
    )
    parser.add_argument(
        "--source",
        default="medical_zh",
        help="Source key, comma-separated keys, or 'all'. Use --list-sources.",
    )
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "medical_qa_finetune",
        help="Output directory for normalized train/test JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum normalized rows per source. Use -1 for no limit.",
    )
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--endpoint",
        default="https://huggingface.co",
        help="HuggingFace endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--include-cot",
        action="store_true",
        help="For medical_o1 sources, prepend Complex_CoT inside <think>...</think>.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable datasets streaming mode. This may download/cache full files.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.endpoint = normalize_endpoint(args.endpoint)
    if args.list_sources:
        list_sources()
        return

    sources = selected_sources(args.source)
    print(f"Using HuggingFace endpoint: {args.endpoint}")
    all_rows: list[dict[str, Any]] = []
    for source in sources:
        all_rows.extend(collect_rows(source, args))

    if not all_rows:
        raise SystemExit("No usable rows were collected.")

    train_rows, test_rows = split_rows(all_rows, test_size=args.test_size, seed=args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all.jsonl", all_rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)
    write_summary(
        output_dir=output_dir,
        source_keys=[source.key for source in sources],
        train_rows=train_rows,
        test_rows=test_rows,
        args=args,
    )

    print(f"Total normalized rows: {len(all_rows)}")
    print(f"Train rows: {len(train_rows)} -> {output_dir / 'train.jsonl'}")
    print(f"Test rows: {len(test_rows)} -> {output_dir / 'test.jsonl'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
