# -*- coding: utf-8 -*-
"""
Fine-tune the local DeepSeek/Qwen2 model with Huatuo QA data using LoRA.

This script saves only LoRA adapter weights. It does not overwrite the base
model weights in model.safetensors.

Typical usage from the project root:
    python DeepSeek-Model/finetune_huatuo_lora.py --check-env
    python DeepSeek-Model/finetune_huatuo_lora.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = {
    "torch": "2.1.0",
    "transformers": "4.44.0",
    "peft": "0.11.0",
    "accelerate": "0.33.0",
    "safetensors": "0.4.3",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_version(version: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(number) for number in numbers[:3])


def version_ok(installed: str, minimum: str) -> bool:
    installed_tuple = parse_version(installed)
    minimum_tuple = parse_version(minimum)
    max_len = max(len(installed_tuple), len(minimum_tuple))
    installed_tuple += (0,) * (max_len - len(installed_tuple))
    minimum_tuple += (0,) * (max_len - len(minimum_tuple))
    return installed_tuple >= minimum_tuple


def dependency_report() -> tuple[list[str], list[str]]:
    lines = []
    problems = []

    for package, minimum in REQUIRED_PACKAGES.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            lines.append(f"{package}: missing, need >= {minimum}")
            problems.append(package)
            continue

        status = "ok" if version_ok(installed, minimum) else "too old"
        lines.append(f"{package}: {installed} ({status}, need >= {minimum})")
        if status != "ok":
            problems.append(package)

    try:
        import torch

        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device: {torch.cuda.get_device_name(0)}")
            lines.append(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
    except Exception as exc:
        lines.append(f"torch_import_failed: {exc}")
        problems.append("torch")

    return lines, problems


def ensure_dependencies() -> None:
    lines, problems = dependency_report()
    if problems:
        print("Environment check failed:")
        for line in lines:
            print(f"  - {line}")
        print()
        print("Install/update dependencies, then rerun training. For example:")
        print("  pip install -r DeepSeek-Model/requirements-lora.txt")
        raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role and content:
                cleaned.append({"role": role, "content": content})
        if cleaned and cleaned[-1]["role"] == "assistant":
            return cleaned

    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError(f"Row lacks usable messages/question/answer fields: {row}")

    return [
        {
            "role": "system",
            "content": "你是一名谨慎的中文医疗问答助手。请给出准确、清晰、简洁的医学科普回答。",
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}: {message['content']}")
    return "\n".join(rendered)


def find_supervised_prefix(full_text: str, assistant_answer: str) -> str:
    answer_start = full_text.rfind(assistant_answer)
    if answer_start < 0:
        raise ValueError("Could not locate assistant answer in rendered chat text")
    return full_text[:answer_start]


def row_label(row: dict[str, Any], row_number: int) -> str:
    parts = [f"row={row_number}"]
    if row.get("id"):
        parts.append(f"id={row['id']}")
    if row.get("source_file"):
        parts.append(f"source_file={row['source_file']}")
    if row.get("source_line"):
        parts.append(f"source_line={row['source_line']}")
    return ", ".join(parts)


def text_preview(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class ChatSFTDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        split_name: str,
        warning_limit: int,
    ):
        self.features = []
        self.skipped = 0
        self.truncated = 0
        self.total_rows = len(rows)
        self._warning_limit = warning_limit

        for row_number, row in enumerate(rows, start=1):
            messages = row_to_messages(row)
            answer = messages[-1]["content"]
            full_text = render_chat(tokenizer, messages)
            prefix_text = find_supervised_prefix(full_text, answer)
            full_token_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            full_token_count = len(full_token_ids)

            if full_token_count > max_length:
                self.truncated += 1
                if self.truncated <= warning_limit:
                    print(
                        f"[max_length warning][{split_name}] "
                        f"{row_label(row, row_number)} has {full_token_count} tokens, "
                        f"which exceeds max_length={max_length}. It will be truncated."
                    )
                    print(f"  question_preview: {text_preview(messages[-2]['content'])}")
                    print(f"  answer_preview: {text_preview(answer)}")

            full_tokens = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            prefix_tokens = tokenizer(
                prefix_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = full_tokens["input_ids"]
            labels = list(input_ids)
            prefix_len = min(len(prefix_tokens["input_ids"]), len(labels))
            labels[:prefix_len] = [-100] * prefix_len

            if not input_ids or all(label == -100 for label in labels):
                self.skipped += 1
                continue

            self.features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

        if self.truncated > warning_limit:
            remaining = self.truncated - warning_limit
            print(
                f"[max_length warning][{split_name}] {remaining} more samples exceeded "
                f"max_length={max_length} and were not printed."
            )
        if self.truncated:
            print(
                f"[max_length summary][{split_name}] {self.truncated}/{self.total_rows} "
                f"samples exceeded max_length={max_length}."
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]


class CausalLMCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_training_arguments(args: argparse.Namespace, use_bf16: bool, use_fp16: bool) -> Any:
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "do_train": True,
        "do_eval": True,
    }

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    ignored = sorted(set(kwargs) - set(filtered_kwargs))
    if ignored:
        print(f"Ignored unsupported TrainingArguments: {', '.join(ignored)}")

    return TrainingArguments(**filtered_kwargs)


def train(args: argparse.Namespace) -> None:
    ensure_dependencies()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

    args.model_dir = args.model_dir.resolve()
    args.train_file = args.train_file.resolve()
    args.eval_file = args.eval_file.resolve()
    args.output_dir = args.output_dir.resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if version_ok(metadata.version("transformers"), "5.0.0"):
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    train_dataset = ChatSFTDataset(
        train_rows,
        tokenizer,
        args.max_length,
        split_name="train",
        warning_limit=args.max_length_warning_limit,
    )
    eval_dataset = ChatSFTDataset(
        eval_rows,
        tokenizer,
        args.max_length,
        split_name="eval",
        warning_limit=args.max_length_warning_limit,
    )

    if len(train_dataset) == 0:
        raise SystemExit("No trainable examples remained after tokenization.")
    if len(eval_dataset) == 0:
        raise SystemExit("No eval examples remained after tokenization.")

    print(f"Train examples: {len(train_dataset)} (skipped {train_dataset.skipped})")
    print(f"Eval examples: {len(eval_dataset)} (skipped {eval_dataset.skipped})")
    print(f"Output adapter directory: {args.output_dir}")

    training_args = build_training_arguments(args, use_bf16=use_bf16, use_fp16=use_fp16)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": CausalLMCollator(tokenizer),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata_path = args.output_dir / "huatuo_lora_training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "base_model_dir": str(args.model_dir),
                "train_file": str(args.train_file),
                "eval_file": str(args.eval_file),
                "epochs": args.epochs,
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": args.target_modules,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for local DeepSeek-Model with Huatuo QA JSONL data."
    )
    parser.add_argument("--check-env", action="store_true", help="Only check dependencies.")
    parser.add_argument("--model-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=project_root / "dataset" / "huatuo_deepseek_finetune" / "train.jsonl",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=project_root / "dataset" / "huatuo_deepseek_finetune" / "test.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "huatuo_lora_adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Set a positive value for a short debug run; default uses epochs.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--max-length-warning-limit",
        type=int,
        default=20,
        help="Maximum number of over-length samples to print per split.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names used as LoRA targets.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_env:
        lines, problems = dependency_report()
        for line in lines:
            print(line)
        if problems:
            raise SystemExit(1)
        return

    train(args)


if __name__ == "__main__":
    main()
