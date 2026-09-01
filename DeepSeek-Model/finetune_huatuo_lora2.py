# -*- coding: utf-8 -*-
"""
Single-pair LoRA overfit training for DeepSeek/Qwen2.

This script is meant for verification, not for real medical capability.
It selects exactly one QA pair, trains LoRA on that pair, and stops only when
both conditions are met:
  1. teacher-forced answer loss <= --target-loss
  2. generated answer similarity >= --target-similarity

Examples from the project root:
    python DeepSeek-Model/finetune_huatuo_lora2.py --dry-run
    python DeepSeek-Model/finetune_huatuo_lora2.py --sample-index 3
    python DeepSeek-Model/finetune_huatuo_lora2.py --selection random --seed 7
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = {
    "torch": "2.1.0",
    "transformers": "4.44.0",
    "peft": "0.11.0",
    "safetensors": "0.4.3",
}

DEFAULT_SYSTEM_PROMPT = (
    "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class SelectedSample:
    row: dict[str, Any]
    index: int


@dataclass
class TrainingState:
    step: int = 0
    loss: float = math.inf
    token_accuracy: float = 0.0
    similarity: float = 0.0
    generated_answer: str = ""
    stopped_reason: str = "max_steps_reached"


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


def row_to_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def answer_from_row(row: dict[str, Any], system_prompt: str) -> str:
    return row_to_messages(row, system_prompt)[-1]["content"]


def question_from_row(row: dict[str, Any], system_prompt: str) -> str:
    messages = row_to_messages(row, system_prompt)
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return str(row.get("question", "")).strip()


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


def row_label(row: dict[str, Any]) -> str:
    parts = []
    if row.get("id"):
        parts.append(f"id={row['id']}")
    if row.get("source_file"):
        parts.append(f"source_file={row['source_file']}")
    if row.get("source_line"):
        parts.append(f"source_line={row['source_line']}")
    return ", ".join(parts) if parts else "selected sample"


def text_preview(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def select_sample(rows: list[dict[str, Any]], args: argparse.Namespace) -> SelectedSample:
    if not rows:
        raise SystemExit(f"No rows found in data file: {args.data_file}")

    if args.sample_index is not None:
        if args.sample_index < 1 or args.sample_index > len(rows):
            raise SystemExit(
                f"--sample-index must be between 1 and {len(rows)}, got {args.sample_index}"
            )
        return SelectedSample(row=rows[args.sample_index - 1], index=args.sample_index)

    indexed_rows = list(enumerate(rows, start=1))
    candidates = []
    for index, row in indexed_rows:
        answer = answer_from_row(row, args.system_prompt)
        if args.max_answer_chars > 0 and len(answer) > args.max_answer_chars:
            continue
        candidates.append((index, row))

    if not candidates:
        candidates = indexed_rows

    if args.selection == "first":
        index, row = candidates[0]
    else:
        index, row = random.Random(args.seed).choice(candidates)

    return SelectedSample(row=row, index=index)


def print_selected_sample(sample: SelectedSample, args: argparse.Namespace) -> None:
    row = sample.row
    question = question_from_row(row, args.system_prompt)
    answer = answer_from_row(row, args.system_prompt)
    print("=" * 80)
    print("Selected QA pair")
    print("=" * 80)
    print(f"data_file: {args.data_file.resolve()}")
    print(f"sample_index: {sample.index}")
    print(f"id: {row.get('id', '')}")
    print(f"source_file: {row.get('source_file', '')}")
    print(f"source_line: {row.get('source_line', '')}")
    print()
    print("[Question]")
    print(question)
    print()
    print("[Expected Answer]")
    print(answer)
    print("=" * 80)


def normalize_for_similarity(text: str) -> str:
    text = re.sub(r".*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？、；：,.!?;:\-—_()\[\]{}（）【】《》\"'“”‘’]", "", text)
    return text.lower()


def text_similarity(generated: str, expected: str) -> float:
    left = normalize_for_similarity(generated)
    right = normalize_for_similarity(expected)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def build_feature(row: dict[str, Any], tokenizer: Any, max_length: int, system_prompt: str) -> dict[str, Any]:
    messages = row_to_messages(row, system_prompt)
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
        print(
            f"[max_length warning] {row_label(row)} has {full_token_count} tokens, "
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
        raise SystemExit("Selected example has no trainable answer tokens after tokenization.")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "full_text": full_text,
        "prefix_text": prefix_text,
        "answer": answer,
        "full_token_count": full_token_count,
        "answer_token_count": sum(1 for label in labels if label != -100),
    }


def tensor_batch(feature: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([feature["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [feature["attention_mask"]], dtype=torch.long, device=device
        ),
        "labels": torch.tensor([feature["labels"]], dtype=torch.long, device=device),
    }


def token_accuracy(logits: Any, labels: Any) -> float:
    mask = labels[:, 1:] != -100
    if mask.sum().item() == 0:
        return 0.0
    predictions = logits[:, :-1, :].argmax(dim=-1)
    correct = (predictions[mask] == labels[:, 1:][mask]).float().mean().item()
    return float(correct)


def generate_answer(
    model: Any,
    tokenizer: Any,
    feature: dict[str, Any],
    device: str,
    args: argparse.Namespace,
) -> str:
    import torch

    prompt = feature["prefix_text"]
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    max_new_tokens = args.generate_max_new_tokens
    if max_new_tokens <= 0:
        max_new_tokens = min(feature["answer_token_count"] + args.generate_extra_tokens, 512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    feature: dict[str, Any],
    device: str,
    args: argparse.Namespace,
) -> TrainingState:
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(**batch)
        loss = float(outputs.loss.detach().cpu())
        acc = token_accuracy(outputs.logits, batch["labels"])

    generated = generate_answer(model, tokenizer, feature, device, args)
    similarity = text_similarity(generated, feature["answer"])
    return TrainingState(
        loss=loss,
        token_accuracy=acc,
        similarity=similarity,
        generated_answer=generated,
    )


def model_dtype_arg(dtype: Any) -> dict[str, Any]:
    transformers_version = metadata.version("transformers")
    if version_ok(transformers_version, "5.0.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_kwargs.update(model_dtype_arg(dtype))
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

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
    model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer, device


def should_stop(state: TrainingState, args: argparse.Namespace) -> bool:
    if state.step < args.min_steps:
        return False
    return state.loss <= args.target_loss and state.similarity >= args.target_similarity


def save_outputs(
    model: Any,
    tokenizer: Any,
    sample: SelectedSample,
    feature: dict[str, Any],
    state: TrainingState,
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    selected_path = args.output_dir / "selected_sample.json"
    selected_path.write_text(
        json.dumps(
            {
                "sample_index": sample.index,
                "row": sample.row,
                "expected_answer": feature["answer"],
                "final_generated_answer": state.generated_answer,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata_path = args.output_dir / "single_pair_lora_training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "base_model_dir": str(args.model_dir.resolve()),
                "data_file": str(args.data_file.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "sample_index": sample.index,
                "sample_id": sample.row.get("id", ""),
                "max_steps": args.max_steps,
                "min_steps": args.min_steps,
                "eval_every": args.eval_every,
                "target_loss": args.target_loss,
                "target_similarity": args.target_similarity,
                "final_step": state.step,
                "final_loss": state.loss,
                "final_token_accuracy": state.token_accuracy,
                "final_similarity": state.similarity,
                "stopped_reason": state.stopped_reason,
                "max_length": args.max_length,
                "full_token_count": feature["full_token_count"],
                "trained_input_token_count": len(feature["input_ids"]),
                "learning_rate": args.learning_rate,
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


def append_log(path: Path, state: TrainingState) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            json.dumps(
                {
                    "step": state.step,
                    "loss": state.loss,
                    "token_accuracy": state.token_accuracy,
                    "similarity": state.similarity,
                    "generated_answer": state.generated_answer,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def train(args: argparse.Namespace) -> None:
    args.model_dir = args.model_dir.resolve()
    args.data_file = args.data_file.resolve()
    args.output_dir = args.output_dir.resolve()

    rows = load_jsonl(args.data_file)
    sample = select_sample(rows, args)
    print_selected_sample(sample, args)

    if args.dry_run:
        print("Dry run enabled; no model was loaded and no training was started.")
        return

    ensure_dependencies()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, device = load_model_and_tokenizer(args)
    feature = build_feature(sample.row, tokenizer, args.max_length, args.system_prompt)
    batch = tensor_batch(feature, device)

    print(f"Device: {device}")
    print(f"Full input tokens before truncation: {feature['full_token_count']}")
    print(f"Input tokens used for training: {len(feature['input_ids'])}")
    print(f"Supervised answer tokens: {feature['answer_token_count']}")
    print(f"Output adapter directory: {args.output_dir}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    log_path = args.output_dir / "training_log.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    best_state = TrainingState()
    initial_state = evaluate(model, tokenizer, batch, feature, device, args)
    initial_state.step = 0
    print(
        f"[eval step 0] loss={initial_state.loss:.6f} "
        f"ppl={math.exp(min(initial_state.loss, 20)):.3f} "
        f"token_acc={initial_state.token_accuracy:.4f} "
        f"similarity={initial_state.similarity:.4f}"
    )
    print(f"[generated step 0] {initial_state.generated_answer}")
    append_log(log_path, initial_state)
    best_state = initial_state

    if should_stop(initial_state, args):
        initial_state.stopped_reason = "target_reached_before_training"
        save_outputs(model, tokenizer, sample, feature, initial_state, args)
        print(f"Target already reached after min_steps={args.min_steps}.")
        print(f"Saved LoRA adapter to: {args.output_dir}")
        return

    for step in range(1, args.max_steps + 1):
        model.train()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                args.max_grad_norm,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every != 0 and step != args.max_steps:
            continue

        state = evaluate(model, tokenizer, batch, feature, device, args)
        state.step = step
        if state.loss < best_state.loss or state.similarity > best_state.similarity:
            best_state = state

        print(
            f"[eval step {step}] loss={state.loss:.6f} "
            f"ppl={math.exp(min(state.loss, 20)):.3f} "
            f"token_acc={state.token_accuracy:.4f} "
            f"similarity={state.similarity:.4f}"
        )
        print(f"[generated step {step}] {state.generated_answer}")
        append_log(log_path, state)

        if should_stop(state, args):
            state.stopped_reason = "target_reached"
            save_outputs(model, tokenizer, sample, feature, state, args)
            print(f"Target reached after step {step}. Saved LoRA adapter to: {args.output_dir}")
            return

    best_state.stopped_reason = "max_steps_reached"
    save_outputs(model, tokenizer, sample, feature, best_state, args)
    print("Reached max steps before both stop conditions were met.")
    print(f"Best observed loss={best_state.loss:.6f}, similarity={best_state.similarity:.4f}")
    print(f"Saved latest LoRA adapter to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Single-pair LoRA overfit training for verifying fine-tuning effects."
    )
    parser.add_argument("--check-env", action="store_true", help="Only check dependencies.")
    parser.add_argument("--dry-run", action="store_true", help="Only select and print one sample.")
    parser.add_argument("--model-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=project_root / "dataset" / "huatuo_deepseek_finetune" / "all_usable.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "huatuo_lora_single_pair_adapter",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "first"),
        default="random",
        help="How to select a sample when --sample-index is not set.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="1-based row index in --data-file. Overrides --selection and answer length filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=500,
        help="Random/first selection only considers answers no longer than this. Use 0 to disable.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum optimization steps. Increase this to train longer.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Do not stop before this many optimization steps, even if targets are met.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument(
        "--target-similarity",
        type=float,
        default=0.90,
        help="Stop when generated answer similarity reaches this value and loss is low enough.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    # parser.add_argument("--generate-max-new-tokens", type=int, default=0)
    parser.add_argument("--generate-max-new-tokens", type=int, default=512)
    parser.add_argument("--generate-extra-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
