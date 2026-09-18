"""Train a text-only DeepSeek LoRA adapter on MIRA manifest train_ids.

Run from the project root:
    python script/finetune_deepseek_mira_lora.py --check-env
    python script/finetune_deepseek_mira_lora.py --dry-run
    python script/finetune_deepseek_mira_lora.py --check-data
    python script/finetune_deepseek_mira_lora.py --epochs 1

Images, captions and extra answer hints are not model inputs. Only the selected
assistant answers (including EOS) contribute to loss. The base model stays
frozen and unchanged on disk; all adapters/checkpoints use a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"G:\Codex_dataset\MIRA-data")
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. Answer the question using "
    "the provided text and options. Give the answer directly without a thinking block."
)
REQUIRED_PACKAGES = {
    "torch": "2.1.0", "transformers": "4.44.0", "peft": "0.12.0",
    "accelerate": "0.33.0", "safetensors": "0.4.3",
}
ID_PATTERN = re.compile(
    r"mira:(train|validation|test):(0|[1-9][0-9]*):"
    r"(open_ended|closed_ended|single_choice|multiple_choice):(0|[1-9][0-9]*)"
)
TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class TextSample:
    sample_id: str
    question: str
    answer: Any
    options: Any = None


def parse_sample_id(sample_id):
    match = ID_PATTERN.fullmatch(sample_id) if isinstance(sample_id, str) else None
    if not match:
        raise ValueError(f"Invalid MIRA QA ID: {sample_id!r}")
    split, row, category, qa = match.groups()
    return split, int(row), category, int(qa)


def load_split_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest must be a JSON object.")
    train_ids, test_ids = payload.get("train_ids"), payload.get("test_ids", [])
    if not isinstance(train_ids, list) or not train_ids or not isinstance(test_ids, list):
        raise ValueError("Manifest requires nonempty train_ids and a test_ids array if present.")
    for label, values in (("train_ids", train_ids), ("test_ids", test_ids)):
        for value in values:
            parse_sample_id(value)
        if len(values) != len(set(values)):
            raise ValueError(f"{label} contains duplicate IDs.")
    if set(train_ids) & set(test_ids):
        raise ValueError("train_ids and test_ids overlap; test data must stay held out.")
    return train_ids, payload


def load_training_samples(data_root, train_ids):
    """Resolve selected QA IDs without reading, decoding or checking images."""
    requested = {}
    for sample_id in train_ids:
        split, row, category, qa = parse_sample_id(sample_id)
        requested.setdefault(split, {}).setdefault(row, []).append((category, qa, sample_id))
    found = {}
    csv.field_size_limit(64 * 1024 * 1024)
    for split, rows in requested.items():
        source = Path(data_root) / f"{split}.csv"
        print(f"Resolving {sum(map(len, rows.values()))} training IDs in {source}", flush=True)
        last_row = max(rows)
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if "vqa_json" not in (reader.fieldnames or []):
                raise ValueError(f"{source} is missing vqa_json.")
            for row_index, row in enumerate(reader):
                if row_index in rows:
                    if None in row or row.get("vqa_json") is None:
                        raise ValueError(f"Malformed CSV at {source}:{row_index} (zero-based record).")
                    data = json.loads(row["vqa_json"])
                    if not isinstance(data, dict):
                        raise ValueError(f"{source}:{row_index}: vqa_json must be an object.")
                    for category, qa_index, sample_id in rows[row_index]:
                        questions = data.get(category, [])
                        if not isinstance(questions, list) or qa_index >= len(questions):
                            raise ValueError(f"QA not found: {sample_id}")
                        qa = questions[qa_index]
                        if not isinstance(qa, dict):
                            raise ValueError(f"Invalid QA object: {sample_id}")
                        question, answer = qa.get("question"), qa.get("answer")
                        if not isinstance(question, str) or not question.strip() or answer is None or answer in ("", {}, []):
                            raise ValueError(f"Missing question or answer: {sample_id}; no samples are silently skipped.")
                        if isinstance(answer, str) and not answer.strip():
                            raise ValueError(f"Empty answer: {sample_id}")
                        found[sample_id] = TextSample(sample_id, question.strip(), answer, qa.get("options"))
                if row_index >= last_row:
                    break
        print(f"Resolved {len(found):,}/{len(train_ids):,} selected QAs", flush=True)
    missing = [sample_id for sample_id in train_ids if sample_id not in found]
    if missing:
        raise ValueError(f"{len(missing)} training IDs not found: {', '.join(missing[:5])}")
    return [found[sample_id] for sample_id in train_ids]


def json_text(value):
    return value.strip() if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sample_question(sample):
    text = f"Question: {sample.question}"
    if sample.options is not None and sample.options not in ("", [], {}):
        text += "\nOptions: " + json_text(sample.options)
    return text


def sample_answer(sample):
    return json_text(sample.answer)


def render_prompt(tokenizer, question, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Use the template's direct-answer prefix, without its generation-time <think>."""
    if not tokenizer.chat_template or not tokenizer.eos_token:
        raise ValueError("The local tokenizer must provide a chat template and EOS token.")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    empty = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=False
    )
    if not empty.endswith(tokenizer.eos_token):
        raise ValueError("Unsupported chat template: empty assistant does not end with EOS.")
    return empty[:-len(tokenizer.eos_token)]


def build_feature(sample, tokenizer, max_length, system_prompt=DEFAULT_SYSTEM_PROMPT):
    question, answer = sample_question(sample), sample_answer(sample)
    prompt = render_prompt(tokenizer, question, system_prompt)
    full = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt}, {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ], tokenize=False, add_generation_prompt=False)
    if full != prompt + answer + tokenizer.eos_token:
        raise ValueError(f"{sample.sample_id}: chat template changed the answer; refusing lossy labels.")
    ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"{sample.sample_id}: {len(ids)} tokens exceed --max-length {max_length}; "
                         "increase the limit. No question/answer was truncated.")
    if ids[:len(prefix)] != prefix:
        raise ValueError(f"{sample.sample_id}: tokenized prompt prefix mismatch; cannot safely mask labels.")
    if len(ids) <= len(prefix) + 1 or ids[-1] != tokenizer.eos_token_id:
        raise ValueError(f"{sample.sample_id}: missing answer tokens or final EOS.")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + ids[len(prefix):]}


class TextDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class TextCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch
        length = max(len(item["input_ids"]) for item in batch)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for item in batch:
            pad = length - len(item["input_ids"])
            for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                result[key].append(item[key] + [fill] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def format_duration(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressCallback:
    """Mixin with TrainerCallback at runtime so dry-run needs only stdlib."""
    def __init__(self, effective_batch_size):
        self.effective_batch_size = effective_batch_size
        self.intervals = deque(maxlen=20)
        self.started = None
        self.elapsed_seconds = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_time = time.perf_counter()
        self.last_step = int(state.global_step)
        self.intervals.clear()
        print(f"LoRA training: {state.max_steps:,} optimizer steps; effective batch={self.effective_batch_size}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is None or state.global_step <= self.last_step:
            return
        now = time.perf_counter()
        self.intervals.append((state.global_step - self.last_step, max(now - self.last_time, 1e-9)))
        speed = sum(steps for steps, _ in self.intervals) / sum(seconds for _, seconds in self.intervals)
        self.elapsed_seconds = now - self.started
        eta = max(state.max_steps - state.global_step, 0) / speed
        print(f"[train] step {state.global_step:,}/{state.max_steps:,} | {speed:.3f} step/s | "
              f"approx {speed * self.effective_batch_size:.2f} QA/s | "
              f"elapsed {format_duration(self.elapsed_seconds)} | ETA {format_duration(eta)}", flush=True)
        self.last_time, self.last_step = now, int(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if isinstance((logs or {}).get("loss"), (int, float)):
            print(f"[loss] step {state.global_step}: {logs['loss']:.6f}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        self.elapsed_seconds = 0.0 if self.started is None else time.perf_counter() - self.started
        print(f"LoRA training time: {format_duration(self.elapsed_seconds)} ({self.elapsed_seconds:.2f} s)", flush=True)


def output_is_safe(model_dir, output_dir):
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    if model_dir == output_dir or model_dir in output_dir.parents or output_dir in model_dir.parents:
        raise ValueError("Output must be outside the base model directory and its ancestors; base weights are protected.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output_dir}")


def version_tuple(version):
    values = [int(part) for part in re.findall(r"\d+", version.split("+", 1)[0])[:3]]
    return tuple(values + [0] * (3 - len(values)))


def dependency_report():
    lines, problems = [], []
    for package, minimum in REQUIRED_PACKAGES.items():
        try:
            installed = metadata.version(package)
            ok = version_tuple(installed) >= version_tuple(minimum)
            lines.append(f"{package}: {installed} ({'ok' if ok else 'too old'}, need >= {minimum})")
        except metadata.PackageNotFoundError:
            ok = False
            lines.append(f"{package}: missing (need >= {minimum})")
        if not ok:
            problems.append(package)
    try:
        import torch
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}; BF16: {torch.cuda.is_bf16_supported()}")
    except Exception as error:
        lines.append(f"PyTorch runtime error: {error}")
        problems.append("torch runtime")
    return lines, problems


def prepare_samples(args):
    train_ids, manifest = load_split_manifest(args.split_manifest)
    selected = train_ids[:args.limit] if args.limit else train_ids
    args.data_root = Path(args.data_root or manifest.get("data_root") or DEFAULT_DATA_ROOT).expanduser().resolve()
    samples = load_training_samples(args.data_root, selected)
    print(f"Manifest: {len(train_ids):,} train / {len(manifest.get('test_ids', [])):,} held-out test IDs; "
          f"selected for training: {len(samples):,}. Images/captions are excluded.", flush=True)
    return samples, manifest


def prepare_tokens(args, samples):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    features = []
    for index, sample in enumerate(samples, start=1):
        features.append(build_feature(sample, tokenizer, args.max_length, args.system_prompt))
        if index % 100 == 0 or index == len(samples):
            print(f"Tokenized {index:,}/{len(samples):,} QAs", flush=True)
    lengths = [len(feature["input_ids"]) for feature in features]
    stats = {"min_tokens": min(lengths), "max_tokens": max(lengths),
             "mean_tokens": sum(lengths) / len(lengths),
             "supervised_tokens": sum(sum(label != -100 for label in f["labels"]) for f in features)}
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, mean={stats['mean_tokens']:.1f}; "
          f"supervised answer/EOS tokens={stats['supervised_tokens']:,}", flush=True)
    return tokenizer, TextDataset(features), stats


def device_and_dtype(args, torch):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or a CUDA PyTorch environment.")
    name = args.dtype
    if name == "auto":
        name = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device == "cuda" else "float32"
    if device == "cpu" and name != "float32":
        raise ValueError("CPU training uses float32; choose --dtype auto or float32.")
    if device == "cuda" and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16.")
    return device, name, getattr(torch, name)


def train(args):
    started = time.perf_counter()
    lines, problems = dependency_report()
    print("\n".join(lines), flush=True)
    if problems:
        raise RuntimeError("Missing/incompatible training dependencies: " + ", ".join(problems) +
                           ". Run: python -m pip install -r script/requirements_deepseek_mira_lora.txt")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed

    output_is_safe(args.model_dir, args.output_dir)
    set_seed(args.seed)
    samples, manifest = prepare_samples(args)
    tokenizer, dataset, stats = prepare_tokens(args, samples)
    device, dtype_name, dtype = device_and_dtype(args, torch)
    print(f"Loading {args.model_dir} on {device} with {dtype_name}", flush=True)
    dtype_key = "dtype" if version_tuple(metadata.version("transformers")) >= (4, 56, 0) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation, **{dtype_key: dtype},
    )
    model.config.use_cache = False
    targets = args.target_modules.split(",")
    available = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    if set(targets) - available:
        raise ValueError(f"LoRA targets missing from model: {sorted(set(targets) - available)}")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none", target_modules=targets,
    ))
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Expected only LoRA parameters to be trainable.")
    model.to(device)
    model.print_trainable_parameters()
    values = dict(
        output_dir=str(args.output_dir), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, lr_scheduler_type="cosine", max_grad_norm=1.0,
        logging_strategy="steps", logging_steps=1, logging_first_step=True,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda", optim="adamw_torch",
        bf16=dtype_name == "bfloat16", fp16=dtype_name == "float16", use_cpu=device == "cpu",
        seed=args.seed, data_seed=args.seed, disable_tqdm=True, label_names=["labels"],
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Transformers 5 always saves safetensors and removed this 4.x option.
    if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
        values["save_safetensors"] = True
    training_args = TrainingArguments(**values)

    class ProgressCallback(TrainingProgressCallback, TrainerCallback):
        pass

    progress = ProgressCallback(args.batch_size * args.gradient_accumulation_steps)
    tokenizer_key = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=TextCollator(tokenizer.pad_token_id), callbacks=[progress],
                      **{tokenizer_key: tokenizer})
    trainer.train()
    trainer.save_model(str(args.output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
    metadata_payload = {
        "base_model_dir": str(args.model_dir), "split_manifest": str(args.split_manifest),
        "data_root": str(args.data_root), "manifest_train_count": len(manifest["train_ids"]),
        "actual_train_count": len(samples), "train_ids": [sample.sample_id for sample in samples],
        "input_mode": "text_only_question_and_options", "images_used": False, "captions_used": False,
        "target": "full_answer_including_nested_fields_and_eos", "token_statistics": stats,
        "prompt_mode": "empty_assistant_template_without_final_eos; no generation-time think prefix",
        "device": device, "dtype": dtype_name,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": targets},
        "training": {"epochs": args.epochs, "max_steps": args.max_steps,
                     "completed_optimizer_steps": trainer.state.global_step,
                     "batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                     "learning_rate": args.learning_rate, "max_length": args.max_length,
                     "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                     "gradient_checkpointing": args.gradient_checkpointing, "seed": args.seed,
                     "system_prompt": args.system_prompt},
        "training_elapsed_seconds": progress.elapsed_seconds,
        "elapsed_seconds_through_adapter_save": time.perf_counter() - started,
        "package_versions": {package: metadata.version(package) for package in REQUIRED_PACKAGES},
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = time.perf_counter() - started
    print(f"Saved LoRA adapter/tokenizer: {args.output_dir}", flush=True)
    print(f"Total time including data, loading, training and saving: {format_duration(total)} ({total:.2f} s)", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-env", action="store_true", help="Check packages/CUDA only; do not load model weights.")
    mode.add_argument("--dry-run", action="store_true", help="Validate all selected IDs/QA text, without ML dependencies.")
    mode.add_argument("--check-data", action="store_true", help="Validate IDs and actual tokenizer labels/lengths, without model weights.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "DeepSeek-Model")
    parser.add_argument("--split-manifest", type=Path, default=Path(__file__).resolve().parent / "mira_split_ids.json")
    parser.add_argument("--data-root", type=Path, help="Default: manifest data_root, then G:/Codex_dataset/MIRA-data.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "deepseek_mira_lora_adapter")
    parser.add_argument("--limit", type=int, default=0, help="First N train_ids for a short experiment; 0 uses all.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=TARGET_MODULES)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    return parser


def validate_args(args):
    for name in ("batch_size", "gradient_accumulation_steps", "max_length", "lora_r", "lora_alpha", "save_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.limit < 0 or args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("Invalid --limit or --max-steps.")
    if not all(math.isfinite(getattr(args, name)) for name in (
        "epochs", "learning_rate", "warmup_ratio", "weight_decay", "lora_dropout"
    )) or args.epochs <= 0 or args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1 or args.weight_decay < 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Invalid epochs, learning rate, warmup, weight decay or dropout.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use python in a single process; distributed launch is not supported.")
    targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not targets or set(targets) - set(TARGET_MODULES.split(",")):
        raise ValueError(f"--target-modules must use Qwen2 projection names from: {TARGET_MODULES}")
    args.target_modules = ",".join(dict.fromkeys(targets))
    for name in ("model_dir", "split_manifest", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    output_is_safe(args.model_dir, args.output_dir)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2" or "Qwen2ForCausalLM" not in config.get("architectures", []):
        raise ValueError("This script targets the local DeepSeek Qwen2ForCausalLM text model.")
    if args.max_length > config.get("max_position_embeddings", 0):
        raise ValueError("--max-length exceeds the base model's context window.")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.check_env:
            lines, problems = dependency_report()
            print("\n".join(lines))
            if problems:
                print("Install with: python -m pip install -r script/requirements_deepseek_mira_lora.txt")
            return 1 if problems else 0
        validate_args(args)
        if args.dry_run or args.check_data:
            samples, _ = prepare_samples(args)
            if args.check_data:
                prepare_tokens(args, samples)
            print("Validation complete; no model weights loaded and no files written.", flush=True)
        else:
            train(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; the original model weights were not modified.", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError, csv.Error, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
