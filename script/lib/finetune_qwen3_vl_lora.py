"""使用 MIRA 图文问答为本地 Qwen3-VL-2B-Instruct 训练 LoRA adapter。

本文件是统一入口调用的 Qwen3-VL 训练后端，不建议直接运行。在项目根目录和
CUDA 版 ``MLMtest`` 环境中使用统一入口：

    python script/finetune_mira_lora.py -- --check-env
    python script/finetune_mira_lora.py -- --dry-run
    python script/finetune_mira_lora.py -- --epochs 1

环境与依赖：
    - 最低版本：torch 2.6、torchvision 0.21、transformers 4.57.3（小于 5）、
      peft 0.17（小于 0.19）、accelerate 1.2（小于 2）、safetensors 0.4.3、
      Pillow 10。应安装在本地 Qwen 推理所用的 CUDA 环境中。
    - 安装命令：python -m pip install "torch>=2.6" "torchvision>=0.21"
      "transformers>=4.57.3,<5" "peft>=0.17,<0.19" "accelerate>=1.2,<2"
      "safetensors>=0.4.3" "Pillow>=10"
    - 本机曾验证的解释器为
      ``D:\\mySoftware2\\anaconda3\\envs\\MLMtest\\python.exe``。

数据与监督规则：
    - 默认基座是项目根目录 ``Qwen3-VL-2B-Instruct``。从
      ``output/mira_split_ids.json`` 读取 ``train_ids`` 并回源到 MIRA CSV；
      保持清单顺序，拒绝重复 ID 或与 ``test_ids`` 重叠，不在训练中使用测试集。
      统一 JSON 中 ``split_manifest: null`` 时使用数据目录 ``train.csv`` 的全部问答。
    - 一条样本包含一个问答及其全部图片。用户输入仅含问题、选项和图片；
      caption 与其他源字段不会作为提示。结构化答案和嵌套解释会完整保留。
    - 缺少问题或答案的问答会显式跳过并记录；缺失 ID/图片会报错。图片在
      collator 中处理，数据集只保存记录和路径，避免长期复制全部图像张量。
    - 只对 assistant 答案和回合结束 token 计算损失，图片、system、user 和
      padding token 的标签为 -100。超长样本会报告 ID，不截断图片或答案。

训练与输出：
    - 默认 1 epoch、batch size 1、梯度累积 8、学习率 2e-4；LoRA rank 16、
      alpha 32、dropout 0.05，启用 SDPA 和梯度检查点。只训练语言解码器投影层
      adapter，基座和视觉权重保持冻结。CUDA 优先 BF16，否则 FP16；CPU FP32。
    - 快速试跑：``--limit 16 --max-steps 2 --output-dir
      output/qwen3_vl_lora_smoke``。显存不足时保持 batch size 1，可将
      ``--max-pixels`` 降至 131072；默认每图像素预算为 4096～262144。
    - 默认输出 ``output/qwen3_vl_2b_lora_adapter``，包含 adapter、processor、
      tokenizer、training_metadata.json、trainer_state.json 和 checkpoint。脚本
      不会向原模型目录写入或合并权重。若目录已有结果，默认改用相邻的
      ``原目录名_run_年月日_时分秒``，保留旧结果；``--on-existing-output error``
      可恢复遇到非空目录即停止的行为。实际保存路径会打印到控制台。
    - 可用 ``--resume-from-checkpoint`` 恢复训练，但应保持相同清单、模型、数据
      和训练设置。最终 adapter 不是独立完整模型；推理时需加载同一 Qwen3-VL
      基座，再用 ``PeftModel.from_pretrained`` 加载 adapter。

每个优化器步骤会输出速度、约 QA/s、耗时和 ETA；最后不足一个有效批量时
QA/s 是近似值，ETA 也不包含最终保存时间。
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
INFERENCE_DIR = PROJECT_ROOT / "inferenceValid"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from embed_mira_chroma import DEFAULT_DATA_ROOT, Sample, image_path, iter_samples  # noqa: E402

try:
    from .terminal_progress import TerminalProgress
except ImportError:  # 兼容直接导入本脚本的离线测试和旧调用方式
    from terminal_progress import TerminalProgress


LOGGER = logging.getLogger("finetune_qwen3_vl_lora")
REQUIRED_PACKAGES = {
    "torch": "2.6.0",
    "transformers": "4.57.3",
    "peft": "0.17.0",
    "accelerate": "1.2.0",
    "safetensors": "0.4.3",
    "Pillow": "10.0.0",
}
DEFAULT_SYSTEM_PROMPT = "你是一名谨慎的医学视觉问答助手。请根据问题和图片给出准确、清晰的回答。"


def parse_version(version: str) -> tuple[int, ...]:
    """Convert a package version into comparable numeric components."""
    values = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(value) for value in values[:3])


def version_at_least(installed: str, minimum: str) -> bool:
    """Compare versions without requiring packaging at import time."""
    width = max(len(parse_version(installed)), len(parse_version(minimum)))
    left = parse_version(installed) + (0,) * (width - len(parse_version(installed)))
    right = parse_version(minimum) + (0,) * (width - len(parse_version(minimum)))
    return left >= right


def dependency_report() -> tuple[list[str], list[str]]:
    """Report required versions and CUDA capability without importing training code."""
    lines, problems = [], []
    for package, minimum in REQUIRED_PACKAGES.items():
        lookup = "Pillow" if package == "Pillow" else package
        try:
            installed = metadata.version(lookup)
        except metadata.PackageNotFoundError:
            lines.append(f"{package}: missing (need >= {minimum})")
            problems.append(package)
            continue
        ok = version_at_least(installed, minimum)
        lines.append(f"{package}: {installed} ({'ok' if ok else 'too old'}, need >= {minimum})")
        if not ok:
            problems.append(package)
    try:
        import torch

        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device: {torch.cuda.get_device_name(0)}")
            lines.append(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
            free, total = torch.cuda.mem_get_info()
            lines.append(f"cuda_memory_free_gb: {free / 1024**3:.2f}/{total / 1024**3:.2f}")
    except Exception as error:
        lines.append(f"torch_runtime_error: {error}")
        if "torch" not in problems:
            problems.append("torch")
    return lines, problems


def ensure_dependencies() -> None:
    """Stop with an actionable message instead of failing halfway through loading 4 GB."""
    lines, problems = dependency_report()
    for line in lines:
        print(line)
    if problems:
        raise RuntimeError(
            "Missing or outdated packages: " + ", ".join(problems) +
            ". Install them in the same environment, for example: "
            "python -m pip install \"torch>=2.6\" \"torchvision>=0.21\" "
            "\"transformers>=4.57.3,<5\" \"peft>=0.17,<0.19\" "
            "\"accelerate>=1.2,<2\" \"safetensors>=0.4.3\" \"Pillow>=10\"")


def load_split_manifest(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Read and validate the generated train/test ID manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("train_ids"), list):
        raise ValueError(f"{path} must contain a JSON object with a train_ids array.")
    train_ids = payload["train_ids"]
    if not train_ids or any(not isinstance(value, str) or not value for value in train_ids):
        raise ValueError("train_ids must be a nonempty array of string IDs.")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("train_ids contains duplicate IDs.")
    test_ids = payload.get("test_ids", [])
    if not isinstance(test_ids, list) or any(not isinstance(value, str) for value in test_ids):
        raise ValueError("test_ids must be an array of string IDs.")
    if set(train_ids).intersection(test_ids):
        raise ValueError("train_ids and test_ids overlap; the test set must remain held out.")
    for sample_id in train_ids:
        id_split(sample_id)
    return train_ids, payload


def id_split(sample_id: str) -> str:
    """Extract the source CSV split from mira:{split}:{row}:{category}:{qa}."""
    match = re.fullmatch(
        r"mira:(train|validation|test):(0|[1-9]\d*):"
        r"(open_ended|closed_ended|single_choice|multiple_choice):(0|[1-9]\d*)", sample_id)
    if match is None:
        raise ValueError(f"Unsupported MIRA QA ID: {sample_id!r}")
    return match.group(1)


def load_training_samples(data_root: Path, train_ids: list[str]) -> list[Sample]:
    """Resolve manifest IDs through the canonical MIRA CSV reader."""
    data_root = data_root.expanduser().resolve()
    requested = set(train_ids)
    found: dict[str, Sample] = {}
    for split in sorted({id_split(sample_id) for sample_id in train_ids}):
        source = data_root / f"{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"CSV for IDs is missing: {source}")
        LOGGER.info("Resolving training IDs in %s", source)
        for sample in iter_samples(data_root, split):
            if sample.id in requested:
                found[sample.id] = sample
    missing = [sample_id for sample_id in train_ids if sample_id not in found]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} training IDs were not found in the CSV files: {preview}")
    return [found[sample_id] for sample_id in train_ids]


def json_text(value: Any) -> str:
    """Render structured answers/options without losing non-string fields."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sample_question(sample: Sample) -> str:
    """Build the user text while keeping the answer exclusively in the assistant turn."""
    question = sample.question or "[not provided in source]"
    parts = [f"问题：{question}"]
    if sample.options:
        parts.append(f"选项：{json_text(sample.options)}")
    return "\n".join(parts)


def sample_answer(sample: Sample) -> str:
    """Render the target answer as text for completion-only supervision."""
    if sample.answer in (None, "", {}, []):
        return "[not provided in source]"
    return json_text(sample.answer)


def read_image(path: Path):
    """Load and detach an RGB image so the file handle never reaches the processor."""
    from PIL import Image, ImageOps

    if not path.is_file():
        raise FileNotFoundError(f"Image referenced by {path} does not exist")
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def messages_for_sample(sample: Sample, data_root: Path, system_prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    """Create Qwen3-VL messages with all QA images in their source order."""
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": sample_question(sample)})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def full_messages(sample: Sample, data_root: Path, system_prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    """Add the assistant answer to the Qwen chat used for teacher forcing."""
    messages = messages_for_sample(sample, data_root, system_prompt, images)
    messages.append({"role": "assistant", "content": sample_answer(sample)})
    return messages


@dataclass
class MIRARecord:
    """Pickle-friendly record kept by the training dataset."""

    sample: Sample
    data_root: Path


class MIRATrainingDataset:
    """A small index of selected MIRA QAs; image tensors are built per batch."""

    def __init__(self, samples: list[Sample], data_root: Path):
        self.records = [MIRARecord(sample, data_root) for sample in samples
                        if not sample.missing_fields]
        self.skipped_ids = [sample.id for sample in samples if sample.missing_fields]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> MIRARecord:
        return self.records[index]


class QwenVLDataCollator:
    """Tokenize one QA at a time, then concatenate variable-size image patches."""

    def __init__(self, processor: Any, tokenizer: Any, max_length: int,
                 min_pixels: int, max_pixels: int, system_prompt: str):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = system_prompt
        configure_processor(processor, min_pixels, max_pixels)

    def _one(self, record: MIRARecord) -> dict[str, Any]:
        import torch

        images = []
        try:
            for name in record.sample.images:
                images.append(read_image(image_path(record.data_root, name)))
            messages = full_messages(record.sample, record.data_root, self.system_prompt, images)
            prompt_messages = messages_for_sample(record.sample, record.data_root, self.system_prompt, images)
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            encoded = self.processor(text=[full_text], images=images or None,
                                     padding=False, truncation=False, return_tensors="pt")
            # The processor expands each image placeholder to many visual tokens.
            # Tokenizer-only prompt lengths would wrongly supervise image/user tokens.
            prefix = self.processor(text=[prompt_text], images=images or None,
                                    padding=False, truncation=False, return_tensors="pt")
            prompt_ids = prefix["input_ids"][0]
            input_ids = encoded["input_ids"][0]
            prefix_length = prompt_ids.shape[0]
            if not torch.equal(input_ids[:prefix_length], prompt_ids):
                raise ValueError(f"{record.sample.id}: full chat does not start with the processed prompt.")
            if input_ids.shape[0] > self.max_length:
                raise ValueError(
                    f"{record.sample.id}: {input_ids.shape[0]} tokens exceed --max-length {self.max_length}. "
                    "Reduce --max-pixels or increase --max-length; no image/answer tokens were truncated.")
            labels = input_ids.clone()
            labels[:prefix_length] = -100
            if bool((labels != -100).sum() == 0):
                raise ValueError(f"{record.sample.id} has no supervised answer tokens after tokenization.")
            item = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
            for key in ("pixel_values", "image_grid_thw"):
                if key in encoded:
                    item[key] = encoded[key]
            return item
        finally:
            for image in images:
                image.close()

    def __call__(self, batch: list[MIRARecord]) -> dict[str, Any]:
        import torch

        items = [self._one(record) for record in batch]
        max_length = max(item["input_ids"].shape[0] for item in items)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids, attention_mask, labels = [], [], []
        for item in items:
            pad_length = max_length - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_length), value=pad_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_length), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_length), value=-100))
        result = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
        pixel_values = [item["pixel_values"] for item in items if "pixel_values" in item]
        image_grid = [item["image_grid_thw"] for item in items if "image_grid_thw" in item]
        if pixel_values:
            result["pixel_values"] = torch.cat(pixel_values, dim=0)
        if image_grid:
            result["image_grid_thw"] = torch.cat(image_grid, dim=0)
        return result


def configure_processor(processor: Any, min_pixels: int, max_pixels: int) -> None:
    """Constrain visual preprocessing so the local 24 GB GPU does not use huge defaults."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if hasattr(image_processor, name):
            setattr(image_processor, name, value)
    if hasattr(image_processor, "size"):
        size = getattr(image_processor, "size")
        if isinstance(size, dict):
            size["shortest_edge"] = min_pixels
            size["longest_edge"] = max_pixels
        else:
            image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, Any, bool, bool]:
    """Load the local vision-language model in BF16/FP16 without touching base files."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForImageTextToText
        model_class = AutoModelForImageTextToText
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.dtype == "auto":
        use_bf16 = bool(device.startswith("cuda") and torch.cuda.is_bf16_supported())
        use_fp16 = bool(device.startswith("cuda") and not use_bf16)
        dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("CPU training requires --dtype auto, float32, or bfloat16.")
    if use_bf16 and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use --dtype auto or float16.")
    LOGGER.info("Loading base model from %s on %s with %s", args.model_dir, device, dtype)
    model = model_class.from_pretrained(
        args.model_dir, local_files_only=True, dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, padding_side="right")
    configure_processor(processor, args.min_pixels, args.max_pixels)
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.to(device)
    return model, processor, device, use_bf16, use_fp16


def build_training_args(args: argparse.Namespace, use_bf16: bool, use_fp16: bool) -> Any:
    """Build TrainingArguments across nearby Transformers releases."""
    from transformers import TrainingArguments

    values = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "optim": "adamw_torch",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "use_cpu": args.device == "cpu",
        "seed": args.seed,
        "data_seed": args.seed,
        "disable_tqdm": True,
        "label_names": ["labels"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": args.max_grad_norm,
        "save_safetensors": True,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    filtered = {key: value for key, value in values.items() if key in parameters}
    return TrainingArguments(**filtered)


def format_duration(seconds: float) -> str:
    """Format ETA/wall time for console output."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressCallback:
    """单行刷新训练进度，并显示速度、loss、ETA 和总训练时间。"""

    def __init__(self, effective_batch_size: int):
        self.effective_batch_size = effective_batch_size
        self.started = None
        self.last_time = None
        self.last_step = 0
        self.steps_per_second = 0.0
        self.current_loss = None
        self.elapsed_seconds = 0.0
        self.last_render_length = 0

    def _render(self, state) -> None:
        total = max(int(state.max_steps or 0), 0)
        step = max(int(state.global_step or 0), 0)
        fraction = min(step / total, 1.0) if total else 0.0
        width = 24
        filled = min(width, int(width * fraction))
        bar = "=" * filled + (">" if filled < width else "") + " " * max(0, width - filled - 1)
        elapsed = self.elapsed_seconds
        qa_per_second = self.steps_per_second * self.effective_batch_size
        remaining = (max(total - step, 0) / self.steps_per_second
                     if self.steps_per_second > 0 else float("inf"))
        loss = f"{self.current_loss:.6f}" if self.current_loss is not None else "--"
        message = (
            f"\r[train] [{bar}] {fraction * 100:5.1f}% {step:,}/{total:,} | "
            f"{self.steps_per_second:.3f} step/s | approx {qa_per_second:.2f} QA/s | "
            f"loss {loss} | elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}"
        )
        # 如果新行比旧行短，用空格清掉残留尾部；回车后覆盖同一行。
        self.last_render_length = max(self.last_render_length, len(message))
        print(message.ljust(self.last_render_length), end="\r", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.last_time = self.started
        self.last_step = int(state.global_step)
        print(f"LoRA training started: {state.max_steps:,} optimizer steps; "
              f"effective batch={self.effective_batch_size}.", flush=True)
        self._render(state)

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is None or not state.global_step or state.global_step == self.last_step:
            return
        now = time.perf_counter()
        elapsed = now - self.started
        self.elapsed_seconds = elapsed
        delta_steps = state.global_step - self.last_step
        delta_time = max(now - self.last_time, 1e-9)
        self.steps_per_second = delta_steps / delta_time
        self.last_time = now
        self.last_step = int(state.global_step)
        self._render(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if isinstance(loss, (int, float)):
            self.current_loss = float(loss)
            self._render(state)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = 0.0 if self.started is None else time.perf_counter() - self.started
        print(flush=True)
        print(f"LoRA training time: {format_duration(elapsed)} ({elapsed:.2f} seconds)", flush=True)


def output_is_safe(model_dir: Path, output_dir: Path, allow_existing: bool = False) -> None:
    """Prevent accidental writes to the base model or an existing adapter directory."""
    model_dir, output_dir = model_dir.resolve(), output_dir.resolve()
    if output_dir == model_dir or model_dir in output_dir.parents or output_dir in model_dir.parents:
        raise ValueError("--output-dir must be outside the base model directory; the original weights are protected.")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"LoRA 输出路径必须是目录，当前路径是文件：{output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise ValueError(f"Output directory is not empty: {output_dir}. "
                         "请修改 output_dir，或将 on_existing_output 设为 new，另存本次结果。")


def select_output_directory(args: argparse.Namespace) -> Path:
    """遇到旧结果时另选新目录；只计算路径，不在参数检查或 dry-run 中创建文件。"""
    requested = args.output_dir
    # 先拒绝与基座重叠及非目录路径，不能靠自动改名绕过保护。
    output_is_safe(args.model_dir, requested, allow_existing=True)
    if args.resume_from_checkpoint or args.allow_existing_output:
        return requested
    if not requested.exists() or not any(requested.iterdir()):
        return requested
    if args.on_existing_output == "error":
        output_is_safe(args.model_dir, requested)
    stem = f"{requested.name}_run_{time.strftime('%Y%m%d_%H%M%S')}"
    candidate = requested.with_name(stem)
    suffix = 2
    while candidate.exists():
        candidate = requested.with_name(f"{stem}_{suffix}")
        suffix += 1
    output_is_safe(args.model_dir, candidate)
    print(f"输出目录已有训练结果，已为本次训练选择新目录：{candidate}", flush=True)
    return candidate


def select_lora_targets(model: Any, suffixes: list[str]) -> list[str]:
    """Restrict adapters to decoder projections; keep the entire vision tower frozen."""
    targets = [name for name, _ in model.named_modules()
               if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in suffixes]
    missing = set(suffixes) - {name.rsplit(".", 1)[-1] for name in targets}
    if not targets or missing:
        raise ValueError(f"No language-model projection found for LoRA targets: {sorted(missing or suffixes)}")
    return targets


def prepare_dataset(args: argparse.Namespace) -> tuple[MIRATrainingDataset, dict[str, Any], list[str]]:
    if args.split_manifest is None:
        args.data_root = Path(args.data_root or DEFAULT_DATA_ROOT).expanduser().resolve()
        train_ids = [sample.id for sample in iter_samples(args.data_root, "train")]
        manifest = {
            "train_ids": train_ids,
            "test_ids": [],
            "data_root": str(args.data_root),
            "source_splits": ["train"],
            "selection_mode": "full_train_split",
        }
        print(f"未指定 split manifest；使用 train.csv 中全部训练问答：{len(train_ids):,} 组。", flush=True)
    else:
        train_ids, manifest = load_split_manifest(args.split_manifest)
    selected_ids = train_ids[:args.limit] if args.limit > 0 else train_ids
    source_root = args.data_root or manifest.get("data_root")
    if not source_root:
        raise ValueError("Provide --data-root or a manifest containing data_root.")
    args.data_root = Path(source_root).expanduser().resolve()
    samples = load_training_samples(args.data_root, selected_ids)
    dataset = MIRATrainingDataset(samples, args.data_root)
    if not len(dataset):
        raise ValueError("No selected QA has both a usable question and answer.")
    print(f"Manifest train IDs: {len(train_ids):,}; selected: {len(samples):,}; "
          f"trainable: {len(dataset):,}; missing question/answer: {len(dataset.skipped_ids):,}", flush=True)
    for sample_id in dataset.skipped_ids[:10]:
        print(f"Skipped incomplete QA: {sample_id}", flush=True)
    for record in dataset.records:
        for name in record.sample.images:
            path = image_path(args.data_root, name)
            if not path.is_file():
                raise FileNotFoundError(f"{record.sample.id}: missing image {path}")
    return dataset, manifest, train_ids


def train(args: argparse.Namespace) -> None:
    """Resolve IDs, load LoRA model, train, and save only adapter artifacts."""
    started = time.perf_counter()
    ensure_dependencies()
    output_is_safe(args.model_dir, args.output_dir, args.allow_existing_output)
    if getattr(args, "_auto_output_dir", False):
        # 自动选出的目录必须由本次训练独占创建；若被另一进程抢先占用，停止而不覆盖。
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset, manifest, train_ids = prepare_dataset(args)
    from transformers import Trainer, TrainerCallback, set_seed
    from transformers.trainer_callback import PrinterCallback
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(args.seed)
    model, processor, device, use_bf16, use_fp16 = load_model_and_processor(args)
    tokenizer = processor.tokenizer
    model_limit = model.config.text_config.max_position_embeddings
    if args.max_length > model_limit:
        raise ValueError(f"--max-length exceeds the model context window ({model_limit}).")
    collator = QwenVLDataCollator(processor, tokenizer, args.max_length,
                                  args.min_pixels, args.max_pixels,
                                  args.system_prompt)

    model.config.use_cache = False
    model.config.text_config.use_cache = False
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("--target-modules must contain at least one module name.")
    target_modules = select_lora_targets(model, target_modules)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    ))
    model.print_trainable_parameters()
    training_args = build_training_args(args, use_bf16, use_fp16)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    class ProgressCallback(TrainingProgressCallback, TrainerCallback):
        pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[ProgressCallback(effective_batch_size)],
    )
    # disable_tqdm=True 时 Trainer 会用 PrinterCallback 逐步打印原始字典，
    # 移除它，避免和自定义的单行训练进度条重复输出。
    trainer.remove_callback(PrinterCallback)
    if args.validate_image_token_inputs:
        # The selected examples must all fit before any optimizer update is made.
        with TerminalProgress("Validated image/token inputs", len(dataset.records)) as progress:
            for index, record in enumerate(dataset.records, start=1):
                collator([record])
                progress.update(index)
    else:
        print(
            "已跳过训练前的图片/token 预检查；图片或输入处理错误可能会在正式训练时出现。",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    trainer.save_state()
    processor.save_pretrained(str(args.output_dir))
    total_elapsed = time.perf_counter() - started
    metadata_payload = {
        "base_model_dir": str(args.model_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split_manifest": (str(args.split_manifest.resolve())
                           if args.split_manifest is not None else None),
        "selection_mode": manifest.get("selection_mode", "split_manifest"),
        "source_splits": manifest.get("source_splits", ["manifest_train_ids"]),
        "manifest_train_count": len(train_ids),
        "actual_train_count": len(dataset),
        "train_ids": [record.sample.id for record in dataset.records],
        "skipped_incomplete_ids": dataset.skipped_ids,
        "data_root": str(args.data_root.resolve()),
        "device": device,
        "dtype": "bfloat16" if use_bf16 else "float16" if use_fp16 else "float32",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout, "target_modules": target_modules},
        "training": {"epochs": args.epochs, "max_steps": args.max_steps,
                      "completed_optimizer_steps": trainer.state.global_step, "seed": args.seed,
                      "batch_size": args.batch_size,
                      "learning_rate": args.learning_rate, "system_prompt": args.system_prompt,
                      "gradient_accumulation_steps": args.gradient_accumulation_steps,
                      "max_length": args.max_length,
                      "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                      "validate_image_token_inputs": args.validate_image_token_inputs},
        "total_elapsed_seconds_including_save": total_elapsed,
        "manifest_metadata": {key: manifest[key] for key in ("seed", "source_splits") if key in manifest},
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter and processor to: {args.output_dir}", flush=True)
    print(f"Total run time including final save: {format_duration(total_elapsed)} ({total_elapsed:.2f} seconds)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Define safe defaults for the local model, manifest, and adapter output."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-env", action="store_true", help="Only report dependencies and CUDA; do not load weights.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve IDs and print counts without loading the model.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
    parser.add_argument("--data-root", type=Path, help="MIRA CSV directory; defaults to data_root in the manifest.")
    parser.add_argument("--split-manifest", type=Path, default=OUTPUT_DIR / "mira_split_ids.json",
                        help="训练 ID 清单；默认读取项目 output/mira_split_ids.json。")
    parser.add_argument("--all-train-data", dest="split_manifest", action="store_const",
                        const=None, default=argparse.SUPPRESS,
                        help="不使用 ID 清单，使用 MIRA 数据目录 train.csv 中的全部问答。")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "qwen3_vl_2b_lora_adapter",
                        help="LoRA 保存目录；默认写入项目 output，不覆盖基座权重。")
    parser.add_argument("--on-existing-output", choices=("new", "error"), default="new",
                        help="输出目录非空时：new 自动另选带时间戳的新目录（默认）；error 停止。续训不改目录。")
    parser.add_argument("--allow-existing-output", action="store_true", help="Allow saving into a nonempty adapter directory.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Trainer checkpoint directory to resume.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N manifest IDs for a debug run; 0 means all.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs for a short run.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-validate-image-token-inputs", action="store_false",
                        dest="validate_image_token_inputs",
                        help="Skip the per-example image/token preflight before training.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.set_defaults(gradient_checkpointing=True, validate_image_token_inputs=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "gradient_accumulation_steps", "logging_steps", "save_steps",
                 "max_length", "lora_r", "lora_alpha"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.limit < 0 or args.epochs <= 0 or args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("Invalid --limit, --epochs or --max-steps.")
    if (not math.isfinite(args.epochs) or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1
            or not 0 <= args.lora_dropout < 1 or args.weight_decay < 0 or args.max_grad_norm < 0):
        raise ValueError("Invalid learning rate, dropout, warmup, weight decay or gradient norm.")
    if not 4096 <= args.min_pixels <= args.max_pixels:
        raise ValueError("Pixel budgets must satisfy 4096 <= min-pixels <= max-pixels.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run this script in one process with python; distributed launch is not supported.")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.split_manifest = (args.split_manifest.expanduser().resolve()
                           if args.split_manifest is not None else None)
    args.output_dir = args.output_dir.expanduser().resolve()
    output_is_safe(args.model_dir, args.output_dir, allow_existing=True)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("--model-dir must contain Qwen3-VL-Instruct, not an embedding or text-only model.")
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("--resume-from-checkpoint must point to a Trainer LoRA checkpoint.")
        if args.output_dir not in checkpoint.parents:
            raise ValueError("Resume using the original --output-dir that contains the checkpoint.")
        args.resume_from_checkpoint = str(checkpoint)
        args.allow_existing_output = True
    requested_output = args.output_dir
    args.output_dir = select_output_directory(args)
    args._auto_output_dir = args.output_dir != requested_output


def main(argv: list[str] | None = None) -> int:
    """Run environment check, dry run, or training."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.check_env:
            lines, problems = dependency_report()
            print("\n".join(lines))
            return 1 if problems else 0
        validate_args(args)
        print(f"本次 LoRA 输出目录：{args.output_dir}", flush=True)
        if args.dry_run:
            prepare_dataset(args)
            print("Dry run complete; no model was loaded and no files were written.", flush=True)
            return 0
        train(args)
        return 0
    except KeyboardInterrupt:
        print("Training interrupted; the base model was not modified.", flush=True)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
