"""评估时只读加载 LoRA，并为答案缓存提供适配器内容指纹。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import warnings


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取 LoRA 相关配置：{path}，原因：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LoRA 相关配置必须为 JSON 对象：{path}")
    return value


def _adapter_files(adapter_path: str | Path) -> tuple[Path, list[Path]]:
    directory = Path(adapter_path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"LoRA 参数目录不存在：{directory}")
    config = directory / "adapter_config.json"
    if not config.is_file():
        raise ValueError(f"LoRA 参数目录缺少 adapter_config.json：{directory}")
    weights = [directory / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    weights = [path for path in weights if path.is_file()]
    if not weights:
        raise ValueError(
            f"LoRA 参数目录缺少 adapter_model.safetensors 或 adapter_model.bin：{directory}"
        )
    if any(path.stat().st_size == 0 for path in weights):
        raise ValueError(f"LoRA 权重文件为空：{directory}")
    return directory, [config, *weights]


def _architecture(config: dict[str, Any]) -> dict[str, Any]:
    # Qwen3-VL 将语言模型尺寸放在 text_config 内；DeepSeek 蒸馏版放在顶层。
    text_config = config.get("text_config")
    language = text_config if isinstance(text_config, dict) else config
    result = {"model_type": config.get("model_type")}
    for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    ):
        if key in language:
            result[key] = language[key]
    return result


def validate_lora_path(model_path: str | Path, adapter_path: str | Path | None) -> None:
    """检查适配器文件以及可核实的基座架构；不要求训练路径与当前路径相同。"""
    if adapter_path is None:
        return
    directory, _ = _adapter_files(adapter_path)
    config = _read_object(directory / "adapter_config.json")
    if config.get("peft_type") != "LORA":
        raise ValueError(f"当前评估仅支持 LORA 适配器：{directory}")
    if config.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"LoRA 的 task_type 必须为 CAUSAL_LM：{directory}")

    current_config = _read_object(Path(model_path).expanduser().resolve() / "config.json")
    current = _architecture(current_config)
    if not current.get("model_type"):
        raise ValueError(f"基座模型配置缺少 model_type：{model_path}")

    # 本项目训练脚本保存的来源优先于 PEFT 的来源；路径可以随项目迁移。
    metadata_path = directory / "training_metadata.json"
    metadata = _read_object(metadata_path) if metadata_path.is_file() else {}
    sources = [metadata.get("base_model_dir"), config.get("base_model_name_or_path")]
    recorded_config = None
    recorded_source = None
    for source in sources:
        if not isinstance(source, str) or not source.strip():
            continue
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = directory / source_path
        candidate = source_path / "config.json"
        if candidate.is_file():
            recorded_config = _read_object(candidate)
            recorded_source = candidate
            break
    if recorded_config is None:
        warnings.warn(
            f"LoRA 记录的训练基座路径已不存在，无法预先核对架构：{directory}；"
            "加载时仍会由 PEFT 检查参数名称和形状。",
            UserWarning,
            stacklevel=2,
        )
        return

    recorded = _architecture(recorded_config)
    mismatches = [
        f"{key}: 训练时为 {value!r}，当前为 {current[key]!r}"
        for key, value in recorded.items()
        if value is not None and key in current and value != current[key]
    ]
    if mismatches:
        raise ValueError(
            "LoRA 与所选基座模型架构不匹配：" + "；".join(mismatches)
            + f"。训练基座配置：{recorded_source}"
        )


def load_lora_adapter(model: Any, adapter_path: str | Path | None) -> Any:
    """加载用于推理的附加参数，不合并、不保存、不覆盖原始模型文件。"""
    if adapter_path is None:
        return model
    directory, _ = _adapter_files(adapter_path)
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("加载 LoRA 需要 peft，请在当前 Python 环境安装 peft。") from exc
    result = PeftModel.from_pretrained(
        model, str(directory), is_trainable=False, local_files_only=True,
    )
    result.eval()
    return result


def adapter_identity(adapter_path: str | Path | None) -> dict[str, Any] | None:
    """用配置和权重内容区分原版、微调版以及同目录重新训练后的版本。"""
    if adapter_path is None:
        return None
    directory, paths = _adapter_files(adapter_path)
    files = []
    # 每次读取内容，重训时即使文件大小或时间戳相同，也不会误用旧答案缓存。
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"name": path.name, "sha256": digest.hexdigest()})
    combined = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"path": str(directory), "sha256": combined, "files": files}
