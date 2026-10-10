"""统一读取项目根目录 prompts.json 中的提示词。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = PROJECT_ROOT / "prompts.json"


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    if not PROMPTS_PATH.is_file():
        raise FileNotFoundError(f"提示词配置文件不存在：{PROMPTS_PATH}")
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("提示词配置必须是 JSON 对象。")
    return data


def prompt(path: str, default: Any = None) -> Any:
    """按点号路径获取提示词；default 仅用于兼容外部旧调用。"""
    value: Any = load_prompts()
    try:
        for part in path.split("."):
            value = value[part]
        return value
    except (KeyError, TypeError):
        if default is not None:
            return default
        raise KeyError(f"提示词配置缺少字段：{path}")


def format_prompt(path: str, **values: Any) -> str:
    return str(prompt(path)).format(**values)
