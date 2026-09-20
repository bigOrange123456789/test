"""根据模型分词器的计数计算经过校验、具有上下限的输出预算。"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


LENGTH_BUDGET_DEFAULTS = {
    "enabled": False,
    "multiplier": 4.0,
    "extraTokens": 256,
    "minNewTokens": 512,
    "maxNewTokens": 4096,
    "retryOnTruncation": True,
}

FACT_LENGTH_BUDGET_DEFAULTS = {
    "enabled": False,
    "multiplier": 2.0,
    "extraTokens": 256,
    "minNewTokens": 1024,
    "maxNewTokens": 8192,
    "retryOnTruncation": True,
}


def normalize_length_budget(value: Any, label: str, *, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并部分配置与独立的默认值副本，并拒绝非法设置。"""
    result = deepcopy(LENGTH_BUDGET_DEFAULTS)
    for supplied in (defaults, value):
        if supplied is None:
            continue
        if not isinstance(supplied, dict):
            raise ValueError(f"{label} 必须是对象。")
        unknown = set(supplied) - set(LENGTH_BUDGET_DEFAULTS)
        if unknown:
            raise ValueError(f"{label} 包含未知字段：{', '.join(sorted(map(str, unknown)))}")
        result.update(deepcopy(supplied))
    for name in ("enabled", "retryOnTruncation"):
        if type(result[name]) is not bool:
            raise ValueError(f"{label}.{name} 必须是布尔值。")
    multiplier = result["multiplier"]
    if type(multiplier) not in (int, float) or not 0 < multiplier <= 32 or not math.isfinite(multiplier):
        raise ValueError(f"{label}.multiplier 必须是大于 0、不超过 32 的有限数值。")
    for name in ("extraTokens", "minNewTokens", "maxNewTokens"):
        lower = 0 if name == "extraTokens" else 1
        if type(result[name]) is not int or not lower <= result[name] <= 32768:
            raise ValueError(f"{label}.{name} 必须是 {lower} 到 32768 之间的整数。")
    if result["minNewTokens"] > result["maxNewTokens"]:
        raise ValueError(f"{label}.minNewTokens 不能大于 maxNewTokens。")
    return result


def calculate_token_budget(token_count: int, settings: dict[str, Any]) -> int:
    """将 ceil(token_count * multiplier) + extraTokens 限制在配置的上下限内。"""
    if type(token_count) is not int or token_count < 0:
        raise ValueError("token_count 必须是非负整数。")
    policy = normalize_length_budget(settings, "lengthBudget")
    # 先比较上限，避免将过大的计数转换成浮点数而溢出。
    if token_count >= policy["maxNewTokens"] / policy["multiplier"]:
        return policy["maxNewTokens"]
    calculated = math.ceil(token_count * policy["multiplier"]) + policy["extraTokens"]
    return min(policy["maxNewTokens"], max(policy["minNewTokens"], calculated))
