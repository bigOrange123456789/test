"""显示有效评分数量与按题自助采样置信区间，不把缺失评分当作零分。"""

from __future__ import annotations

import math
import numpy as np


def summarize_values(values, *, seed=42, bootstrap_samples=1000, binary=False):
    values = list(values)
    valid = [float(value) for value in values if value is not None]
    if not np.isfinite(valid).all():
        raise ValueError("指标必须为有限数值或 None。")
    n = len(valid)
    result = {"mean": None, "std": None, "n": n, "total": len(values),
              "coverage": n / len(values) if values else 0.0, "ci95": [None, None],
              "ci_method": "按问答对 bootstrap 均值区间，仅覆盖有效评分；不反映裁判系统偏差"}
    if not n:
        return result
    array = np.asarray(valid, dtype=np.float64)
    result.update(mean=float(array.mean()), std=float(array.std(ddof=0)))
    if binary:
        if any(value not in (0.0, 1.0) for value in valid):
            raise ValueError("二元准确率必须为0或1。")
        # 全对/全错的小样本也有不确定性，使用 Wilson 区间而不是退化的 bootstrap。
        z = 1.959963984540054
        p = result["mean"]
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        result["ci95"] = [max(0.0, center - radius), min(1.0, center + radius)]
        result["ci_method"] = "二元准确率Wilson区间；假设问答独立，不涵盖解析/标注偏差"
        return result
    if n == 1:
        result["ci_method"] = "仅一个有效样本，不估计置信区间"
        return result
    rng = np.random.default_rng(seed)
    means = []
    # 分批处理，避免测试集很大时分配 bootstrap_samples × N 的巨型矩阵。
    batch_size = max(1, min(100, 1_000_000 // n))
    for start in range(0, bootstrap_samples, batch_size):
        batch = min(batch_size, bootstrap_samples - start)
        means.extend(array[rng.integers(0, n, size=(batch, n))].mean(axis=1).tolist())
    result["ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def paired_summary(after, before, **kwargs):
    """在同一题的两个有效评分间取差，并报告有效配对数量。"""
    if len(after) != len(before):
        raise ValueError("配对差值要求两组样本长度一致。")
    return summarize_values([a - b if a is not None and b is not None else None
                             for a, b in zip(after, before)], **kwargs)
