"""推理端和独立评分端共用的文本指标实现。"""

from __future__ import annotations

import warnings
from functools import lru_cache

import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def chinese_tokens(text: str) -> list[str]:
    """使用 jieba 精确分词，过滤空白词元。"""
    return [token.strip() for token in jieba.lcut(text or "", cut_all=False) if token.strip()]


def compute_bleu4(prediction: str, reference: str, smoothing: str = "none") -> float:
    """句级 BLEU-4，返回 0～1。"""
    pred, ref = chinese_tokens(prediction), chinese_tokens(reference)
    if not pred or not ref:
        return 0.0
    if smoothing not in {"none", "method1"}:
        raise ValueError("BLEU smoothing 只能为 none 或 method1。")
    smooth = SmoothingFunction().method1 if smoothing == "method1" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(sentence_bleu([ref], pred, weights=(.25, .25, .25, .25),
                                   smoothing_function=smooth, auto_reweigh=False))


class _WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


@lru_cache(maxsize=1)
def _rouge_scorer():
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                    tokenizer=_WhitespaceTokenizer())


def compute_rouge_l(prediction: str, reference: str) -> float:
    """jieba 分词后计算 ROUGE-L F-measure，返回 0～1。"""
    pred = " ".join(chinese_tokens(prediction))
    ref = " ".join(chinese_tokens(reference))
    return float(_rouge_scorer().score(ref, pred)["rougeL"].fmeasure)
