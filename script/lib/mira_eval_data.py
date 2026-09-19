"""Read MIRA CSV question-answer pairs for the evaluation pipeline.

IDs and source parsing match the embedding pipeline. The default source is
train.csv, as in sample_mira_ids.py; this source split is separate from the
train/test subsets subsequently selected for an evaluation run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inferenceValid.embed_mira_chroma import image_path, iter_samples, json_text


LOGGER = logging.getLogger("rag_eval")
SOURCE_SPLITS = ("train", "validation", "test")


def load_mira_dataset(data_root: Path, source_splits=("train",)) -> list[dict]:
    """Load complete QA pairs without opening or checking image files.

    The question includes answer choices but never source captions or extra
    annotation fields. Structured answers are serialized losslessly as JSON;
    text answers remain text. Incomplete QA pairs are warned about and skipped,
    preserving the original row/category/QA indices of all remaining IDs.
    """
    data_root = Path(data_root).expanduser().resolve()
    if isinstance(source_splits, str):
        raise ValueError("source_splits 必须是原始划分名称的列表或元组。")
    source_splits = tuple(source_splits)
    if not source_splits or any(split not in SOURCE_SPLITS for split in source_splits):
        raise ValueError("source_splits 只能包含 train、validation、test，且不能为空。")
    if len(set(source_splits)) != len(source_splits):
        raise ValueError("source_splits 不能重复。")
    if not data_root.is_dir():
        raise FileNotFoundError(f"未找到 MIRA 数据目录：{data_root}")
    for split in source_splits:
        source = data_root / f"{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"未找到 MIRA 原始划分：{source}")

    rows, image_cache = [], {}
    skipped = 0
    for split in source_splits:
        accepted = 0
        for sample in iter_samples(data_root, split):
            if sample.missing_fields or (
                isinstance(sample.answer, str) and not sample.answer.strip()
            ):
                skipped += 1
                missing = sample.missing_fields or ["answer"]
                LOGGER.warning("跳过无法评估的 MIRA 问答 %s：缺少有效 %s。", sample.id, missing)
                continue
            question = sample.question
            if sample.options:
                question += "\nOptions: " + json_text(sample.options)
            reference = sample.answer if isinstance(sample.answer, str) else json_text(sample.answer)
            images = []
            for name in sample.images:
                if name not in image_cache:
                    # abspath normalizes relative segments without reading image files.
                    image_cache[name] = os.path.abspath(image_path(data_root, name))
                images.append(image_cache[name])
            rows.append({"id": sample.id, "question": question,
                         "images": images, "reference": reference})
            accepted += 1
        LOGGER.info("MIRA %s.csv：读取 %d 条可评估问答。", split, accepted)
    if skipped:
        LOGGER.warning("MIRA 总计跳过 %d 条缺少问题或答案的问答；保留 %d 条。", skipped, len(rows))
    if not rows:
        raise ValueError("MIRA 所选划分没有包含有效问题和答案的问答。")
    return rows
