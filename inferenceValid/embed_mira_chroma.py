# -*- coding: utf-8 -*-
"""将 MIRA CSV 中的每道问答及其全部图片编码为一个向量，持久化到 ChromaDB。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"G:\Codex_dataset\MIRA-data")
DEFAULT_MODEL_DIR = PROJECT_ROOT / "Qwen3-VL-Embedding-2B"
CATEGORIES = ("open_ended", "closed_ended", "single_choice", "multiple_choice")
DIMENSION = 2048
PIPELINE_VERSION = 1
INSTRUCTION = "Represent the user's input."
LOGGER = logging.getLogger("mira")


def json_text(value: Any) -> str:
    """将结构化问答、选项或路径列表转换成可还原的紧凑 JSON。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    """分块计算文件摘要，用于检查续跑时标注和模型配置是否发生变化。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析数据位置、模型参数、入库批次和小规模验证选项。"""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="MIRA 数据根目录。")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="本地 Embedding 模型目录，不是 Instruct 模型。")
    parser.add_argument("--db-dir", type=Path, default=None, help="Chroma 和断点目录；默认在数据目录旁的 MIRA-chroma。")
    parser.add_argument("--collection", default="mira_qwen3_vl_embedding", help="Chroma 集合名称。")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train"], help="处理的数据划分；训练集对应 1,120,031 道问答。")
    parser.add_argument("--device", default="auto", help="计算设备：auto、cpu、cuda 或 cuda:0 等。")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto", help="模型计算精度。")
    parser.add_argument("--batch-size", type=int, default=1, help="每次模型前向计算的问答数；显存充足时可增大。")
    parser.add_argument("--commit-every", type=int, default=16, help="每处理多少道题写入 Chroma 并保存断点。")
    parser.add_argument("--max-seq-length", type=int, default=8192, help="每组图文允许的最大 token 数，包含图片 token；超长会报错，不截断。")
    parser.add_argument("--min-pixels", type=int, default=4096, help="每张图片预处理后的最小像素数。")
    parser.add_argument("--max-pixels", type=int, default=262144, help="每张图片预处理后的最大像素数。")
    parser.add_argument("--include-caption", action="store_true", help="在问答和图片之外，将原始 caption 一起编码。")
    parser.add_argument("--limit", type=int, default=0, help="本次最多处理的问答数，0 表示不限；下次会继续处理。")
    parser.add_argument("--expected-count", type=int, default=None, help="覆盖预期问答总数；默认读取 selection_manifest.json，0 表示不校验。")
    parser.add_argument("--scan-only", action="store_true", help="只流式检查并统计问答，不加载模型、不写数据库。")
    parser.add_argument("--check-images", action="store_true", help="扫描模式下额外检查图片文件是否存在。")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.commit_every, args.min_pixels) <= 0:
        parser.error("batch-size、commit-every、min-pixels 必须大于 0。")
    if not 1 <= args.max_seq_length <= 32768:
        parser.error("max-seq-length 必须在 1 到 32768 之间。")
    if args.max_pixels < args.min_pixels or args.limit < 0 or (args.expected_count is not None and args.expected_count < 0):
        parser.error("像素范围或数量参数不合法。")
    if len(set(args.splits)) != len(args.splits):
        parser.error("splits 不能重复。")
    args.data_root = args.data_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.db_dir = (args.db_dir or args.data_root.parent / "MIRA-chroma").resolve()
    return args


@dataclass
class Sample:
    """一组问答及其关联图片，连同稳定 ID 和下一条问答的读取位置。"""

    id: str
    split: str
    row_index: int
    category: str
    qa_index: int
    question: str
    answer: Any
    options: Any
    caption: str
    images: list[str]
    next_cursor: dict[str, int]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_fields(self) -> list[str]:
        """标注源数据缺失的字段，保留不完整题目而不编造其内容。"""
        fields = []
        if not self.question:
            fields.append("question")
        if self.answer in (None, "", {}, []):
            fields.append("answer")
        return fields

    def document(self, include_caption: bool = False) -> str:
        """保留问题、选项和完整答案，包括答案解释及视觉证据。"""
        parts = [f"Question: {self.question or '[not provided in source]'}"]
        if self.options:
            parts.append(f"Options: {json_text(self.options)}")
        answer = self.answer if isinstance(self.answer, str) else json_text(self.answer)
        if "answer" in self.missing_fields:
            answer = "[not provided in source]"
        parts.append(f"Answer: {answer}")
        if self.extra:
            parts.append(f"Additional source fields: {json_text(self.extra)}")
        if include_caption and self.caption:
            parts.append(f"Caption: {self.caption}")
        return "\n".join(parts)

    def metadata(self) -> dict[str, Any]:
        """保存回源需要的位置、题型、图像路径和原始图片说明。"""
        return {"split": self.split, "source_csv": f"{self.split}.csv",
                "source_row": self.row_index, "category": self.category,
                "qa_index": self.qa_index, "image_paths": json_text(self.images),
                "image_count": len(self.images), "caption": self.caption,
                "has_question": "question" not in self.missing_fields,
                "has_answer": "answer" not in self.missing_fields,
                "missing_fields": json_text(self.missing_fields)}


def parse_image_paths(value: Any) -> list[str]:
    """接受单图片路径或 JSON/Python 字面量路径列表，保留所有图片的顺序。"""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(("[", "(")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = ast.literal_eval(value)
        else:
            value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("图片路径必须是非空字符串或非空路径列表。")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("图片路径列表中存在空值或非字符串。")
    return [item.strip().replace("\\", "/") for item in value]


def image_path(data_root: Path, name: str) -> Path:
    """将 CSV 中的图片路径解析到本地文件；不自动联网下载图片。"""
    if "://" in name:
        raise ValueError(f"需要本地图片，不能使用远程地址：{name}")
    path = Path(name)
    return path if path.is_absolute() else data_root / path


def flatten_row(row: dict[str, str]) -> list[tuple[str, int, dict[str, Any]]]:
    """把一行 vqa_json 的四类题目逐条展开，不把整行当成一道题。"""
    data = json.loads(row["vqa_json"])
    if not isinstance(data, dict) or set(data) - set(CATEGORIES):
        raise ValueError("vqa_json 必须是包含四种已知题型的对象，发现了不支持的结构。")
    pairs = []
    for category in CATEGORIES:
        items = data.get(category, [])
        if not isinstance(items, list):
            raise ValueError(f"题型 {category} 必须是列表。")
        for index, qa in enumerate(items):
            if not isinstance(qa, dict):
                raise ValueError(f"{category}[{index}] 题目结构不正确。")
            pairs.append((category, index, qa))
    if not pairs:
        raise ValueError("该图片行没有任何问答。")
    return pairs


def iter_samples(data_root: Path, split: str, cursor: dict[str, int] | None = None) -> Iterator[Sample]:
    """流式读取 CSV；用文本位置和行内题号直接续读，兼容带换行的 CSV 字段。"""
    cursor = cursor or {"offset": 0, "row": 0, "qa": 0}
    path = data_root / f"{split}.csv"
    csv.field_size_limit(64 * 1024 * 1024)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        # 使用 readline 迭代以保留 tell/seek 能力；csv 模块负责处理引号和字段内换行。
        lines = iter(stream.readline, "")
        header = next(csv.reader(lines))
        if not {"image_path", "vqa_json"}.issubset(header):
            raise ValueError(f"{path} 缺少 image_path 或 vqa_json 列。")
        if cursor["offset"]:
            stream.seek(cursor["offset"])
        reader = csv.DictReader(lines, fieldnames=header, strict=True)
        row_index, start_qa = cursor["row"], cursor["qa"]
        while True:
            row_start = stream.tell()
            try:
                row = next(reader)
            except StopIteration:
                return
            row_end = stream.tell()
            try:
                if None in row or any(row[key] is None for key in ("image_path", "vqa_json")):
                    raise ValueError("CSV 字段数量不正确。")
                pairs = flatten_row(row)
                if start_qa >= len(pairs):
                    raise ValueError("断点中的题号超过该行题目数。")
                for pair_index in range(start_qa, len(pairs)):
                    category, qa_index, qa = pairs[pair_index]
                    paths = parse_image_paths(qa.get("image_paths", qa.get("image_path", row["image_path"])))
                    last = pair_index + 1 == len(pairs)
                    next_cursor = {"offset": row_end if last else row_start,
                                   "row": row_index + 1 if last else row_index,
                                   "qa": 0 if last else pair_index + 1}
                    extra = {key: value for key, value in qa.items()
                             if key not in {"question", "answer", "options", "image_path", "image_paths"}}
                    original_question = qa.get("question")
                    question = original_question.strip() if isinstance(original_question, str) else ""
                    if original_question is not None and not isinstance(original_question, str):
                        extra["original_question"] = original_question
                    sample = Sample(f"mira:{split}:{row_index}:{category}:{qa_index}", split,
                                    row_index, category, qa_index, question, qa.get("answer"),
                                    qa.get("options"), row.get("caption", "") or "", paths, next_cursor, extra)
                    if sample.missing_fields:
                        LOGGER.warning("%s 原始字段缺失=%s；保留此题和图片，元数据中记录缺失情况。", sample.id, sample.missing_fields)
                    yield sample
            except Exception as error:
                raise ValueError(f"{path.name} 数据行 {row_index + 1}（不含表头）: {error}") from error
            row_index += 1
            start_qa = 0


def expected_count(args: argparse.Namespace) -> int | None:
    """从本地清单读取所选划分的问答数量，允许命令行显式覆盖。"""
    if args.expected_count is not None:
        return args.expected_count or None
    path = args.data_root / "selection_manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    splits = manifest["actual"]["splits"]
    return sum(int(splits[split]["questions"]) for split in args.splits)


def model_identity(model_dir: Path) -> dict[str, Any]:
    """检查 Embedding 配置，并记录权重文件信息，防止续跑时混入另一模型。"""
    required = ["config.json", "sentence_bert_config.json", "1_Pooling/config.json"]
    if any(not (model_dir / name).is_file() for name in required):
        raise ValueError(f"{model_dir} 缺少 Embedding 配置；请使用 Qwen3-VL-Embedding-2B 目录，不是 Instruct 目录。")
    pooling = json.loads((model_dir / required[-1]).read_text(encoding="utf-8"))
    task = json.loads((model_dir / required[1]).read_text(encoding="utf-8"))
    if pooling.get("pooling_mode") != "lasttoken" or pooling.get("embedding_dimension") != DIMENSION or task.get("transformer_task") != "feature-extraction":
        raise ValueError("模型配置不符合 Qwen3-VL-Embedding-2B 的 lasttoken/2048 维特征提取设置。")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"模型权重尚未下载：{model_dir}")
    config_files = sorted(path for path in model_dir.rglob("*") if path.is_file()
                          and ".cache" not in path.parts and path.suffix in {".json", ".jinja", ".txt"})
    return {"path": str(model_dir),
            "config_hashes": {path.relative_to(model_dir).as_posix(): sha256_file(path) for path in config_files},
            "weights": [{"file": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in weights]}


def pipeline_config(args: argparse.Namespace) -> dict[str, Any]:
    """记录影响向量内容的配置和标注摘要；改变批次、设备或 limit 不影响续跑。"""
    return {"version": PIPELINE_VERSION, "data_root": str(args.data_root),
            "sources": {split: sha256_file(args.data_root / f"{split}.csv") for split in args.splits},
            "model": model_identity(args.model_dir), "instruction": INSTRUCTION,
            "dimension": DIMENSION, "pooling": "last_nonpadding_token_l2", "dtype": args.dtype,
            "context_policy": "no_truncation", "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels, "include_caption": args.include_caption}


class Checkpoint:
    """保存每个 CSV 的读取位置；先成功入库，再原子更新此文件。"""

    def __init__(self, path: Path, config: dict[str, Any], splits: list[str]):
        """读取已有断点并检查配置一致性，或创建一个空的读取状态。"""
        self.path = path
        self.signature = hashlib.sha256(json_text(config).encode("utf-8")).hexdigest()
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data["signature"] != self.signature:
                raise ValueError("数据、模型或编码配置已改变，不能混入现有向量库。请恢复原参数，或指定新的 --db-dir/--collection。")
        else:
            self.data = {"signature": self.signature, "config": config,
                         "progress": {split: {"cursor": {"offset": 0, "row": 0, "qa": 0},
                                              "processed": 0, "done": False} for split in splits}}

    @property
    def processed(self) -> int:
        """统计断点已经确认入库的问答数。"""
        return sum(value["processed"] for value in self.data["progress"].values())

    def save(self) -> None:
        """先刷盘临时 JSON，再原子替换断点；中断不会留下半个 JSON 文件。"""
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)


def load_embedding_processor(args: argparse.Namespace):
    """兼容不同 Transformers 版本的 PIL 处理器，保持入库和查询的预处理一致。"""
    if __package__:
        from .qwen_vl_compat import load_qwen_vl_processor
    else:
        # 兼容直接运行本文件，以及旧脚本通过 sys.path 导入的方式。
        from qwen_vl_compat import load_qwen_vl_processor
    return load_qwen_vl_processor(args.model_dir, min_pixels=args.min_pixels,
                                  max_pixels=args.max_pixels, padding_side="right")


class QwenEmbeddingEncoder:
    """使用完整图文模型的最终隐藏状态生成归一化检索向量。"""

    def __init__(self, args: argparse.Namespace):
        """只加载本地权重和处理器，不调用远程 API，也不加载聊天模型的输出头。"""
        import torch
        from transformers import Qwen3VLModel

        self.args = args
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 没有可用 GPU。")
        # 先验证处理器，配置/依赖不兼容时不占用模型显存。
        self.processor = load_embedding_processor(args)
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        # 4.57.x 的裸特征模型前缀为空，不会自动剥离权重中的 model.。
        # 仅映射加载时的名称；有原生前缀处理的版本交给库自身，避免重复剥离。
        load_options = {"key_mapping": {r"^model\.": ""}} if not Qwen3VLModel.base_model_prefix else {}
        self.model, loading = Qwen3VLModel.from_pretrained(
            args.model_dir, local_files_only=True, dtype=dtype, output_loading_info=True,
            **load_options,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"模型权重加载不完整：{loading}")
        self.model.to(device).eval()
        LOGGER.info("已加载 Embedding 模型：%s，设备=%s，精度=%s", args.model_dir, device, self.model.dtype)
        if device == "cpu":
            LOGGER.warning("当前使用 CPU；百万条图文编码耗时很长，建议先用 --limit 10 测量速度。")

    def encode(self, samples: list[Sample]) -> list[list[float]]:
        """把每道题的问答文本与全部图片放入同一上下文，只产生一个 2048 维向量。"""
        return self.encode_inputs([sample.document(self.args.include_caption) for sample in samples],
                                  [[image_path(self.args.data_root, name) for name in sample.images] for sample in samples])

    def encode_inputs(self, texts: list[str], image_groups: list[list[Path]]) -> list[list[float]]:
        """按官方 last-token + L2 方法编码图文，也可传空图片列表编码未来的文字查询。"""
        import torch
        from PIL import Image, ImageOps

        if not texts or len(texts) != len(image_groups):
            raise ValueError("文字和图片分组数量不一致。")
        conversations, images = [], []
        with ExitStack() as resources:
            for text, paths in zip(texts, image_groups):
                content = []
                for path in paths:
                    original = resources.enter_context(Image.open(path))
                    picture = ImageOps.exif_transpose(original).convert("RGB")
                    resources.callback(picture.close)
                    images.append(picture)
                    content.append({"type": "image"})
                content.append({"type": "text", "text": text})
                conversations.append([
                    {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
                    {"role": "user", "content": content},
                ])
            prompts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=prompts, images=images or None,
                                    padding=True, truncation=False, return_tensors="pt")
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        if max(lengths) > self.args.max_seq_length:
            raise ValueError(f"图文长度 {lengths} 超过 max-seq-length={self.args.max_seq_length}，未截断图片或答案。"
                             "可增大 --max-seq-length 后从原库续跑；若减小 --max-pixels，则需指定新的数据库目录。")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            hidden = self.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"]
            columns = mask.shape[1] - 1 - mask.flip(dims=[1]).long().argmax(dim=1)
            vectors = hidden[torch.arange(hidden.shape[0], device=hidden.device), columns].float()
            norms = vectors.norm(p=2, dim=-1, keepdim=True)
            if vectors.shape[1] != DIMENSION or not torch.isfinite(vectors).all() or (norms <= 0).any():
                raise RuntimeError("模型返回了无效特征，未写入数据库。")
            return (vectors / norms).cpu().tolist()


def scan_dataset(args: argparse.Namespace) -> int:
    """不加载模型，遍历并核对真实问答数量；可选检查全部关联图片是否存在。"""
    total = 0
    for split in args.splits:
        counts = dict.fromkeys(CATEGORIES, 0)
        last_row = -1
        checked_images = set()
        for sample in iter_samples(args.data_root, split):
            if args.check_images:
                if sample.row_index != last_row:
                    checked_images.clear()
                for name in sample.images:
                    if name not in checked_images and not image_path(args.data_root, name).is_file():
                        raise FileNotFoundError(f"{sample.id} 缺少图片：{name}")
                    checked_images.add(name)
            total += 1
            counts[sample.category] += 1
            last_row = sample.row_index
            if total % 100000 == 0:
                LOGGER.info("已扫描 %d 道问答", total)
            if args.limit and total >= args.limit:
                LOGGER.info("扫描达到 limit=%d；该数量不代表全量。", args.limit)
                return total
        LOGGER.info("%s：图片记录行数=%d，问答数=%d，分类=%s", split, last_row + 1, sum(counts.values()), counts)
    target = expected_count(args)
    if target is not None and total != target:
        raise ValueError(f"扫描问答数={total}，预期={target}，请检查数据划分或清单。")
    LOGGER.info("扫描完成：%d 道问答，未创建向量库。", total)
    return total


def open_collection(args: argparse.Namespace, checkpoint: Checkpoint):
    """打开显式传入向量的本地余弦索引，并检查它与断点属于同一次编码任务。"""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(args.db_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name=args.collection, embedding_function=None,
        metadata={"pipeline_signature": checkpoint.signature, "dimension": DIMENSION,
                  "model": "Qwen3-VL-Embedding-2B", "data_root": str(args.data_root)},
        configuration={"hnsw": {"space": "cosine"}},
    )
    if (collection.metadata or {}).get("pipeline_signature") != checkpoint.signature:
        raise ValueError("同名 Chroma 集合的模型或数据配置不一致，请使用新的 --collection。")
    if collection.count() < checkpoint.processed:
        raise ValueError("数据库条数少于断点记录，可能数据库被删除或替换；请恢复完整数据库或指定新的 --db-dir。")
    return client, collection


def run_index(args: argparse.Namespace, encoder_factory=QwenEmbeddingEncoder) -> dict[str, int]:
    """按批次续读、去重、编码、入库，并且只在入库成功后提交断点。"""
    from filelock import FileLock

    if not args.collection or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.collection):
        raise ValueError("集合名称只能包含英文字母、数字、下划线和短横线。")
    args.db_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.db_dir / ".mira-writer.lock"), timeout=0):
        LOGGER.info("正在核对标注文件和模型配置摘要……")
        checkpoint = Checkpoint(args.db_dir / f"{args.collection}.checkpoint.json", pipeline_config(args), args.splits)
        client, collection = open_collection(args, checkpoint)
        checkpoint.save()
        target = expected_count(args)
        LOGGER.info("数据=%s；集合=%s；已有向量=%d；已确认进度=%d；预期=%s",
                    args.data_root, args.collection, collection.count(), checkpoint.processed, target or "未指定")
        LOGGER.info("Chroma 与断点目录：%s", args.db_dir)
        encoder = None
        processed_now, encoded_now = 0, 0
        started = time.monotonic()
        block_size = min(args.commit_every, client.get_max_batch_size())
        for split in args.splits:
            progress = checkpoint.data["progress"][split]
            if progress["done"]:
                continue
            iterator = iter_samples(args.data_root, split, progress["cursor"])
            try:
                while not args.limit or processed_now < args.limit:
                    size = min(block_size, args.limit - processed_now) if args.limit else block_size
                    samples = list(itertools.islice(iterator, size))
                    if not samples:
                        progress["done"] = True
                        checkpoint.save()
                        break
                    existing = set(collection.get(ids=[sample.id for sample in samples], include=[])["ids"])
                    missing = [sample for sample in samples if sample.id not in existing]
                    if missing and encoder is None:
                        encoder = encoder_factory(args)
                    embeddings = []
                    for start in range(0, len(missing), args.batch_size):
                        batch = missing[start:start + args.batch_size]
                        try:
                            embeddings.extend(encoder.encode(batch))
                        except Exception as error:
                            raise RuntimeError(f"编码失败，当前批次 ID={[sample.id for sample in batch]}：{error}") from error
                    if missing:
                        collection.upsert(ids=[sample.id for sample in missing], embeddings=embeddings,
                                          documents=[sample.document(args.include_caption) for sample in missing],
                                          metadatas=[sample.metadata() for sample in missing])
                    # Chroma 的持久化写入先于断点；若在两者之间中断，下次 get(ids) 会跳过已入库向量。
                    progress["cursor"] = samples[-1].next_cursor
                    progress["processed"] += len(samples)
                    checkpoint.save()
                    processed_now += len(samples)
                    encoded_now += len(missing)
                    speed = processed_now / max(time.monotonic() - started, 0.001)
                    remaining = max(0, target - checkpoint.processed) / speed / 3600 if target else None
                    LOGGER.info("已确认=%d/%s；本批新编码=%d，已存在=%d；%.3f 道/秒；预计剩余=%s 小时",
                                checkpoint.processed, target or "?", len(missing), len(existing), speed,
                                f"{remaining:.2f}" if remaining is not None else "?")
            finally:
                iterator.close()
            if args.limit and processed_now >= args.limit:
                break
        count = collection.count()
        complete = all(value["done"] for value in checkpoint.data["progress"].values())
        if complete:
            if count != checkpoint.processed or (target is not None and count != target):
                raise ValueError(f"全量校验失败：Chroma={count}，读取问答={checkpoint.processed}，预期={target}。")
            LOGGER.info("全部完成：Chroma 中共有 %d 个向量。", count)
        else:
            LOGGER.info("本次处理已停止：库中 %d 个向量；再次执行同一命令会从断点继续。", count)
        return {"count": count, "processed_now": processed_now, "encoded_now": encoded_now}


def main(argv: list[str] | None = None) -> int:
    """配置中文日志并运行扫描或入库；中断或异常时明确提示续跑方式。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    args = parse_args(argv)
    try:
        if not args.data_root.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{args.data_root}。本机实际路径通常是 G:\\Codex_dataset\\MIRA-data。")
        if args.scan_only:
            scan_dataset(args)
        else:
            run_index(args)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("已中断。已写入的向量和断点会保留，重新执行同一命令即可继续。")
        return 130
    except Exception:
        LOGGER.exception("处理失败，未将失败批次标记完成。修复原因后重新执行同一命令即可续跑。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
