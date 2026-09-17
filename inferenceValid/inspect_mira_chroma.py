# -*- coding: utf-8 -*-
"""查看已有 MIRA Chroma 集合的向量格式，随机打印记录并检查基本有效性。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DB_DIR = Path(r"G:\Codex_dataset\MIRA-chroma")
DEFAULT_COLLECTION = "mira_qwen3_vl_embedding"
EXPECTED_DIMENSION = 2048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析已有数据库的位置、随机抽样数量和控制台显示长度。"""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-h", "--help", action="help", help="显示中文帮助并退出。")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="已有 Chroma 数据库目录；不会新建数据库。")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="要查看的已有集合名称。")
    parser.add_argument("--sample-size", type=int, default=3, help="随机抽取多少条记录，超过总数时查看全部。")
    parser.add_argument("--seed", type=int, default=None, help="随机种子；数据库不变时可复现抽样结果。")
    parser.add_argument("--vector-values", type=int, default=16, help="每个向量显示前多少个数，0 表示完整输出。")
    parser.add_argument("--text-limit", type=int, default=1200, help="问答和元数据各自最多显示的字符数，0 表示完整输出。")
    args = parser.parse_args(argv)
    if args.sample_size <= 0 or args.vector_values < 0 or args.text_limit < 0:
        parser.error("sample-size 必须大于 0，vector-values 和 text-limit 不能小于 0。")
    args.db_dir = args.db_dir.resolve()
    return args


def open_collection(args: argparse.Namespace):
    """只获取已存在的集合，不调用新增、更新或删除接口，也不加载编码模型。"""
    if not (args.db_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"找不到已有数据库：{args.db_dir / 'chroma.sqlite3'}；请检查 --db-dir。")
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError

    client = chromadb.PersistentClient(path=str(args.db_dir), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(args.collection, embedding_function=None)
    except NotFoundError as error:
        available = [item.name for item in client.list_collections(limit=20)]
        raise ValueError(f"集合 {args.collection!r} 不存在。已有集合（最多 20 个）：{available}") from error
    return client, collection


def shorten(text: str, limit: int) -> str:
    """仅缩短控制台展示，不修改数据库中的完整原文。"""
    if limit and len(text) > limit:
        return text[:limit] + f"\n... [省略 {len(text) - limit} 个字符；--text-limit 0 可查看完整内容]"
    return text


def vector_stats(embedding: Any) -> dict[str, Any]:
    """检查返回向量的形状、浮点类型、有限性和 L2 范数，不把抽样当作全库验证。"""
    vector = np.asarray(embedding)
    issues = []
    if vector.shape != (EXPECTED_DIMENSION,):
        issues.append(f"形状应为 ({EXPECTED_DIMENSION},)，实际为 {vector.shape}")
    floating = np.issubdtype(vector.dtype, np.floating)
    finite = bool(np.isfinite(vector).all()) if floating else False
    if not floating:
        issues.append("不是浮点向量")
    elif not finite:
        issues.append("包含 NaN 或 Inf")
    norm = None
    if floating and finite and vector.ndim == 1 and vector.size:
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=0, atol=1e-3):
            issues.append(f"L2 范数为 {norm}，不接近 1，可能未正确归一化")
    return {"shape": vector.shape, "dtype": str(vector.dtype), "finite": finite,
            "norm": norm, "issues": issues}


def print_record(result: dict[str, Any], index: int, offset: int, args: argparse.Namespace) -> dict[str, Any]:
    """显示一条记录的 ID、数值向量、问答及元数据，并返回检查结果。"""
    embeddings = result.get("embeddings")
    embedding = embeddings[0] if embeddings is not None and len(embeddings) else None
    stats = vector_stats(embedding)
    vector = np.asarray(embedding)
    documents = result.get("documents")
    metadatas = result.get("metadatas")
    document = documents[0] if documents is not None and len(documents) else None
    metadata = metadatas[0] if metadatas is not None and len(metadatas) else None
    if not isinstance(document, str) or not document.strip():
        stats["issues"].append("缺少问答原文")
    if not isinstance(metadata, dict) or not metadata:
        stats["issues"].append("缺少元数据")

    print(f"\n========== 随机记录 {index}（读取偏移 {offset}） ==========")
    print(f"ID: {result['ids'][0]}")
    print(f"向量格式: numpy.ndarray，shape={stats['shape']}，读取 dtype={stats['dtype']}")
    print(f"有限数值: {stats['finite']}；L2 范数: {stats['norm']}")
    values = vector.reshape(-1)
    shown = values if args.vector_values == 0 else values[:args.vector_values]
    print(f"向量内容（显示 {shown.size}/{values.size} 个数）:")
    print(json.dumps(shown.tolist(), ensure_ascii=False))
    if shown.size < values.size:
        print("其余数值仅在显示时省略；使用 --vector-values 0 输出完整向量。")
    print("问答原文 (document):")
    print(shorten(document, args.text_limit) if isinstance(document, str) else "<缺失>")
    print("元数据 (metadata，包含图片路径):")
    print(shorten(json.dumps(metadata, ensure_ascii=False, indent=2), args.text_limit))
    print("检查结果: " + ("；".join(stats["issues"]) if stats["issues"] else "通过"))
    return stats


def inspect_collection(collection, args: argparse.Namespace) -> int:
    """按随机位置只读取少量记录，避免将百万条 ID 或向量全部加载到内存。"""
    total = collection.count()
    print("========== 数据库概况 ==========")
    print(f"数据库目录: {args.db_dir}")
    print(f"集合: {args.collection}")
    print(f"当前已保存记录数: {total:,}")
    print("记录结构: id(str) + embedding(一维浮点向量) + document(str) + metadata(dict)")
    print(f"本项目期望维度: {EXPECTED_DIMENSION}；期望 L2 范数: 约 1")
    print("注意：读取 dtype 是 Chroma SDK 返回的类型，不等同于磁盘存储精度。")
    print("集合元数据: " + json.dumps(collection.metadata, ensure_ascii=False))
    if total == 0:
        print("集合为空：尚无已保存向量，无法确认编码成功。")
        return 1

    # 抽取互不重复的位置；每次 get 只请求一条记录及它的向量/原文/元数据。
    offsets = random.Random(args.seed).sample(range(total), min(args.sample_size, total))
    print(f"随机抽样数量: {len(offsets)}；随机种子: {args.seed}")
    seen = set()
    failed = 0
    for index, offset in enumerate(offsets, start=1):
        result = collection.get(limit=1, offset=offset, include=["embeddings", "documents", "metadatas"])
        ids = result.get("ids", [])
        if len(ids) != 1 or ids[0] in seen:
            print(f"警告：偏移 {offset} 未返回唯一记录，集合可能正在变化，请暂停入库后再查看。")
            failed += 1
            continue
        seen.add(ids[0])
        stats = print_record(result, index, offset, args)
        failed += bool(stats["issues"])

    final_count = collection.count()
    print("\n========== 抽样结论 ==========")
    if final_count != total:
        print(f"查看期间记录数从 {total:,} 变为 {final_count:,}；本次读取不是全库一致性快照。")
    if failed:
        print(f"发现 {failed} 条异常或未能读取的记录，请查看上述详情。")
    else:
        print(f"抽取的 {len(seen)} 条记录均可读取，向量维度/数值/归一化及原文/元数据检查通过。")
    print("这只证明所抽记录的基本格式与持久化正常，不证明全量完成，也不评价语义检索效果。")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """设置中文输出并查看集合；对不存在的库、缺少依赖和中断给出明确提示。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        client, collection = open_collection(args)
        return inspect_collection(collection, args)
    except KeyboardInterrupt:
        print("\n已停止查看，未修改向量记录或编码断点。")
        return 130
    except Exception as error:
        print(f"查看失败：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
