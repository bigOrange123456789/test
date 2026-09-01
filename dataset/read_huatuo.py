# -*- coding: utf-8 -*-
"""
Huatuo-26M 数据读取与可视化脚本
================================
论文: Huatuo-26M, a Large-scale Chinese Medical QA Dataset (arXiv:2305.01526)

【为什么缓存文件里看不到中文？】
你用 datasets.load_dataset(..., cache_dir=...) 下载后，数据并不是以 txt/json
明文保存的，而是被 HuggingFace datasets 库序列化成了 Apache Arrow 二进制
格式（cache 目录下的 *.arrow 文件）。直接用文本编辑器打开当然看不到中文，
必须通过 datasets 库（或 pyarrow）读取解码后才能看到中文文本。

本脚本功能：
1. 从你已下载的 cache_dir 直接加载 4 个数据集（不会重复下载）；
2. 打印每个数据集的 split、字段、样本数；
3. 把嵌套列表格式的 question/answer 展开为可读字符串；
4. 在控制台彩色打印若干条中文样本（Windows 下自动处理 UTF-8 编码）；
5. 把样本导出为 .txt（人看）和 .jsonl（程序用）两种明文文件。

依赖: pip install datasets
"""

import io
import os
import sys
import json

# ---- Windows 控制台 UTF-8 处理（Linux/Mac 上无副作用）----
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset

# =====================================================================
# 1. 数据集配置：名称 -> 你下载时用的 cache_dir
#    保持和你下载代码里一致即可，load_dataset 会直接复用缓存、不重新下载
# =====================================================================
DATASETS = {
    "knowledge_graph": {
        "repo": "FreedomIntelligence/huatuo_knowledge_graph_qa",
        "cache_dir": "./my_data_cache/data1",
        "desc": "医疗知识图谱问答（答案为简短实体/短语）",
    },
    "encyclopedia": {
        "repo": "FreedomIntelligence/huatuo_encyclopedia_qa",
        "cache_dir": "./my_data_cache/data2",
        "desc": "在线医疗百科问答（答案为长段落科普文本）",
    },
    "consultation": {
        "repo": "FreedomIntelligence/huatuo_consultation_qa",
        "cache_dir": "./my_data_cache/data3",
        "desc": "在线问诊记录（注意：answer 字段是 51zyzy.com 的 URL，不是正文！）",
    },
    "testdatasets": {
        "repo": "FreedomIntelligence/huatuo26M-testdatasets",
        "cache_dir": "./my_data_cache/data4",
        "desc": "论文使用的测试集（多来源混合抽样）",
    },
}

OUTPUT_DIR = "./huatuo_samples"   # 导出明文样本的目录
N_SAMPLES = 5                     # 每个 split 打印/导出多少条样本


# =====================================================================
# 2. 工具函数：把 HuggingFace 返回的嵌套列表展开成纯字符串
#    官方数据形如:
#      {'question': ["颜面部凹陷的手术治疗有些什么？"],
#       'answer':   ["自体颗粒脂肪移植；自体脂肪移植；..."]}
#    百科数据 question 还可能是 [["曲匹地尔片的用法用量"]] 这种双层嵌套
# =====================================================================
def flatten_to_text(value):
    """递归展开任意层级的 list，拼成可读字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [flatten_to_text(v) for v in value]
        parts = [p for p in parts if p]          # 去掉空串
        return "\n".join(parts)                   # 多条之间换行
    return str(value).strip()


def show_sample(idx, example):
    """在控制台漂亮地打印一条中文样本。"""
    print(f"\n{'─' * 70}")
    print(f"📝 样本 #{idx + 1}")
    print(f"{'─' * 70}")

    # 先打印 question / answer 两个主字段
    q = flatten_to_text(example.get("question"))
    a = flatten_to_text(example.get("answer"))
    print(f"【问题 Question】\n{q}")
    print(f"\n【回答 Answer】\n{a}")

    # 再打印可能存在的其他字段（如 Huatuo-Lite 的 label/related_diseases，
    # 测试集若有额外字段也能一并显示）
    extra_keys = [k for k in example.keys() if k not in ("question", "answer")]
    for k in extra_keys:
        v = flatten_to_text(example[k])
        if v:
            print(f"\n【{k}】\n{v}")


def export_samples(split_name, examples, out_path_txt, out_path_jsonl):
    """把样本导出为明文 txt 和 jsonl。"""
    with open(out_path_txt, "w", encoding="utf-8") as f_txt, \
         open(out_path_jsonl, "w", encoding="utf-8") as f_json:
        for i, ex in enumerate(examples):
            q = flatten_to_text(ex.get("question"))
            a = flatten_to_text(ex.get("answer"))

            f_txt.write(f"===== 样本 #{i + 1} =====\n")
            # f_txt.write(f"【问题】\n{q}\n\n")
            # f_txt.write(f"【回答】\n{a}\n")
            for k in ex.keys():
                if k not in ("question", "answer"):
                    v = flatten_to_text(ex[k])
                    if v:
                        f_txt.write(f"【{k}】{v}\n")
            f_txt.write("\n")

            # jsonl: question/answer 统一拍平为字符串，方便后续程序直接用
            row = {"question": q, "answer": a}
            for k in ex.keys():
                if k not in ("question", "answer"):
                    row[k] = flatten_to_text(ex[k])
            f_json.write(json.dumps(row, ensure_ascii=False) + "\n")


# =====================================================================
# 3. 主流程
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for short_name, cfg in DATASETS.items():
        print("\n" + "=" * 70)
        print(f"📦 数据集: {short_name}  ({cfg['repo']})")
        print(f"   说明: {cfg['desc']}")
        print(f"   缓存目录: {cfg['cache_dir']}")
        print("=" * 70)

        # 加载——指定 cache_dir 后直接读本地 Arrow 缓存
        ds = load_dataset(cfg["repo"], cache_dir=cfg["cache_dir"])

        # DatasetDict: 包含 train / validation / test 等 split
        for split_name, split_ds in ds.items():
            print(f"\n▶ split = {split_name} | 样本数 = {len(split_ds)} | 字段 = {split_ds.column_names}")

            n_show = min(N_SAMPLES, len(split_ds))
            samples = [split_ds[i] for i in range(n_show)]

            # 控制台可视化
            for i, ex in enumerate(samples):
                show_sample(i, ex)

            # 导出明文文件
            tag = f"{short_name}_{split_name}"
            txt_path = os.path.join(OUTPUT_DIR, f"{tag}_samples.txt")
            jsonl_path = os.path.join(OUTPUT_DIR, f"{tag}_samples.jsonl")
            export_samples(split_name, samples, txt_path, jsonl_path)
            print(f"\n✅ 已导出: {txt_path}")
            print(f"✅ 已导出: {jsonl_path}")

        # consultation 数据集的特别提醒
        if short_name == "consultation":
            print("\n" + "!" * 70)
            print("⚠️  注意：huatuo_consultation_qa 的 answer 只是一个 URL，例如")
            print("    https://www.51zyzy.com/question/detail/10391424.html")
            print("    论文作者为防止数据滥用，公开版本不提供问诊正文。")
            print("    需要全文须向作者申请（仅限科研用途），或自行按 URL 爬取。")
            print("!" * 70)

    print("\n🎉 全部完成！导出的明文文件在目录:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
