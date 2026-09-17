# MIRA 图文向量入库

## 默认路径与数据范围

- 脚本：`D:\Codex\MLMtest\inferenceValid\embed_mira_chroma.py`
- 数据：`G:\Codex_dataset\MIRA-data`。实际目录名是 `Codex_dataset`，不是 `Codex\_dataset`。
- 模型：`D:\Codex\MLMtest\Qwen3-VL-Embedding-2B`，不是同名大小的 `Qwen3-VL-2B-Instruct` 聊天模型。
- 数据库及断点：`G:\Codex_dataset\MIRA-chroma`
- Chroma 集合：`mira_qwen3_vl_embedding`

默认只处理 `train.csv`。该文件有 178,003 行图片记录，每行的 `vqa_json` 包含多道题，
展开后合计 **1,120,031** 道问答。`validation.csv` 和 `test.csv` 分别有 33,056 和 8,279 道题。
脚本默认从 `selection_manifest.json` 读取并核对预期总数。

## 运行

在项目根目录、目标 Python 环境中执行：

```powershell
python -m pip install -r inferenceValid/requirements_mira.txt
python inferenceValid/embed_mira_chroma.py --scan-only --check-images
python inferenceValid/embed_mira_chroma.py --limit 10
python inferenceValid/embed_mira_chroma.py
```

`--scan-only` 只扫描数据，检查数量及结构，不加载模型或创建向量库。
`--limit 10` 是本次处理上限；随后直接运行不带 limit 的命令，会从第 11 道题继续。
`Ctrl+C` 或进程意外退出后，重新执行相同命令即可续跑，不需要手动指定起点。

本机 `lab2_3` 的 PyTorch 是 CPU 版本。已补装与其匹配的 `torchvision 0.26.0+cpu`；
这是 Transformers 加载完整 Qwen3-VL 处理器所需的依赖，脚本本身只使用图片和文字。
使用 GPU 时，需要目标环境安装支持该 GPU 的 PyTorch/torchvision 配套版本。

### 本机 GPU 环境 MLMtest

环境路径：`D:\mySoftware2\anaconda3\envs\MLMtest`，独立于原有 `lab2_3` 环境。
使用 Python 3.11、PyTorch 2.11.0 / torchvision 0.26.0（CUDA 12.8 版本）、
Transformers 5.5.1 和 ChromaDB 1.5.9。以下命令在项目根目录执行：

```powershell
conda activate MLMtest
python inferenceValid/embed_mira_chroma.py --device cuda:0 --batch-size 4 --commit-every 32
```

不指定 `--device` 时，脚本也会自动选择可用的 CUDA GPU；显式指定 `cuda:0` 可以在
GPU 不可用时报错，避免不知情地退回 CPU。若后续较长图文导致显存不足，将
`--batch-size` 降为 `1` 后重跑即可保留原断点；不要同时启动多个进程写同一库。

重建环境时使用以下版本和官方 CUDA wheel 源，无需修改原来的 CPU 环境：

```powershell
conda create -n MLMtest python=3.11 pip -y
conda activate MLMtest
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.5.1 chromadb==1.5.9 -r inferenceValid/requirements_mira.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

CUDA wheel 版本参考：[PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)。

本机已验证 RTX 3090 的 CUDA 运算及 BF16 图文编码：真实数据先入库 4 条、重启后续跑 2 条，
共 6 个 2048 维归一化向量，断点与数据库数量一致。`pip check` 和 9 项回归测试均通过。
验证库位于 `C:\Users\HQ\Documents\ChatGPT\test\MLMtest_gpu_smoke`，未启动正式全量提取。

常用选项：

```powershell
python inferenceValid/embed_mira_chroma.py --device cuda:0 --batch-size 4 --commit-every 32
python inferenceValid/embed_mira_chroma.py --data-root "G:\Codex_dataset\MIRA-data" --db-dir "G:\Codex_dataset\MIRA-chroma"
python inferenceValid/embed_mira_chroma.py --help
```

如需包含验证集和测试集，使用另一个集合：

```powershell
python inferenceValid/embed_mira_chroma.py --splits train validation test --collection mira_all_splits
```

## 向量内容

每道题单独形成一条 Chroma 记录。输入包含该题的问题、选项、完整答案（含解释、视觉证据）、
其他原始题目字段，以及它关联的全部图片。单图片路径、JSON 图片路径列表和 Python 字面量图片路径列表均支持。
一张拼图文件仍按一张图片处理，不猜测或拆分其中的子图。

采用 Embedding 模型的完整图文前向计算，取最后一个有效 token 的最终隐藏状态并做 L2 归一化，
生成一个 **2048 维向量**。这不是词嵌入平均或图片特征与文字特征的直接拼接。
系统指令使用模型官方默认的 `Represent the user's input.`。

默认图片像素范围为 4,096 到 262,144；全部图片都会保留，分别等比例缩放。
默认上下文上限为 8,192 token，包含所有图片 token。不会自动截断答案、丢弃图片或分成多个向量。
超长时会报告出错批次和 ID；可增大 `--max-seq-length`（不超过 32,768）后继续。
若改动图片像素范围、是否编码 caption、模型或源标注，需另建集合或数据库，防止混合不同编码结果。

默认 caption 保存在元数据里；需要参与编码时加 `--include-caption`，并使用一个新集合。

训练集有 6 道题缺少答案，其中 1 道的 question 字段是异常对象而非问题字符串。
脚本保留这些记录，在文档中标注源字段缺失，保留异常原始对象，并在元数据中设置
`has_answer`、`has_question`、`missing_fields`，不会虚构问题或答案。
缺失/损坏图片、不可解析的 JSON、模型错误会停止任务并报告位置，不会静默跳过。

## 存储与断点

- `ids`：由划分、CSV 数据行序号、题型、题内序号组成。相同文本的不同原始题目仍保留各自 ID。
- `embeddings`：归一化 float32 向量；Chroma 使用 cosine 索引。
- `documents`：可用于 RAG 上下文的问答及相关文字。
- `metadatas`：划分、源 CSV、从 0 开始的数据行序号、题型、题号、caption、图片路径及质量标记。
- `image_paths`：JSON 字符串列表，相对 `data_root` 的路径可用于找回原图；数据库不复制图片文件。
- `<集合名>.checkpoint.json`：读取位置、行内题号、已处理数量及模型/标注配置摘要。

处理过程只保留一个有限大小的数据批次。先调用 Chroma `upsert`，成功后才原子保存断点。
如果在两者之间中断，重启会先查询当前批次已有 ID，跳过已入库条目的模型计算。
没有保存的计算结果可能需要重算，默认最多涉及当前 16 道题，但不会新增重复记录。
进度位置支持直接 seek 到 CSV 记录，包括字段内有换行及在一行中间停止的情况。

请同时保留整个 Chroma 目录和断点文件。脚本还使用进程锁，避免多个提取进程同时写入该目录。
续跑会校验 CSV 内容摘要、模型配置及权重文件属性，避免把变更后的数据混入原索引。
增加批次大小、切换设备、修改 limit 或增大上下文上限可以继续使用原库。

仅 1,120,031 个 2048 维 float32 向量就约占 **9.18 GB**；Chroma 的索引、SQLite、
元数据和文字还会额外占用磁盘与内存。百万条图文编码在 CPU 上耗时很长，先用限量运行评估速度和资源。

## 后续检索示例

在项目根目录启动 Python，使用同一模型和相同图片预处理设置编码查询。
不要使用 `lab2.py` 中之前为控制台展示而计算的平均向量查询此库。

```python
import chromadb
from chromadb.config import Settings
from inferenceValid.embed_mira_chroma import parse_args, QwenEmbeddingEncoder

args = parse_args([])
encoder = QwenEmbeddingEncoder(args)
vectors = encoder.encode_inputs(["What are the imaging findings of hemorrhage?"], [[]])
client = chromadb.PersistentClient(path=str(args.db_dir), settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(args.collection, embedding_function=None)
results = collection.query(
    query_embeddings=vectors,
    n_results=5,
    where={"has_answer": True},
    include=["documents", "metadatas", "distances"],
)
print(results)
```

如查询同时带图片，将 `[[]]` 替换为该条查询的图片路径分组，例如 `[[Path(".../image.jpg")]]`。
返回的 `distances` 越小越相似；余弦相似度为 `1 - distance`，不是正确匹配概率。

## 验证

### 查看已入库向量

激活 `MLMtest` 后可运行以下查看脚本，不加载模型或使用 GPU，不新增/修改向量及编码断点：

```powershell
python D:\Codex\MLMtest\inferenceValid\inspect_mira_chroma.py
python D:\Codex\MLMtest\inferenceValid\inspect_mira_chroma.py --sample-size 5 --seed 42
python D:\Codex\MLMtest\inferenceValid\inspect_mira_chroma.py --sample-size 1 --vector-values 0 --text-limit 0
```

默认查看正式数据库 `G:\Codex_dataset\MIRA-chroma` 中的 `mira_qwen3_vl_embedding` 集合。
其他数据库使用 `--db-dir` 和 `--collection` 指定。路径或集合不存在时会报错，不创建空库。
输出当前记录数、记录结构、随机 ID、向量形状/读取 dtype/L2 范数、向量数值、问答原文和图片路径等元数据。
默认随机抽取 3 条且不重复，向量展示前 16 个数；`--vector-values 0` 展示完整向量。
`--text-limit 0` 可展示完整问答和元数据。它只按随机偏移读取所需的记录，不将全部向量载入内存。

抽样检查通过仅说明这些记录已保存且基本格式正常，不代表全量完成，也不代表检索效果合格。
读取 dtype 是 SDK 返回数组的类型，不用于推断磁盘存储精度。若编码仍在运行，数量可能继续增长；
需要稳定、可复现的结果时，先用 Ctrl+C 暂停编码，再查看，之后可继续原编码命令。
脚本退出码为 0 表示抽样通过，1 表示异常或空库，130 表示手动中断。
查询接口参考：[Chroma Get 官方文档](https://docs.trychroma.com/docs/querying-collections/query-and-get)。

### 回归测试

```powershell
python -m unittest discover -s inferenceValid/tests -p "test_*mira_chroma.py" -v
```

测试使用独立临时数据库和测试编码器，验证 CSV 多行字段、题目展开、行内续读、去重、
失败不提交进度，以及入库后断点写入前中断的恢复。真实模型另外用少量图文数据验证。

方法参考：[Qwen3-VL-Embedding 官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
和 [Chroma upsert 文档](https://docs.trychroma.com/docs/collections/update-data)。
