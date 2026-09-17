# 病例分析 RAG

在项目目录运行：

```powershell
conda activate lab2_3
cd D:\Codex\MLMtest
python cardio_ai_platform.py
```

进入「病例分析」，勾选「RAG 检索增强」，设置参考组数 K（默认 3，范围 1 到 10），选择生成模型后开始分析。不上传图片时，使用病例文本检索；关闭 RAG 时不查询向量库。

通义视觉和本地 `Qwen3-VL-2B-Instruct` 会接收当前病例、上传图片以及每个命中组的完整问答和全部图片。原本 DeepSeek、LoRA DeepSeek 和远程文本 DeepSeek 不支持图片输入：它们可使用图文检索得到的问答文本，但不会接收图片像素。需要完整图文 RAG 时请选择通义视觉或本地 Qwen。

## 本地 Qwen 视觉生成

前端选择 `Qwen3-VL-2B-Instruct（本地视觉）`，对应 `inferenceValid/config.json` 中已有的 `myQwen` 配置。默认模型目录为项目下的 `Qwen3-VL-2B-Instruct`，使用 `Qwen3VLForConditionalGeneration`，与 RAG 用的 Embedding 模型分开加载。支持纯文字、单张上传图片，以及 RAG 中每组的多张参考图片，全程不调用远程生成 API。

首次选用时加载并缓存。也可在启动时预加载此模型：

```powershell
python cardio_ai_platform.py --preload-local-models qwen3-vl-local
```

`myQwen` 可配置 `device`（默认 `auto`）、`dtype`（默认 `auto`）、`min_pixels`（4096）、`max_pixels`（262144）和 `max_context_tokens`（16384，包含生成预算）。自动模式优先 GPU，加载前空闲显存不足 6 GB 时改用 CPU；日志显示实际设备。修改这些加载参数后需要重启服务。修改本地生成模型的图片像素参数不会改变 RAG 入库编码方式。

## 配置

检索配置在 `inferenceValid/rag_config.json`，与生成模型的 `inferenceValid/config.json` 分开：

| 字段 | 默认值或用途 |
| --- | --- |
| `data_root` | `G:/Codex_dataset/MIRA-data`，回源图片的位置 |
| `db_dir` | `G:/Codex_dataset/MIRA-chroma`，已有 Chroma 数据库和入库断点 |
| `collection` | `mira_qwen3_vl_embedding` |
| `model_dir` | `Qwen3-VL-Embedding-2B`，相对路径以项目目录为基准 |
| `device` | `auto`；也可设为 `cpu` 或 `cuda:0` |
| `max_seq_length` | 8192，查询图文总 token 上限；超出时报错，不截断 |
| `max_k` | 10；前端同样限制为 1 到 10，可在后端进一步降低上限 |
| `max_context_chars` | 200000，原始病例和完整参考问答的字符总上限 |
| `max_reference_image_bytes` | 67108864，参考图片的二进制总大小上限 |

生成接口本身的上下文和图片数量限制仍然生效。超限时显示错误，可减小 K；不会悄悄丢掉某组图片或截短答案。

## 编码与检索

查询直接调用 `embed_mira_chroma.QwenEmbeddingEncoder.encode_inputs`，共用入库时的聊天模板、指令、图片预处理、最后一个非填充 token 池化及 L2 归一化，生成 2048 维向量。此实现与 [Qwen 官方 Embedding 模型](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 使用的图文检索流程一致。

程序读取集合对应的 `.checkpoint.json`，继承入库时的精度、像素范围和 caption 设置，并核对模型配置摘要、权重文件名及大小、流水线签名、向量维度和余弦索引。查询文本只包含患者材料，不包含输出格式要求和检索资料。

使用 Chroma 的 `query_embeddings` 检索，返回 document、metadata 和 distance；界面显示的余弦相似度为 `1 - distance`，不是医学诊断置信度。接口及距离定义见 [Chroma 查询文档](https://docs.trychroma.com/docs/querying-collections/query-and-get) 和 [索引配置文档](https://docs.trychroma.com/docs/collections/configure)。

一条向量对应一组原始问答，document 已保留问题、选项、答案和视觉证据等字段；图片由 metadata 的 `image_paths` 回源。不同问答可能共享图片，仍按原始问答组保留，不把同一图片的所有问答合并成一条。模型收到的参考组有独立的 `[MIRA-1]` 等编号，提示词要求区分参考样本与当前患者。

只打开已经存在的集合，不创建或重新编码数据库。尚未完成全量入库时，只检索当前已入库的数据，界面显示当前条数。如果条数少于 K，使用全部可用记录；空库、缺失图片、配置不匹配时明确报错。

持续入库时，如果 Chroma 报告索引 pickle 元数据 `EOF while parsing`，读取端间隔 1 秒重试，最多三次；不会删除、重建或覆盖向量库。持续失败时显示具体错误，需要检查入库日志及索引状态。

## 进度与预加载

服务启动时在后台预加载 RAG 编码器，后续请求复用。首次点击会等待正在进行的预加载，避免重复加载。使用 `--no-preload-rag` 可以改为首次开启 RAG 时才加载；`--preload-local-models none` 可以跳过本地 DeepSeek 的预加载。

`auto` 在 CUDA 可用且空闲显存不少于 6 GB 时尝试 GPU，否则使用 CPU。6 GB 是加载前的初步检查，长文本和多张图片仍可能需要更多显存。CPU 查询可能明显较慢；后台日志会输出实际设备、模型加载状态和耗时。缺少依赖时，在同一虚拟环境安装 `inferenceValid/requirements_mira.txt`。

中间栏由真实后端 SSE 事件更新：接收材料、连接向量库、编码、Top-K 检索、回源、组装上下文、模型生成。回源完成后，在分析过程下方按组显示完整问答和全部对应图片，不必等待模型生成结束。每组保留来源 ID、CSV 数据行、题型、相似度和图片数量；问答不再截短为 1200 字符，后续进度更新也不会替换正在查看的资料。每 10 秒发送处理心跳，长时间编码时仍能看到已耗时。错误包含请求编号和后端堆栈，便于排查。

图片预览保持完整画面和原始比例，点击可在新标签页查看完整尺寸图片。浏览器通过 `/api/rag/image?id=记录ID&index=图片序号` 加载预览，追加 `&full=1` 查看大图；后端根据 Chroma 中记录的图片元数据回源，并将文件读取范围限制在配置的 MIRA 数据目录内。图片加载失败时保留问答并显示错误提示；关闭 RAG 时隐藏检索资料，开始新分析时清除上一轮资料。

通义接口若返回 `HTTP 400 / Arrearage`，表示 API Key 所属账号欠费或账户状态异常，需核对阿里云费用状态；参见 [阿里云错误码说明](https://help.aliyun.com/zh/model-studio/error-code#overdue-payment)。此时若编码、检索、回源步骤已完成，失败发生在远程生成调用阶段。

若出现 `WinError 10054` 或 `ConnectionResetError`，表示网络连接被关闭。TLS 握手阶段的断连发生在模型处理请求之前，不能据此判断为 RAG 编码错误或模型不支持多图。传输层对暂时断连、超时和部分服务繁忙错误默认最多尝试 3 次，每次重新连接并保留完整请求；证书校验错误、欠费、鉴权失败及普通参数错误不会重试。日志包含尝试次数、请求字节数、图片数、代理主机和耗时，不打印 API Key 或图片 Base64。

可在 `inferenceValid/config.json` 对应的远程模型（如 `tongyi`）中添加以下可选字段，保留原有密钥、地址和模型名：

```json
"api_proxy_mode": "system",
"api_max_attempts": 3,
"api_timeout": 90
```

`system` 沿用当前系统或环境代理；设为 `direct` 则仅此模型请求直连，不更改系统代理。代理网络不稳定时可用 `direct` 对比测试。TLS 证书校验始终开启。重试次数可设为 1 到 5；设为 1 可关闭重试，避免生成请求超时后可能发生的重复调用。修改配置后需要重启服务。

## 验证

```powershell
python -X utf8 -m unittest discover -s inferenceValid/tests -p test_rag_mira.py -v
node --test inferenceValid/tests/test_analysis_stream.cjs
```

后端回归测试使用临时真实 Chroma 集合和可控编码向量，覆盖 Top-K 排序、多图分组、完整问答传输、图片预览与原图、图片路径越界、缓存、纯文字、缺失图片、错误配置、上下文限制、RAG 开关及 SSE 错误。生成 API 在测试中被替代，不会产生付费调用。前端测试覆盖中文逐字节分块、CRLF 边界、即时进度、JSON 回退和错误事件。

已启动本地服务且安装 Playwright 时，可运行 `node inferenceValid/tests/test_rag_browser.cjs` 验证模拟结果的多图显示和桌面/手机布局；运行 `node inferenceValid/tests/test_rag_display_live.cjs` 可验证真实 MIRA 检索、图片接口及本地 Qwen 生成，不调用远程付费 API。默认服务地址为 `http://127.0.0.1:8771/#diagnosis`，可通过环境变量 `RAG_PREVIEW_URL` 修改。
