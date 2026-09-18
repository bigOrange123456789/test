# DeepSeek：使用 MIRA 训练集进行纯文本 LoRA 微调

脚本为 `script/finetune_deepseek_mira_lora.py`，适配本地 `DeepSeek-Model` 中的
`Qwen2ForCausalLM` 文本模型。只保存 LoRA adapter，不合并或覆盖原模型。

## 运行

在 `D:\Codex\MLMtest` 下执行：

```powershell
python script/finetune_deepseek_mira_lora.py --check-env
python script/finetune_deepseek_mira_lora.py --dry-run
python script/finetune_deepseek_mira_lora.py --check-data
python script/finetune_deepseek_mira_lora.py --epochs 1
```

`--check-env` 检查依赖和 CUDA；`--dry-run` 只检查清单与 CSV 问答，只需 Python
标准库；`--check-data` 还使用真实 tokenizer 检查全部训练样本的标签与长度，
不加载模型权重、不启动训练、不写出文件。

若依赖不齐全，在同一个 Python 环境中安装：

```powershell
python -m pip install -r script/requirements_deepseek_mira_lora.txt
```

本机已配置 `MLMtest`：PyTorch `2.11.0+cu128`、Transformers `4.57.3`、
PEFT `0.18.0`、Accelerate `1.13.0`，可以识别 RTX 3090 和 BF16。
解释器路径为 `D:\mySoftware2\anaconda3\envs\MLMtest\python.exe`。
`lab2_3` 的 PyTorch 是 CPU 构建，建议使用 `MLMtest` 训练：

```powershell
conda activate MLMtest
python script/finetune_deepseek_mira_lora.py --check-env
python script/finetune_deepseek_mira_lora.py --device cuda --epochs 1
```

脚本不会自动安装依赖。显式指定 `--device cuda` 时，CUDA 不可用会报错；默认
`--device auto` 优先 CUDA，否则使用 CPU。CPU 运行完整模型会明显慢一些。

## 数据与训练目标

- 原始模型默认：`D:\Codex\MLMtest\DeepSeek-Model`。
- 清单默认：`D:\Codex\MLMtest\script\mira_split_ids.json`。
- 只使用清单中的 `train_ids`，保持其顺序，拒绝重复 ID 或与 `test_ids` 重叠。
  训练期间不使用测试集进行评估或参数选择。
- 数据目录按 `--data-root`、清单的 `data_root`、`G:\Codex_dataset\MIRA-data`
  的优先级确定。此机器实际目录为 `G:\Codex_dataset\MIRA-data`。
- ID 按零基 CSV 数据记录及题型内问答索引回源，一条样本是一个问答对。
- 用户输入只有 `Question: ...` 和（若有）`Options: ...`。不读取图片、
  不检查图片是否存在、不使用 `caption` 或 QA 的额外字段作为输入提示。
- 监督目标是完整 `answer`：若为对象或列表，序列化成 JSON，保留正确选项、
  答案解释、视觉证据等原有嵌套内容。这些只作为答案标签，不作为输入。
- 只对 assistant 答案及末尾 EOS 计算损失；system、问题、选项、padding
  的标签为 `-100`。EOS 与 padding 即使共用 token ID，真正的 EOS 也参与监督。
- 找不到 ID、缺少问题/答案、超过 `--max-length` 或模板改变答案时明确报错；
  不静默删除样本或截断答案。所有样本在加载模型、开始优化前完成分词校验。

## 训练参数与进度

默认 1 epoch，batch size 1，梯度累积 8，最大长度 2048，学习率 `2e-4`，
LoRA rank 16 / alpha 32 / dropout 0.05。只训练 attention 与 MLP 投影上的
LoRA 参数，原始参数冻结。默认启用 SDPA 和梯度检查点，单进程、单设备运行。
CUDA 自动选择 BF16（支持时）或 FP16；CPU 使用 FP32。

每次优化器更新打印当前步数、step/s、约 QA/s、已耗时和 ETA，并打印 loss。
速度使用最近最多 20 次更新的平均值；QA/s 按配置的有效批量估算，末尾不足批量
时有偏差。ETA 估计剩余训练时间，不包含最终保存。训练结束打印训练耗时，
保存完成后再打印包括数据读取、模型加载和保存的本次总耗时。

先做一个短测试（使用独立目录）：

```powershell
python script/finetune_deepseek_mira_lora.py --limit 16 --max-steps 2 --output-dir script/deepseek_mira_smoke_adapter
```

`--limit` 仅取清单开头 N 个训练 ID，默认 0 表示全部；`--max-steps` 为正数时
覆盖 epochs。显存不足时保持 `--batch-size 1`，可以降低 `--max-length`，
但应先运行 `--check-data` 确保完整问答能容纳。LoRA 不是 4-bit QLoRA。

## 输出与原模型保护

默认输出到 `script/deepseek_mira_lora_adapter/`，该目录已加入 `.gitignore`。

- `adapter_model.safetensors`、`adapter_config.json`：LoRA 参数和配置。
- tokenizer 文件：本次训练使用的分词器。
- `training_metadata.json`：实际训练 ID、文本输入约定、超参数、版本及耗时。
- `trainer_state.json`：训练状态。
- `checkpoint-*`：每 `--save-steps` 次更新保存，默认 100 次，最多保留两个。
  当前脚本启动新训练，不提供断点续训参数；这些检查点可用于加载阶段性 adapter。

输出目录必须不存在或为空；输出目录不能是原模型目录、其子目录或其祖先目录。
重复实验请指定新的 `--output-dir`，已有训练结果不会被覆盖。
非默认实验目录需要自行添加 Git 忽略规则。

## 对比微调前后

adapter 不是完整模型。基线加载 `DeepSeek-Model`；微调版加载相同基座，
然后用 `PeftModel.from_pretrained(base, adapter_dir)` 加载新增参数即可。
该脚本不会改动平台推理配置或原来的华佗 adapter。

本地聊天模板的 `add_generation_prompt=True` 会额外插入 `<think>`，而
MIRA 的训练答案是直接回答。评测两组模型时，应使用相同的直接回答前缀、
相同问题和选项、相同生成参数。下面示例在项目目录运行：

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from script.finetune_deepseek_mira_lora import (
    DEFAULT_SYSTEM_PROMPT, TextSample, sample_question, render_prompt,
)

base_dir = r"D:\Codex\MLMtest\DeepSeek-Model"
adapter_dir = r"D:\Codex\MLMtest\script\deepseek_mira_lora_adapter"
tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base_dir, local_files_only=True)
# 测基线时省略下一行；两次使用相同的输入和生成参数。
model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

sample = TextSample("example", "What is the diagnosis?", answer=None,
                    options=["A. ...", "B. ..."])
prompt = render_prompt(tokenizer, sample_question(sample), DEFAULT_SYSTEM_PROMPT)
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

训练答案中的结构化 JSON 是原始 MIRA 标注格式，并非前端病例分析所要求的
`diagnosis/findings/analysis/advice` 四字段格式；这次微调目标是纯文本问答对比。
