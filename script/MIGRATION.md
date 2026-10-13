# 推理与指标评估迁移说明

项目现在支持将“模型推理”和“指标/裁判评分”拆成两个阶段。这样可以先长时间生成并缓存答案，再在不重新加载被测模型的情况下重复调整评分配置。

## 旧结果迁移

旧版生成的 `generations.jsonl` 无需重新推理，可以直接交给新的评分脚本。旧的一步式入口文件已迁移为 `inference_rag.py`；当前项目不再保留 `evaluate_rag.py` 入口。

## 推荐的两阶段流程

### 1. 只做推理

将每个运行配置的顶层 `skipEvaluation` 设为 `true`（`inferenceOnly: true` 也可以作为别名）：

```json
{
  "name": "Qwen",
  "skipEvaluation": true,
  "useRAG": false,
  "model": "Qwen3-VL-2B-Instruct"
}
```

然后执行：

```powershell
python script/inference_rag.py --config script/evaluate_rag.json
```

推理阶段仍执行数据划分、RAG 检索、模型生成和缓存写入，但不会加载裁判模型，也不会计算 FActScore、Accuracy、ROUGE-L 或配对差值。每个任务目录会写出 `generations.jsonl` 和可供续跑的 `predictions.jsonl`；`summary.json`、`comparison.json`、`suite_comparison.json`/`.md` 会保留，但明确标记 `skipped`/`inference_only`，不能把其中的占位字段当成评分结果。

也可以用命令行覆盖配置：

```powershell
python script/inference_rag.py --skip_evaluation --config script/evaluate_rag.json
python script/inference_rag.py --no_skip_evaluation --config script/evaluate_rag.json
```

同一批配置应保持相同的跳过设置。`skipEvaluation` 与 `inferenceOnly` 同时出现时按 `true` 处理，并记录警告。

### 2. 单独计算指标

把推理阶段生成的文件交给独立评估脚本：

```powershell
python script/evaluate_rag_metrics.py \
  --generations output/evaluation/Qwen/no_rag/generations.jsonl \
               output/evaluation/Qwen_lora/no_rag/generations.jsonl \
  --output_dir output/evaluation_metrics
```

评估脚本不加载被测模型，也不读取图片；需要开放题 FActScore 或语义裁判时，只加载配置指定的固定裁判模型。它会在输出根目录生成 `suite_comparison.json`、`suite_comparison.md`、`question_type_comparison.json` 和 `comparison.json`，并在每个标签子目录写出 `predictions.jsonl`、`summary.json` 与 `review_needed.json`。

也可以使用 JSON 配置：

```json
{
  "generations": [
    "../output/evaluation/Qwen/no_rag/generations.jsonl",
    "../output/evaluation/Qwen_lora/no_rag/generations.jsonl"
  ],
  "labels": {
    "../output/evaluation/Qwen/no_rag/generations.jsonl": "Qwen_noRAG",
    "../output/evaluation/Qwen_lora/no_rag/generations.jsonl": "Qwen_lora_noRAG"
  },
  "outputDir": "../output/evaluation_metrics",
  "resume": true,
  "evaluation": {
    "judgeScope": "open_ended",
    "prima": {"enabled": true, "separateQuestionTypes": true}
  }
}
```

### 缓存边界

- 推理端 `--resume` 只复用相同数据划分、模型、LoRA、RAG 和生成提示/参数对应的答案缓存；生成指纹变化时会重新生成。
- 评估端 `--resume` 只复用已有 `generations.jsonl` 的评分结果。修改裁判模型、裁判提示词、PRIMA 开关或预算后，只有匹配指纹的 `judge_cache` 才能复用。
- `judge_cache` 属于评估阶段，不会让推理端跳过模型生成；不要把推理输出目录中的占位 `summary.json` 当作最终指标。
- 为生成结果指定稳定且唯一的标签；若两个输入文件的 `config_name` 相同，请在 JSON 的 `labels` 中显式改名。

## 迁移旧脚本引用

新代码和测试应从 `script.inference_rag` 导入推理入口。历史输出中的 `generations.jsonl` 保持字段兼容，可直接交给 `evaluate_rag_metrics.py`，缺少题型或选项字段时才需要通过 `datasetPath` 回源补充。
