# PRIMA 题型评估说明

在 `MLMtest` 环境运行 `python script/evaluate_rag.py`。所有实验参数仍来自同目录的 `evaluate_rag.json`，四组模型的 `evaluation` 配置必须一致。

实验指标参照 PRIMA（DOI: 10.1109/TMI.2026.3728817）第 7 页 “Metrics and Implementation Details” 和第 8 页 Table II。

| 题型 | JSON 名称 | 答案指标 | 解释指标 |
| --- | --- | --- | --- |
| 开放式 | `open_ended` | FActScore：支持的原子事实数 / 全部候选原子事实数 | 解释 ROUGE-L |
| 封闭式 | `closed_ended` | Accuracy：明确 Yes/No 是否正确 | 解释 ROUGE-L |
| 单选 | `single_choice` | Accuracy：唯一所选选项是否正确 | 解释 ROUGE-L |
| 多选 | `multiple_choice` | Accuracy：所选集合是否与标准集合完全一致 | 解释 ROUGE-L |

多选题的字母顺序不影响得分，例如 `A,C` 与 `C,A` 等价。少选、多选、错选均为 0，不用部分正确的选项 F1 代替 Accuracy。模型无法给出明确最终答案时计 0，保留在有效参考题的分母内。

## JSON 开关

当前四组配置都已开启下面的设置。它们位于每组的 `evaluation.prima` 对象中：

```json
{
  "enabled": true,
  "separateQuestionTypes": true,
  "announceQuestionType": true,
  "questionTypes": ["open_ended", "closed_ended", "single_choice", "multiple_choice"],
  "factScore": true,
  "explanationRougeL": true,
  "factScoreMaxNewTokens": 1024,
  "factScoreBatchSize": 8
}
```

| 参数 | 作用 |
| --- | --- |
| `enabled` | 总开关。`false` 返回先前的参考语义评分流程；其余 PRIMA 参数不生效。 |
| `separateQuestionTypes` | 按题型分别执行生成、显示进度并保存独立目录。关闭后合并执行和主表展示，summary 仍保留逐题型统计。 |
| `announceQuestionType` | 明确提示开放/封闭/单选/多选；多选要求选出全部正确选项，不泄露答案数量。 |
| `questionTypes` | 选择要评估的题型，必须为非空、不重复列表。例如只测试选择题时填 `["single_choice", "multiple_choice"]`。 |
| `factScore` | 开放题原子事实评估开关。关闭后 FActScore 为不适用，不调用整体语义分替代它。 |
| `explanationRougeL` | 要求模型给出 `Answer` 与 `Explanation`，并比较最终解释。关闭后不计该指标。 |
| `factScoreMaxNewTokens` | 事实拆分/每批核验的最大输出 token 数，默认 1024；截断 JSON 会记失败并按设置重试，不能使用残缺数组评分。 |
| `factScoreBatchSize` | 每批核验的事实数，默认 8；剩余事实继续下一批，没有只评分前几条的上限。 |

`evaluation.judgeModel` / `judgeModelPath` 决定所有被测模型共同使用的原版裁判，默认本地原版千问，不装载被测 LoRA。`judgeRetries` 控制各阶段格式失败的重试次数（0 或 1）。`judgeScope`、`judgeMaxNewTokens` 仅控制关闭 PRIMA 后的旧语义评分。

只想关闭新增功能：把四组 `evaluation.prima.enabled` 都改为 `false`。为保留不同设置的实验，建议同时更改 `outputDir`。单独调整某个开关也必须对四组保持一致。

题型过滤在调试参数 `--N` 之前进行。筛掉的测试题仍然从 RAG 知识库中排除，不会变成检索材料。

## FActScore 的具体口径

1. 从模型的最终回答与公开解释中拆出全部可核验的原子事实；不从隐藏 `<think>` 中提取，不向拆分器提供参考答案。
2. 逐条以参考答案及其解释为唯一证据，判断 `supported=true/false`。不支持和证据不足都记为 false，没有 0.5 的部分事实分。
3. 每题分数为支持数 / 事实总数，再对有效开放题取均值；不是把全数据所有事实混在一起加权。

这是**参考答案版 FActScore**。论文未交代其裁判和证据检索实现，当前代码也未调用原论文的外部知识库，因此不能声称完全复现论文结果。该指标衡量事实精度，不衡量答案对所有要点的覆盖率；参考答案不完整或小型裁判误判都会影响结果。

空最终回答或合法提取到零事实时，本项目明确计 0，避免拒答获取高分。事实拆分失败或任一核验批次失败时，整题计 `null`，不能丢弃失败的事实后缩小分母。成功的拆分和核验批次可缓存；断点恢复时复用这些阶段。

## 解释 ROUGE-L 与缺失值

优先读取 `explanation`、`rationale`、`reasoning`，或最终文本中的 `Explanation:` / `解释:`。封闭题 `Yes/No, ...` 后的说明也可提取。只使用用户可见的最终解释，不使用隐藏思考、`visual_evidence` 或图片标题。

参考缺少明确解释时计 `null`，不把开放题整段 `text` 当成解释；参考解释存在但模型没有生成解释时计 0。当前这份 50 题清单有 31 题存在参考解释（单选 17、多选 4、封闭 10），另 19 题缺少独立解释（开放 15、封闭 4），所以解释得分需要结合有效数量/覆盖率阅读。

ROUGE-L 采用项目原有的 jieba 分词 + 最长公共子序列 F1，表示解释的文字相似度，不等同于解释的医学正确性。

## 输出与复核

当前配置的结果根目录为 `output/evaluation_prima`，保留四组名称 `Qwen`、`Qwen_lora`、`Deepseek`、`Deepseek_lora`。

- `suite_comparison.md`：四类题型主表与微调前后变化。
- `suite_comparison.json`：完整统计、置信区间和配对差值。
- `question_type_comparison.json`：便于后续绘图的逐模型、逐题型表。
- `<模型名>/<RAG模式>/by_question_type/<题型>/`：独立的 `predictions.jsonl`、`summary.json`、`test_ids.json`、`review_needed.json`。无该题型时生成空明细和 `n=0` 汇总，不伪造零分。
- `predictions.jsonl` 中的 `factscore_details`：每条原子事实、是否支持、理由，以及裁判原始输出。

在同一输出目录切换协议或关闭分组时，不再使用的题型报告会移入所在目录的 `_previous_reports`，保留可恢复副本。每次启动也会归档上轮的题型总表，避免本轮尚未完成时误读旧结果。

主表的 FActScore、Accuracy 和解释 ROUGE-L 都以 0～100 展示；JSON 内部仍为 0～1。表内 `±` 是有效题目分数的总体标准差，**不是论文多次实验的标准差**；95% 置信区间单独存在 JSON 中。

本流程先完成四组答案生成，再加载一次固定裁判。相比一次性整体语义评分，真正的事实拆分和核验需要更多模型调用。成功阶段缓存用于减少重复工作；首次完整评估仍需相应时间。

使用 `python script/evaluate_rag.py --check_data` 可以只检查四组路径、题型数量和参考解释覆盖，不加载模型、不生成正式结果。
