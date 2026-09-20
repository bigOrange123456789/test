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
| `factScoreMaxNewTokens` | 关闭动态裁判预算时，事实拆分/每批核验的固定输出 token 上限，默认 1024。 |
| `factScoreBatchSize` | 每批核验的事实数，默认 8；剩余事实继续下一批，没有只评分前几条的上限。 |
| `factScoreLengthBudget` | 动态裁判预算，当前 JSON 已启用，见下方说明。 |

`evaluation.judgeModel` / `judgeModelPath` 决定所有被测模型共同使用的原版裁判，默认本地原版千问，不装载被测 LoRA。`judgeRetries` 控制各阶段格式失败的重试次数（0 或 1）。`judgeScope`、`judgeMaxNewTokens` 仅控制关闭 PRIMA 后的旧语义评分。

只想关闭新增功能：把四组 `evaluation.prima.enabled` 都改为 `false`。为保留不同设置的实验，建议同时更改 `outputDir`。单独调整某个开关也必须对四组保持一致。

题型过滤在调试参数 `--N` 之前进行。筛掉的测试题仍然从 RAG 知识库中排除，不会变成检索材料。

## 按文本长度设置输出 token 预算

回答生成和裁判评分使用独立预算。原来的 `generation.maxNewTokens=2048` 控制模型回答；`evaluation.prima.factScoreMaxNewTokens=1024` 控制事实拆分/核验，1024 并不是所有生成步骤共用的上限。

当前四组 JSON 的 `generation` 都增加了：

```json
"referenceLengthBudget": {
  "enabled": true,
  "multiplier": 4.0,
  "extraTokens": 256,
  "minNewTokens": 512,
  "maxNewTokens": 4096,
  "retryOnTruncation": true
}
```

程序用各基础模型自己的 tokenizer 计算参考答案及公开解释的 token 数，不用字数近似，不计聊天模板标记，也不计结构化 `visual_evidence` 字段。每题预算为 `ceil(参考 token 数 × multiplier) + extraTokens`，再限制在 `minNewTokens` 与 `maxNewTokens` 之间。例如参考答案为 100 tokens 时分配 656；为 500 tokens 时分配 2256。缺少可解析参考文本时使用原 `maxNewTokens`，同样受动态上下限约束。

这只是最大允许生成量，模型遇到结束标记会提前停止。若第一次确实触顶且尚未达到动态上限，`retryOnTruncation=true` 会用 4096 的上限重新生成一次，保留第二次完整回答；不会把第一次的残句当续写上下文。仍然触顶时保存截断标记并列入复核。扩容重试发生错误时保留第一次回答及错误信息，不能冒充完整答案。两次调用的耗时会合计，并在 `generation_info.attempts` 中分别记录。

设 `enabled=false` 可恢复固定 `generation.maxNewTokens`；显式命令行 `--max_new_tokens` 也会关闭回答的动态预算并优先采用该值。回答预算与 PRIMA 总开关独立，关闭 PRIMA 并不会自动关闭回答的动态预算。

裁判的动态配置位于 `evaluation.prima.factScoreLengthBudget`，字段相同，目前为：

```json
{
  "enabled": true,
  "multiplier": 2.0,
  "extraTokens": 256,
  "minNewTokens": 1024,
  "maxNewTokens": 8192,
  "retryOnTruncation": true
}
```

裁判输出长度主要由待处理内容决定：拆分阶段按模型实际回答计数，核验阶段按本批事实 JSON 计数，而不是只看参考答案长度。长回答即使对应很短的参考，也不会因此获得不足的拆分预算。触顶时可在既有 `judgeRetries` 次数内扩容重试，格式错误仍须通过严格 JSON 校验；失败仍记 `null`。关闭后恢复固定 `factScoreMaxNewTokens`。预算与计数依据保存在 `factscore_details.budget_audit` 和各次 `attempts` 中。

参考答案的内容不会进入被测模型提示词，但按测试参考长度分配预算确实使用了测试标注的长度信息。四组应保持同样的预算规则，原版与其 LoRA 使用同一个基础 tokenizer；不同模型 tokenizer 的 token 数可能不同。正式报告应注明这一设置，不应把动态预算结果与固定预算结果当作完全相同的实验条件。FActScore 的评分公式不随预算变化，预算只影响生成和格式完整性。

改变回答预算会使生成缓存失效；仅调整裁判预算可复用已生成的回答，但会重新执行受影响的裁判阶段。当前 JSON 将新结果写到 `output/evaluation_prima_adaptive`，保留先前的 `output/evaluation_prima`。参考预算只是估算，无法保证更快或杜绝截断；模型实际上下文窗口仍是硬限制。

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

当前配置的结果根目录为 `output/evaluation_prima_adaptive`，保留四组名称 `Qwen`、`Qwen_lora`、`Deepseek`、`Deepseek_lora`。

- `suite_comparison.md`：四类题型主表与微调前后变化。
- `suite_comparison.json`：完整统计、置信区间和配对差值。
- `question_type_comparison.json`：便于后续绘图的逐模型、逐题型表。
- `<模型名>/<RAG模式>/by_question_type/<题型>/`：独立的 `predictions.jsonl`、`summary.json`、`test_ids.json`、`review_needed.json`。无该题型时生成空明细和 `n=0` 汇总，不伪造零分。
- `predictions.jsonl` 中的 `factscore_details`：每条原子事实、是否支持、理由，以及裁判原始输出。

在同一输出目录切换协议或关闭分组时，不再使用的题型报告会移入所在目录的 `_previous_reports`，保留可恢复副本。每次启动也会归档上轮的题型总表，避免本轮尚未完成时误读旧结果。

主表的 FActScore、Accuracy 和解释 ROUGE-L 都以 0～100 展示；JSON 内部仍为 0～1。表内 `±` 是有效题目分数的总体标准差，**不是论文多次实验的标准差**；95% 置信区间单独存在 JSON 中。

本流程先完成四组答案生成，再加载一次固定裁判。相比一次性整体语义评分，真正的事实拆分和核验需要更多模型调用。成功阶段缓存用于减少重复工作；首次完整评估仍需相应时间。

使用 `python script/evaluate_rag.py --check_data` 可以只检查四组路径、题型数量和参考解释覆盖，不加载模型、不生成正式结果。
