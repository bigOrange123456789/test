"""PRIMA 题型协议的离线回归：真实数据选择、评分、缓存与报告，模拟模型推理。"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import evaluate_rag as evaluation


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class PrimaEvaluationTests(unittest.TestCase):
    """独立 fixture，不继承旧版测试，避免重复执行整套旧版用例。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evaluation-prima-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "results"
        self.dataset = self.root / "dataset.jsonl"
        self.manifest = self.root / "split.json"
        self.config = self.root / "evaluation.json"
        self.models, self.adapters = {}, {}
        for name, model_type in (("Qwen3-VL-2B-Instruct", "qwen3_vl"), ("DeepSeek-Model", "qwen2")):
            base = self.root / name
            base.mkdir()
            (base / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
            adapter = self.root / (name + "_adapter")
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": str(base),
            }), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"offline adapter marker")
            self.models[name], self.adapters[name] = base, adapter
        self.types = ("open_ended", "single_choice", "multiple_choice", "closed_ended")
        self.samples, self.responses = [], {}
        explanation = "Irregular rhythm supports atrial fibrillation."
        for index, kind in enumerate(self.types):
            question = f"Case {index}: provide the requested diagnosis."
            if kind == "open_ended":
                answer = {"text": "Atrial fibrillation.", "explanation": explanation}
                response = "Answer: Atrial fibrillation.\nExplanation: " + explanation
            elif kind == "single_choice":
                answer = {"correct_option": "A", "explanation": explanation}
                response = "Answer: A\nExplanation: " + explanation
            elif kind == "multiple_choice":
                answer = {"correct_options": ["A", "C"], "explanation": explanation}
                response = "Answer: C, A\nExplanation: " + explanation
            else:
                answer = {"text": "Yes", "explanation": explanation}
                response = "Answer: Yes\nExplanation: " + explanation
            answer["visual_evidence"] = "PRIVATE_VISUAL_ANNOTATION"
            row = {
                "id": f"mira:test:{index}:{kind}:0", "question": question,
                "reference": json.dumps(answer), "answer": answer,
                "question_type": kind, "images": [],
            }
            if kind in {"single_choice", "multiple_choice"}:
                row["options"] = {"A": "Atrial fibrillation", "B": "Healthy", "C": "Tachycardia"}
            self.samples.append(row)
            self.responses[question] = response
        self.knowledge = {
            "id": "mira:train:9:open_ended:0", "question": "Knowledge only.",
            "reference": "Knowledge answer.", "question_type": "open_ended", "images": [],
        }
        self.entries = [
            self.entry("Qwen", "Qwen3-VL-2B-Instruct"),
            self.entry("Qwen_lora", "Qwen3-VL-2B-Instruct", lora=True),
            self.entry("Deepseek", "DeepSeek-Model"),
            self.entry("Deepseek_lora", "DeepSeek-Model", lora=True),
        ]
        self.write_data()
        self.write_config()

    def entry(self, name, model, lora=False):
        return {
            "name": name, "model": model, "pathLora": str(self.adapters[model]) if lora else None,
            "useRAG": False, "datasetPath": str(self.dataset), "datasetFilter": str(self.manifest),
            "evaluation": {
                "judgeModel": "Qwen3-VL-2B-Instruct", "judgeScope": "open_ended",
                "judgeMaxNewTokens": 256, "judgeRetries": 1, "bootstrapSamples": 100,
                "prima": {
                    "enabled": True, "separateQuestionTypes": True, "announceQuestionType": True,
                    "questionTypes": ["open_ended", "closed_ended", "single_choice", "multiple_choice"],
                    "factScore": True, "explanationRougeL": True,
                    "factScoreMaxNewTokens": 1024, "factScoreBatchSize": 8,
                },
            },
        }

    def write_data(self, reserved=None):
        self.dataset.write_text("\n".join(json.dumps(row) for row in [*self.samples, self.knowledge]),
                                encoding="utf-8")
        self.manifest.write_text(json.dumps({
            "test_ids": reserved if reserved is not None else [row["id"] for row in self.samples],
            "train_ids": [self.knowledge["id"]],
        }), encoding="utf-8")

    def write_config(self):
        self.config.write_text(json.dumps(self.entries), encoding="utf-8")

    def set_prima(self, **settings):
        for entry in self.entries:
            entry["evaluation"]["prima"].update(settings)
        self.write_config()

    def resolve(self, *extra):
        with patch.object(evaluation, "MODEL_DIRECTORIES", self.models):
            return evaluation.resolve_run_configs(evaluation.build_parser().parse_args([
                "--config", str(self.config), "--output_dir", str(self.output), *extra,
            ]))

    def run_main(self, *, bad_verification=False, extra=()):
        events, answer_prompts, judge_prompts, active = [], [], [], set()

        def factory(args):
            self.assertFalse(active, "加载下一模型前必须释放上一模型。")
            is_judge = args.run_name == "固定语义裁判"
            name = args.run_name
            if is_judge:
                self.assertEqual(args.model, "Qwen3-VL-2B-Instruct")
                self.assertEqual(Path(args.model_path), self.models[args.model])
                self.assertIsNone(args.lora_path)
                self.assertTrue(args.text_only)
                # 有生成调用的组必须已经卸载；完全缓存命中的组可以不加载。
                self.assertEqual(sum(event[0] == "create_generator" for event in events),
                                 sum(event[0] == "close_generator" for event in events))
            active.add(name)
            events.append(("create_judge" if is_judge else "create_generator", name))

            def chat(messages, max_new_tokens):
                self.assertFalse(is_judge)
                contents = json.dumps(messages, ensure_ascii=False)
                question = next((text for text in self.responses if text in contents), None)
                self.assertIsNotNone(question, contents)
                answer_prompts.append((name, question, contents))
                events.append(("answer", name))
                events.append(("answer_budget", name, max_new_tokens))
                generator.last_generation_info = {
                    "hit_token_limit": False, "output_tokens": 12, "elapsed_seconds": 0.25,
                }
                return self.responses[question]

            def text(prompt, max_new_tokens=256):
                self.assertTrue(is_judge, "被测模型不得评分自身的输出。")
                judge_prompts.append(prompt)
                events.append(("judge_budget", name, max_new_tokens))
                generator.last_generation_info = {
                    "hit_token_limit": False, "output_tokens": 20, "elapsed_seconds": 0.1,
                }
                if "原子事实拆分器" in prompt:
                    events.append(("extract", name))
                    return json.dumps({"facts": ["Atrial fibrillation.", "The rhythm is irregular."]})
                if "原子事实核验器" in prompt:
                    events.append(("verify", name))
                    if bad_verification:
                        return "invalid verification output"
                    return json.dumps({"verdicts": [
                        {"id": 0, "supported": True, "reason": "Reference supports this fact."},
                        {"id": 1, "supported": False, "reason": "No support for this fact."},
                    ]})
                self.fail("预期只调用原子事实拆分/核验；不能退回旧语义评分。")

            def close():
                if name in active:
                    active.remove(name)
                    events.append(("close_judge" if is_judge else "close_generator", name))

            def count_tokens(text):
                count = len(text.split())
                events.append(("count_tokens", name, text, count))
                return count

            generator = SimpleNamespace(
                args=args, supports_images=False if is_judge else args.model != "DeepSeek-Model",
                chat=chat, text=text, close=close, count_tokens=count_tokens,
                last_generation_info={"hit_token_limit": False, "output_tokens": 12},
            )
            return generator

        def close_handlers(**kwargs):
            for handler in kwargs.get("handlers", []):
                handler.close()

        stdout, stderr = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, "MODEL_DIRECTORIES", self.models))
            stack.enter_context(patch.object(evaluation, "create_generator", side_effect=factory))
            stack.enter_context(patch.object(evaluation, "_model_identity", side_effect=lambda path, revision: {"path": path}))
            stack.enter_context(patch.object(evaluation.logging, "basicConfig", side_effect=close_handlers))
            stack.enter_context(patch.dict(sys.modules, {"rag_reports": None}))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            code = evaluation.main(["--config", str(self.config), "--output_dir", str(self.output), *extra])
        self.assertFalse(active)
        return code, events, answer_prompts, judge_prompts, stdout.getvalue(), stderr.getvalue()

    def rows(self, name="Qwen"):
        path = self.output / name / "no_rag" / "predictions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def summary(self, name="Qwen"):
        return read_json(self.output / name / "no_rag" / "summary.json")

    def test_four_models_share_one_fixed_two_stage_fact_judge_after_generation(self):
        code, events, _, judge_prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(sum(event[0] == "create_generator" for event in events), 4)
        self.assertEqual(sum(event[0] == "answer" for event in events), 16)
        self.assertEqual(sum(event[0] == "create_judge" for event in events), 1)
        self.assertEqual(sum(event[0] == "extract" for event in events), 1)
        self.assertEqual(sum(event[0] == "verify" for event in events), 1)
        self.assertEqual(len(judge_prompts), 2, "四组相同回答应共享两阶段成功缓存。")
        self.assertNotIn("create_generator", [event[0] for event in events[events.index(("create_judge", "固定语义裁判")):]])
        for entry in self.entries:
            summary, rows = self.summary(entry["name"]), self.rows(entry["name"])
            self.assertEqual(set(summary["metrics"]), {"factscore", "answer_accuracy", "explanation_rouge_l"})
            self.assertEqual(summary["metrics"]["factscore"]["mean"], 0.5)
            self.assertEqual(summary["metrics"]["factscore"]["total"], 1)
            self.assertEqual(summary["metrics"]["answer_accuracy"]["mean"], 1)
            self.assertEqual(summary["metrics"]["answer_accuracy"]["total"], 3)
            self.assertEqual(summary["metrics"]["explanation_rouge_l"]["mean"], 1)
            self.assertIsNone(rows[0]["answer_accuracy"])
            self.assertTrue(all(row["factscore"] is None for row in rows[1:]))
            self.assertEqual(summary["judge_failures"], 0)

    def test_separate_question_type_outputs_and_percent_accuracy_table(self):
        code, _, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        for entry in self.entries:
            for sample in self.samples:
                directory = self.output / entry["name"] / "no_rag" / "by_question_type" / sample["question_type"]
                self.assertEqual(read_json(directory / "test_ids.json"), [sample["id"]])
                rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["id"] for row in rows], [sample["id"]])
                self.assertEqual(read_json(directory / "summary.json")["num_samples"], 1)
                self.assertEqual(read_json(directory / "review_needed.json"), [])
        markdown = (self.output / "suite_comparison.md").read_text(encoding="utf-8")
        for label in ("开放", "封闭", "单选", "多选"):
            self.assertIn(label, markdown)
        self.assertRegex(markdown, r"Accuracy[^\n]*%")
        self.assertRegex(markdown, r"100(?:\.0+)?")

    def test_prompts_announce_choices_without_leaking_reference_or_answer_count(self):
        code, _, prompts, judge_prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        for _, question, prompt in prompts:
            sample = next(row for row in self.samples if row["question"] == question)
            self.assertNotIn("PRIVATE_VISUAL_ANNOTATION", prompt)
            self.assertNotIn(sample["answer"]["explanation"], prompt)
            self.assertIn("Explanation:", prompt)
            if sample["question_type"] == "single_choice":
                self.assertIn("single-choice", prompt)
                self.assertIn("单选", prompt)
            if sample["question_type"] == "multiple_choice":
                self.assertIn("multiple-choice", prompt)
                self.assertIn("全部", prompt)
                self.assertNotRegex(prompt, r"(?:恰好|一共|共有|正好)\s*(?:2|两|二)\s*(?:个|项)")
        self.assertTrue(all("PRIVATE_VISUAL_ANNOTATION" not in prompt for prompt in judge_prompts))

    def test_multiple_choice_accuracy_requires_exact_option_set(self):
        self.entries = self.entries[:1]
        self.set_prima(factScore=False)
        for answer in ("A", "A, B, C", "B, C"):
            with self.subTest(answer=answer):
                self.responses[self.samples[2]["question"]] = "Answer: " + answer + "\nExplanation: Selected options."
                code, _, _, _, _, errors = self.run_main()
                self.assertEqual(code, 0, errors)
                row = next(row for row in self.rows() if row["id"] == self.samples[2]["id"])
                self.assertEqual(row["answer_accuracy"], 0)
                summary = self.summary()
                self.assertAlmostEqual(summary["metrics"]["answer_accuracy"]["mean"], 2 / 3)
                self.assertEqual(summary["by_question_type"]["multiple_choice"]["metrics"]["answer_accuracy"]["mean"], 0)

    def test_explanation_metric_distinguishes_missing_reference_from_missing_prediction(self):
        self.entries = self.entries[:1]
        self.samples[1]["answer"] = {"correct_option": "A", "visual_evidence": "PRIVATE_VISUAL_ANNOTATION"}
        self.samples[1]["reference"] = json.dumps(self.samples[1]["answer"])
        self.responses[self.samples[2]["question"]] = "<think>Irregular rhythm supports atrial fibrillation.</think>\nAnswer: A, C"
        self.write_data()
        self.set_prima(factScore=False)
        code, _, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        rows = {row["id"]: row for row in self.rows()}
        single, multiple = rows[self.samples[1]["id"]], rows[self.samples[2]["id"]]
        self.assertIsNone(single["explanation_rouge_l"], "参考缺少解释不能用答案字母或视觉标注替代。")
        self.assertEqual(multiple["explanation_rouge_l"], 0, "隐藏思考块不能当作面向用户的解释。")
        self.assertEqual(multiple["answer_accuracy"], 1)
        self.assertEqual(single["answer_details"]["explanation_status"], "reference_missing")
        self.assertEqual(multiple["answer_details"]["explanation_status"], "prediction_missing")

    def test_fact_judge_failure_stays_null_and_is_listed_for_review(self):
        self.entries = self.entries[:1]
        self.write_config()
        code, _, _, _, _, errors = self.run_main(bad_verification=True)
        self.assertEqual(code, 2, errors)
        self.assertIsNone(self.rows()[0]["factscore"])
        summary = self.summary()
        self.assertIsNone(summary["metrics"]["factscore"]["mean"])
        self.assertEqual(summary["metrics"]["factscore"]["coverage"], 0)
        self.assertEqual(summary["judge_failures"], 1)
        review = read_json(self.output / "Qwen" / "no_rag" / "review_needed.json")
        self.assertEqual([row["id"] for row in review], [self.samples[0]["id"]])

    def test_default_and_disabled_protocol_keep_five_legacy_metrics(self):
        self.entries = self.entries[:1]
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                if enabled is None:
                    self.entries[0]["evaluation"].pop("prima", None)
                else:
                    self.entries[0]["evaluation"]["prima"] = {"enabled": False}
                self.entries[0]["evaluation"]["judgeScope"] = "none"
                self.write_config()
                jobs = self.resolve()
                self.assertFalse(jobs[0].evaluation["prima"]["enabled"])
                code, events, _, _, _, errors = self.run_main()
                self.assertEqual(code, 0, errors)
                self.assertEqual(set(self.summary()["metrics"]), set(evaluation.METRICS))
                self.assertEqual(len(self.summary()["metrics"]), 5)
                self.assertFalse(any(event[0] == "create_judge" for event in events))

    def test_each_optional_new_feature_can_be_disabled_in_json(self):
        self.entries = self.entries[:1]
        self.set_prima(factScore=False, explanationRougeL=False, separateQuestionTypes=False,
                       announceQuestionType=False)
        code, events, prompts, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertFalse(any(event[0] == "create_judge" for event in events))
        self.assertTrue(all("题型：" not in prompt for _, _, prompt in prompts))
        self.assertFalse((self.output / "Qwen" / "no_rag" / "by_question_type").exists())
        self.assertEqual(self.summary()["metrics"]["factscore"]["n"], 0)
        self.assertEqual(self.summary()["metrics"]["explanation_rouge_l"]["n"], 0)
        self.assertEqual(self.summary()["metrics"]["answer_accuracy"]["mean"], 1)

    def test_question_type_selection_precedes_limit_and_reserves_all_test_ids_for_rag(self):
        self.entries = self.entries[:1]
        self.set_prima(questionTypes=["multiple_choice"], factScore=False)
        args = self.resolve("--N", "1")[0]
        test, knowledge, metadata = evaluation.prepare_protocol_data(args, [{"name": "rag", "use_rag": True, "top_k": 1}])
        self.assertEqual([row["id"] for row in test], [self.samples[2]["id"]])
        self.assertEqual([row["id"] for row in knowledge], [self.knowledge["id"]])
        self.assertEqual(set(metadata["reserved_test_ids"]), {row["id"] for row in self.samples})
        code, _, prompts, _, _, errors = self.run_main(extra=("--N", "1"))
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(prompts), 1)
        self.assertEqual([row["id"] for row in self.rows()], [self.samples[2]["id"]])

    def test_invalid_prima_settings_rejected_before_loading_models(self):
        self.entries = self.entries[:1]
        original = copy.deepcopy(self.entries[0]["evaluation"]["prima"])
        invalid = [
            None, [], {"unknown": True}, {"enabled": "true"}, {"separateQuestionTypes": 1},
            {"announceQuestionType": None}, {"factScore": 1}, {"explanationRougeL": "false"},
            {"questionTypes": []}, {"questionTypes": "open_ended"},
            {"questionTypes": ["single_choice", "single_choice"]}, {"questionTypes": ["unknown"]},
            {"factScoreMaxNewTokens": True}, {"factScoreMaxNewTokens": 0},
            {"factScoreBatchSize": 0}, {"factScoreBatchSize": 100},
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.entries[0]["evaluation"]["prima"] = value if not isinstance(value, dict) else original | value
                self.write_config()
                with self.assertRaises(ValueError):
                    self.resolve()
        self.assertFalse(self.output.exists())

    def test_different_prima_protocols_in_same_comparison_are_rejected(self):
        self.entries[-1]["evaluation"]["prima"]["announceQuestionType"] = False
        self.write_config()
        code, events, _, _, _, errors = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(events, [])
        self.assertIn("evaluation", errors)
        self.assertFalse(self.output.exists())

    def test_no_matching_questions_is_an_error_not_a_zero_score(self):
        self.entries = self.entries[:1]
        self.write_data(reserved=[self.samples[0]["id"]])
        self.set_prima(questionTypes=["multiple_choice"], factScore=False)
        code, events, _, _, _, _ = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(events, [])

    def test_changing_question_type_prompt_invalidates_generation_cache(self):
        self.entries = self.entries[:1]
        self.entries[0]["resume"] = True
        self.set_prima(factScore=False, announceQuestionType=False)
        code, _, prompts, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(prompts), 4)
        old_fingerprints = [row["generation_fingerprint"] for row in self.rows()]
        self.set_prima(announceQuestionType=True)
        code, _, prompts, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(prompts), 4, "改变题型提示后必须重新生成，不能复用旧提示的回答。")
        self.assertNotEqual(old_fingerprints, [row["generation_fingerprint"] for row in self.rows()])
        self.assertTrue(all(not row["generation_from_cache"] for row in self.rows()))

    def test_disabled_reports_are_archived_when_reusing_the_same_output_directory(self):
        self.entries = self.entries[:1]
        self.set_prima(factScore=False)
        code, _, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        mode_directory = self.output / "Qwen" / "no_rag"
        group_directory = mode_directory / "by_question_type"
        # 保存完整旧报告快照，后续验证归档不仅存在，而且内容没有被覆盖。
        previous_groups = {path.relative_to(group_directory): path.read_bytes()
                           for path in group_directory.rglob("*") if path.is_file()}
        self.assertEqual(len(previous_groups), len(self.types) * 4)
        active_comparison = self.output / "question_type_comparison.json"
        self.assertEqual({row["question_type"] for row in read_json(active_comparison)["rows"]},
                         set(self.types))

        self.set_prima(separateQuestionTypes=False)
        code, _, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertFalse(group_directory.exists(), "关闭分组后不能留下看似仍有效的旧分组报告。")
        archives = list((mode_directory / "_previous_reports").glob("by_question_type-*"))
        self.assertEqual(len(archives), 1)
        archived_groups = {path.relative_to(archives[0]): path.read_bytes()
                           for path in archives[0].rglob("*") if path.is_file()}
        self.assertEqual(archived_groups, previous_groups)
        self.assertFalse(self.summary()["evaluation"]["prima"]["separateQuestionTypes"])
        self.assertTrue(self.summary()["evaluation"]["prima"]["enabled"])
        merged = read_json(active_comparison)
        self.assertEqual([row["question_type"] for row in merged["rows"]], ["all"])
        self.assertEqual(merged["rows"][0]["num_samples"], len(self.samples))
        markdown = (self.output / "suite_comparison.md").read_text(encoding="utf-8")
        data_lines = [line for line in markdown.splitlines() if line.startswith("| Qwen |")]
        self.assertEqual(len(data_lines), 1)
        self.assertIn("| 合并 |", data_lines[0])
        self.assertIn("Accuracy (%)", markdown)
        merged_bytes = active_comparison.read_bytes()
        previous_suite_archives = set((self.output / "_previous_reports").glob("question_type_comparison.json-*"))

        self.entries[0]["evaluation"]["judgeScope"] = "none"
        self.set_prima(enabled=False)
        code, _, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertFalse(active_comparison.exists(), "关闭协议后旧题型总表应移出活动报告位置。")
        suite_archives = set((self.output / "_previous_reports").glob("question_type_comparison.json-*"))
        self.assertTrue(previous_suite_archives.issubset(suite_archives))
        added_archives = list(suite_archives - previous_suite_archives)
        self.assertEqual(len(added_archives), 1)
        self.assertEqual(added_archives[0].read_bytes(), merged_bytes)
        self.assertFalse(group_directory.exists())
        self.assertEqual(list((mode_directory / "_previous_reports").glob("by_question_type-*")), archives)
        self.assertEqual({path.relative_to(archives[0]): path.read_bytes()
                          for path in archives[0].rglob("*") if path.is_file()}, previous_groups)
        self.assertFalse(self.summary()["evaluation"]["prima"]["enabled"])
        self.assertEqual(set(self.summary()["metrics"]), set(evaluation.METRICS))
        suite = read_json(self.output / "suite_comparison.json")
        self.assertNotIn("question_type_comparison", suite)
        self.assertEqual(set(suite["summaries"][0]["metrics"]), set(evaluation.METRICS))
        markdown = (self.output / "suite_comparison.md").read_text(encoding="utf-8")
        self.assertIn("BLEU-4", markdown)
        self.assertNotIn("# PRIMA题型评估", markdown)
        self.assertNotIn("Accuracy (%)", markdown)

    def test_dynamic_answer_and_fact_budgets_flow_through_pipeline_and_saved_audits(self):
        self.entries = self.entries[:1]
        answer_policy = {
            "enabled": True, "multiplier": 10, "extraTokens": 64,
            "minNewTokens": 128, "maxNewTokens": 1024, "retryOnTruncation": True,
        }
        fact_policy = {
            "enabled": True, "multiplier": 2, "extraTokens": 128,
            "minNewTokens": 256, "maxNewTokens": 2048, "retryOnTruncation": True,
        }
        self.entries[0]["generation"] = {"referenceLengthBudget": answer_policy}
        self.entries[0]["evaluation"]["prima"]["factScoreLengthBudget"] = fact_policy
        self.write_config()
        code, events, prompts, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(sum(event[0] == "count_tokens" and event[1] == "Qwen" for event in events), 4)
        self.assertEqual(sum(event[0] == "count_tokens" and event[1] == "固定语义裁判" for event in events), 2)
        self.assertEqual([event[2] for event in events if event[0] == "judge_budget"], [256, 256])
        summary = self.summary()
        self.assertEqual(summary["generation_settings"]["reference_length_budget"], {
            "version": "reference-length-v1", **answer_policy,
        })
        self.assertEqual(summary["evaluation"]["prima"]["factScoreLengthBudget"], fact_policy)
        manifest = read_json(self.output / "Qwen" / "run_config.json")
        self.assertEqual(manifest["arguments"]["reference_length_budget"], answer_policy)
        for row in self.rows():
            info = row["generation_info"]
            audit = info["token_budget"]
            count = len(row["answer_details"]["semantic_reference"].split())
            expected = min(1024, max(128, count * 10 + 64))
            self.assertEqual(audit["reference_tokens"], count)
            self.assertEqual(audit["policy"], answer_policy)
            self.assertEqual(audit["initial_max_new_tokens"], expected)
            self.assertEqual(info["max_new_tokens"], expected)
            self.assertEqual(info["total_output_tokens"], 12)
            self.assertEqual(len(info["attempts"]), 1)
        open_row = next(row for row in self.rows() if row["id"] == self.samples[0]["id"])
        details = open_row["factscore_details"]
        self.assertEqual(details["score"], 0.5)
        self.assertEqual(details["length_budget"], fact_policy)
        self.assertEqual([audit["stage"] for audit in details["budget_audit"]], ["extract", "verify"])
        self.assertTrue(all(audit["calculated_max_new_tokens"] == 256 for audit in details["budget_audit"]))
        self.assertTrue(all("PRIVATE_VISUAL_ANNOTATION" not in event[2]
                            for event in events if event[0] == "count_tokens"))
        self.assertTrue(all(self.samples[0]["answer"]["explanation"] not in prompt
                            for _, _, prompt in prompts))


if __name__ == "__main__":
    unittest.main()
