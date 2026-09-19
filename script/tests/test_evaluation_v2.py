"""新评估链路的离线集成检查：独立生成、统一裁判、共享缓存和缺失评分。

模型工厂使用轻量模拟；数据选择、答案归一化、客观评分、参考裁判缓存和结果文件
仍走真实实现，不加载模型、不占用 GPU。
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
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
from script.lib.evaluation_statistics import paired_summary, summarize_values


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class StatisticsTests(unittest.TestCase):
    def test_null_excluded_from_mean_with_explicit_coverage(self):
        result = summarize_values([1, None, 0, 0.5], bootstrap_samples=200)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(len(result["ci95"]), 2)
        self.assertGreaterEqual(result["ci95"][0], 0)
        self.assertLessEqual(result["ci95"][1], 1)

    def test_all_missing_and_empty_do_not_invent_zero_scores(self):
        for values, total in (([None, None], 2), ([], 0)):
            with self.subTest(values=values):
                result = summarize_values(values, bootstrap_samples=100)
                self.assertIsNone(result["mean"])
                self.assertIsNone(result["std"])
                self.assertEqual(result["ci95"], [None, None])
                self.assertEqual(result["n"], 0)
                self.assertEqual(result["total"], total)
                self.assertEqual(result["coverage"], 0)

    def test_single_sample_does_not_claim_confidence_interval(self):
        result = summarize_values([None, 0.5], bootstrap_samples=100)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["std"], 0)
        self.assertEqual(result["ci95"], [None, None])
        self.assertIn("一个", result["ci_method"])

    def test_bootstrap_reproducible_for_seed_and_constant_scores(self):
        values = [1, 0, 0.5, 1, 0, 0.5, 0]
        first = summarize_values(values, seed=19, bootstrap_samples=500)
        second = summarize_values(values, seed=19, bootstrap_samples=500)
        self.assertEqual(first, second)
        constant = summarize_values([0.5] * 10, bootstrap_samples=100)
        self.assertEqual(constant["ci95"], [0.5, 0.5])

    def test_nonfinite_scores_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                summarize_values([invalid])

    def test_binary_wilson_interval_keeps_uncertainty_for_small_perfect_samples(self):
        perfect = summarize_values([1] * 35, binary=True)
        self.assertAlmostEqual(perfect["ci95"][0], 0.9010990077)
        self.assertAlmostEqual(perfect["ci95"][1], 1)
        failed = summarize_values([0] * 35, binary=True)
        self.assertAlmostEqual(failed["ci95"][0], 0)
        self.assertAlmostEqual(failed["ci95"][1], 1 - perfect["ci95"][0])
        single = summarize_values([1], binary=True)
        self.assertLess(single["ci95"][0], 0.21)
        self.assertIn("Wilson", single["ci_method"])
        with self.assertRaises(ValueError):
            summarize_values([0.5], binary=True)

    def test_paired_difference_uses_same_questions_with_both_scores_present(self):
        result = paired_summary([1, None, 0.5, 0], [0, 1, None, 1], bootstrap_samples=100)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["std"], 1)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["coverage"], 0.5)

    def test_paired_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary([1], [0, 1])


class EvaluationV2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evaluation-v2-")
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
        self.samples = []
        self.responses = {}
        for index in range(35):
            if index < 15:
                kind, answer, response = "single_choice", "A", "A"
            elif index < 25:
                kind, answer, response = "multiple_choice", ["A", "C"], "A, C"
            else:
                kind, answer, response = "closed_ended", "Yes", "Yes"
            question = f"Objective question number {index}: choose the final answer."
            row = {
                "id": f"mira:test:{index}:{kind}:0", "question": question,
                "reference": json.dumps({"text": answer}) if isinstance(answer, list) else answer,
                "answer": {"text": answer} if isinstance(answer, list) else answer,
                "question_type": kind, "images": [],
            }
            if kind in {"single_choice", "multiple_choice"}:
                row["options"] = {"A": "Atrial fibrillation", "B": "Healthy", "C": "Tachycardia"}
            self.samples.append(row)
            self.responses[question] = response
        for index in range(2):
            question = f"Open question number {index}: explain the diagnosis."
            self.samples.append({
                "id": f"mira:test:{35 + index}:open_ended:0", "question": question,
                "reference": json.dumps({"text": "Atrial fibrillation", "visual_evidence": "annotation only"}),
                "question_type": "open_ended", "images": [],
            })
            self.responses[question] = json.dumps({"text": "Atrial fibrillation", "visual_evidence": "ignored"})
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.samples), encoding="utf-8")
        self.manifest.write_text(json.dumps({"test_ids": [row["id"] for row in self.samples], "train_ids": []}),
                                 encoding="utf-8")
        self.entries = [
            self.entry("Qwen", "Qwen3-VL-2B-Instruct"),
            self.entry("Qwen_lora", "Qwen3-VL-2B-Instruct", lora=True),
            self.entry("Deepseek", "DeepSeek-Model"),
            self.entry("Deepseek_lora", "DeepSeek-Model", lora=True),
        ]
        self.write_config()

    def entry(self, name, model, lora=False):
        return {
            "name": name, "model": model, "pathLora": str(self.adapters[model]) if lora else None,
            "useRAG": False, "datasetPath": str(self.dataset), "datasetFilter": str(self.manifest),
            "evaluation": {
                "judgeModel": "Qwen3-VL-2B-Instruct", "judgeScope": "open_ended",
                "judgeMaxNewTokens": 256, "judgeRetries": 1, "bootstrapSamples": 100,
            },
        }

    def write_config(self, *, scope=None):
        if scope is not None:
            for entry in self.entries:
                entry["evaluation"]["judgeScope"] = scope
        self.config.write_text(json.dumps(self.entries), encoding="utf-8")

    def run_main(self, *, bad_judge=False, prediction_override=None, extra=()):
        events, prompts, active, generated = [], [], set(), []

        def factory(args):
            self.assertFalse(active, "加载下一模型之前必须先卸载当前模型。")
            is_judge = len(generated) == len(self.entries)
            name = "fixed_judge" if is_judge else args.run_name
            if is_judge:
                self.assertEqual(args.model, "Qwen3-VL-2B-Instruct")
                self.assertEqual(Path(args.model_path), self.models["Qwen3-VL-2B-Instruct"])
                self.assertIsNone(args.lora_path, "统一裁判不得继承某组的 LoRA。")
                self.assertEqual(sum(event[0] == "close_generator" for event in events), len(self.entries))
            else:
                self.assertEqual(name, self.entries[len(generated)]["name"])
                generated.append(name)
            active.add(name)
            events.append(("create_judge" if is_judge else "create_generator", name))

            def chat(messages, max_new_tokens):
                self.assertFalse(is_judge, "语义裁判应走纯文本接口。")
                contents = json.dumps(messages, ensure_ascii=False)
                match = next((question for question in self.responses if question in contents), None)
                self.assertIsNotNone(match, contents)
                events.append(("answer", name, match))
                return prediction_override(match) if prediction_override else self.responses[match]

            def text(prompt, max_new_tokens=256):
                self.assertTrue(is_judge, "生成模型不得兼任自己的语义裁判。")
                self.assertEqual(len(generated), len(self.entries))
                prompts.append(prompt)
                events.append(("judge", name))
                if bad_judge and "Open question number 1" in prompt:
                    return "unparseable judge output"
                return '{"correctness": 1, "reason": "参考答案与主要结论一致"}'

            def close():
                if name in active:
                    active.remove(name)
                    events.append(("close_judge" if is_judge else "close_generator", name))

            return SimpleNamespace(
                args=args, supports_images=False if is_judge else args.model != "DeepSeek-Model",
                chat=chat, text=text, close=close,
                last_generation_info={"hit_token_limit": False, "generated_tokens": 3},
            )

        def close_handlers(**kwargs):
            # 不把临时目录日志句柄留给 Windows 的删除操作。
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
            code = evaluation.main([
                "--config", str(self.config), "--output_dir", str(self.output), *extra,
            ])
        self.assertFalse(active, "结束后必须卸载裁判和所有生成模型。")
        return code, events, prompts, stdout.getvalue(), stderr.getvalue()

    def rows(self, name):
        path = self.output / name / "no_rag" / "predictions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def summary(self, name):
        return read_json(self.output / name / "no_rag" / "summary.json")

    def test_all_generators_finish_before_one_fixed_judge_and_repeated_answers_share_cache(self):
        code, events, prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(sum(event[0] == "create_generator" for event in events), 4)
        self.assertEqual(sum(event[0] == "create_judge" for event in events), 1)
        self.assertEqual(sum(event[0] == "answer" for event in events), 37 * 4)
        self.assertEqual(len(prompts), 2, "4组相同开放题回答只需评分一次。")
        self.assertTrue(all("Objective question" not in prompt for prompt in prompts))
        self.assertTrue(all("visual_evidence" not in prompt for prompt in prompts))
        self.assertTrue(all("annotation only" not in prompt for prompt in prompts))
        self.assertEqual(events[-1][0], "close_judge")
        for index, entry in enumerate(self.entries):
            name = entry["name"]
            summary, rows = self.summary(name), self.rows(name)
            self.assertEqual(summary["judge_required"], 2)
            self.assertEqual(summary["judge_failures"], 0)
            self.assertEqual(summary["judge_pending"], 0)
            self.assertEqual(summary["metrics"]["answer_accuracy"]["n"], 35)
            self.assertEqual(summary["metrics"]["answer_accuracy"]["mean"], 1)
            self.assertEqual(summary["metrics"]["semantic_score"]["n"], 2)
            self.assertEqual(summary["metrics"]["semantic_score"]["coverage"], 1)
            self.assertEqual(summary["metrics"]["semantic_score"]["mean"], 1)
            self.assertEqual(summary["scoring_status"], "complete")
            self.assertEqual(read_json(self.output / name / "no_rag" / "review_needed.json"), [])
            self.assertTrue(all(row["semantic_score"] is None for row in rows[:35]))
            self.assertTrue(all(row["semantic_details"]["from_cache"] is (index > 0) for row in rows[35:]))
        suite = read_json(self.output / "suite_comparison.json")
        for delta in suite["paired_deltas"]:
            self.assertEqual(delta["metrics"]["answer_accuracy"]["total"], 35)
            self.assertEqual(delta["metrics"]["answer_accuracy"]["coverage"], 1)
            self.assertEqual(delta["metrics"]["semantic_score"]["total"], 2)
            self.assertEqual(delta["metrics"]["semantic_score"]["coverage"], 1)

    def test_judge_failures_remain_null_reduce_coverage_and_require_review(self):
        code, _, prompts, _, errors = self.run_main(bad_judge=True)
        self.assertEqual(code, 2, errors)
        self.assertGreater(len(prompts), 2)
        for entry in self.entries:
            name = entry["name"]
            summary, rows = self.summary(name), self.rows(name)
            metric = summary["metrics"]["semantic_score"]
            self.assertEqual(metric["mean"], 1, "无法解析的评分不得作为0拉低均值。")
            self.assertEqual(metric["n"], 1)
            self.assertEqual(metric["total"], 2)
            self.assertEqual(metric["coverage"], 0.5)
            self.assertEqual(summary["judge_failures"], 1)
            self.assertEqual(summary["scoring_status"], "needs_review")
            self.assertIsNone(rows[-1]["semantic_score"])
            self.assertEqual(rows[-1]["semantic_details"]["status"], "judge_failed")
            review = read_json(self.output / name / "no_rag" / "review_needed.json")
            self.assertEqual([row["id"] for row in review], [self.samples[-1]["id"]])

    def test_scope_none_never_creates_a_judge(self):
        self.write_config(scope="none")
        code, events, prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(prompts, [])
        self.assertFalse(any(event[0] == "create_judge" for event in events))
        for entry in self.entries:
            summary = self.summary(entry["name"])
            self.assertEqual(summary["judge_required"], 0)
            self.assertEqual(summary["metrics"]["semantic_score"]["n"], 0)
            self.assertIsNone(summary["metrics"]["semantic_score"]["mean"])
            self.assertEqual(summary["metrics"]["answer_accuracy"]["mean"], 1)

    def test_scope_all_scores_every_question_with_same_fixed_judge(self):
        self.write_config(scope="all")
        code, events, prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(sum(event[0] == "create_judge" for event in events), 1)
        self.assertEqual(len(prompts), 37)
        for entry in self.entries:
            summary = self.summary(entry["name"])
            self.assertEqual(summary["judge_required"], 37)
            self.assertEqual(summary["metrics"]["semantic_score"]["n"], 37)

    def test_different_judge_settings_rejected_before_generation(self):
        self.entries[-1]["evaluation"]["judgeScope"] = "all"
        self.write_config()
        code, events, _, _, errors = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(events, [])
        self.assertIn("evaluation", errors)
        self.assertFalse(self.output.exists())

    def test_all_scope_preserves_medical_explanations_for_judge(self):
        sample = self.samples[0]
        sample["answer"] = {"correct_option": "A", "explanation": "Reference explanation for diagnosis"}
        sample["reference"] = json.dumps(sample["answer"])
        self.responses[sample["question"]] = json.dumps({
            "correct_option": "A", "explanation": "Candidate explanation for diagnosis",
            "visual_evidence": "PRIVATE_IMAGE_ANNOTATION",
        })
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.samples), encoding="utf-8")
        self.write_config(scope="all")
        code, _, prompts, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        prompt = next(text for text in prompts if sample["question"] in text)
        self.assertIn("Reference explanation for diagnosis", prompt)
        self.assertIn("Candidate explanation for diagnosis", prompt)
        self.assertNotIn("PRIVATE_IMAGE_ANNOTATION", prompt)
        self.assertEqual(self.rows("Qwen")[0]["answer_accuracy"], 1)

    def test_unparseable_objective_predictions_are_zero_and_in_review(self):
        self.write_config(scope="none")

        def changed_answer(question):
            return "A or B" if question == self.samples[0]["question"] else self.responses[question]

        code, _, _, _, errors = self.run_main(prediction_override=changed_answer)
        self.assertEqual(code, 0, errors)
        for entry in self.entries:
            name = entry["name"]
            metric = self.summary(name)["metrics"]["answer_accuracy"]
            self.assertEqual(metric["n"], 35)
            self.assertAlmostEqual(metric["mean"], 34 / 35)
            self.assertEqual(self.rows(name)[0]["answer_accuracy"], 0)
            review = read_json(self.output / name / "no_rag" / "review_needed.json")
            self.assertEqual([row["id"] for row in review], [self.samples[0]["id"]])

    def test_invalid_objective_reference_reduces_coverage_and_is_visible(self):
        self.samples[0]["answer"] = self.samples[0]["reference"] = "A or B"
        self.dataset.write_text("\n".join(json.dumps(row) for row in self.samples), encoding="utf-8")
        self.write_config(scope="none")
        code, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        summary = self.summary("Qwen")
        self.assertEqual(summary["invalid_references"], 1)
        metric = summary["metrics"]["answer_accuracy"]
        self.assertEqual(metric["total"], 35)
        self.assertEqual(metric["n"], 34)
        self.assertAlmostEqual(metric["coverage"], 34 / 35)
        self.assertEqual(metric["mean"], 1)
        self.assertIsNone(self.rows("Qwen")[0]["answer_accuracy"])
        review = read_json(self.output / "Qwen" / "no_rag" / "review_needed.json")
        self.assertEqual([row["id"] for row in review], [self.samples[0]["id"]])

    def test_different_chroma_sources_prevent_claiming_lora_only_effect(self):
        self.write_config(scope="none")
        code, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        suite = read_json(self.output / "suite_comparison.json")
        # 在真实已生成的汇总上模拟两组使用了不同向量库；无需加载嵌入或生成模型。
        for entry in self.entries:
            path = self.output / entry["name"] / "comparison.json"
            comparison = read_json(path)
            for summary in comparison["summaries"]:
                summary.update(use_rag=True, top_k=5, chroma_collection="same_collection",
                               chroma_id_prefix="mira:",
                               chroma_db_dir="source_B" if entry["pathLora"] else "source_A")
            path.write_text(json.dumps(comparison), encoding="utf-8")
        result = evaluation.write_suite_comparison(self.output, suite["runs"], status="complete", elapsed=1)
        self.assertEqual(result["paired_deltas"], [])


if __name__ == "__main__":
    unittest.main()
