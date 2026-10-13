"""只推理模式的最小回归测试：不加载裁判，输出可供后续评分的生成文件。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import inference_rag as inference


class InferenceSkipEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="inference-only-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset.jsonl"
        self.config = self.root / "config.json"
        self.output = self.root / "output"
        records = [{
            "id": "case-1", "question": "What is the diagnosis?", "reference": "Atrial fibrillation.",
            "answer": {"text": "Atrial fibrillation."}, "question_type": "open_ended", "images": [],
        }]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        self.config.write_text(json.dumps({
            "name": "inference_only", "skipEvaluation": True, "model": "Qwen3-VL-2B-Instruct",
            "useRAG": False, "datasetPath": str(self.dataset), "datasetFilter": None,
            "outputDir": str(self.output), "evaluation": {
                "prima": {"enabled": True, "factScore": True, "separateQuestionTypes": True,
                           "questionTypes": ["open_ended"]},
            },
        }), encoding="utf-8")

    def test_skip_evaluation_writes_generations_without_judge(self):
        class FakeGenerator:
            args = SimpleNamespace()
            supports_images = False
            last_generation_info = {"hit_token_limit": False, "output_tokens": 3}

            def __init__(self):
                self.args = SimpleNamespace(
                    evaluation={"prima": {"enabled": True, "announceQuestionType": True,
                                           "explanationRougeL": True}},
                    reference_length_budget={"enabled": False}, max_new_tokens=32,
                )

            def chat(self, messages, max_new_tokens):
                return "Answer: Atrial fibrillation."

            def close(self):
                return None

        # 替换模型边界和报告依赖；裁判相关入口若被调用应让测试失败。
        with patch.object(inference, "validate_model_choice"), \
                patch.object(inference, "validate_lora_path"), \
                patch.object(inference, "create_generator", return_value=FakeGenerator()), \
                patch.object(inference, "prepare_retrieval", return_value={"case-1": []}), \
                patch.object(inference, "score_answer", side_effect=AssertionError("skip模式不可解析/评分")), \
                patch.object(inference, "compute_bleu4", side_effect=AssertionError("skip模式不可算指标")), \
                patch.object(inference, "compute_rouge_l", side_effect=AssertionError("skip模式不可算指标")), \
                patch.object(inference, "run_semantic_scoring", side_effect=AssertionError("不应启动裁判")), \
                patch.dict(inference.sys.modules, {"rag_reports": None}):
            code = inference.main(["--config", str(self.config), "--output_dir", str(self.output)])
        self.assertEqual(code, 0)
        mode = self.output / "inference_only"
        generation_path = mode / "no_rag" / "generations.jsonl"
        self.assertTrue(generation_path.is_file())
        row = json.loads(generation_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["id"], "case-1")
        self.assertEqual(row["prediction"], "Answer: Atrial fibrillation.")
        prediction = json.loads((mode / "no_rag" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(prediction["metrics_skipped"])
        self.assertIsNone(prediction["answer_accuracy"])
        summary = json.loads((mode / "no_rag" / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["skipped"])
        self.assertIsNone(summary["metrics"])
        comparison = json.loads((mode / "comparison.json").read_text(encoding="utf-8"))
        self.assertTrue(comparison["skipped"])
        self.assertEqual(comparison["status"], "inference_only")

    def test_skip_evaluation_with_dynamic_budget_does_not_score(self):
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["generation"] = {"maxNewTokens": 32, "maxInputTokens": 128,
                                 "referenceLengthBudget": {
                                     "enabled": True, "multiplier": 2.0, "extraTokens": 4,
                                     "minNewTokens": 8, "maxNewTokens": 32,
                                     "retryOnTruncation": False}}
        self.config.write_text(json.dumps(config), encoding="utf-8")

        class FakeGenerator:
            supports_images = False
            last_generation_info = {"hit_token_limit": False, "output_tokens": 3}

            def __init__(self):
                self.args = SimpleNamespace(
                    evaluation={"prima": {"enabled": True, "announceQuestionType": True,
                                           "explanationRougeL": True}},
                    reference_length_budget=config["generation"]["referenceLengthBudget"], max_new_tokens=32,
                )

            def count_tokens(self, text):
                return 5

            def chat(self, messages, max_new_tokens):
                return "Answer: Atrial fibrillation."

            def close(self):
                return None

        with patch.object(inference, "validate_model_choice"), \
                patch.object(inference, "validate_lora_path"), \
                patch.object(inference, "create_generator", return_value=FakeGenerator()), \
                patch.object(inference, "prepare_retrieval", return_value={"case-1": []}), \
                patch.object(inference, "score_answer", side_effect=AssertionError("动态预算只推理时不可评分")), \
                patch.object(inference, "run_semantic_scoring", side_effect=AssertionError("不应启动裁判")):
            code = inference.main(["--config", str(self.config), "--output_dir", str(self.output)])
        self.assertEqual(code, 0)
        row = json.loads((self.output / "inference_only" / "no_rag" / "generations.jsonl").read_text(
            encoding="utf-8").splitlines()[0])
        self.assertEqual(row["prediction"], "Answer: Atrial fibrillation.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
