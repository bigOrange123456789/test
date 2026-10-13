"""针对仅评分入口的轻量端到端测试，不加载任何真实模型。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import evaluate_rag_metrics as metrics


class MetricsFromGenerationsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="metrics-from-generations-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "metrics"
        self.base_path = self.root / "base" / "generations.jsonl"
        self.lora_path = self.root / "lora" / "generations.jsonl"

        # 只包含客观题，因此无需实例化裁判模型；两份结果具有相同 ID 集合。
        base = [
            self.record("case-1", "single_choice", {"correct_option": "A"}, "Answer: A", "base"),
            self.record("case-2", "multiple_choice", {"correct_options": ["A", "C"]}, "Answer: A, C", "base"),
        ]
        lora = [
            self.record("case-1", "single_choice", {"correct_option": "A"}, "Answer: B", "lora"),
            self.record("case-2", "multiple_choice", {"correct_options": ["A", "C"]}, "Answer: A, C", "lora"),
        ]
        self.write_records(self.base_path, base)
        self.write_records(self.lora_path, lora)

    @staticmethod
    def record(identifier, question_type, answer, prediction, config_name):
        return {
            "id": identifier,
            "question": "Choose the correct answer.",
            "question_type": question_type,
            "options": {"A": "Option A", "B": "Option B", "C": "Option C"},
            "answer": answer,
            "reference": json.dumps(answer),
            "prediction": prediction,
            "images": [],
            "config_name": config_name,
            "generation_fingerprint": f"fingerprint-{config_name}",
            "retrieved_evidence": [],
            "generation_info": {"elapsed_seconds": 0.01},
        }

    @staticmethod
    def write_records(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_metrics_from_generations_and_paired_delta(self):
        # 该 fixture 是纯客观题，若实现意外加载裁判，该断言会立即发现。
        with patch.object(metrics, "_TextJudgeModel", side_effect=AssertionError("不应加载裁判")):
            code = metrics.main([
                "--generations", str(self.base_path), str(self.lora_path),
                "--output_dir", str(self.output),
            ])
        self.assertEqual(code, 0)

        suite = self.read_json(self.output / "suite_comparison.json")
        self.assertEqual({item["name"] for item in suite["summaries"]}, {"base", "lora"})
        self.assertEqual(len(suite["paired_deltas"]), 1)
        delta = suite["paired_deltas"][0]
        self.assertEqual((delta["name"], delta["baseline"], delta["num_samples"]), ("lora", "base", 2))
        self.assertAlmostEqual(delta["metrics"]["answer_accuracy"]["mean"], -0.5)

        base_summary = self.read_json(self.output / "base" / "summary.json")
        lora_summary = self.read_json(self.output / "lora" / "summary.json")
        self.assertEqual(base_summary["metrics"]["answer_accuracy"]["mean"], 1.0)
        self.assertEqual(lora_summary["metrics"]["answer_accuracy"]["mean"], 0.5)
        self.assertTrue((self.output / "suite_comparison.md").is_file())

    def test_mismatched_ids_do_not_create_paired_delta(self):
        altered = [self.record("different-id", "single_choice", {"correct_option": "A"}, "Answer: A", "lora")]
        self.write_records(self.lora_path, altered)
        code = metrics.main([
            "--generations", str(self.base_path), str(self.lora_path),
            "--output_dir", str(self.output),
        ])
        self.assertEqual(code, 0)
        suite = self.read_json(self.output / "suite_comparison.json")
        self.assertEqual(suite["paired_deltas"], [])

    def test_windows_device_labels_and_traversal_are_rejected(self):
        for label in (".", "..", "CON", "CON.txt", "LPT1.log", "name."):
            with self.subTest(label=label), self.assertRaises(ValueError):
                metrics._safe_label(label)

    def test_image_paths_participate_in_test_fingerprint(self):
        records = [self.record("same-id", "single_choice", {"correct_option": "A"}, "A", "base")]
        first = metrics._summary("first", self.base_path, records, [
            {**records[0], "generation_info": {}, "config_name": "base"}],
            metrics._normalize_evaluation({}, self.root), 42, 0.0)
        records_with_other_image = [{**records[0], "images": ["other/image.png"]}]
        second = metrics._summary("second", self.base_path, records_with_other_image, [
            {**records_with_other_image[0], "generation_info": {}, "config_name": "base"}],
            metrics._normalize_evaluation({}, self.root), 42, 0.0)
        self.assertNotEqual(first["test_fingerprint"], second["test_fingerprint"])

    def test_incompatible_generation_settings_do_not_pair(self):
        base_sidecar = {"model": "Qwen3-VL-2B-Instruct", "model_path": "model",
                        "lora_path": None, "use_rag": False, "top_k": 0,
                        "dataset_fingerprint": "dataset", "generation_settings": {"max_new_tokens": 128}}
        lora_sidecar = {**base_sidecar, "lora_path": "adapter",
                        "generation_settings": {"max_new_tokens": 256}}
        evaluation = metrics._normalize_evaluation({}, self.root)
        base = metrics._summary("base", self.base_path, [], [
            {"config_name": "base", "generation_info": {}}], evaluation, 42, 0.0, base_sidecar)
        lora = metrics._summary("lora", self.lora_path, [], [
            {"config_name": "lora", "generation_info": {}}], evaluation, 42, 0.0, lora_sidecar)
        # 这些必需元数据不一致时，即便 ID 完全一致也不能做逐题配对比较。
        for summary in (base, lora):
            summary.update(num_samples=1, metrics={}, by_question_type={}, judge_failures=0)
        row = {"id": "same-id", "answer_details": {"question_type": "single_choice"},
               "generation_info": {}}
        result = metrics._write_reports(self.output, [base, lora], {"base": [row], "lora": [row]}, evaluation, 42)
        self.assertEqual(result["paired_deltas"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
