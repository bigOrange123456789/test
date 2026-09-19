"""批量对比原版/LoRA 模型的离线回归测试；不加载真实模型。"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import evaluate_rag as evaluation


class EvaluationSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evaluation-suite-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "results"
        self.config_path = self.root / "evaluate_rag.json"
        self.filter_path = self.root / "split.json"
        self.dataset_path = self.root / "dataset.jsonl"
        self.ids = ["sample-2", "sample-0"]
        self.filter_path.write_text(json.dumps({"test_ids": self.ids, "train_ids": ["sample-1"]}),
                                    encoding="utf-8")
        rows = [{"id": f"sample-{index}", "question": f"Patient {index}: likely diagnosis?",
                 "reference": "This is a possible myocardial infarction.", "images": []}
                for index in range(3)]
        self.dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.models = {}
        self.adapters = {}
        for name, model_type in (("Qwen3-VL-2B-Instruct", "qwen3_vl"), ("DeepSeek-Model", "qwen2")):
            base = self.root / name
            base.mkdir()
            (base / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
            adapter = self.root / (name + "_adapter")
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": str(base),
            }), encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"fake adapter weights for offline tests")
            self.models[name], self.adapters[name] = base, adapter
        self.entries = [
            self.entry("Qwen", "Qwen3-VL-2B-Instruct"),
            self.entry("Qwen_lora", "Qwen3-VL-2B-Instruct", lora=True),
            self.entry("Deepseek", "DeepSeek-Model"),
            self.entry("Deepseek_lora", "DeepSeek-Model", lora=True),
        ]
        self.write_config(self.entries)

    def entry(self, name, model, lora=False):
        return {"name": name, "model": model, "useRAG": False,
                "datasetPath": str(self.dataset_path), "datasetFilter": str(self.filter_path),
                "chromaPath": None, "pathLora": str(self.adapters[model]) if lora else None}

    def write_config(self, value):
        self.config_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")

    def resolve(self, *extra):
        with patch.object(evaluation, "MODEL_DIRECTORIES", self.models):
            return evaluation.resolve_run_configs(evaluation.build_parser().parse_args([
                "--config", str(self.config_path), "--output_dir", str(self.output), *extra,
            ]))

    def run_main(self, *extra):
        events = []
        active = set()
        messages = []

        def factory(args):
            self.assertFalse(active, "下一组模型加载前必须先释放上一组模型")
            name = args.run_name
            active.add(name)
            events.append(("create", name, args.model, args.lora_path))

            def chat(prompt, token_budget):
                self.assertIn(name, active)
                messages.append((name, prompt))
                return "This is a possible myocardial infarction."

            def close():
                active.discard(name)
                events.append(("close", name))

            return SimpleNamespace(args=args, supports_images=args.model != "DeepSeek-Model",
                                   chat=chat, close=close)

        def close_handlers(**kwargs):
            # Windows 临时目录删除前关闭测试创建的日志文件句柄。
            for handler in kwargs.get("handlers", []):
                handler.close()

        stdout, stderr = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(evaluation, "MODEL_DIRECTORIES", self.models))
            stack.enter_context(patch.object(evaluation, "create_generator", side_effect=factory))
            stack.enter_context(patch.object(evaluation, "_model_identity", return_value={"path": "mock-base"}))
            stack.enter_context(patch.object(evaluation.logging, "basicConfig", side_effect=close_handlers))
            stack.enter_context(patch.dict(sys.modules, {"rag_reports": None}))
            stack.enter_context(patch.dict(os.environ))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            code = evaluation.main([
                "--config", str(self.config_path), "--output_dir", str(self.output),
                "--factscore_method", "keyword", *extra,
            ])
        self.assertFalse(active, "评估结束必须释放最后一个模型")
        return code, events, messages, stdout.getvalue(), stderr.getvalue()

    def test_four_jobs_keep_config_order_default_full_test_and_separate_output(self):
        jobs = self.resolve()
        self.assertEqual([job.run_name for job in jobs], [entry["name"] for entry in self.entries])
        self.assertEqual([job.lora_path for job in jobs], [entry["pathLora"] for entry in self.entries])
        for job in jobs:
            self.assertIsNone(job.N)
            self.assertEqual(Path(job.output_dir), self.output / job.run_name)
            self.assertEqual(Path(job.dataset_filter), self.filter_path)
            self.assertEqual(Path(job.suite_output_dir), self.output)

    def test_relative_dataset_filter_and_lora_paths_resolve_from_config_directory(self):
        entry = copy.deepcopy(self.entries[1])
        for field in ("datasetPath", "datasetFilter", "pathLora"):
            entry[field] = Path(entry[field]).name
        self.write_config([entry])
        job = self.resolve()[0]
        self.assertEqual(Path(job.dataset_path), self.dataset_path)
        self.assertEqual(Path(job.dataset_filter), self.filter_path)
        self.assertEqual(Path(job.lora_path), self.adapters[entry["model"]])

    def test_duplicate_or_unsafe_names_are_rejected(self):
        for name in ("../escape", "a/b", "a\\b", "CON", "LPT1", " bad", "", "a."):
            with self.subTest(name=name):
                entry = dict(self.entries[0], name=name)
                self.write_config([entry])
                with self.assertRaises(ValueError):
                    self.resolve()
        self.write_config([self.entries[0], dict(self.entries[1], name="qwen")])
        with self.assertRaises(ValueError):
            self.resolve()

    def test_null_values_are_supported_but_invalid_path_types_are_rejected(self):
        self.write_config([dict(self.entries[0], pathLora=None, datasetFilter=None)])
        job = self.resolve()[0]
        self.assertIsNone(job.lora_path)
        self.assertIsNone(job.dataset_filter)
        for field in ("pathLora", "datasetFilter"):
            for value in (False, 0, [], ""):
                with self.subTest(field=field, value=value):
                    self.write_config([dict(self.entries[0], **{field: value})])
                    with self.assertRaises(ValueError):
                        self.resolve()

    def test_four_evaluations_load_sequentially_keep_test_ids_and_write_independent_results(self):
        code, events, messages, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual([event[1] for event in events if event[0] == "create"],
                         [entry["name"] for entry in self.entries])
        self.assertEqual([event[0] for event in events], ["create", "close"] * 4)
        self.assertEqual(len(messages), 8)
        fingerprints = []
        for entry in self.entries:
            directory = self.output / entry["name"]
            self.assertEqual(json.loads((directory / "test_ids.json").read_text(encoding="utf-8")), self.ids)
            predictions = [json.loads(line) for line in
                           (directory / "no_rag" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in predictions], self.ids)
            self.assertNotIn("sample-1", [row["id"] for row in predictions])
            self.assertFalse(any(row["generation_from_cache"] for row in predictions))
            summary = json.loads((directory / "no_rag" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["model"], entry["model"])
            self.assertEqual(summary["run_name"], entry["name"])
            self.assertEqual(summary["lora_path"], entry["pathLora"])
            self.assertEqual(summary["num_samples"], 2)
            self.assertEqual(set(summary["metrics"]), set(evaluation.METRICS))
            manifest = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            fingerprints.append(predictions[0]["generation_fingerprint"])
        self.assertEqual(len(set(fingerprints)), 4)
        self.assertTrue((self.output / "suite_comparison.json").is_file())
        self.assertTrue((self.output / "suite_comparison.md").is_file())
        comparison = json.loads((self.output / "suite_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual([row["name"] for row in comparison["runs"]],
                         [entry["name"] for entry in self.entries])
        self.assertEqual([row["name"] for row in comparison["summaries"]],
                         [entry["name"] for entry in self.entries])
        self.assertTrue(all(row["status"] == "complete" for row in comparison["runs"]))

    def test_base_cache_does_not_supply_lora_predictions_even_with_same_name(self):
        entry = dict(self.entries[0])
        self.write_config([entry])
        code, _, messages, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(messages), 2)
        entry["pathLora"] = str(self.adapters[entry["model"]])
        self.write_config([entry])
        code, _, messages, _, errors = self.run_main("--resume")
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(messages), 2, "原版缓存不得直接填充 LoRA 版本的答案")
        predictions = [json.loads(line) for line in
                       (self.output / entry["name"] / "no_rag" / "predictions.jsonl")
                       .read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(not row["generation_from_cache"] for row in predictions))

    def test_resume_uses_changed_filter_instead_of_overriding_it_with_old_test_ids(self):
        self.write_config([self.entries[0]])
        code, _, _, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        old_ids = self.output / self.entries[0]["name"] / "test_ids.json"
        self.assertEqual(json.loads(old_ids.read_text(encoding="utf-8")), self.ids)
        self.filter_path.write_text(json.dumps({"test_ids": ["sample-0"], "train_ids": ["sample-1"]}),
                                    encoding="utf-8")
        code, events, messages, _, errors = self.run_main("--resume")
        self.assertEqual(code, 0, errors)
        self.assertEqual([event[0] for event in events], ["create", "close"])
        self.assertEqual(len(messages), 1)
        self.assertEqual(json.loads(old_ids.read_text(encoding="utf-8")), ["sample-0"])
        predictions = [json.loads(line) for line in
                       (old_ids.parent / "no_rag" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in predictions], ["sample-0"])
        self.assertFalse(predictions[0]["generation_from_cache"])

    def test_check_data_verifies_all_jobs_without_model_or_output(self):
        code, events, messages, output, errors = self.run_main("--check_data")
        self.assertEqual(code, 0, errors)
        self.assertEqual(events, [])
        self.assertEqual(messages, [])
        self.assertFalse(self.output.exists())
        for entry in self.entries:
            self.assertIn(entry["name"], output)

    def test_resume_reuses_matching_answers_without_new_generation(self):
        self.write_config([self.entries[1]])
        code, _, messages, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(messages), 2)
        code, _, messages, _, errors = self.run_main("--resume")
        self.assertEqual(code, 0, errors)
        self.assertEqual(messages, [])
        predictions = [json.loads(line) for line in
                       (self.output / self.entries[1]["name"] / "no_rag" / "predictions.jsonl")
                       .read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in predictions], self.ids)
        self.assertTrue(all(row["generation_from_cache"] for row in predictions))

    def test_invalid_later_adapter_fails_before_any_model_or_output(self):
        entries = copy.deepcopy(self.entries)
        entries[-1]["pathLora"] = str(self.root / "missing_adapter")
        self.write_config(entries)
        code, events, messages, _, errors = self.run_main()
        self.assertNotEqual(code, 0, errors)
        self.assertEqual(events, [])
        self.assertEqual(messages, [])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
