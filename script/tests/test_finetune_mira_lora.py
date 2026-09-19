"""统一 MIRA LoRA 入口的离线测试，不加载模型权重。"""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from script import finetune_mira_lora as unified


class UnifiedFinetuneTests(unittest.TestCase):
    def write_config(self, payload):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "finetune_mira_lora.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_default_file_selects_deepseek_backend(self):
        payload = unified.load_config(unified.DEFAULT_CONFIG_PATH)
        self.assertEqual(payload["model"], "DeepSeek-Model")
        backend, argv = unified.resolve_backend_argv(payload, [])
        self.assertEqual(backend.__name__.rsplit(".", 1)[-1], "finetune_deepseek_mira_lora")
        args = backend.build_parser().parse_args(argv)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.device, "auto")

    def test_qwen_selection_uses_original_backend_and_defaults(self):
        payload = {"model": "Qwen3-VL-2B-Instruct", "model_arguments": {}}
        backend, argv = unified.resolve_backend_argv(payload, [])
        self.assertEqual(backend.__name__.rsplit(".", 1)[-1], "finetune_qwen3_vl_lora")
        unified_args = backend.build_parser().parse_args(argv)
        direct_args = backend.build_parser().parse_args([])
        self.assertEqual(vars(unified_args), vars(direct_args))

    def test_both_default_profiles_and_backends_use_project_output(self):
        payload = unified.load_config(unified.DEFAULT_CONFIG_PATH)
        for model, adapter_name in (
            ("DeepSeek-Model", "deepseek_mira_lora_adapter"),
            ("Qwen3-VL-2B-Instruct", "qwen3_vl_2b_lora_adapter"),
        ):
            with self.subTest(model=model):
                backend, argv = unified.resolve_backend_argv(dict(payload, model=model), [])
                for args in (backend.build_parser().parse_args(argv), backend.build_parser().parse_args([])):
                    self.assertEqual(args.split_manifest, unified.SCRIPT_DIR.parent / "output" / "mira_split_ids.json")
                    self.assertEqual(args.output_dir, unified.SCRIPT_DIR.parent / "output" / adapter_name)

    def test_json_paths_use_config_directory_and_cli_paths_remain_explicit(self):
        payload = {
            "model": "Qwen3-VL-2B-Instruct",
            "model_arguments": {"Qwen3-VL-2B-Instruct": {
                "split_manifest": "../output/custom_ids.json",
                "output_dir": "../output/custom_adapter",
                "resume_from_checkpoint": "../output/custom_adapter/checkpoint-4",
            }},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            backend, argv = unified.resolve_backend_argv(payload, [], root / "config")
            args = backend.build_parser().parse_args(argv)
            self.assertEqual(args.split_manifest, root / "output" / "custom_ids.json")
            self.assertEqual(args.output_dir, root / "output" / "custom_adapter")
            self.assertEqual(args.resume_from_checkpoint, str(root / "output" / "custom_adapter" / "checkpoint-4"))
            _, argv = unified.resolve_backend_argv(
                payload, ["--output-dir", "explicit_cli_adapter"], root / "config",
            )
            self.assertEqual(backend.build_parser().parse_args(argv).output_dir, Path("explicit_cli_adapter"))
        self.assertEqual(payload["model_arguments"][payload["model"]]["output_dir"], "../output/custom_adapter")

    def test_json_values_and_cli_overrides_follow_backend_argparse(self):
        payload = {
            "model": "DeepSeek-Model",
            "model_arguments": {"DeepSeek-Model": {
                "epochs": 1, "gradient_checkpointing": False, "limit": 8,
            }},
        }
        backend, argv = unified.resolve_backend_argv(payload, ["--epochs", "2", "--limit", "3"])
        args = backend.build_parser().parse_args(argv)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.limit, 3)
        self.assertFalse(args.gradient_checkpointing)

    def test_main_delegates_exact_resolved_argv_to_original_main(self):
        path = self.write_config({
            "model": "DeepSeek-Model",
            "model_arguments": {"DeepSeek-Model": {"dry_run": True, "limit": 4}},
        })
        backend = mock.Mock()
        parser = argparse.ArgumentParser()
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--epochs", type=float, default=1)
        backend.build_parser.return_value = parser
        backend.main.return_value = 0
        with mock.patch.object(unified, "import_backend", return_value=backend):
            result = unified.main(["--config", str(path), "--", "--epochs", "2"])
        self.assertEqual(result, 0)
        backend.main.assert_called_once_with(["--dry-run", "--limit", "4", "--epochs", "2"])

    def test_invalid_models_profiles_and_arguments_are_rejected(self):
        bad_payloads = [
            {},
            {"model": "unknown"},
            {"model": "DeepSeek-Model", "extra": 1},
            {"model": "DeepSeek-Model", "model_arguments": []},
            {"model": "DeepSeek-Model", "model_arguments": {"unknown": {}}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                path = self.write_config(payload)
                with self.assertRaises(ValueError):
                    unified.load_config(path)
        with self.assertRaisesRegex(ValueError, "不支持"):
            unified.resolve_backend_argv(
                {"model": "DeepSeek-Model", "model_arguments": {"DeepSeek-Model": {"max_pixels": 1}}}, [])


if __name__ == "__main__":
    unittest.main()
