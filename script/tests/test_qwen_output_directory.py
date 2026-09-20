"""Qwen 重复微调的输出目录保护测试；不加载模型或训练数据。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script import finetune_mira_lora as unified
from script.lib import finetune_qwen3_vl_lora as finetune


class QwenOutputDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-output-policy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "models" / "base"
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
        }), encoding="utf-8")
        self.output = self.root / "output" / "adapter"
        self.manifest = self.root / "split.json"

    def arguments(self, *extra):
        return finetune.build_parser().parse_args([
            "--model-dir", str(self.model),
            "--output-dir", str(self.output),
            "--split-manifest", str(self.manifest),
            *extra,
        ])

    def previous_adapter(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "adapter_model.safetensors"
        sentinel.write_bytes(b"previous adapter must remain unchanged")
        return sentinel

    def checkpoint(self, root=None):
        checkpoint = (root or self.output) / "checkpoint-63"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_config.json", "trainer_state.json"):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        return checkpoint

    def test_parser_defaults_to_preserving_existing_results_in_a_new_run(self):
        self.assertEqual(self.arguments().on_existing_output, "new")

    def test_new_output_path_remains_unchanged_without_creating_it(self):
        args = self.arguments()
        console = io.StringIO()
        with redirect_stdout(console):
            selected = finetune.select_output_directory(args)
        self.assertEqual(selected, self.output)
        self.assertEqual(args.output_dir, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(console.getvalue(), "")

    def test_existing_empty_directory_can_be_used_without_renaming(self):
        self.output.mkdir(parents=True)
        args = self.arguments()
        console = io.StringIO()
        with redirect_stdout(console):
            selected = finetune.select_output_directory(args)
        self.assertEqual(selected, self.output)
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(console.getvalue(), "")

    def test_existing_adapter_selects_sibling_run_and_preserves_old_files(self):
        sentinel = self.previous_adapter()
        before = sentinel.read_bytes()
        args = self.arguments()
        console = io.StringIO()
        with redirect_stdout(console), patch.object(finetune.time, "strftime", return_value="20260920_120000"):
            selected = finetune.select_output_directory(args)
        self.assertEqual(selected, self.output.with_name("adapter_run_20260920_120000"))
        self.assertEqual(args.output_dir, self.output)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertFalse(selected.exists())
        self.assertIn(str(selected), console.getvalue())

    def test_same_second_collisions_skip_existing_directories_and_files(self):
        sentinel = self.previous_adapter()
        candidate = self.output.with_name("adapter_run_20260920_120000")
        candidate.mkdir()
        occupied_file = candidate.with_name(candidate.name + "_2")
        occupied_file.write_bytes(b"not a directory")
        args = self.arguments()
        with patch.object(finetune.time, "strftime", return_value="20260920_120000"):
            selected = finetune.select_output_directory(args)
        self.assertEqual(selected, candidate.with_name(candidate.name + "_3"))
        self.assertFalse(selected.exists())
        self.assertEqual(list(candidate.iterdir()), [])
        self.assertEqual(occupied_file.read_bytes(), b"not a directory")
        self.assertTrue(sentinel.is_file())

    def test_error_policy_still_rejects_nonempty_output(self):
        sentinel = self.previous_adapter()
        args = self.arguments("--on-existing-output", "error")
        with self.assertRaisesRegex(ValueError, "not empty"):
            finetune.select_output_directory(args)
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_explicit_allow_existing_remains_compatible_with_error_policy(self):
        self.previous_adapter()
        args = self.arguments("--on-existing-output", "error", "--allow-existing-output")
        self.assertEqual(finetune.select_output_directory(args), self.output)

    def test_resume_selection_does_not_create_an_unrelated_run(self):
        checkpoint = self.checkpoint()
        args = self.arguments("--resume-from-checkpoint", str(checkpoint))
        self.assertEqual(finetune.select_output_directory(args), self.output)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_validate_assigns_selected_path_without_writing_any_run_files(self):
        sentinel = self.previous_adapter()
        args = self.arguments()
        with patch.object(finetune.time, "strftime", return_value="20260920_120000"):
            finetune.validate_args(args)
        self.assertEqual(args.output_dir, self.output.with_name("adapter_run_20260920_120000"))
        self.assertFalse(args.output_dir.exists())
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_valid_resume_keeps_checkpoint_parent_and_allows_existing_output(self):
        checkpoint = self.checkpoint()
        args = self.arguments("--resume-from-checkpoint", str(checkpoint))
        finetune.validate_args(args)
        self.assertEqual(args.output_dir, self.output)
        self.assertEqual(args.resume_from_checkpoint, str(checkpoint))
        self.assertTrue(args.allow_existing_output)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_invalid_resume_does_not_silently_start_a_new_run(self):
        self.previous_adapter()
        args = self.arguments("--resume-from-checkpoint", str(self.output / "missing-checkpoint"))
        with self.assertRaisesRegex(ValueError, "Trainer LoRA checkpoint"):
            finetune.validate_args(args)
        self.assertEqual(args.output_dir, self.output)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_resume_from_another_output_directory_is_rejected(self):
        self.previous_adapter()
        checkpoint = self.checkpoint(self.root / "different-run")
        args = self.arguments("--resume-from-checkpoint", str(checkpoint))
        with self.assertRaisesRegex(ValueError, "original --output-dir"):
            finetune.validate_args(args)
        self.assertEqual(args.output_dir, self.output)

    def test_output_file_is_rejected_even_with_explicit_allow(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"existing file")
        for extra in ([], ["--allow-existing-output"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                finetune.validate_args(self.arguments(*extra))
        self.assertEqual(self.output.read_bytes(), b"existing file")

    def test_model_directory_ancestors_and_descendants_stay_protected(self):
        paths = (self.model, self.model / "adapter", self.model.parent, self.root)
        for output in paths:
            for extra in ([], ["--allow-existing-output"]):
                args = self.arguments("--output-dir", str(output), *extra)
                with self.subTest(output=output, extra=extra), self.assertRaisesRegex(ValueError, "original weights"):
                    finetune.validate_args(args)
        self.assertEqual(list(self.model.iterdir()), [self.model / "config.json"])

    def test_dry_run_prints_new_path_without_creating_it_or_training(self):
        sentinel = self.previous_adapter()
        console = io.StringIO()
        argv = [
            "--model-dir", str(self.model), "--output-dir", str(self.output),
            "--split-manifest", str(self.manifest), "--dry-run",
        ]
        with redirect_stdout(console), patch.object(finetune, "prepare_dataset") as prepare, \
                patch.object(finetune, "train") as train, \
                patch.object(finetune.time, "strftime", return_value="20260920_120000"):
            result = finetune.main(argv)
        self.assertEqual(result, 0)
        prepare.assert_called_once()
        train.assert_not_called()
        selected = prepare.call_args.args[0].output_dir
        self.assertEqual(selected, self.output.with_name("adapter_run_20260920_120000"))
        self.assertFalse(selected.exists())
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertIn(str(selected), console.getvalue())

    def test_concurrent_selection_cannot_claim_the_same_automatic_run_directory(self):
        sentinel = self.previous_adapter()
        before = sentinel.read_bytes()
        first, second = self.arguments(), self.arguments()
        with patch.object(finetune.time, "strftime", return_value="20260920_120000"):
            finetune.validate_args(first)
            finetune.validate_args(second)
        self.assertEqual(first.output_dir, second.output_dir)
        self.assertFalse(first.output_dir.exists())
        # 第一轮只占用目录，在解析数据前停止；不导入 Trainer 或加载权重。
        with patch.object(finetune, "ensure_dependencies"), \
                patch.object(finetune, "prepare_dataset", side_effect=RuntimeError("stop before data")) as prepare, \
                patch.object(finetune, "load_model_and_processor") as load:
            with self.assertRaisesRegex(RuntimeError, "stop before data"):
                finetune.train(first)
            prepare.assert_called_once_with(first)
            load.assert_not_called()
            self.assertTrue(first.output_dir.is_dir())
            self.assertEqual(list(first.output_dir.iterdir()), [])
            prepare.reset_mock()
            with self.assertRaises(FileExistsError):
                finetune.train(second)
            prepare.assert_not_called()
            load.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_json_policy_is_forwarded_and_cli_can_override_it(self):
        for policy in ("new", "error"):
            with self.subTest(policy=policy):
                payload = {
                    "model": "Qwen3-VL-2B-Instruct",
                    "model_arguments": {"Qwen3-VL-2B-Instruct": {
                        "output_dir": "../output/adapter",
                        "on_existing_output": policy,
                    }},
                }
                backend, argv = unified.resolve_backend_argv(payload, [], self.root / "config")
                args = backend.build_parser().parse_args(argv)
                self.assertEqual(args.on_existing_output, policy)
                self.assertEqual(args.output_dir, self.output)
                override = "new" if policy == "error" else "error"
                backend, argv = unified.resolve_backend_argv(
                    payload, ["--", "--on-existing-output", override], self.root / "config",
                )
                self.assertEqual(backend.build_parser().parse_args(argv).on_existing_output, override)
                self.assertEqual(payload["model_arguments"]["Qwen3-VL-2B-Instruct"]["on_existing_output"], policy)


if __name__ == "__main__":
    unittest.main()
