"""抽样脚本的 JSON 配置测试，不扫描真实数据集，也不访问真实 Chroma。"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import sample_mira_ids as sampling


@contextmanager
def changed_directory(path):
    """临时改变启动目录，验证配置相对路径不受终端位置影响。"""
    original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


class SampleMiraIdsConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sample-mira-config-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_dir = self.root / "settings"
        self.config_dir.mkdir()
        self.config_path = self.config_dir / "sample_mira_ids.json"
        self.output = self.config_dir / "result" / "selected.json"
        self.payload = {
            "train_count": 7, "test_count": 3,
            "data_root": "../dataset", "keywords_file": "keywords.txt", "db_dir": "../vectors",
            "collection": "custom_mira", "splits": ["train", "validation"], "seed": 1234,
            "exclude_shared_images": True, "output": "result/selected.json", "check_config": False,
        }
        self.write_config(self.payload)

    def write_config(self, value, path=None):
        target = path or self.config_path
        target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8-sig")
        return target

    def run_main(self, argv=None, *, sampler_error=None):
        signature = inspect.signature(sampling.sample_dataset)
        calls = []

        def sample(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            calls.append(values)
            if sampler_error is not None:
                raise sampler_error
            return {"train_count": values["train_count"], "test_count": values["test_count"],
                    "train_ids": ["训练题"], "test_ids": ["测试题"]}

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sampling, "DEFAULT_CONFIG_PATH", self.config_path), \
                patch.object(sampling, "sample_dataset", side_effect=sample) as sampler, \
                patch.object(sys, "argv", ["sample_mira_ids.py"]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = sampling.main(argv)
        return code, calls, sampler.call_count, stdout.getvalue(), stderr.getvalue()

    def resolve(self, argv=()):
        with patch.object(sampling, "DEFAULT_CONFIG_PATH", self.config_path):
            parsed = sampling.build_parser().parse_args(list(argv))
            return sampling.resolve_run_config(parsed)

    def test_no_arguments_load_default_json_and_pass_every_sampling_parameter(self):
        # main() 使用默认 argv=None，模拟直接 python sample_mira_ids.py。
        code, calls, count, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(count, 1)
        actual = calls[0]
        self.assertEqual(actual["train_count"], 7)
        self.assertEqual(actual["test_count"], 3)
        self.assertEqual(Path(actual["data_root"]), self.root / "dataset")
        self.assertEqual(Path(actual["keywords_file"]), self.config_dir / "keywords.txt")
        self.assertEqual(Path(actual["db_dir"]), self.root / "vectors")
        self.assertEqual(actual["collection"], "custom_mira")
        self.assertEqual(actual["splits"], ["train", "validation"])
        self.assertEqual(actual["seed"], 1234)
        self.assertIs(actual["exclude_shared_images"], True)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["train_count"], 7)
        self.assertEqual(saved["test_count"], 3)
        self.assertEqual(saved["train_ids"], ["训练题"])

    def test_load_config_returns_namespace_and_resolves_paths_without_existing_dataset(self):
        args = sampling.load_config(self.config_path)
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(Path(args.data_root), self.root / "dataset")
        self.assertEqual(Path(args.keywords_file), self.config_dir / "keywords.txt")
        self.assertEqual(Path(args.db_dir), self.root / "vectors")
        self.assertEqual(Path(args.output), self.output)
        self.assertFalse((self.root / "dataset").exists())
        self.assertFalse(self.output.parent.exists())

    def test_json_relative_paths_remain_based_on_config_directory_from_another_cwd(self):
        elsewhere = self.root / "another_terminal_directory"
        elsewhere.mkdir()
        with changed_directory(elsewhere):
            args = self.resolve()
        self.assertEqual(Path(args.data_root), self.root / "dataset")
        self.assertEqual(Path(args.keywords_file), self.config_dir / "keywords.txt")
        self.assertEqual(Path(args.db_dir), self.root / "vectors")
        self.assertEqual(Path(args.output), self.output)

    def test_explicit_cli_values_override_json_and_cli_relative_paths_use_cwd(self):
        elsewhere = self.root / "terminal"
        elsewhere.mkdir()
        with changed_directory(elsewhere):
            args = self.resolve([
                "--train-count", "5", "--test-count", "2", "--data-root", "cli_data",
                "--keywords-file", "cli_keywords.txt", "--db-dir", "cli_vectors",
                "--collection", "cli_collection", "--splits", "test", "--seed", "2026",
                "--output", "cli_ids.json", "--check-config",
            ])
        self.assertEqual(args.train_count, 5)
        self.assertEqual(args.test_count, 2)
        self.assertEqual(Path(args.data_root), elsewhere / "cli_data")
        self.assertEqual(Path(args.keywords_file), elsewhere / "cli_keywords.txt")
        self.assertEqual(Path(args.db_dir), elsewhere / "cli_vectors")
        self.assertEqual(Path(args.output), elsewhere / "cli_ids.json")
        self.assertEqual(args.collection, "cli_collection")
        self.assertEqual(args.splits, ["test"])
        self.assertEqual(args.seed, 2026)
        self.assertIs(args.check_config, True)
        self.assertIs(args.exclude_shared_images, True)

    def test_explicit_config_replaces_default_config(self):
        custom = self.write_config(dict(self.payload, train_count=11, test_count=4),
                                   self.config_dir / "custom.json")
        args = self.resolve(["--config", str(custom)])
        self.assertEqual(args.train_count, 11)
        self.assertEqual(args.test_count, 4)

    def test_parser_omitted_sampling_parameters_are_none_for_config_merging(self):
        args = sampling.build_parser().parse_args([])
        for field in ("train_count", "test_count", "data_root", "keywords_file", "db_dir",
                      "collection", "splits", "seed", "exclude_shared_images", "output", "check_config"):
            with self.subTest(field=field):
                self.assertIsNone(getattr(args, field))

    def test_optional_settings_may_be_omitted_and_default_output_name_is_preserved(self):
        self.write_config({"train_count": 500, "test_count": 50})
        args = sampling.load_config(self.config_path)
        self.assertEqual(args.train_count, 500)
        self.assertEqual(args.test_count, 50)
        self.assertEqual(args.splits, ["train"])
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.collection, sampling.DEFAULT_COLLECTION)
        self.assertIs(args.exclude_shared_images, False)
        self.assertIs(args.check_config, False)
        self.assertEqual(Path(args.output), sampling.PROJECT_ROOT / "output" / "mira_split_ids.json")

    def test_null_db_uses_sibling_default_and_null_keywords_enable_match_all(self):
        self.write_config(dict(self.payload, db_dir=None, keywords_file=None))
        args = self.resolve()
        self.assertEqual(Path(args.db_dir), self.root / "MIRA-chroma")
        self.assertIsNone(args.keywords_file)

    def test_null_db_default_follows_cli_data_root_but_null_keywords_stay_disabled(self):
        self.write_config(dict(self.payload, db_dir=None, keywords_file=None))
        data_root = self.root / "cli_parent" / "mira_data"
        args = self.resolve(["--data-root", str(data_root)])
        self.assertEqual(Path(args.data_root), data_root)
        self.assertEqual(Path(args.db_dir), data_root.parent / "MIRA-chroma")
        self.assertIsNone(args.keywords_file)

    def test_null_keywords_are_forwarded_as_match_all_mode(self):
        self.write_config(dict(self.payload, keywords_file=None))
        code, calls, count, _, errors = self.run_main()
        self.assertEqual(code, 0, errors)
        self.assertEqual(count, 1)
        self.assertIsNone(calls[0]["keywords_file"])
        self.assertIs(calls[0]["all_samples_match_keywords"], True)

    def test_optional_chinese_description_and_explicit_false_boolean_override(self):
        self.write_config(dict(self.payload, _说明="这里修改训练集和测试集数量。"))
        args = self.resolve(["--no-exclude-shared-images"])
        self.assertIs(args.exclude_shared_images, False)
        self.assertFalse(hasattr(args, "_说明"))
        self.write_config(dict(self.payload, _说明=False))
        with self.assertRaises(ValueError):
            sampling.load_config(self.config_path)

    def test_rejects_missing_or_nonpositive_or_noninteger_counts(self):
        cases = [{}, {"train_count": 1}, {"test_count": 1}]
        for field in ("train_count", "test_count"):
            for value in (None, True, False, 0, -1, 2.0, "2", [], {}):
                cases.append(dict(self.payload, **{field: value}))
        for payload in cases:
            with self.subTest(payload=payload):
                self.write_config(payload)
                with self.assertRaises(ValueError):
                    sampling.load_config(self.config_path)

    def test_rejects_unknown_fields_nonobjects_and_invalid_optional_types(self):
        cases = [[], [self.payload], None, "text", 42, dict(self.payload, trainCount=5)]
        for field in ("data_root", "keywords_file", "db_dir", "collection", "output"):
            for value in ("", "  ", True, 12, [], {}):
                cases.append(dict(self.payload, **{field: value}))
        for field in ("data_root", "collection", "output"):
            cases.append(dict(self.payload, **{field: None}))
        for field in ("exclude_shared_images", "check_config"):
            for value in (None, "false", 0, 1, [], {}):
                cases.append(dict(self.payload, **{field: value}))
        for value in (None, True, 1.5, "123", [], {}):
            cases.append(dict(self.payload, seed=value))
        for value in (None, [], "train", ["unknown"], ["train", "train"], [True], [1]):
            cases.append(dict(self.payload, splits=value))
        for payload in cases:
            with self.subTest(payload=payload):
                self.write_config(payload)
                with self.assertRaises(ValueError):
                    sampling.load_config(self.config_path)

    def test_missing_or_malformed_config_fails_without_sampling_or_creating_output(self):
        for content in (None, "{broken json", "[]"):
            with self.subTest(content=content):
                if content is None:
                    self.config_path.unlink(missing_ok=True)
                else:
                    self.config_path.write_text(content, encoding="utf-8")
                code, calls, count, _, _ = self.run_main()
                self.assertNotEqual(code, 0)
                self.assertEqual(calls, [])
                self.assertEqual(count, 0)
                self.assertFalse(self.output.parent.exists())

    def test_invalid_json_is_not_silently_repaired_by_cli_override(self):
        self.write_config(dict(self.payload, train_count=True))
        code, calls, count, _, _ = self.run_main(["--train-count", "5"])
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(count, 0)
        self.assertFalse(self.output.exists())

    def test_check_config_from_json_or_cli_never_samples_or_creates_output(self):
        for from_json in (True, False):
            with self.subTest(from_json=from_json):
                self.write_config(dict(self.payload, check_config=from_json))
                code, calls, count, output, errors = self.run_main(
                    None if from_json else ["--check-config"])
                self.assertEqual(code, 0, errors)
                self.assertEqual(calls, [])
                self.assertEqual(count, 0)
                self.assertFalse(self.output.parent.exists())
                self.assertIn("train_count", output)
                self.assertIn("test_count", output)

    def test_output_cannot_overwrite_config_or_keyword_input(self):
        keywords = self.config_dir / "keywords.json"
        keywords.write_text("cardiovascular disease\n", encoding="utf-8")
        for target in (self.config_path.name, "./" + self.config_path.name, keywords.name):
            with self.subTest(target=target):
                self.write_config(dict(self.payload, output=target, keywords_file=keywords.name))
                config_before, keywords_before = self.config_path.read_bytes(), keywords.read_bytes()
                code, calls, count, _, _ = self.run_main()
                self.assertNotEqual(code, 0)
                self.assertEqual(calls, [])
                self.assertEqual(count, 0)
                self.assertEqual(self.config_path.read_bytes(), config_before)
                self.assertEqual(keywords.read_bytes(), keywords_before)

    def test_invalid_output_extension_or_sampling_error_creates_no_output(self):
        self.write_config(dict(self.payload, output="result/selected.txt"))
        code, _, count, _, _ = self.run_main()
        self.assertNotEqual(code, 0)
        self.assertEqual(count, 0)
        self.assertFalse(self.output.parent.exists())
        self.write_config(self.payload)
        code, _, count, _, _ = self.run_main(sampler_error=ValueError("可用样本数量不足"))
        self.assertNotEqual(code, 0)
        self.assertEqual(count, 1)
        self.assertFalse(self.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
