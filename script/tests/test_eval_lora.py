"""适配器来源检查、只读加载和缓存隔离测试，无需加载真实模型。"""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.lib.eval_lora import adapter_identity, load_lora_adapter, validate_lora_path


class LoraTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evaluate-lora-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        self.write_json(self.base / "config.json", {"model_type": "qwen2", "hidden_size": 1536})
        self.config = {
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "base_model_name_or_path": str(self.base),
        }
        self.write_json(self.adapter / "adapter_config.json", self.config)
        (self.adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_null_is_unchanged_base(self):
        model = object()
        self.assertIs(load_lora_adapter(model, None), model)
        self.assertIsNone(adapter_identity(None))
        self.assertIsNone(validate_lora_path("missing-model", None))

    def test_valid_local_adapter(self):
        self.assertIsNone(validate_lora_path(self.base, self.adapter))

    def test_missing_weights_fail(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "缺少 adapter_model"):
            validate_lora_path(self.base, self.adapter)

    def test_empty_weights_fail(self):
        (self.adapter / "adapter_model.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "为空"):
            validate_lora_path(self.base, self.adapter)

    def test_unsupported_adapter_and_task_rejected(self):
        for key, value in (("peft_type", "PREFIX_TUNING"), ("task_type", "SEQ_CLS")):
            with self.subTest(key=key):
                self.write_json(self.adapter / "adapter_config.json", {**self.config, key: value})
                with self.assertRaises(ValueError):
                    validate_lora_path(self.base, self.adapter)

    def test_different_architecture_rejected(self):
        other = self.root / "qwen-vl"
        other.mkdir()
        self.write_json(other / "config.json", {
            "model_type": "qwen3_vl", "text_config": {"hidden_size": 2048},
        })
        with self.assertRaisesRegex(ValueError, "架构不匹配"):
            validate_lora_path(other, self.adapter)

    def test_moved_compatible_base_accepted(self):
        other = self.root / "moved-base"
        other.mkdir()
        self.write_json(other / "config.json", {"model_type": "qwen2", "hidden_size": 1536})
        validate_lora_path(other, self.adapter)

    def test_training_metadata_takes_precedence(self):
        other = self.root / "wrong-base"
        other.mkdir()
        self.write_json(other / "config.json", {"model_type": "qwen3_vl"})
        self.write_json(self.adapter / "adapter_config.json", {
            **self.config, "base_model_name_or_path": str(other),
        })
        self.write_json(self.adapter / "training_metadata.json", {"base_model_dir": str(self.base)})
        validate_lora_path(self.base, self.adapter)

    def test_unknown_old_location_warns_but_allows_relocation(self):
        self.write_json(self.adapter / "adapter_config.json", {
            **self.config, "base_model_name_or_path": str(self.root / "old-location"),
        })
        with self.assertWarnsRegex(UserWarning, "无法预先核对架构"):
            validate_lora_path(self.base, self.adapter)

    def test_fingerprint_detects_same_size_same_timestamp_weight_change(self):
        before = adapter_identity(self.adapter)
        weights = self.adapter / "adapter_model.safetensors"
        old = weights.stat()
        weights.write_bytes(b"ADAPTER-WEIGHTS")
        os.utime(weights, ns=(old.st_atime_ns, old.st_mtime_ns))
        after = adapter_identity(self.adapter)
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_fingerprint_detects_config_change(self):
        before = adapter_identity(self.adapter)
        self.write_json(self.adapter / "adapter_config.json", {**self.config, "r": 32})
        self.assertNotEqual(before["sha256"], adapter_identity(self.adapter)["sha256"])

    def test_loading_is_local_readonly_inference(self):
        wrapped = Mock()
        factory = Mock(return_value=wrapped)
        fake_peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=factory))
        model = object()
        with patch.dict(sys.modules, {"peft": fake_peft}):
            result = load_lora_adapter(model, self.adapter)
        self.assertIs(result, wrapped)
        factory.assert_called_once_with(
            model, str(self.adapter.resolve()), is_trainable=False, local_files_only=True,
        )
        wrapped.eval.assert_called_once_with()
        wrapped.merge_and_unload.assert_not_called()
        wrapped.save_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
