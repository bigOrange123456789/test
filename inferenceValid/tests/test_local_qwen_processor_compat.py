"""验证本地生成模型复用兼容处理器；使用临时配置且不加载真实权重。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from inferenceValid import local_qwen_vl as qwen


class LocalQwenProcessorCompatibilityTests(unittest.TestCase):
    """覆盖旧版 Transformers 的加载、缓存和失败时的加载顺序。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_path = Path(temporary.name).resolve()
        (self.model_path / "config.json").write_text(
            json.dumps({
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }),
            encoding="utf-8",
        )
        self.cfg = {
            "model_path": str(self.model_path),
            "device": "cpu",
            "dtype": "float32",
            "min_pixels": 8192,
            "max_pixels": 131072,
        }
        self.torch = ModuleType("torch")
        self.torch.float32 = object()
        self.torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        self.transformers = ModuleType("transformers")
        # 只提供旧版慢速类；若生产代码仍硬导入 Pil，测试将直接失败。
        self.image_processor = SimpleNamespace(
            size={"shortest_edge": 4096, "longest_edge": 1310720},
            patch_size=16,
        )
        self.image_loader = Mock(return_value=self.image_processor)
        self.transformers.Qwen2VLImageProcessor = SimpleNamespace(
            __name__="Qwen2VLImageProcessor", from_pretrained=self.image_loader,
        )
        self.processor = SimpleNamespace(image_processor=None)
        self.auto_loader = Mock(return_value=self.processor)
        self.transformers.AutoProcessor = SimpleNamespace(from_pretrained=self.auto_loader)
        self.model = Mock()
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model_loader = Mock(return_value=(self.model, {}))
        self.transformers.Qwen3VLForConditionalGeneration = SimpleNamespace(
            from_pretrained=self.model_loader,
        )
        self.service = qwen.LocalQwenVL()

    def test_legacy_processor_uses_left_padding_and_cached_model(self):
        with patch.dict(sys.modules, {"transformers": self.transformers, "torch": self.torch}):
            first = self.service.load(self.cfg)
            second = self.service.load(dict(self.cfg))

        self.assertIs(first[0], self.model)
        self.assertIs(first[1], self.processor)
        self.assertIs(second[0], first[0])
        self.assertIs(second[1], first[1])
        self.assertIs(self.processor.image_processor, self.image_processor)
        self.assertEqual(self.image_processor.size, {"shortest_edge": 8192, "longest_edge": 131072})
        self.auto_loader.assert_called_once_with(
            self.model_path, local_files_only=True, padding_side="left", use_fast=False,
        )
        self.image_loader.assert_called_once_with(
            self.model_path,
            local_files_only=True,
            size={"shortest_edge": 8192, "longest_edge": 131072},
            min_pixels=8192,
            max_pixels=131072,
        )
        self.model_loader.assert_called_once()
        self.model.to.assert_called_once_with("cpu")
        self.model.eval.assert_called_once_with()

    def test_processor_failure_prevents_model_weights_from_loading(self):
        """处理器依赖错误原样抛出，并保持服务缓存为空。"""
        dependency_error = ImportError("PIL processor dependency is unavailable")
        self.image_loader.side_effect = dependency_error

        with patch.dict(sys.modules, {"transformers": self.transformers, "torch": self.torch}):
            with self.assertRaises(ImportError) as caught:
                self.service.load(self.cfg)

        self.assertIs(caught.exception, dependency_error)
        self.image_loader.assert_called_once()
        self.model_loader.assert_not_called()
        self.assertIsNone(self.service.model)
        self.assertIsNone(self.service.processor)
        self.assertIsNone(self.service.signature)


if __name__ == "__main__":
    unittest.main()
