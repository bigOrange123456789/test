"""验证新旧 Transformers 的 PIL 图像处理器兼容，不加载权重或 GPU。"""

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from inferenceValid import embed_mira_chroma as mira


class EmbeddingProcessorCompatibilityTests(unittest.TestCase):
    """用独立的 Transformers 替身覆盖版本选择和失败传播。"""

    def setUp(self):
        self.args = SimpleNamespace(
            model_dir=Path("local-embedding-model"),
            min_pixels=4096,
            max_pixels=262144,
            device="cpu",
            dtype="auto",
        )
        self.transformers = ModuleType("transformers")
        self.processor = SimpleNamespace(image_processor=object())
        self.auto_loader = Mock(return_value=self.processor)
        self.transformers.AutoProcessor = SimpleNamespace(from_pretrained=self.auto_loader)

    def image_processor_class(self):
        """模拟旧版载入后 size 未跟随像素参数更新的情况。"""
        image_processor = SimpleNamespace(
            size={"shortest_edge": 4096, "longest_edge": 1310720},
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        return SimpleNamespace(__name__="FakePILImageProcessor", from_pretrained=Mock(return_value=image_processor))

    def load(self):
        with patch.dict(sys.modules, {"transformers": self.transformers}):
            return mira.load_embedding_processor(self.args)

    def test_prefers_new_pil_processor_and_preserves_model_parameters(self):
        preferred = self.image_processor_class()
        legacy = self.image_processor_class()
        self.transformers.Qwen2VLImageProcessorPil = preferred
        self.transformers.Qwen2VLImageProcessor = legacy

        actual = self.load()

        self.assertIs(actual, self.processor)
        self.assertIs(actual.image_processor, preferred.from_pretrained.return_value)
        preferred.from_pretrained.assert_called_once_with(
            self.args.model_dir,
            local_files_only=True,
            size={"shortest_edge": 4096, "longest_edge": 262144},
            min_pixels=4096,
            max_pixels=262144,
        )
        legacy.from_pretrained.assert_not_called()
        self.auto_loader.assert_called_once_with(
            self.args.model_dir,
            local_files_only=True,
            padding_side="right",
            use_fast=False,
        )
        image_processor = actual.image_processor
        self.assertEqual(image_processor.size, {"shortest_edge": 4096, "longest_edge": 262144})
        self.assertEqual(
            (image_processor.patch_size, image_processor.temporal_patch_size, image_processor.merge_size),
            (16, 2, 2),
        )
        self.assertEqual(image_processor.image_mean, [0.5, 0.5, 0.5])
        self.assertEqual(image_processor.image_std, [0.5, 0.5, 0.5])

    def test_uses_legacy_slow_processor_when_pil_name_is_absent(self):
        legacy = self.image_processor_class()
        fast = self.image_processor_class()
        self.transformers.Qwen2VLImageProcessor = legacy
        self.transformers.Qwen2VLImageProcessorFast = fast
        self.args.min_pixels = 8192
        self.args.max_pixels = 131072

        actual = self.load()

        self.assertIs(actual.image_processor, legacy.from_pretrained.return_value)
        legacy.from_pretrained.assert_called_once_with(
            self.args.model_dir,
            local_files_only=True,
            size={"shortest_edge": 8192, "longest_edge": 131072},
            min_pixels=8192,
            max_pixels=131072,
        )
        self.assertEqual(actual.image_processor.size, {"shortest_edge": 8192, "longest_edge": 131072})
        fast.from_pretrained.assert_not_called()

    def test_missing_both_pil_names_reports_actionable_error(self):
        fast = self.image_processor_class()
        self.transformers.Qwen2VLImageProcessorFast = fast

        with self.assertRaisesRegex(RuntimeError, "PIL"):
            self.load()

        fast.from_pretrained.assert_not_called()

    def test_lazy_import_dependency_failure_is_not_hidden_by_fallback(self):
        legacy = self.image_processor_class()
        self.transformers.Qwen2VLImageProcessor = legacy
        dependency_error = ImportError("PIL backend dependency is unavailable")

        def lazy_getattr(name):
            if name == "Qwen2VLImageProcessorPil":
                raise dependency_error
            raise AttributeError(name)

        self.transformers.__getattr__ = lazy_getattr

        with self.assertRaises(ImportError) as caught:
            self.load()

        self.assertIs(caught.exception, dependency_error)
        legacy.from_pretrained.assert_not_called()

    def test_selected_processor_load_failure_does_not_try_another_backend(self):
        preferred = self.image_processor_class()
        legacy = self.image_processor_class()
        self.transformers.Qwen2VLImageProcessorPil = preferred
        self.transformers.Qwen2VLImageProcessor = legacy
        dependency_error = ImportError("image loader dependency is unavailable")
        preferred.from_pretrained.side_effect = dependency_error

        with self.assertRaises(ImportError) as caught:
            self.load()

        self.assertIs(caught.exception, dependency_error)
        legacy.from_pretrained.assert_not_called()

    def test_encoder_checks_processor_before_loading_model_weights(self):
        """处理器不兼容时及时报错，不能先占用模型内存。"""
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        model_loader = Mock()
        self.transformers.Qwen3VLModel = SimpleNamespace(from_pretrained=model_loader)
        processor_error = RuntimeError("PIL processor failed")

        with patch.dict(sys.modules, {"transformers": self.transformers, "torch": fake_torch}):
            with patch.object(mira, "load_embedding_processor", side_effect=processor_error) as load_processor:
                with self.assertRaises(RuntimeError) as caught:
                    mira.QwenEmbeddingEncoder(self.args)

        self.assertIs(caught.exception, processor_error)
        load_processor.assert_called_once_with(self.args)
        model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
