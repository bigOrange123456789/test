"""用微型真实 Qwen3-VL 权重验证前缀兼容，避免仅靠模型替身掩盖加载失败。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLModel

from inferenceValid import embed_mira_chroma as mira


class EmbeddingWeightLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        config = Qwen3VLConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 1, "num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 8,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 4, "out_hidden_size": 32,
                           "deepstack_visual_indexes": [0]},
        )
        self.original = Qwen3VLModel(config).eval()
        config.save_pretrained(self.path)
        self.args = SimpleNamespace(model_dir=self.path, device="cpu", dtype="float32")

    def save_weights(self, prefix, missing=None):
        weights = {prefix + key: value.clone() for key, value in self.original.state_dict().items()
                   if key != missing}
        save_file(weights, self.path / "model.safetensors", metadata={"format": "pt"})

    def test_prefixed_and_bare_weights_load_exactly(self):
        """两种命名都逐张量比对，确保没有随机初始化或遗漏视觉参数。"""
        for prefix in ("model.", ""):
            with self.subTest(prefix=prefix):
                self.save_weights(prefix)
                with patch.object(mira, "load_embedding_processor", return_value=object()):
                    encoder = mira.QwenEmbeddingEncoder(self.args)
                actual = encoder.model.state_dict()
                expected = self.original.state_dict()
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_missing_weight_still_fails(self):
        """修复不能通过放宽完整性校验来隐藏真实权重缺失。"""
        self.save_weights("model.", missing="language_model.embed_tokens.weight")
        with patch.object(mira, "load_embedding_processor", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "模型权重加载不完整"):
                mira.QwenEmbeddingEncoder(self.args)


if __name__ == "__main__":
    unittest.main()
