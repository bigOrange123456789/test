"""验证本地视觉模型路由、图像顺序及纯文字兼容，不加载真实权重。"""

import base64
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cardio_ai_platform as app
from inferenceValid import local_qwen_vl as qwen
from inferenceValid import rag_mira


class LocalQwenTests(unittest.TestCase):
    @staticmethod
    def image(color):
        from PIL import Image
        buffer = io.BytesIO()
        with Image.new("RGB", (32, 32), color) as picture:
            picture.save(buffer, "PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def test_conversion_retains_all_images_and_text_order(self):
        original = [{"role": "system", "content": "系统指令"}, {"role": "user", "content": [
            {"type": "text", "text": "当前病例"}, {"type": "image_url", "image_url": {"url": self.image("red")}},
            {"type": "text", "text": "参考组一"}, {"type": "image_url", "image_url": {"url": self.image("blue")}},
        ]}]
        with contextlib.ExitStack() as stack:
            messages, pictures = qwen.prepare_messages(original, stack)
            self.assertEqual([p["type"] for p in messages[1]["content"]], ["text", "image", "text", "image"])
            self.assertEqual(pictures[0].getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(pictures[1].getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(original[1]["content"][1]["type"], "image_url")

    def test_plain_text_and_invalid_image(self):
        with contextlib.ExitStack() as stack:
            messages, images = qwen.prepare_messages([{"role": "user", "content": "纯文字"}], stack)
            self.assertEqual(images, [])
            self.assertEqual(messages[0]["content"][0]["text"], "纯文字")
            for url in ("https://example.com/image.png", "data:image/png;base64,wrong"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    qwen.prepare_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}], stack)

    def test_backend_uses_local_vision_for_text_image_and_rag(self):
        config = {"myQwen": {"localhost": True, "model_path": "Qwen3-VL-2B-Instruct"}}
        module = SimpleNamespace(remove_thinking_text=lambda text: text, is_local_config=lambda model_id, cfg: cfg["localhost"],
                                 initialize_conversations=lambda configs, prompt: configs["myQwen"].update(messages=[{"role": "system", "content": prompt}]))
        payload = {"model": "qwen3-vl-local", "inputs": {"caseInput": "原始病例"}, "rag": {"enabled": False, "k": 3}}
        answer = json.dumps(dict.fromkeys(app.REPORT_FIELD_ORDER, "测试结果"))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(app, "_load_inference_module", return_value=module))
            stack.enter_context(patch.object(app, "_load_inference_config", return_value=config))
            remote = stack.enter_context(patch.object(app, "_call_remote_chat_completion", side_effect=AssertionError("本地模型不能调用远程 API")))
            generate = stack.enter_context(patch.object(qwen.SERVICE, "generate", return_value=answer))
            app._analyze_case(payload)
            self.assertIsInstance(generate.call_args.args[0][-1]["content"], str)
            payload["image"] = {"dataUrl": self.image("red")}
            app._analyze_case(payload)
            self.assertEqual(sum(p["type"] == "image_url" for p in generate.call_args.args[0][-1]["content"]), 1)
            payload["rag"]["enabled"] = True
            groups = [{"image_paths": ["one", "two"]}]
            stack.enter_context(patch.object(rag_mira.SERVICE, "retrieve", return_value=groups))
            augment = stack.enter_context(patch.object(rag_mira.SERVICE, "augment", return_value=("完整提示", [
                {"type": "text", "text": "完整参考组"},
                {"type": "image_url", "image_url": {"url": self.image("blue")}},
                {"type": "image_url", "image_url": {"url": self.image("green")}},
            ])))
            app._analyze_case(payload)
            self.assertTrue(augment.call_args.args[2])
            content = generate.call_args.args[0][-1]["content"]
            self.assertIn("原始病例", content[0]["text"])
            self.assertEqual(sum(p["type"] == "image_url" for p in content), 3)
            remote.assert_not_called()

    def test_preload_uses_vision_loader(self):
        config = {"myQwen": {"localhost": True}}
        module = SimpleNamespace(is_local_config=lambda model_id, cfg: True)
        with patch.object(app, "_load_inference_config", return_value=config), \
             patch.object(app, "_load_inference_module", return_value=module), \
             patch.object(qwen.SERVICE, "load") as load:
            app._preload_local_models("qwen3-vl-local")
            load.assert_called_once_with(config["myQwen"])

    def generation_fixture(self):
        import torch
        from transformers import BatchEncoding

        model = MagicMock()
        model.device, model.dtype = torch.device("cpu"), torch.float32
        model.config.text_config.max_position_embeddings = 32768
        model.generation_config = SimpleNamespace()
        model.generate.return_value = torch.tensor([[1, 2, 3, 9, 10]])
        processor = MagicMock()
        processor.apply_chat_template.return_value = "格式化提示"
        processor.return_value = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]]),
                                                "attention_mask": torch.ones((1, 3), dtype=torch.long)})
        processor.batch_decode.return_value = ["模型新增内容"]
        args = SimpleNamespace(max_new_tokens=8, temperature=0.0, top_p=0.95, repetition_penalty=1.1)
        return model, processor, args

    def test_generate_decodes_only_new_tokens_and_uses_one_config(self):
        model, processor, args = self.generation_fixture()
        service = qwen.LocalQwenVL()
        with patch.object(service, "load", return_value=(model, processor)):
            answer = service.generate([{"role": "user", "content": "测试"}], {}, args)
        self.assertEqual(answer, "模型新增内容")
        self.assertEqual(processor.batch_decode.call_args.args[0].tolist(), [[9, 10]])
        kwargs = model.generate.call_args.kwargs
        self.assertNotIn("max_new_tokens", kwargs)
        self.assertEqual(kwargs["generation_config"].max_new_tokens, 8)
        self.assertFalse(kwargs["generation_config"].do_sample)

    def test_context_budget_fails_without_truncating(self):
        model, processor, args = self.generation_fixture()
        service = qwen.LocalQwenVL()
        with patch.object(service, "load", return_value=(model, processor)):
            with self.assertRaisesRegex(ValueError, "未截断"):
                service.generate([{"role": "user", "content": "测试"}], {"max_context_tokens": 10}, args)
        model.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
