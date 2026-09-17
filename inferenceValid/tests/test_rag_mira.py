"""验证真实 Chroma 检索、图文组装和病例分析 SSE 协议；不调用付费 API。"""

import base64
import contextlib
import http.client
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cardio_ai_platform as app
from inferenceValid import embed_mira_chroma as mira
from inferenceValid import rag_mira as rag


class QueryEncoder:
    """仅替换昂贵的模型前向，实际查询仍使用 Chroma 的余弦索引。"""

    def __init__(self, args):
        self.model = SimpleNamespace(device="cpu")
        self.calls = []

    def encode_inputs(self, texts, images):
        self.calls.append((texts, [len(group) for group in images]))
        return [[1.0] + [0.0] * (mira.DIMENSION - 1)]


class RagTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.model = self.root / "model"
        self.data.mkdir()
        (self.model / "1_Pooling").mkdir(parents=True)
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "sentence_bert_config.json").write_text(json.dumps({"transformer_task": "feature-extraction"}), encoding="utf-8")
        (self.model / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode": "lasttoken", "embedding_dimension": 2048}), encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"test-fixture")
        (self.data / "train.csv").write_text("fixture", encoding="utf-8")
        for name in ("a.png", "b.png"):
            with Image.new("RGB", (32, 32), "red") as image:
                image.save(self.data / name)
        self.args = mira.parse_args(["--data-root", str(self.data), "--model-dir", str(self.model),
                                     "--db-dir", str(self.root / "db"), "--device", "cpu"])
        self.args.db_dir.mkdir()
        checkpoint = mira.Checkpoint(self.args.db_dir / f"{self.args.collection}.checkpoint.json",
                                     mira.pipeline_config(self.args), ["train"])
        checkpoint.save()
        self.client, self.collection = mira.open_collection(self.args, checkpoint)
        self.samples = [mira.Sample(f"mira:train:{i}:open_ended:0", "train", i, "open_ended", 0,
                                    f"问题{i}", f"完整答案{i}", None, "", ["a.png", "b.png"] if i == 0 else ["a.png"], {})
                        for i in range(4)]
        vectors = [[score, math.sqrt(1-score*score)] + [0.0] * (mira.DIMENSION - 2)
                   for score in (1.0, 0.9, 0.5, -0.5)]
        self.collection.upsert(ids=[s.id for s in self.samples], embeddings=vectors,
                               documents=[s.document() for s in self.samples], metadatas=[s.metadata() for s in self.samples])
        self.config = self.root / "rag_config.json"
        self.config.write_text(json.dumps({"data_root": str(self.data), "db_dir": str(self.args.db_dir),
                                           "collection": self.args.collection, "model_dir": str(self.model), "device": "cpu"}), encoding="utf-8")
        self.service = rag.MiraRAG(self.config)
        self.events = []
        self.emit = lambda stage, state, message, details: self.events.append((stage, state, message, details))
        self.encoder_patch = patch.object(mira, "QwenEmbeddingEncoder", QueryEncoder)
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)

    def tearDown(self):
        from chromadb.api.shared_system_client import SharedSystemClient
        for system in list(SharedSystemClient._identifier_to_system.values()):
            system.stop()
        SharedSystemClient.clear_system_cache()

    def retrieve(self, image=None, k=3):
        return self.service.retrieve({"caseInput": "胸痛", "exams": "检查材料"}, image, k, self.emit)

    def test_top_k_full_group_and_query_image(self):
        image = {"data_url": "data:image/png;base64," + base64.b64encode((self.data / "a.png").read_bytes()).decode()}
        groups = self.retrieve(image)
        self.assertEqual([g["id"] for g in groups], [s.id for s in self.samples[:3]])
        self.assertEqual([len(g["image_paths"]) for g in groups], [2, 1, 1])
        self.assertAlmostEqual(groups[1]["similarity"], 0.9, places=5)
        self.assertEqual(self.service.encoder.calls[0][1], [1])
        self.assertIn("病例输入：胸痛", self.service.encoder.calls[0][0][0])
        self.assertEqual(groups[0]["document"], self.samples[0].document())
        prompt, content = self.service.augment("原始材料", groups, True)
        self.assertTrue(prompt.startswith("原始材料"))
        self.assertEqual(sum(p["type"] == "image_url" for p in content), 4)
        self.assertIn("[MIRA-3]", prompt)

    def test_text_only_query_and_model_cache(self):
        self.retrieve()
        encoder = self.service.encoder
        self.retrieve(k=1)
        self.assertIs(self.service.encoder, encoder)
        self.assertEqual(encoder.calls[0][1], [0])

    def test_display_event_contains_full_document_and_each_image(self):
        document = "Question: " + "长问题" * 600 + "\nAnswer: 完整答案末尾"
        self.collection.update(ids=[self.samples[0].id], documents=[document],
                               embeddings=[[1.0] + [0.0] * (mira.DIMENSION - 1)])
        self.retrieve()
        matches = next(details["matches"] for stage, status, _, details in self.events if stage == "sources" and status == "complete")
        self.assertEqual(matches[0]["document"], document)
        self.assertEqual([image["name"] for image in matches[0]["images"]], ["a.png", "b.png"])
        self.assertEqual([len(match["images"]) for match in matches], [2, 1, 1])
        query = parse_qs(urlsplit(matches[0]["images"][1]["url"]).query)
        self.assertEqual(query, {"id": [self.samples[0].id], "index": ["1"]})

    def test_display_image_preview_and_original(self):
        import io
        from PIL import Image
        with Image.new("RGB", (1600, 900), "blue") as original:
            original.save(self.data / "a.png")
        preview, mime = self.service.read_display_image(self.samples[0].id, 0)
        self.assertEqual(mime, "image/png")
        with Image.open(io.BytesIO(preview)) as picture:
            self.assertEqual(picture.size, (960, 540))
        original, mime = self.service.read_display_image(self.samples[0].id, 0, True)
        self.assertEqual(original, (self.data / "a.png").read_bytes())
        self.assertIsNone(self.service.encoder)
        with self.assertRaises(ValueError):
            self.service.read_display_image(self.samples[0].id, -1)
        with self.assertRaises(FileNotFoundError):
            self.service.read_display_image(self.samples[0].id, 8)
        with self.assertRaises(FileNotFoundError):
            self.service.read_display_image("missing-id", 0)
        self.collection.update(ids=[self.samples[0].id], metadatas=[{"image_paths": '["../outside.png"]'}])
        with self.assertRaisesRegex(ValueError, "不在 MIRA"):
            self.service.read_display_image(self.samples[0].id, 0)

    def test_display_image_http_endpoint(self):
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.CardioAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(rag, "SERVICE", self.service):
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                for url, status in ((f"/api/rag/image?id={self.samples[0].id}&index=1", 200),
                                    ("/api/rag/image?id=missing&index=0", 404),
                                    ("/api/rag/image?id=missing&index=bad", 400)):
                    connection.request("GET", url)
                    response = connection.getresponse()
                    body = response.read()
                    self.assertEqual(response.status, status)
                    if status == 200:
                        self.assertEqual(response.getheader("Content-Type"), "image/png")
                        self.assertTrue(body.startswith(b"\x89PNG"))
                self.assertIsNone(self.service.encoder)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(5)

    def test_k_above_count_uses_available_records(self):
        self.assertEqual(len(self.retrieve(k=10)), 4)

    def test_text_model_receives_documents_without_image_parts(self):
        groups = self.retrieve()
        prompt, content = self.service.augment("原始材料", groups, False)
        self.assertTrue(all(p["type"] == "text" for p in content))
        self.assertIn("完整答案0", prompt)
        self.assertIn("未发送参考图片像素", prompt)

    def test_missing_image_fails_explicitly(self):
        (self.data / "b.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "缺少图片"):
            self.retrieve()

    def test_foreign_image_path_rejected(self):
        self.collection.update(ids=[self.samples[0].id], metadatas=[{"image_paths": '["../outside.png"]'}])
        with self.assertRaisesRegex(ValueError, "不在 MIRA"):
            self.retrieve()

    def test_empty_index_does_not_generate_without_evidence(self):
        self.collection.delete(ids=[s.id for s in self.samples])
        with self.assertRaisesRegex(ValueError, "集合为空"):
            self.retrieve()

    def test_pipeline_mismatch_does_not_encode(self):
        self.collection.modify(metadata={"pipeline_signature": "wrong"})
        with self.assertRaisesRegex(ValueError, "不匹配"):
            self.retrieve()
        self.assertIsNone(self.service.encoder)

    def test_context_limit_does_not_silently_truncate(self):
        groups = self.retrieve()
        self.service.config["max_context_chars"] = 2
        with self.assertRaisesRegex(ValueError, "未截断"):
            self.service.augment("原始病例", groups, False)

    def test_backend_sends_original_and_all_reference_images(self):
        config = {"tongyi": {"model": "qwen-vl"}}
        module = SimpleNamespace(remove_thinking_text=lambda text: text,
                                 initialize_conversations=lambda configs, prompt: configs["tongyi"].update(messages=[{"role": "system", "content": prompt}]))
        payload = {"model": "remote-api-2", "inputs": {"caseInput": "当前病例"}, "rag": {"enabled": True, "k": 3},
                   "image": {"dataUrl": "data:image/png;base64," + base64.b64encode((self.data / "a.png").read_bytes()).decode()}}
        answer = json.dumps(dict.fromkeys(app.REPORT_FIELD_ORDER, "测试结果"))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(app, "_load_inference_config", return_value=config))
            stack.enter_context(patch.object(app, "_load_inference_module", return_value=module))
            stack.enter_context(patch.object(app, "_is_local_model", return_value=False))
            stack.enter_context(patch.object(rag, "SERVICE", self.service))
            remote = stack.enter_context(patch.object(app, "_call_remote_chat_completion", return_value=answer))
            report = app._analyze_case(payload)
            self.assertEqual(report["analysis"], "测试结果")
            parts = remote.call_args.args[1]["messages"][-1]["content"]
            self.assertIn("当前病例", parts[0]["text"])
            self.assertEqual(sum(p["type"] == "image_url" for p in parts), 5)
            payload["rag"]["enabled"] = False
            with patch.object(self.service, "retrieve", side_effect=AssertionError("关闭 RAG 不应检索")):
                app._analyze_case(payload)
            self.assertEqual(sum(p["type"] == "image_url" for p in remote.call_args.args[1]["messages"][-1]["content"]), 1)


class ProtocolTests(unittest.TestCase):
    def test_partial_index_metadata_retries_but_other_errors_do_not(self):
        from chromadb.errors import InternalError
        message = "Error deserializing pickle file: eval error at offset 0: EOF while parsing"
        service = rag.MiraRAG()
        operation = MagicMock(side_effect=[InternalError(message), 123])
        with patch.object(rag.time, "sleep"):
            self.assertEqual(service._read_index(operation, lambda *args: None, "index"), 123)
            operation = MagicMock(side_effect=InternalError(message))
            with self.assertRaisesRegex(RuntimeError, "已重试三次"):
                service._read_index(operation, lambda *args: None, "index")
            self.assertEqual(operation.call_count, 3)
            operation = MagicMock(side_effect=InternalError("其他错误"))
            with self.assertRaises(InternalError):
                service._read_index(operation, lambda *args: None, "index")
            self.assertEqual(operation.call_count, 1)

    def test_options_and_bad_image(self):
        self.assertEqual(rag.parse_options({}), (False, 3))
        for k in (0, -1, 11, 1.5, True, "3"):
            with self.subTest(k=k), self.assertRaises(ValueError):
                rag.parse_options({"rag": {"enabled": True, "k": k}})
        with self.assertRaises(ValueError):
            rag.uploaded_image_bytes({"data_url": "data:image/png;base64,wrong"})

    def test_http_progress_precedes_completion_and_error_is_an_event(self):
        release = threading.Event()
        def analyze(payload, request_id=None, on_progress=None):
            on_progress({"type": "progress", "stage": "retrieval", "state": "running", "message": "检索中"})
            if not release.wait(5):
                raise RuntimeError("测试超时")
            raise RuntimeError("测试：图片丢失")
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.CardioAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(app, "_analyze_case", side_effect=analyze):
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request("POST", "/api/analyze", body=json.dumps({"inputs": {}}), headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn("event-stream", response.getheader("Content-Type"))
                first = json.loads(response.readline().decode()[6:])
                self.assertEqual(first["status"], "started")
                response.readline()
                progress = json.loads(response.readline().decode()[6:])
                self.assertEqual(progress["stage"], "retrieval")
                release.set()
                rest = response.read().decode()
                self.assertIn('"type": "error"', rest)
                self.assertIn("图片丢失", rest)
                self.assertIn("data: [DONE]", rest)
            finally:
                release.set()
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(5)


if __name__ == "__main__":
    unittest.main()
