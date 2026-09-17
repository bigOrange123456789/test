"""断连重试、代理模式、HTTP 错误及脱敏日志的回归测试。"""

import contextlib
import io
import json
import ssl
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inferenceValid import remote_transport as transport


class RemoteTransportTests(unittest.TestCase):
    def setUp(self):
        self.logs, self.progress = [], []
        self.body = {"model": "test-vision", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "原始病例"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}},
        ]}]}

    def call(self, config=None):
        return transport.post_json("https://example.com/v1/chat/completions", self.body, "secret-test-key", config or {}, 90,
                                   self.logs.append, lambda message, details: self.progress.append((message, details)))

    def response(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'{"choices":[]}'
        return response

    def test_reset_then_success_rebuilds_connection_and_keeps_images(self):
        openers = [MagicMock(), MagicMock()]
        openers[0].open.side_effect = urllib.error.URLError(ConnectionResetError(10054, "连接断开"))
        openers[1].open.return_value = self.response()
        with patch.object(transport.urllib.request, "build_opener", side_effect=openers), patch.object(transport.time, "sleep") as sleep:
            self.assertEqual(json.loads(self.call()), {"choices": []})
        first = openers[0].open.call_args.args[0]
        second = openers[1].open.call_args.args[0]
        self.assertEqual(first.data, second.data)
        self.assertEqual(json.loads(second.data), self.body)
        self.assertEqual(len(self.progress), 1)
        sleep.assert_called_once_with(1)
        self.assertNotIn("secret-test-key", " ".join(self.logs))
        self.assertNotIn("base64,fixture", " ".join(self.logs))

    def test_retry_exhaustion_is_bounded(self):
        opener = MagicMock()
        opener.open.side_effect = ConnectionResetError(10054, "断开")
        with patch.object(transport.urllib.request, "build_opener", return_value=opener), patch.object(transport.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "3 次尝试"):
                self.call()
        self.assertEqual(opener.open.call_count, 3)

    def test_billing_and_auth_errors_are_not_retried(self):
        for status in (400, 401, 403):
            with self.subTest(status=status):
                opener = MagicMock()
                opener.open.side_effect = urllib.error.HTTPError("https://example.com", status, "denied", {}, io.BytesIO(b'{"code":"Arrearage"}'))
                with patch.object(transport.urllib.request, "build_opener", return_value=opener), patch.object(transport.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, "Arrearage"):
                        self.call()
                    sleep.assert_not_called()
                self.assertEqual(opener.open.call_count, 1)

    def test_certificate_error_does_not_disable_validation_or_retry(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.URLError(ssl.SSLCertVerificationError("证书无效"))
        with patch.object(transport.urllib.request, "build_opener", return_value=opener), patch.object(transport.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "证书无效"):
                self.call()
            sleep.assert_not_called()

    def test_direct_proxy_and_timeout_are_per_request(self):
        opener = MagicMock()
        opener.open.return_value = self.response()
        with patch.object(transport.urllib.request, "ProxyHandler", wraps=transport.urllib.request.ProxyHandler) as proxy, \
             patch.object(transport.urllib.request, "build_opener", return_value=opener):
            self.call({"api_proxy_mode": "direct", "api_timeout": 25})
            proxy.assert_called_once_with({})
            self.assertEqual(opener.open.call_args.kwargs["timeout"], 25)

    def test_proxy_credentials_redacted(self):
        with patch.object(transport.urllib.request, "getproxies", return_value={"https": "http://name:password@127.0.0.1:7890"}):
            summary = transport.proxy_summary("system")
            self.assertIn("127.0.0.1:7890", summary)
            self.assertNotIn("password", summary)
            self.assertNotIn("name", summary)

    def test_bad_configuration(self):
        for config in ({"api_max_attempts": 0}, {"api_max_attempts": 10}, {"api_proxy_mode": "unknown"}, {"api_timeout": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.call(config)


if __name__ == "__main__":
    unittest.main()
