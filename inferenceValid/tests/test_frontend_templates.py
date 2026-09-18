"""前端模板的命令行选项与 HTTP 入口回归测试，不加载模型。"""

import contextlib
import io
import sys
import threading
import unittest
from functools import partial
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cardio_ai_platform as app


class FrontendTemplateTests(unittest.TestCase):
    def start_server(self, template=None):
        handler = app.CardioAIHandler if template is None else partial(
            app.CardioAIHandler, frontend_template=template
        )
        server = app.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(cleanup)
        return server

    def request(self, server, path, method="GET", headers=None):
        connection = HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_default_and_explicit_templates_on_all_entry_paths(self):
        original = app.APP_INDEX.read_bytes()
        for template in (None, "original", "compact"):
            server = self.start_server(template)
            for path in ("/", "/?v=1", "/cardio_ai_platform/", "/cardio_ai_platform/index.html?v=1"):
                with self.subTest(template=template, path=path):
                    status, headers, body = self.request(server, path)
                    self.assertEqual(status, 200)
                    self.assertIn(f'data-frontend-template="{template or "original"}"', body.decode("utf-8"))
                    self.assertEqual(headers["Cache-Control"], "no-store")
                    self.assertEqual(int(headers["Content-Length"]), len(body))
                    self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
                    head_status, head_headers, head_body = self.request(server, path, "HEAD")
                    self.assertEqual(head_status, 200)
                    self.assertEqual(head_headers["Content-Length"], headers["Content-Length"])
                    self.assertEqual(head_body, b"")
            # 入口不允许 304，避免重启时拿到上次选择的模板。
            self.assertEqual(self.request(server, "/", headers={
                "If-Modified-Since": "Fri, 01 Jan 2100 00:00:00 GMT"
            })[0], 200)
        self.assertEqual(app.APP_INDEX.read_bytes(), original)

    def test_static_assets_unchanged(self):
        server = self.start_server("compact")
        for name in ("app.js", "styles.css"):
            with self.subTest(name=name):
                status, _, body = self.request(server, f"/cardio_ai_platform/{name}")
                self.assertEqual(status, 200)
                self.assertEqual(body, (app.APP_INDEX.parent / name).read_bytes())

    def test_cli_passes_template_to_handler_and_preserves_default(self):
        for options, expected in (([], "original"), (["--template", "original"], "original"),
                                  (["--template", "compact"], "compact")):
            with self.subTest(options=options), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["cardio_ai_platform.py", "--no-open",
                                                            "--no-preload-rag", "--preload-local-models", "none", *options]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(app, "pick_port", return_value=8765))
                preload = stack.enter_context(patch.object(app, "_preload_local_models"))
                factory = stack.enter_context(patch.object(app, "ThreadingHTTPServer"))
                app.main()
                handler = factory.call_args.args[1]
                self.assertIs(handler.func, app.CardioAIHandler)
                self.assertEqual(handler.keywords, {"frontend_template": expected})
                preload.assert_called_once_with("none")
                factory.return_value.serve_forever.assert_called_once()
                factory.return_value.server_close.assert_called_once()

    def test_cli_rejects_unknown_template_before_preloading(self):
        with patch.object(sys, "argv", ["cardio_ai_platform.py", "--template", "unknown"]), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch.object(app, "_preload_local_models") as preload:
            with self.assertRaises(SystemExit) as error:
                app.main()
            self.assertEqual(error.exception.code, 2)
            preload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
