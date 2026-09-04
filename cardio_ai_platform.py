#!/usr/bin/env python3
"""Start the cardiovascular AI research front-end prototype locally."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
APP_INDEX = ROOT_DIR / "cardio_ai_platform" / "index.html"


def _load_http_server():
    """Import http.server without letting this file shadow stdlib html."""
    original_path = sys.path[:]
    blocked = {"", str(ROOT_DIR), str(Path.cwd().resolve())}
    sys.path[:] = [entry for entry in sys.path if entry not in blocked]
    try:
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    finally:
        sys.path[:] = original_path
    return SimpleHTTPRequestHandler, ThreadingHTTPServer


SimpleHTTPRequestHandler, ThreadingHTTPServer = _load_http_server()


class CardioAIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in {"", "/"}:
            self.path = "/cardio_ai_platform/index.html"
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[CardioAI] {self.address_string()} - {format % args}")


def pick_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, preferred))
            return preferred
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cardiovascular AI diagnosis and knowledge fusion UI."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred local port.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the local URL without opening a browser automatically.",
    )
    args = parser.parse_args()

    if not APP_INDEX.exists():
        raise FileNotFoundError(
            f"Front-end entry not found: {APP_INDEX}. Please keep cardio_ai_platform next to html.py."
        )

    port = pick_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), CardioAIHandler)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{port}/"

    print("")
    print("心血管疾病人工智能诊疗与知识融合平台已启动")
    print(f"本地链接: {url}")
    print("按 Ctrl+C 停止服务")
    print("")

    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止本地服务...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
