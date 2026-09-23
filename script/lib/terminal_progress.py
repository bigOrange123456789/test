"""用于长时间数据预处理循环的轻量单行进度条。"""

from __future__ import annotations

import sys
import time
from typing import TextIO


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TerminalProgress:
    """以回车刷新同一行，并限频更新速度和剩余时间。"""

    def __init__(self, label: str, total: int, *, stream: TextIO | None = None,
                 width: int = 28, min_interval: float = 0.2):
        self.label = label
        self.total = max(0, int(total))
        self.stream = stream or sys.stdout
        self.width = max(10, width)
        self.min_interval = max(0.0, min_interval)
        self.started = time.perf_counter()
        self.last_render = 0.0
        self.closed = False

    def update(self, current: int) -> None:
        if self.closed:
            return
        current = min(max(0, int(current)), self.total)
        now = time.perf_counter()
        elapsed = now - self.started
        if current < self.total and now - self.last_render < self.min_interval:
            return
        fraction = current / self.total if self.total else 1.0
        filled = min(self.width, int(self.width * fraction))
        bar = "=" * filled + ">" * (filled < self.width) + " " * max(0, self.width - filled - 1)
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else 0.0
        eta_text = _duration(eta) if current < self.total and rate > 0 else "--:--:--"
        self.stream.write(
            f"\r{self.label} [{bar}] {fraction * 100:5.1f}% "
            f"{current:,}/{self.total:,} {rate:.2f}/s ETA {eta_text}"
        )
        self.stream.flush()
        self.last_render = now
        if current >= self.total:
            self.finish()

    def finish(self) -> None:
        if not self.closed:
            self.stream.write("\n")
            self.stream.flush()
            self.closed = True

    def __enter__(self) -> "TerminalProgress":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.finish()
