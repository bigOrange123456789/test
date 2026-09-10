# -*- coding: utf-8 -*-
"""
Download Hugging Face models with automatic resume support.

断点续传说明：
  - huggingface_hub 会保留未完成的下载缓存和临时文件。
  - 脚本中断后，重新运行同一个 repo_id/local_dir/cache_dir 组合即可继续下载。
  - 如果远程服务或镜像站不支持 HTTP Range，库会自动退化为重新下载对应文件。

示例：
  python inferenceValid/download.py
  python inferenceValid/download.py --repo-id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir DeepSeek-Model
  python inferenceValid/download.py --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import os
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "Qwen3-VL-2B-Instruct"
DEFAULT_CACHE_DIR = None


def configure_stdout() -> None:
    """将控制台输出设置为 UTF-8，减少 Windows 终端中文乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def format_bytes(size: int | float) -> str:
    """把字节数格式化成 Hugging Face 进度条常见的十进制单位。"""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.2f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1000
    return f"{value:.2f}TB"


def format_eta(seconds: float | None) -> str:
    """把预计剩余秒数格式化成易读文本。"""
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def parse_csv_patterns(value: str | None) -> list[str] | None:
    """把逗号分隔的 allow/ignore patterns 参数解析成列表。"""
    if not value:
        return None
    patterns = [pattern.strip() for pattern in value.split(",")]
    return [pattern for pattern in patterns if pattern]


def matches_any_pattern(path: str, patterns: list[str] | None) -> bool:
    """判断文件路径是否匹配任意一个通配符规则。"""
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def should_include_path(path: str, allow_patterns: list[str] | None, ignore_patterns: list[str] | None) -> bool:
    """根据 allow_patterns 和 ignore_patterns 判断文件是否会被本次下载包含。"""
    if allow_patterns and not matches_any_pattern(path, allow_patterns):
        return False
    if ignore_patterns and matches_any_pattern(path, ignore_patterns):
        return False
    return True


def build_snapshot_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """根据命令行参数构建 snapshot_download 的参数，并兼容不同 huggingface_hub 版本。"""
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "local_dir": local_dir,
        "cache_dir": cache_dir,
        "force_download": args.force_download,
        "local_files_only": args.local_files_only,
        "max_workers": args.max_workers,
        "allow_patterns": parse_csv_patterns(args.allow_patterns),
        "ignore_patterns": parse_csv_patterns(args.ignore_patterns),
    }

    if args.token:
        kwargs["token"] = args.token

    if args.endpoint:
        kwargs["endpoint"] = args.endpoint
        os.environ["HF_ENDPOINT"] = args.endpoint

    supported_params = inspect.signature(snapshot_download).parameters

    if "resume_download" in supported_params:
        kwargs["resume_download"] = True

    if "local_dir_use_symlinks" in supported_params:
        kwargs["local_dir_use_symlinks"] = False

    return {key: value for key, value in kwargs.items() if value is not None}


def get_repo_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    """读取远程仓库文件清单和大小，用于找出需要重点显示进度的大文件。"""
    if args.local_files_only:
        return []

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    allow_patterns = parse_csv_patterns(args.allow_patterns)
    ignore_patterns = parse_csv_patterns(args.ignore_patterns)
    api = HfApi(endpoint=args.endpoint) if args.endpoint else HfApi()
    files: list[dict[str, Any]] = []

    for item in api.list_repo_tree(
        args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        recursive=True,
        expand=True,
        token=args.token or None,
    ):
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        if not path or size is None:
            continue
        if should_include_path(path, allow_patterns, ignore_patterns):
            files.append({"path": path, "size": int(size)})

    return sorted(files, key=lambda file: file["size"], reverse=True)


def choose_big_file_to_watch(args: argparse.Namespace) -> dict[str, Any] | None:
    """选择本次下载中最大的文件，并在控制台提示它的具体大小。"""
    if not args.show_big_file_progress:
        return None

    try:
        files = get_repo_files(args)
    except Exception as error:
        print(f"无法读取远程文件大小清单，跳过大文件单独进度显示：{error}")
        return None

    if not files:
        return None

    big_file = files[0]
    print(f"最大文件: {big_file['path']} ({format_bytes(big_file['size'])})")
    return big_file


def iter_incomplete_files(local_dir: Path, cache_dir: Path | None) -> list[Path]:
    """查找 Hugging Face 下载过程中保留的 .incomplete 临时文件。"""
    roots = [local_dir / ".cache" / "huggingface" / "download"]
    if cache_dir:
        roots.append(cache_dir)

    incomplete_files: list[Path] = []
    for root in roots:
        if root.exists():
            incomplete_files.extend(path for path in root.rglob("*.incomplete") if path.is_file())
    return incomplete_files


def get_downloaded_bytes_for_file(big_file: dict[str, Any], args: argparse.Namespace) -> int:
    """根据目标文件和 .incomplete 临时文件估算大文件当前已下载字节数。"""
    local_dir = args.local_dir.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    expected_size = int(big_file["size"])
    final_path = local_dir / big_file["path"]
    candidates: list[int] = []

    if final_path.exists():
        candidates.append(final_path.stat().st_size)

    for incomplete_file in iter_incomplete_files(local_dir, cache_dir):
        candidates.append(incomplete_file.stat().st_size)

    if not candidates:
        return 0
    return min(max(candidates), expected_size)


def print_big_file_progress(big_file: dict[str, Any], args: argparse.Namespace, previous: tuple[int, float] | None) -> tuple[int, float]:
    """打印最大文件当前下载进度，并返回本次进度快照用于计算速度。"""
    now = time.monotonic()
    downloaded = get_downloaded_bytes_for_file(big_file, args)
    total = int(big_file["size"])
    percent = downloaded / total * 100 if total else 0.0

    speed_text = "--/s"
    eta_text = "--"
    if previous is not None:
        previous_bytes, previous_time = previous
        elapsed = max(now - previous_time, 0.001)
        speed = max(downloaded - previous_bytes, 0) / elapsed
        speed_text = f"{format_bytes(speed)}/s"
        eta_text = format_eta((total - downloaded) / speed) if speed > 0 else "--"

    print(
        f"[big-file] {big_file['path']}: "
        f"{format_bytes(downloaded)}/{format_bytes(total)} "
        f"({percent:.2f}%), speed={speed_text}, eta={eta_text}",
        flush=True,
    )
    return downloaded, now


def monitor_big_file_progress(big_file: dict[str, Any], args: argparse.Namespace, stop_event: Event) -> None:
    """后台定时打印最大文件进度，避免大文件下载时只看到总进度条卡住。"""
    previous: tuple[int, float] | None = None
    while not stop_event.is_set():
        previous = print_big_file_progress(big_file, args, previous)
        if previous[0] >= int(big_file["size"]):
            return
        stop_event.wait(args.progress_interval)


def download_with_retry(args: argparse.Namespace) -> str:
    """执行模型下载，并在网络波动时按指定次数重试。"""
    kwargs = build_snapshot_kwargs(args)
    big_file = choose_big_file_to_watch(args)
    last_error: Exception | None = None

    for attempt in range(1, args.retries + 2):
        stop_event = Event()
        monitor_thread: Thread | None = None
        try:
            print("========== 下载配置 ==========")
            print(f"repo_id:   {kwargs['repo_id']}")
            print(f"revision:  {kwargs.get('revision') or '<default>'}")
            print(f"local_dir: {kwargs['local_dir']}")
            print(f"cache_dir: {kwargs.get('cache_dir') or '<default/local_dir .cache>'}")
            print(f"endpoint:  {kwargs.get('endpoint') or '<huggingface default>'}")
            print(f"resume:    enabled")
            print("==============================")

            if big_file:
                monitor_thread = Thread(
                    target=monitor_big_file_progress,
                    args=(big_file, args, stop_event),
                    daemon=True,
                )
                monitor_thread.start()

            downloaded_path = snapshot_download(**kwargs)
            if big_file:
                print_big_file_progress(big_file, args, None)
            return downloaded_path
        except KeyboardInterrupt:
            print("\n下载已中断；下次运行同一命令会自动尝试从未完成位置继续。")
            raise
        except Exception as error:
            last_error = error
            if attempt > args.retries:
                break
            print(f"\n第 {attempt} 次下载失败：{error}")
            print(f"{args.retry_sleep} 秒后重试；已下载部分会被保留用于续传。")
            time.sleep(args.retry_sleep)
        finally:
            stop_event.set()
            if monitor_thread:
                monitor_thread.join(timeout=1)

    raise SystemExit(f"下载失败：{last_error}")


def parse_args() -> argparse.Namespace:
    """解析下载参数，允许指定模型、保存目录、镜像站、缓存目录和重试策略。"""
    parser = argparse.ArgumentParser(description="Download Hugging Face model snapshots with resume support.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR, help="Directory to save model files.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Optional cache directory used for resumable downloads.")
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"), help="Hugging Face repo type.")
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit id to download.")
    parser.add_argument("--endpoint", default=None, help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--token", default=None, help="Optional Hugging Face access token for private repos.")
    parser.add_argument("--allow-patterns", default=None, help="Comma-separated file patterns to include.")
    parser.add_argument("--ignore-patterns", default=None, help="Comma-separated file patterns to exclude.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download worker count.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count after network errors.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Seconds to wait between retries.")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between big-file progress prints.")
    parser.add_argument("--no-big-file-progress", action="store_false", dest="show_big_file_progress", help="Disable separate progress prints for the largest file.")
    parser.add_argument("--force-download", action="store_true", help="Force redownload instead of reusing cache.")
    parser.add_argument("--local-files-only", action="store_true", help="Use local cache only without network access.")
    parser.set_defaults(show_big_file_progress=True)
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析参数、下载模型，并打印最终保存路径。"""
    configure_stdout()
    args = parse_args()
    path = download_with_retry(args)
    print("Path to dataset/model files:", path)


if __name__ == "__main__":
    main()
