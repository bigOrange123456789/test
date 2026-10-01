"""PyTorch/Hugging Face 分布式微调的轻量辅助函数。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def set_local_rank_from_argument(value: int | None) -> None:
    """兼容旧 launcher 通过 --local-rank 传参的方式。"""
    if value is None:
        return
    current = os.environ.get("LOCAL_RANK")
    if current is not None and int(current) != value:
        raise ValueError(
            f"--local-rank={value} 与环境变量 LOCAL_RANK={current} 不一致。"
        )
    os.environ["LOCAL_RANK"] = str(value)


def validate_local_gpu_count(local_size: int, device_count: int) -> None:
    """在初始化进程组前，确保本机每个 worker 都能分配独立 GPU。"""
    if local_size < 1:
        raise RuntimeError(f"LOCAL_WORLD_SIZE 必须是正整数，当前值为 {local_size}。")
    if local_size > device_count:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={local_size} 大于当前进程可见 GPU 数量 {device_count}；"
            "请减少 torchrun --nproc_per_node，或检查 CUDA_VISIBLE_DEVICES。"
        )


def clear_output_directory(path: str | Path) -> None:
    """清理训练输出目录内容，但保留目录本身。"""
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        # 符号链接只删除链接本身，避免误删链接目标。
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    return rank() == 0


def main_process_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed() -> None:
    """在每个 torchrun worker 加载模型前绑定本地 GPU 并初始化进程组。"""
    size = world_size()
    if size <= 1:
        return

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("多进程微调需要每个 worker 都能使用 CUDA GPU。")
    local = local_rank()
    device_count = torch.cuda.device_count()
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(size)))
    validate_local_gpu_count(local_size, device_count)
    if local < 0 or local >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local} 超出当前进程可见 GPU 数量 {device_count}；"
            "请确认 CUDA_VISIBLE_DEVICES 和 torchrun --nproc_per_node 设置。"
        )
    if not dist.is_available():
        raise RuntimeError("当前 PyTorch 未编译分布式训练支持。")
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    if backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeError("PyTorch 同时缺少 NCCL 与 Gloo 分布式后端。")

    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", world_size=size, rank=rank())
    main_process_print(
        f"多 GPU DDP 已启用：world_size={size}, backend={backend}, "
        f"local GPUs={device_count}。"
    )


def broadcast_from_main(value: Any) -> Any:
    """广播 rank 0 计算出的短小配置值，例如自动选择的输出路径。"""
    if not is_distributed():
        return value
    import torch.distributed as dist

    payload = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def run_on_main_process(action: Callable[[], Any], description: str) -> Any:
    """只让 rank 0 执行动作，并把成功/失败状态广播给所有 worker。"""
    if not is_distributed():
        return action()

    result: dict[str, Any] | None = None
    if is_main_process():
        try:
            result = {"ok": True, "value": action()}
        except Exception as error:  # 广播错误，避免其它 rank 永久等待 collective。
            result = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    result = broadcast_from_main(result)
    if not result["ok"]:
        raise RuntimeError(
            f"主进程执行{description}失败 ({result['error_type']}): {result['error']}"
        )
    return result.get("value")


def cleanup_distributed() -> None:
    """释放本脚本初始化的进程组。"""
    if not is_distributed():
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class SilentProgress:
    """非主进程继续执行预处理循环，但不重复输出终端进度条。"""

    def __init__(self, *_args: Any, **_kwargs: Any):
        pass

    def update(self, _current: int) -> None:
        pass

    def finish(self) -> None:
        pass

    def __enter__(self) -> "SilentProgress":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        pass
