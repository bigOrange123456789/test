"""本地 Qwen3-VL-Instruct 图文生成，复用病例分析的多模态消息结构。"""

from __future__ import annotations

import base64
import copy
import io
import json
import threading
import time
from contextlib import ExitStack
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_messages(messages: list[dict], resources: ExitStack) -> tuple[list[dict], list]:
    """按消息中的原顺序解码每张图片，保留病例与参考组的对应关系。"""
    from PIL import Image, ImageOps

    conversations, images = [], []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted = []
        for part in content:
            if part.get("type") == "text":
                converted.append({"type": "text", "text": str(part.get("text", ""))})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("本地 Qwen 仅接收上传或检索得到的 Base64 图片，不自动下载远程图片。")
                try:
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    original = resources.enter_context(Image.open(io.BytesIO(raw)))
                    picture = ImageOps.exif_transpose(original).convert("RGB")
                    resources.callback(picture.close)
                except Exception as error:
                    raise ValueError(f"本地 Qwen 无法解码第 {len(images) + 1} 张图片：{error}") from error
                images.append(picture)
                converted.append({"type": "image"})
            else:
                raise ValueError(f"本地 Qwen 不支持消息类型：{part.get('type')}")
        conversations.append({"role": message["role"], "content": converted})
    return conversations, images


class LocalQwenVL:
    """缓存本地视觉模型；加载和生成串行执行，避免同一模型重复占用显存。"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.signature = None
        self.lock = threading.RLock()

    def load(self, cfg: dict, emit=None):
        """只读取本地 Instruct 权重，使用视觉生成模型类而非纯文本模型类。"""
        import torch
        from transformers import AutoProcessor, Qwen2VLImageProcessorPil, Qwen3VLForConditionalGeneration

        model_path = (PROJECT_ROOT / cfg.get("model_path", "Qwen3-VL-2B-Instruct")).resolve()
        requested_device = str(cfg.get("device", "auto"))
        dtype_name = str(cfg.get("dtype", "auto"))
        min_pixels = int(cfg.get("min_pixels", 4096))
        max_pixels = int(cfg.get("max_pixels", 262144))
        if dtype_name not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("本地 Qwen dtype 必须为 auto、float32、float16 或 bfloat16。")
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("本地 Qwen 的图片像素范围无效。")
        signature = (str(model_path), requested_device, dtype_name, min_pixels, max_pixels)
        with self.lock:
            if self.model is not None:
                if signature != self.signature:
                    raise ValueError("本地 Qwen 加载参数已改变，请重启服务使新配置生效。")
                return self.model, self.processor
            if not (model_path / "config.json").is_file():
                raise FileNotFoundError(f"本地 Qwen 模型路径不存在：{model_path}")
            config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
                raise ValueError("本地 Qwen 生成需要 Qwen3-VL-2B-Instruct，不能使用 Embedding 权重。")
            device = requested_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda" and torch.cuda.mem_get_info()[0] < 6 * 1024**3:
                    device = "cpu"
                    if emit:
                        emit("GPU 空闲显存不足 6 GB，本地 Qwen 改用 CPU，生成可能较慢。", {})
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("指定了 CUDA，但当前环境的 PyTorch 没有可用 GPU。")
            dtype = (torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                     else torch.float16 if device.startswith("cuda") else torch.float32) if dtype_name == "auto" else getattr(torch, dtype_name)
            if emit:
                emit("正在加载本地 Qwen3-VL-2B-Instruct。", {"device": device, "dtype": str(dtype)})
            print(f"[CardioAI][本地Qwen] 加载开始 路径={model_path} 设备={device} 精度={dtype}", flush=True)
            started = time.perf_counter()
            model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise RuntimeError(f"本地 Qwen 权重加载不完整：{loading}")
            model.to(device).eval()
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, padding_side="left")
            processor.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
                model_path, local_files_only=True,
                size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            self.model, self.processor, self.signature = model, processor, signature
            print(f"[CardioAI][本地Qwen] 加载完成 耗时={time.perf_counter() - started:.2f}秒", flush=True)
            return model, processor

    def generate(self, messages: list[dict], cfg: dict, args, emit=None) -> str:
        """处理文字、单图或 RAG 多图，只解码新增 token，不把提示词当作模型答案。"""
        import torch

        with self.lock:
            model, processor = self.load(cfg, emit)
            with ExitStack() as resources:
                conversations, images = prepare_messages(messages, resources)
                prompt = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=images or None, padding=True,
                                   truncation=False, return_tensors="pt")
                image_count = len(images)
            input_tokens = inputs["input_ids"].shape[1]
            context_limit = int(cfg.get("max_context_tokens", 16384))
            model_limit = getattr(model.config.text_config, "max_position_embeddings", context_limit)
            if input_tokens + args.max_new_tokens > min(context_limit, model_limit):
                raise ValueError(f"本地 Qwen 输入 {input_tokens} token 加输出预算 {args.max_new_tokens} 超过上下文上限；请减小 K 或精简病例，未截断任何参考组。")
            inputs = inputs.to(model.device)
            # 浮点图像张量与模型精度保持一致；token ID 仍保持整型。
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    inputs[key] = value.to(model.dtype)
            generation_config = copy.deepcopy(model.generation_config)
            generation_config.do_sample = args.temperature > 0
            generation_config.temperature = args.temperature if generation_config.do_sample else 1.0
            generation_config.top_p = args.top_p if generation_config.do_sample else 1.0
            generation_config.top_k = 50
            generation_config.max_new_tokens = args.max_new_tokens
            generation_config.repetition_penalty = args.repetition_penalty
            if emit:
                emit("本地 Qwen 正在生成结构化分析结果。",
                     {"device": str(model.device), "input_tokens": input_tokens,
                      "images_sent": image_count, "max_new_tokens": args.max_new_tokens})
            print(f"[CardioAI][本地Qwen] 生成开始 输入token={input_tokens} 图片数={image_count} 输出上限={args.max_new_tokens}", flush=True)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation_config)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError("本地 Qwen 显存不足；请减小 K、减少图片像素上限，或在 myQwen 配置中设置 device 为 cpu 后重启。") from error
            answer = processor.batch_decode(generated[:, input_tokens:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0].strip()
            print(f"[CardioAI][本地Qwen] 生成完成 耗时={time.perf_counter() - started:.2f}秒 输出字符={len(answer)}", flush=True)
            if not answer:
                raise RuntimeError("本地 Qwen 未返回任何文本。")
            return answer


SERVICE = LocalQwenVL()
