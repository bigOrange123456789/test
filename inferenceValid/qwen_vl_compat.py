"""为本地 Qwen 视觉生成和 MIRA 向量编码提供一致的图像处理器加载方式。"""

import logging


def load_qwen_vl_processor(model_dir, *, min_pixels, max_pixels, padding_side="right"):
    """兼容新旧 Transformers 的 PIL 后端，沿用模型配置和指定的像素范围。"""
    import transformers

    # 新版单独命名为 Pil；4.57.x 的无后缀类即为原来的 PIL 慢速实现。
    # 不回退到 Fast，避免更换缩放实现后影响既有向量的可比性。
    image_processor_class = getattr(transformers, "Qwen2VLImageProcessorPil", None)
    if image_processor_class is None:
        image_processor_class = getattr(transformers, "Qwen2VLImageProcessor", None)
    if image_processor_class is None:
        raise RuntimeError("当前 Transformers 缺少 Qwen2-VL 的 PIL 图像处理器，无法保持既有图像预处理方式。")
    size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    image_processor = image_processor_class.from_pretrained(
        model_dir, local_files_only=True, size=size,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    # 4.57.x 加载配置时只覆盖 min/max_pixels，不同步 size；统一辅助 token 计数的像素范围。
    image_processor.size = size.copy()
    processor = transformers.AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, padding_side=padding_side, use_fast=False,
    )
    processor.image_processor = image_processor
    logging.getLogger(__name__).info("Qwen 图像处理器：%s，像素范围=%d～%d",
                                     image_processor_class.__name__, min_pixels, max_pixels)
    return processor
