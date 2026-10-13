
from huggingface_hub import snapshot_download

'''
【DeepSeek】:
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

【Qwen-VL】:
【Qwen2-VL】:
【Qwen2.5-VL】:
【Qwen3-VL】:   
    Qwen/Qwen3-VL-2B-Instruct   (https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
    Qwen/Qwen3-VL-Embedding-2B  (https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)

【LLaVA】

'''

#DeepSeek-R1-Distill-Qwen-1.5B/7B 或 DeepSeek-V2-Lite
model_id = "Qwen/Qwen3-VL-2B-Instruct"
"deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
"deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# 下载模型到指定目录
snapshot_download(
    repo_id=model_id,
    local_dir=model_id.split("/")[-1],  # 本地保存路径 (Qwen3-VL-2B-Instruct 文件夹)
    ignore_patterns=["model.safetensors", "*.safetensors"] # 忽略大文件，其他全部下载
    resume_download=True,          # 启用断点续传
    local_dir_use_symlinks=False   # 不使用符号链接，复制实际文件
)
print("请到huggingface上手动下载模型的model.safetensors文件: https://huggingface.co/models")
