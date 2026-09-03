# -*- coding: utf-8 -*-
"""
Config-driven inference script for local DeepSeek/LoRA models and remote APIs.

配置说明：
  - config.json 中每个顶层 key 都是一个可切换的对话目标。
  - 每个目标通过 "localhost": true/false 区分本地模型和远程 API。
  - 本地目标可通过 "lora_adapter" 指定 LoRA adapter；值为 null 时只用基础模型。

交互命令：
  - switch-模型ID：切换到指定对话目标。
  - models：查看 config.json 中可用的对话目标。
  - check-models：检查远程 API 当前配置的模型是否可用。
  - exit / quit / q：退出。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}


def configure_stdout() -> None:
    """将控制台输出设置为 UTF-8，避免 Windows 终端打印中文时出现乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_version(version: str) -> tuple[int, ...]:
    """从版本字符串中提取数字版本号，方便后续比较依赖版本。"""
    numbers = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(number) for number in numbers[:3])


def version_at_least(installed: str, minimum: str) -> bool:
    """判断已安装依赖版本是否不低于脚本要求的最低版本。"""
    installed_tuple = parse_version(installed)
    minimum_tuple = parse_version(minimum)
    max_len = max(len(installed_tuple), len(minimum_tuple))
    installed_tuple += (0,) * (max_len - len(installed_tuple))
    minimum_tuple += (0,) * (max_len - len(minimum_tuple))
    return installed_tuple >= minimum_tuple


def check_local_dependencies(use_lora: bool) -> None:
    """检查本地 DeepSeek/LoRA 推理所需的 Python 依赖是否存在且版本足够。"""
    required = {
        "torch": "2.1.0",
        "transformers": "4.44.0",
        "safetensors": "0.4.3",
    }
    if use_lora:
        required["peft"] = "0.11.0"

    problems = []
    for package, minimum in required.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            problems.append(f"{package} missing, need >= {minimum}")
            continue
        if not version_at_least(installed, minimum):
            problems.append(f"{package} {installed} is too old, need >= {minimum}")

    if problems:
        print("当前环境缺少本地推理依赖：")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("可以先在可用环境里安装：")
        print("  pip install -r DeepSeek-Model/requirements-lora.txt")
        raise SystemExit(1)


def model_dtype_arg(dtype: Any) -> dict[str, Any]:
    """根据 transformers 版本选择 dtype 参数名，兼容 4.x 与 5.x。"""
    transformers_version = metadata.version("transformers")
    if version_at_least(transformers_version, "5.0.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """读取 config.json，并返回以模型 ID 为 key 的配置字典。"""
    config_path = path or (SCRIPT_DIR / "config.json")
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_config_bool(value: Any, default: bool = False) -> bool:
    """把 config.json 中的布尔字段解析成 Python bool，兼容 true/false 字符串。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def is_local_config(_model_id: str, cfg: dict[str, Any]) -> bool:
    """判断某个配置目标是否为本地模型，唯一依据是配置项里的 localhost 字段。"""
    return parse_config_bool(cfg.get("localhost"), default=False)


def resolve_existing_path(path: str | Path | None) -> Path | None:
    """解析配置中的相对路径，依次尝试当前目录、脚本目录和项目根目录。"""
    if path is None or path == "":
        return None

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        SCRIPT_DIR / raw_path,
        PROJECT_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def mask_api_key(api_key: str | None) -> str:
    """在控制台展示 API key 摘要时进行脱敏，避免泄露完整密钥。"""
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "<set>"
    return api_key[:4] + "..." + api_key[-4:]


def short_error_message(error: Exception) -> str:
    """把异常内容压缩成一行短文本，方便在控制台打印。"""
    message = str(error).replace("\n", " ").strip()
    message = re.sub(r"\s+", " ", message)
    return message[:300] + ("..." if len(message) > 300 else "")


def normalize_model_ids(models_response: Any) -> list[str]:
    """从 OpenAI 兼容的 models.list() 返回值中提取模型 ID 列表。"""
    model_ids = []
    for model in getattr(models_response, "data", []):
        model_id = getattr(model, "id", None)
        if model_id:
            model_ids.append(model_id)
    return sorted(set(model_ids))


def initialize_conversations(config: dict[str, dict[str, Any]], system_prompt: str) -> None:
    """为 config.json 中的每个对话目标创建独立的 messages 对话历史。"""
    for cfg in config.values():
        cfg["messages"] = [{"role": "system", "content": system_prompt}]


def print_config_summary(config: dict[str, dict[str, Any]]) -> None:
    """打印当前配置中的全部对话目标，并区分本地模型和远程 API。"""
    print("\n========== 对话目标 ==========")
    for model_id, cfg in config.items():
        if is_local_config(model_id, cfg):
            lora_adapter = cfg.get("lora_adapter") or "<disabled>"
            print(f"[{model_id}] 本地模型 model_path={cfg.get('model_path', '')} lora_adapter={lora_adapter}")
        else:
            print(
                f"[{model_id}] 远程 API base_url={cfg.get('base_url', '')} "
                f"model={cfg.get('model', '')} api_key={mask_api_key(cfg.get('api_key'))}"
            )
    print("==============================\n")


def group_remote_configs(config: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    """按 base_url 和 api_key 对远程目标分组，减少重复查询远程模型列表。"""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for model_id, cfg in config.items():
        if is_local_config(model_id, cfg):
            continue
        api_key = cfg.get("api_key")
        base_url = cfg.get("base_url")
        if api_key and base_url:
            grouped[(base_url, api_key)].append(model_id)
        else:
            print(f"[{model_id}] 跳过远程模型检查：缺少 api_key 或 base_url")
    return grouped


def print_available_remote_models(config: dict[str, dict[str, Any]]) -> None:
    """检查远程 API 的当前模型是否可用；只有不可用或查询失败时才打印可用列表。"""
    try:
        import openai
    except ImportError:
        print("当前环境未安装 openai，无法检查远程 API 模型列表。")
        return

    print("========== 远程 API 可用模型检查 ==========")
    grouped = group_remote_configs(config)
    if not grouped:
        print("没有发现远程 API 配置。")
        print("==========================================\n")
        return

    for (base_url, api_key), model_ids in grouped.items():
        title = ", ".join(model_ids)
        print(f"\n[{title}]")
        print(f"base_url: {base_url}")
        print(f"api_key: {mask_api_key(api_key)}")

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=15)
            available_models = normalize_model_ids(client.models.list())
        except Exception as error:
            print("状态：查询失败，api_key 可能失效，或 base_url 不支持 /models。")
            print(f"错误：{short_error_message(error)}")
            continue

        if not available_models:
            print("状态：查询成功，但没有返回模型列表。")
            continue

        unavailable_model_ids = []
        for model_id in model_ids:
            configured_model = config[model_id].get("model", "")
            if configured_model in available_models:
                print(f"当前配置 [{model_id}].model = {configured_model}：可用")
            else:
                unavailable_model_ids.append(model_id)
                print(f"当前配置 [{model_id}].model = {configured_model}：未在可用模型列表中")

        if unavailable_model_ids:
            print(f"可用模型数量：{len(available_models)}")
            print("可用模型：")
            for available_model in available_models:
                print(f"  - {available_model}")

    print("\n==========================================\n")


def choose_lora_adapter(cfg: dict[str, Any], args: argparse.Namespace) -> Path | None:
    """根据 config.json 和命令行参数决定本地目标是否加载 LoRA adapter。"""
    if args.use_lora is False:
        return None

    lora_adapter = cfg.get("lora_adapter")
    if not lora_adapter:
        if args.use_lora is True:
            raise FileNotFoundError("当前本地目标没有配置 lora_adapter，无法强制启用 LoRA")
        return None

    return resolve_existing_path(lora_adapter)


def choose_device(args: argparse.Namespace) -> tuple[str, Any]:
    """根据命令行参数和 CUDA 可用性决定本地模型推理设备与浮点精度。"""
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def tokenizer_source_path(model_path: Path, lora_adapter: Path | None) -> Path:
    """选择 tokenizer 加载路径；LoRA 目录没有 tokenizer 时回退到基础模型目录。"""
    if lora_adapter and (lora_adapter / "tokenizer_config.json").exists():
        return lora_adapter
    return model_path


def load_local_target(model_id: str, cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Any, Any, str]:
    """按需加载一个本地对话目标，并缓存 model、tokenizer 和 device。"""
    if model_id in LOCAL_MODEL_CACHE:
        return LOCAL_MODEL_CACHE[model_id]

    lora_adapter = choose_lora_adapter(cfg, args)
    check_local_dependencies(use_lora=lora_adapter is not None)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_existing_path(cfg.get("model_path"))
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"本地模型路径不存在：{cfg.get('model_path')}")
    if lora_adapter is not None and not lora_adapter.exists():
        raise FileNotFoundError(f"LoRA adapter 路径不存在：{cfg.get('lora_adapter')}")

    device, dtype = choose_device(args)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source_path(model_path, lora_adapter),
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_kwargs.update(model_dtype_arg(dtype))
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if lora_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter)

    model.to(device)
    model.eval()
    LOCAL_MODEL_CACHE[model_id] = (model, tokenizer, device)

    print(f"[{model_id}] Loaded base model: {model_path}")
    if lora_adapter is not None:
        print(f"[{model_id}] Loaded LoRA adapter: {lora_adapter}")
    else:
        print(f"[{model_id}] LoRA adapter: disabled")
    print(f"[{model_id}] Device: {device}")
    return LOCAL_MODEL_CACHE[model_id]


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """把当前对话历史转换成本地模型可接收的 prompt 文本。"""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_lines = [f"{message['role']}: {message['content']}" for message in messages]
    prompt_lines.append("assistant:")
    return "\n".join(prompt_lines)


def remove_thinking_text(text: str) -> str:
    """去掉部分推理模型输出中 </think> 之前的思考文本。"""
    return re.sub(r".*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_local_answer(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    """使用本地 DeepSeek 或 DeepSeek+LoRA 根据对话历史生成回答。"""
    import torch

    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if args.strip_thinking:
        answer = remove_thinking_text(answer)
    return answer.strip()


def call_remote_model(model_id: str, cfg: dict[str, Any], args: argparse.Namespace) -> str:
    """使用 OpenAI 兼容接口调用远程大模型 API，并返回模型回答文本。"""
    try:
        import openai
    except ImportError as error:
        raise ImportError("当前环境未安装 openai，无法调用远程 API；请先执行 pip install openai") from error

    client = openai.OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=args.api_timeout,
    )
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=cfg["messages"],
        extra_body=cfg.get("extra_body", {}),
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stream=args.stream,
    )

    if args.stream:
        answer = ""
        print(f" {model_id}:", end="", flush=True)
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                answer += content
                print(content, end="", flush=True)
        print()
    else:
        answer = response.choices[0].message.content or ""

    if args.strip_thinking:
        answer = remove_thinking_text(answer)
    return answer.strip()


def chat_with_model(model_id: str, question: str, config: dict[str, dict[str, Any]], args: argparse.Namespace) -> str:
    """根据 model_id 自动选择本地模型或远程 API，并维护该目标的多轮对话历史。"""
    if not question:
        return "Please enter a question~"
    if model_id not in config:
        return f"未知模型目标：{model_id}"

    cfg = config[model_id]
    cfg["messages"].append({"role": "user", "content": question})

    try:
        if is_local_config(model_id, cfg):
            model, tokenizer, device = load_local_target(model_id, cfg, args)
            answer = generate_local_answer(model, tokenizer, device, cfg["messages"], args)
        else:
            answer = call_remote_model(model_id, cfg, args)
    except Exception as error:
        return f"Failed to call model: {short_error_message(error)}"

    cfg["messages"].append({"role": "assistant", "content": answer})
    return answer


def list_targets(config: dict[str, dict[str, Any]]) -> None:
    """在控制台列出所有可切换的对话目标。"""
    print_config_summary(config)


def choose_initial_model(config: dict[str, dict[str, Any]], requested_model_id: str | None) -> str:
    """决定启动脚本后的默认对话目标，优先使用命令行指定的 model_id。"""
    if not config:
        raise SystemExit("config.json 中没有任何模型配置。")
    if requested_model_id:
        if requested_model_id not in config:
            raise SystemExit(f"config.json 中不存在模型目标：{requested_model_id}")
        return requested_model_id
    return next(iter(config))


def interactive_loop(config: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    """启动命令行多轮对话循环，并支持切换模型、列出模型和检查远程模型。"""
    model_id = choose_initial_model(config, args.model_id)
    print_config_summary(config)
    print(f"当前对话目标：{model_id}")
    print("输入问题开始对话；输入 switch-模型ID 切换；输入 models 查看目标；输入 exit / quit / q 退出。")

    while True:
        question = input("our: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        if question == "models":
            list_targets(config)
            continue
        if question == "check-models":
            print_available_remote_models(config)
            continue
        if question.startswith("switch-"):
            target = question.split("switch-", 1)[1].strip()
            if target in config:
                model_id = target
                print("已将 model_id 切换为:", model_id)
            else:
                print("无法识别的切换目标:", target)
            continue

        answer = chat_with_model(model_id, question, config, args)
        if not args.stream:
            print(f" {model_id}: {answer}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，控制配置文件、默认模型、LoRA 开关和生成参数。"""
    parser = argparse.ArgumentParser(
        description="Config-driven inference for local DeepSeek/LoRA models and remote APIs."
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json", help="Path to config.json.")
    parser.add_argument("--model-id", default=None, help="Initial model target id in config.json.")
    parser.add_argument("--question", default="", help="Run one question and exit.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.00001)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--api-timeout", type=float, default=60)
    parser.add_argument("--stream", action="store_true", help="Stream remote API output.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--check-remote-models", action="store_true", help="Check remote API model availability on startup.")

    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--use-lora", action="store_true", dest="use_lora", help="Force local targets to use lora_adapter.")
    lora_group.add_argument("--no-lora", action="store_false", dest="use_lora", help="Disable LoRA even if config has lora_adapter.")

    parser.add_argument(
        "--keep-thinking",
        action="store_false",
        dest="strip_thinking",
        help="Keep text before </think> if the model emits it.",
    )
    parser.set_defaults(strip_thinking=True, use_lora=None)
    return parser.parse_args()


def main() -> None:
    """脚本入口：读取配置、初始化对话目标，并进入单次问答或交互式问答模式。"""
    configure_stdout()
    args = parse_args()
    config_path = resolve_existing_path(args.config)
    config = load_config(config_path)
    initialize_conversations(config, args.system_prompt)

    if args.check_remote_models:
        print_available_remote_models(config)

    model_id = choose_initial_model(config, args.model_id)
    if args.question:
        answer = chat_with_model(model_id, args.question, config, args)
        print(answer)
    else:
        interactive_loop(config, args)


if __name__ == "__main__":
    main()
