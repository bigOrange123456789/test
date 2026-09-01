# -*- coding: utf-8 -*-
"""
Run inference with the local DeepSeek model, optionally with the fine-tuned
LoRA adapter.

Default paths:
    base model:    DeepSeek-Model
    LoRA adapter:  DeepSeek-Model/huatuo_lora_adapter

Examples:
    python inferenceLoRa.py --question "孕妇甲状腺激素低意味着什么？"
    python inferenceLoRa.py --no-lora --question "孕妇甲状腺激素低意味着什么？"
    python inferenceLoRa.py --use-lora --question "孕妇甲状腺激素低意味着什么？"
    python inferenceLoRa.py
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = "你是一名谨慎的中文医疗问答助手。请基于用户问题给出准确、清晰、简洁的医学科普回答。"#"请用简洁、简短的语言回答用户的问题"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_version(version: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(number) for number in numbers[:3])


def version_at_least(installed: str, minimum: str) -> bool:
    installed_tuple = parse_version(installed)
    minimum_tuple = parse_version(minimum)
    max_len = max(len(installed_tuple), len(minimum_tuple))
    installed_tuple += (0,) * (max_len - len(installed_tuple))
    minimum_tuple += (0,) * (max_len - len(minimum_tuple))
    return installed_tuple >= minimum_tuple


def check_dependencies(use_lora: bool) -> None:
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
        print("当前环境缺少推理依赖：")
        for problem in problems:
            print(f"  - {problem}")
        print()
        print("可以先在可用环境里安装：")
        print("  pip install -r DeepSeek-Model/requirements-lora.txt")
        raise SystemExit(1)


def model_dtype_arg(dtype):
    transformers_version = metadata.version("transformers")
    if version_at_least(transformers_version, "5.0.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_model(args):
    check_dependencies(use_lora=args.use_lora)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = args.base_model.resolve()
    lora_adapter = args.lora_adapter.resolve()
    if not base_model.exists():
        raise SystemExit(f"Base model path does not exist: {base_model}")
    if args.use_lora and not lora_adapter.exists():
        raise SystemExit(f"LoRA adapter path does not exist: {lora_adapter}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer_path = lora_adapter if args.use_lora else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_kwargs.update(model_dtype_arg(dtype))
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if args.use_lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_adapter)

    model.to(device)
    model.eval()

    print(f"Loaded base model: {base_model}")
    if args.use_lora:
        print(f"Loaded LoRA adapter: {lora_adapter}")
        print("Inference mode: base model + LoRA")
    else:
        print("Loaded LoRA adapter: disabled")
        print("Inference mode: base model only")
    print(f"Device: {device}")
    return model, tokenizer, device


def build_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"system: {system_prompt}\nuser: {question}\nassistant:"


def remove_thinking_text(text: str) -> str:
    return re.sub(r".*?</think>\s*", "", text, flags=re.DOTALL).strip()


def generate_answer(model, tokenizer, device: str, args, question: str) -> str:
    import torch

    prompt = build_prompt(tokenizer, args.system_prompt, question)
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


def interactive_loop(model, tokenizer, device: str, args) -> None:
    assistant_name = "LoRA" if args.use_lora else "Base"
    print("输入问题开始对话；输入 exit / quit / q 退出。")
    while True:
        question = input("our: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        answer = generate_answer(model, tokenizer, device, args, question)
        print(f"{assistant_name}: {answer}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Inference script for DeepSeek-Model with Huatuo LoRA adapter."
    )
    parser.add_argument("--base-model", type=Path, default=project_root / "DeepSeek-Model")
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        # default=project_root / "DeepSeek-Model" / "huatuo_lora_adapter",
        default=project_root / "DeepSeek-Model" / "huatuo_lora_single_pair_adapter",
    )
    parser.add_argument("--question", default="", help="Run one question and exit.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    # parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.00001)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora",
        action="store_true",
        dest="use_lora",
        help="Load the LoRA adapter on top of the base model. This is the default.",
    )
    lora_group.add_argument(
        "--no-lora",
        action="store_false",
        dest="use_lora",
        help="Run the base DeepSeek model without the LoRA adapter.",
    )
    parser.add_argument(
        "--keep-thinking",
        action="store_false",
        dest="strip_thinking",
        help="Keep text before </think> if the model emits it.",
    )
    parser.set_defaults(strip_thinking=True, use_lora=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_model(args)

    if args.question:
        answer = generate_answer(model, tokenizer, device, args, args.question)
        print(answer)
    else:
        interactive_loop(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
