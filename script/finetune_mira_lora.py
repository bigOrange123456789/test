"""MIRA LoRA 微调的统一入口。

DeepSeek 文本模型与 Qwen3-VL 视觉语言模型的数据管线差异较大，不适合把内部
实现合并成一个充满模型判断的脚本。本入口采用“统一配置、独立后端”的结构：

* ``DeepSeek-Model`` 由 ``lib/finetune_deepseek_mira_lora.py`` 后端执行；
* ``Qwen3-VL-2B-Instruct`` 由 ``lib/finetune_qwen3_vl_lora.py`` 后端执行。

``DeepSeek-Model`` 保留为配置中的模型选择标识，实际默认权重目录为项目根目录的
``DeepSeek-R1-Distill-Qwen-1.5B``；选择标识与磁盘上的文件夹名称分别维护。

因为最终调用的是原后端的 ``main(argv)``，未重新实现训练过程，所以相同参数下
的数据选择、提示词、token/图片处理、随机种子、LoRA 层、优化器和输出均与直接
运行原脚本一致。

默认读取同目录的 ``finetune_mira_lora.json``：

    python script/finetune_mira_lora.py --check-config
    python script/finetune_mira_lora.py

也可以在 JSON 参数之后追加临时命令行覆盖。例如 JSON 中配置了 epochs=1，下面
的命令会把最终值覆盖为 2：

    python script/finetune_mira_lora.py -- --epochs 2

JSON 结构：

    {
      "model": "DeepSeek-Model",
      "model_arguments": {
        "DeepSeek-Model": {"epochs": 1, "device": "auto"},
        "Qwen3-VL-2B-Instruct": {"epochs": 1, "device": "auto"}
      }
    }

参数名使用后端命令行参数的下划线形式，例如 ``gradient_accumulation_steps``。
未写入 JSON 的参数继续使用原后端默认值。布尔值会正确转换为 ``--check-env``、
``--dry-run``、``--no-gradient-checkpointing`` 等开关。

默认从项目 ``output/mira_split_ids.json`` 读取训练 ID，并分别将参数保存到
``output/deepseek_mira_lora_adapter`` 或 ``output/qwen3_vl_2b_lora_adapter``。
JSON 中的相对路径以配置文件所在目录为基准；临时命令行路径仍以当前工作目录
为基准。可在 JSON 中修改 ``split_manifest`` 和 ``output_dir`` 指定其他位置。

Qwen 配置可设置 ``on_existing_output: "new"``（默认）：若输出目录已有结果，
自动另存到相邻的 ``原目录名_run_年月日_时分秒``，保留之前的微调参数；
``"error"`` 则停止并提示修改目录。断点续训继续使用原输出目录。训练开始和结束
都会打印实际保存位置；评估新结果时，将评估 JSON 的 ``pathLora`` 指向该位置。

DeepSeek 配置可设置 ``incomplete_samples: "skip"``（默认），跳过原始问题或答案
缺失的问答并输出数量；``"error"`` 则遇缺失即停止。不会从测试集补齐，也不会用
visual_evidence 代替答案。训练开始前保存 ``data_selection.json`` 审计清单，完成后
``training_metadata.json`` 记录实际训练编号及所有跳过编号。清单中的无效 ID、损坏
CSV/JSON 仍报错；该选项不改变 Qwen 后端的现有处理。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "finetune_mira_lora.json"
BACKENDS = {
    "DeepSeek-Model": "lib.finetune_deepseek_mira_lora",
    "Qwen3-VL-2B-Instruct": "lib.finetune_qwen3_vl_lora",
}


def load_config(path: Path) -> dict[str, Any]:
    """读取并验证统一微调配置，不导入模型或训练依赖。"""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"微调配置文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("微调配置必须是 JSON 对象。")
    unknown = set(payload) - {"model", "model_arguments"}
    if unknown:
        raise ValueError(f"微调配置包含未知字段：{sorted(unknown)}")
    model = payload.get("model")
    if model not in BACKENDS:
        raise ValueError(f"model 只能是：{', '.join(BACKENDS)}")
    profiles = payload.get("model_arguments", {})
    if not isinstance(profiles, dict):
        raise ValueError("model_arguments 必须是 JSON 对象。")
    invalid_profiles = set(profiles) - set(BACKENDS)
    if invalid_profiles:
        raise ValueError(f"model_arguments 包含未知模型：{sorted(invalid_profiles)}")
    for name, arguments in profiles.items():
        if not isinstance(arguments, dict):
            raise ValueError(f"model_arguments.{name} 必须是 JSON 对象。")
    return payload


def import_backend(model: str):
    """导入所选后端；兼容作为模块或直接脚本运行。"""
    module_name = BACKENDS[model]
    if __package__:
        return importlib.import_module(f"{__package__}.{module_name}")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module(module_name)


def parser_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """建立参数名到 argparse 动作的映射，防止 JSON 悄悄传入无效参数。"""
    return {action.dest: action for action in parser._actions
            if action.dest not in {"help", argparse.SUPPRESS}}


def option_for(action: argparse.Action) -> str:
    """优先使用清晰的长选项名。"""
    options = [value for value in action.option_strings if value.startswith("--")]
    if not options:
        raise ValueError(f"参数 {action.dest} 没有可用的命令行选项。")
    return options[0]


def arguments_to_argv(parser: argparse.ArgumentParser, values: dict[str, Any]) -> list[str]:
    """将 JSON 参数转换为原后端 argv，并保持 argparse 的原始校验行为。"""
    actions = parser_actions(parser)
    unknown = set(values) - set(actions)
    if unknown:
        raise ValueError(f"所选模型不支持这些参数：{sorted(unknown)}")
    argv: list[str] = []
    for name, value in values.items():
        action = actions[name]
        option = option_for(action)
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(f"{name} 必须是 JSON 布尔值。")
            if value:
                argv.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not isinstance(value, bool):
                raise ValueError(f"{name} 必须是 JSON 布尔值。")
            if not value:
                argv.append(option)
        else:
            if value is None or isinstance(value, (dict, list, bool)):
                raise ValueError(f"{name} 必须是字符串或数字，不能是 {type(value).__name__}。")
            argv.extend((option, str(value)))
    return argv


def resolve_backend_argv(
    payload: dict[str, Any], extra_args: list[str], config_dir: Path | None = None,
) -> tuple[Any, list[str]]:
    """选择后端并组合 JSON 参数与临时命令行参数。"""
    model = payload["model"]
    backend = import_backend(model)
    profiles = payload.get("model_arguments", {})
    configured = dict(profiles.get(model, {}))
    parser = backend.build_parser()
    actions = parser_actions(parser)
    config_dir = (config_dir or SCRIPT_DIR).expanduser().resolve()
    # 配置文件不依赖启动位置；命令行覆盖在路径转换后追加，保留其原有语义。
    for name, value in configured.items():
        action = actions.get(name)
        if isinstance(value, str) and action is not None and (
            action.type is Path or name == "resume_from_checkpoint"
        ):
            path = Path(value).expanduser()
            configured[name] = str((path if path.is_absolute() else config_dir / path).resolve())
    argv = arguments_to_argv(parser, configured)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    argv.extend(extra_args)
    return backend, argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="统一微调 JSON；默认使用脚本同目录的同名 JSON。")
    parser.add_argument("--check-config", action="store_true",
                        help="只检查 JSON 和后端参数，不加载数据或模型。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取统一配置，并把执行完整委托给原模型后端。"""
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    try:
        payload = load_config(args.config)
        backend, backend_argv = resolve_backend_argv(
            payload, extra_args, args.config.expanduser().resolve().parent,
        )
        # 先解析一次以验证 JSON 与临时参数组合；真正执行时后端会再次解析同一 argv。
        backend.build_parser().parse_args(backend_argv)
        print(f"微调模型：{payload['model']}", flush=True)
        backend_path = BACKENDS[payload["model"]].replace(".", "/") + ".py"
        print(f"执行后端：{backend_path}", flush=True)
        if args.check_config:
            print("配置检查通过；未加载数据、模型，也未写入训练结果。", flush=True)
            return 0
        return int(backend.main(backend_argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
