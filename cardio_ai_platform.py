#!/usr/bin/env python3
"""Start the cardiovascular AI research front-end prototype locally."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parent
APP_INDEX = ROOT_DIR / "cardio_ai_platform" / "index.html"
INFERENCE_SCRIPT = ROOT_DIR / "inferenceValid" / "inferenceLoRa.py"
INFERENCE_CONFIG = ROOT_DIR / "inferenceValid" / "config.json"

MODEL_ID_ALIASES = {
    "deepseek-original": "myDeepSeek",
    "deepseek-finetuned": "myDeepSeek_LoRA",
    "remote-api-1": "deepseek",
    "remote-api-2": "tongyi",
}
DEFAULT_PRELOAD_LOCAL_MODELS = "deepseek-original"

REPORT_FIELD_ORDER = ("diagnosis", "findings", "analysis", "advice")
REPORT_FIELD_ALIASES = {
    "diagnosis": ("diagnosis", "综合诊断结果", "综合诊断", "诊断结果", "result"),
    "findings": ("findings", "关键发现", "key_findings", "keyFindings"),
    "analysis": ("analysis", "AI分析", "AI 分析", "ai_analysis", "aiAnalysis"),
    "advice": ("advice", "advise", "临床建议", "recommendation", "recommendations"),
}

# ANALYSIS_SYSTEM_PROMPT = (
#     "你是一名谨慎的中文心血管医学病例分析助手。"
#     "你只做科研演示和临床辅助分析，不替代医生诊断。"
#     "禁止输出思考过程、解释过程、步骤说明或 Markdown。"
#     "你的回复第一个字符必须是 {，最后一个字符必须是 }。"
#     'JSON 必须包含四个字符串字段："diagnosis"、"findings"、"analysis"、"advice"。'
# )
ANALYSIS_SYSTEM_PROMPT = (
    "你是一名谨慎的中文心血管医学病例分析助手。"
    "禁止输出思考过程、解释过程、步骤说明或 Markdown。"
    "你的回复第一个字符必须是 {，最后一个字符必须是 }。"
    'JSON 必须包含四个字符串字段："diagnosis"、"findings"、"analysis"、"advice"。'
)

_INFERENCE_MODULE = None
_INFERENCE_CONFIG_CACHE = None
_MODEL_CALL_LOCK = threading.Lock()

LOCAL_DEPENDENCIES = {
    "torch": "2.1.0",
    "transformers": "4.44.0",
    "safetensors": "0.4.3",
}


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


def _load_inference_module():
    global _INFERENCE_MODULE
    if _INFERENCE_MODULE is not None:
        return _INFERENCE_MODULE
    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(f"DeepSeek inference script not found: {INFERENCE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("cardio_inference_lora", INFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import inference script: {INFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _INFERENCE_MODULE = module
    return module


def _load_inference_config() -> dict:
    global _INFERENCE_CONFIG_CACHE
    if _INFERENCE_CONFIG_CACHE is not None:
        return _INFERENCE_CONFIG_CACHE
    if not INFERENCE_CONFIG.exists():
        raise FileNotFoundError(f"DeepSeek config not found: {INFERENCE_CONFIG}")
    module = _load_inference_module()
    _INFERENCE_CONFIG_CACHE = module.load_config(INFERENCE_CONFIG)
    return _INFERENCE_CONFIG_CACHE


def _analysis_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_new_tokens=768,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.1,
        device="auto",
        api_timeout=90,
        trust_remote_code=False,
        use_lora=None,
        stream=False,
        strip_thinking=True,
    )


def _parse_version(version: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(number) for number in numbers[:3])


def _version_at_least(installed: str, minimum: str) -> bool:
    installed_tuple = _parse_version(installed)
    minimum_tuple = _parse_version(minimum)
    max_len = max(len(installed_tuple), len(minimum_tuple))
    installed_tuple += (0,) * (max_len - len(installed_tuple))
    minimum_tuple += (0,) * (max_len - len(minimum_tuple))
    return installed_tuple >= minimum_tuple


def _dependency_state(package: str, minimum: str) -> tuple[str, str | None]:
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        return f"{package} 未安装，需要 >= {minimum}", None
    if not _version_at_least(installed, minimum):
        return f"{package} 当前版本 {installed}，需要 >= {minimum}", installed
    return "", installed


def _local_dependency_problems(use_lora: bool) -> list[str]:
    required = dict(LOCAL_DEPENDENCIES)
    if use_lora:
        required["peft"] = "0.11.0"
    problems = []
    for package, minimum in required.items():
        problem, _installed = _dependency_state(package, minimum)
        if problem:
            problems.append(problem)
    return problems


def _ensure_local_dependencies(model_id: str, use_lora: bool) -> None:
    problems = _local_dependency_problems(use_lora)
    if not problems:
        return

    details = "；".join(problems)
    message = (
        f"本地模型 {model_id} 的推理依赖未就绪：{details}。"
        "请在当前虚拟环境中执行：pip install -r DeepSeek-Model/requirements-lora.txt"
    )
    print(f"[CardioAI] {message}", flush=True)
    raise RuntimeError(message)


def _preload_target_model_ids(preload_spec: str, config: dict) -> list[str]:
    normalized = (preload_spec or "").strip()
    if normalized.lower() in {"", "0", "false", "none", "off", "skip", "disabled"}:
        return []
    if normalized.lower() == "all":
        raw_targets = list(config.keys())
    else:
        raw_targets = [part.strip() for part in re.split(r"[,，]", normalized) if part.strip()]

    model_ids = []
    for raw_target in raw_targets:
        model_id = MODEL_ID_ALIASES.get(raw_target, raw_target)
        if model_id not in config:
            print(f"[CardioAI] 跳过未知预加载模型：{raw_target}", flush=True)
            continue
        if model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def _preload_local_models(preload_spec: str) -> None:
    config = _load_inference_config()
    module = _load_inference_module()
    args = _analysis_args()
    model_ids = _preload_target_model_ids(preload_spec, config)
    if not model_ids:
        print("[CardioAI] 本地模型预加载已关闭", flush=True)
        return

    print(f"[CardioAI] 本地模型预加载目标：{', '.join(model_ids)}", flush=True)
    for model_id in model_ids:
        cfg = copy.deepcopy(config[model_id])
        if not module.is_local_config(model_id, cfg):
            print(f"[CardioAI] 跳过远程模型预加载：{model_id}", flush=True)
            continue

        lora_adapter = module.choose_lora_adapter(cfg, args)
        started_at = time.perf_counter()
        print(
            f"[CardioAI] 本地模型预加载开始：model_id={model_id} "
            f"use_lora={lora_adapter is not None}",
            flush=True,
        )
        try:
            _ensure_local_dependencies(model_id, use_lora=lora_adapter is not None)
            with _MODEL_CALL_LOCK:
                module.load_local_target(model_id, cfg, args)
        except Exception as error:
            print(f"[CardioAI] 本地模型预加载失败：model_id={model_id} error={error}", flush=True)
            traceback.print_exc()
            continue
        print(
            f"[CardioAI] 本地模型预加载完成：model_id={model_id} "
            f"elapsed={time.perf_counter() - started_at:.2f}s",
            flush=True,
        )


def _resolve_model_id(frontend_model_id: str | None) -> str:
    config = _load_inference_config()
    requested = (frontend_model_id or "deepseek-original").strip()
    model_id = MODEL_ID_ALIASES.get(requested, requested)
    if model_id not in config:
        available = ", ".join(sorted(config))
        raise ValueError(f"未知模型：{requested}；后端可用模型：{available}")
    return model_id


def _is_local_model(model_id: str, cfg: dict) -> bool:
    module = _load_inference_module()
    return module.is_local_config(model_id, cfg)


def _analysis_log(message: str, request_id: str | None = None) -> None:
    request_label = f" request_id={request_id}" if request_id else ""
    print(f"[CardioAI] /api/analyze{request_label} {message}", flush=True)


def _preview_text(text: str, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _field_lengths(report: dict[str, str]) -> str:
    return ", ".join(f"{field}={len(str(report.get(field) or ''))}" for field in REPORT_FIELD_ORDER)


def _build_case_prompt(inputs: dict) -> str:
    age = str(inputs.get("age") or "").strip()
    sex = str(inputs.get("sex") or "").strip()
    bmi = str(inputs.get("bmi") or "").strip()
    blood_pressure = str(inputs.get("bloodPressure") or "").strip()
    heart_rate = str(inputs.get("heartRate") or "").strip()
    family_history = str(inputs.get("familyHistory") or "").strip()
    case_input = str(inputs.get("caseInput") or "").strip()
    symptoms = str(inputs.get("symptoms") or "").strip()
    exams = str(inputs.get("exams") or "").strip()
    diagnosis_report = str(inputs.get("diagnosisReport") or "").strip()
    return f"""
请基于以下心血管病例材料完成结构化病例分析。

输出要求：
1. 只返回一个合法 JSON 对象。
2. 不要使用 Markdown 代码块。
3. 四个字段必须全部存在，字段值必须是中文字符串。
4. findings 可以用换行分隔多个关键发现。
5. advice 需要包含“需由临床医生结合实际检查复核”的安全提醒。
6. 不要先写“我现在需要”“首先”“总结”等分析过程，直接输出 JSON。
7. 回复第一个字符必须是 {{，最后一个字符必须是 }}。

患者结构化信息：
年龄：{age or "未填写"}
性别：{sex or "未填写"}
BMI：{bmi or "未填写"}
血压：{blood_pressure or "未填写"}
心率：{heart_rate or "未填写"}
家族史：{family_history or "未填写"}

病例输入：
{case_input}

临床症状：
{symptoms}

检查结果：
{exams}

病例诊断报告：
{diagnosis_report}

请严格按以下结构返回：
{{"diagnosis":"...","findings":"...","analysis":"...","advice":"..."}}
""".strip()


def _call_remote_chat_completion(model_id: str, cfg: dict, args: SimpleNamespace) -> str:
    api_key = cfg.get("api_key")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    configured_model = cfg.get("model")
    if not api_key or not base_url or not configured_model:
        raise RuntimeError(f"远程模型 {model_id} 缺少 api_key、base_url 或 model 配置")

    endpoint = f"{base_url}/chat/completions"
    body = {
        "model": configured_model,
        "messages": cfg["messages"],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "stream": False,
    }
    extra_body = cfg.get("extra_body") or {}
    if isinstance(extra_body, dict):
        body.update(extra_body)

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.api_timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"远程模型 {model_id} 请求失败：HTTP {error.code} {error.reason}；{detail[:600]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"远程模型 {model_id} 网络请求失败：{error.reason}") from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"远程模型 {model_id} 返回内容不是 JSON：{raw[:600]}") from error

    if "error" in payload:
        raise RuntimeError(f"远程模型 {model_id} 返回错误：{payload['error']}")

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"远程模型 {model_id} 没有返回 choices：{raw[:600]}")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        content = choices[0].get("text")
    if content is None:
        raise RuntimeError(f"远程模型 {model_id} 没有返回 message.content：{raw[:600]}")
    return str(content).strip()


def _stringify_report_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_report_fields(payload: dict) -> dict[str, str]:
    source = payload
    for wrapper in ("data", "result", "delta"):
        nested = payload.get(wrapper)
        if isinstance(nested, dict):
            source = nested
            break

    report = {}
    for field, aliases in REPORT_FIELD_ALIASES.items():
        report[field] = ""
        for alias in aliases:
            if alias in source:
                report[field] = _stringify_report_value(source[alias]).strip()
                break
    return report


def _json_candidates(text: str) -> list[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(cleaned[index : index + end])
    return candidates


def _extract_heading_section(text: str, labels: tuple[str, ...]) -> str:
    heading_pattern = "|".join(
        re.escape(label)
        for aliases in REPORT_FIELD_ALIASES.values()
        for label in aliases
        if re.search(r"[\u4e00-\u9fff]", label)
    )
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[：:]\s*(.*?)(?=\n\s*(?:{heading_pattern})\s*[：:]|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _split_medical_items(text: str) -> list[str]:
    items = []
    for part in re.split(r"[\n；;、，,]+", text):
        item = part.strip(" -•\t\r\n")
        if item:
            items.append(item)
    return items


def _fallback_report_from_inputs(inputs: dict, answer: str) -> dict[str, str]:
    age = str(inputs.get("age") or "").strip()
    sex = str(inputs.get("sex") or "").strip()
    bmi = str(inputs.get("bmi") or "").strip()
    blood_pressure = str(inputs.get("bloodPressure") or "").strip()
    heart_rate = str(inputs.get("heartRate") or "").strip()
    family_history = str(inputs.get("familyHistory") or "").strip()
    case_input = str(inputs.get("caseInput") or "").strip()
    symptoms = str(inputs.get("symptoms") or "").strip()
    exams = str(inputs.get("exams") or "").strip()
    diagnosis_report = str(inputs.get("diagnosisReport") or "").strip()
    profile_text = "\n".join(
        [
            f"年龄：{age}",
            f"性别：{sex}",
            f"BMI：{bmi}",
            f"血压：{blood_pressure}",
            f"心率：{heart_rate}",
            f"家族史：{family_history}",
        ]
    )
    all_text = "\n".join([profile_text, case_input, symptoms, exams, diagnosis_report])

    findings = []
    profile_items = [
        label
        for label in (
            f"年龄 {age}" if age else "",
            f"性别 {sex}" if sex else "",
            f"BMI {bmi}" if bmi else "",
            f"血压 {blood_pressure}" if blood_pressure else "",
            f"心率 {heart_rate}" if heart_rate else "",
            family_history if family_history else "",
        )
        if label
    ]
    if profile_items:
        findings.append(f"患者信息：{'；'.join(profile_items)}")
    for item in _split_medical_items(symptoms):
        findings.append(f"症状：{item}")
    for line in exams.splitlines():
        line = line.strip()
        if line:
            findings.append(f"检查：{line}")
    for keyword in ("高血压", "LDL-C", "ST-T", "冠脉", "钙化斑块", "胸痛", "胸闷", "气短"):
        if keyword in all_text and not any(keyword in item for item in findings):
            findings.append(f"相关线索：{keyword}")
    findings = findings[:8] or ["模型未返回标准关键发现，需结合原始病例材料复核。"]

    if any(keyword in all_text for keyword in ("冠脉", "冠心病", "ST-T", "胸痛", "LDL-C")):
        diagnosis = "疑似冠心病或冠状动脉粥样硬化相关风险，需进一步临床评估。"
    elif any(keyword in all_text for keyword in ("呼吸困难", "水肿", "BNP", "EF")):
        diagnosis = "疑似心功能异常相关风险，需进一步临床评估。"
    else:
        diagnosis = "心血管相关症状或危险因素待评估。"

    analysis = (
        "模型已返回内容，但未严格遵循四字段 JSON 格式；后端已根据病例输入整理为结构化摘要。"
        "当前材料提示存在症状、危险因素或检查异常之间的关联，应结合病史、心电图、血脂、影像学检查"
        "及医生查体进行综合判断。"
    )
    if answer.strip():
        analysis += " 原始模型输出未直接展示，以避免将非结构化思考过程呈现在临床结果区。"

    return {
        "diagnosis": diagnosis,
        "findings": "\n".join(findings),
        "analysis": analysis,
        "advice": "建议由心血管专科医生结合实际检查复核；如出现持续胸痛、呼吸困难、晕厥或症状加重，应及时就医。",
    }


def _normalize_model_answer(answer: str, inputs: dict) -> dict[str, str]:
    module = _load_inference_module()
    cleaned = module.remove_thinking_text(answer).strip()
    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        report = _extract_report_fields(parsed)
        if any(report.values()):
            return report

    report = {
        "diagnosis": _extract_heading_section(cleaned, ("综合诊断结果", "综合诊断", "诊断结果")),
        "findings": _extract_heading_section(cleaned, ("关键发现",)),
        "analysis": _extract_heading_section(cleaned, ("AI分析", "AI 分析")),
        "advice": _extract_heading_section(cleaned, ("临床建议",)),
    }
    if any(report.values()):
        return report

    return _fallback_report_from_inputs(inputs, cleaned)


def _complete_report(report: dict[str, str]) -> dict[str, str]:
    defaults = {
        "diagnosis": "暂未生成综合诊断结果。",
        "findings": "暂未生成关键发现。",
        "analysis": "暂未生成 AI 分析。",
        "advice": "暂未生成临床建议；最终判断需由临床医生结合实际检查复核。",
    }
    completed = {field: (report.get(field) or defaults[field]).strip() for field in REPORT_FIELD_ORDER}
    if "临床医生" not in completed["advice"] and "医生" not in completed["advice"]:
        completed["advice"] += " 最终判断需由临床医生结合实际检查复核。"
    return completed


def _analyze_case(payload: dict, request_id: str | None = None) -> dict[str, str]:
    frontend_model = payload.get("model")
    model_id = _resolve_model_id(payload.get("model"))
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("请求体缺少 inputs 对象")

    module = _load_inference_module()
    base_config = _load_inference_config()
    request_config = {model_id: copy.deepcopy(base_config[model_id])}
    module.initialize_conversations(request_config, ANALYSIS_SYSTEM_PROMPT)

    prompt = _build_case_prompt(inputs)
    print("prompt为:",prompt)
    args = _analysis_args()
    cfg = request_config[model_id]
    is_local = _is_local_model(model_id, cfg)
    _analysis_log(
        f"frontend_model={frontend_model!r} resolved_model={model_id!r} local={is_local}",
        request_id,
    )
    input_keys = ", ".join(sorted(str(key) for key in inputs.keys()))
    input_lengths = ", ".join(
        f"{key}={len(str(value or ''))}" for key, value in sorted(inputs.items())
    )
    _analysis_log(f"input_keys=[{input_keys}]", request_id)
    _analysis_log(f"input_lengths={{ {input_lengths} }}", request_id)
    _analysis_log(
        f"prompt_chars={len(prompt)} max_new_tokens={args.max_new_tokens} "
        f"temperature={args.temperature} stream={args.stream}",
        request_id,
    )

    started_at = time.perf_counter()
    if is_local:
        lora_adapter = module.choose_lora_adapter(cfg, args)
        _analysis_log(
            f"local_prepare use_lora={lora_adapter is not None} adapter={str(lora_adapter) if lora_adapter else 'disabled'}",
            request_id,
        )
        _ensure_local_dependencies(model_id, use_lora=lora_adapter is not None)
        try:
            with _MODEL_CALL_LOCK:
                _analysis_log("model_call begin", request_id)
                cfg["messages"].append({"role": "user", "content": prompt})
                model, tokenizer, device = module.load_local_target(model_id, cfg, args)
                answer = module.generate_local_answer(
                    model_id,
                    model,
                    tokenizer,
                    device,
                    cfg["messages"],
                    args,
                )
                cfg["messages"].append({"role": "assistant", "content": answer})
        except SystemExit as error:
            raise RuntimeError(f"本地模型 {model_id} 调用提前退出：{error.code}") from error
    else:
        cfg["messages"].append({"role": "user", "content": prompt})
        _analysis_log(
            f"remote_call begin base_url={str(cfg.get('base_url') or '').rstrip('/')} model={cfg.get('model')}",
            request_id,
        )
        answer = _call_remote_chat_completion(model_id, cfg, args)

    elapsed = time.perf_counter() - started_at
    _analysis_log(
        f"model_call end elapsed={elapsed:.2f}s answer_chars={len(answer)} "
        f"answer_preview={_preview_text(answer)!r}",
        request_id,
    )
    if answer.startswith("Failed to call model"):
        _analysis_log(f"model_call returned failure={_preview_text(answer, 500)!r}", request_id)
        raise RuntimeError(answer)

    report = _complete_report(_normalize_model_answer(answer, inputs))
    _analysis_log(f"normalized_report field_lengths={{ {_field_lengths(report)} }}", request_id)
    return report


def _error_report(error: Exception) -> dict[str, str]:
    message = str(error).replace("\n", " ").strip()
    message = re.sub(r"\s+", " ", message)
    return {
        "diagnosis": "后端分析失败。",
        "findings": "未生成关键发现。",
        "analysis": message[:800] or "未知错误。",
        "advice": "请检查 inferenceValid/config.json、模型路径/API Key、依赖安装和控制台日志；最终判断需由临床医生结合实际检查复核。",
    }


class CardioAIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in {"", "/"}:
            self.path = "/cardio_ai_platform/index.html"
        return super().do_GET()

    def do_OPTIONS(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/api/analyze":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            return
        self.send_error(404, "Unknown endpoint")

    def do_POST(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path != "/api/analyze":
            self.send_error(404, "Unknown endpoint")
            return

        request_id = f"{int(time.time() * 1000) % 1000000:06d}-{threading.get_ident()}"
        request_started_at = time.perf_counter()
        _analysis_log(
            f"received content_length={self.headers.get('Content-Length', '0')} "
            f"accept={self.headers.get('Accept', '')!r}",
            request_id,
        )
        try:
            payload = self._read_json_body()
        except ValueError as error:
            _analysis_log(f"bad_request error={error}", request_id)
            self._send_json({"error": str(error)}, status=400)
            return

        input_payload = payload.get("inputs")
        input_count = len(input_payload) if isinstance(input_payload, dict) else 0
        _analysis_log(
            f"payload model={payload.get('model')!r} top_keys={sorted(payload.keys())} "
            f"inputs_type={type(input_payload).__name__} inputs_count={input_count}",
            request_id,
        )
        self._send_sse_headers()
        self._write_sse({"status": "started", "request_id": request_id})
        try:
            report = _analyze_case(payload, request_id=request_id)
        except SystemExit as error:
            message = f"模型调用提前退出：{error.code}；请查看控制台中的依赖检查或模型加载日志。"
            _analysis_log(f"failed: {message}", request_id)
            report = _error_report(RuntimeError(message))
        except Exception as error:
            _analysis_log(f"failed: {error}", request_id)
            report = _error_report(error)

        _analysis_log(f"sse_report field_lengths={{ {_field_lengths(report)} }}", request_id)
        for field in REPORT_FIELD_ORDER:
            self._write_sse({"delta": {field: report[field]}})
            _analysis_log(f"sse_delta_sent field={field} chars={len(report[field])}", request_id)
            time.sleep(0.04)
        self._write_sse(report)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        _analysis_log(
            f"completed elapsed={time.perf_counter() - request_started_at:.2f}s",
            request_id,
        )

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0:
            raise ValueError("Empty request body")
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _write_sse(self, payload: dict):
        body = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {body}\n\n".encode("utf-8"))
        self.wfile.flush()

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    parser.add_argument(
        "--preload-local-models",
        default=DEFAULT_PRELOAD_LOCAL_MODELS,
        help=(
            "Comma-separated frontend/backend local model ids to preload before serving. "
            "Use 'all' to preload every local target, or 'none' to disable."
        ),
    )
    args = parser.parse_args()

    if not APP_INDEX.exists():
        raise FileNotFoundError(
            f"Front-end entry not found: {APP_INDEX}. Please keep cardio_ai_platform next to html.py."
        )

    print("")
    print("正在初始化心血管疾病人工智能诊疗与知识融合平台...")
    _preload_local_models(args.preload_local_models)

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
