"""从最终答案计算客观题准确率；不把解释、推理或 JSON 格式相似度当作正确性。"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


QUESTION_TYPES = {"single_choice", "multiple_choice", "closed_ended", "open_ended"}
_LABELLED = re.compile(r"^\(?([A-Z])\)?[.、:：)\-]\s*(.+)$", re.I | re.S)
_ANSWER_MARKER = re.compile(
    r"(?:^|\n|(?<=[.!?。]))\s*(?:#{1,6}\s*)?(?:therefore[,，]?\s*)?"
    r"(?:based\s+on[^\n]+?,\s*)?"
    r"(?:the\s+)?(?:(?:final|correct|most\s+appropriate)\s+)?"
    r"(?:answers?|choices?|selected\s+options?)\s*(?::|are\b|is\b)\s*[:：]?\s*"
    r"|(?:^|\n)\s*(?:最终答案|正确答案|答案|选择|所选选项)\s*[:：]\s*",
    re.I,
)


def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(" \t\r\n.。!！;；")


def _without_thinking(value: Any) -> tuple[str, bool]:
    """只保留完整思考块之外的最终回答；思考块未闭合时明确判为无有效答案。"""
    if not isinstance(value, str):
        return "", False
    text = value.strip()
    while re.search(r"<think\b[^>]*>", text, flags=re.I):
        opening = re.search(r"<think\b[^>]*>", text, flags=re.I)
        closing = re.search(r"</think\s*>", text[opening.end():], flags=re.I)
        if closing is None:
            return "", False
        end = opening.end() + closing.end()
        text = text[:opening.start()] + text[end:]
    if re.search(r"</?think\b", text, flags=re.I):
        return "", False
    return text.strip(), bool(text.strip())


def _decode(value: Any) -> tuple[Any, bool]:
    """仅解码整个 JSON 或完整代码围栏，不从解释段落中搜寻 JSON 片段。"""
    if not isinstance(value, str):
        return value, True
    text = value.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.I | re.S)
        if match is None:
            return text, False
        text = match.group(1).strip()
    if text.startswith(("{", "[", '"')):
        try:
            return json.loads(text), True
        except (ValueError, TypeError):
            return text, False
    return text, True


def _answer_field(value: Any, question_type: str) -> tuple[Any, bool]:
    """只选答案字段，explanation、visual_evidence 不参与客观题或文本相似度。"""
    value, valid = _decode(value)
    if not valid:
        return value, False
    if not isinstance(value, dict):
        return value, True
    preferred = {"single_choice": ("correct_option", "correct_options", "answer", "text"),
                 "multiple_choice": ("correct_options", "correct_option", "answer", "text"),
                 "closed_ended": ("text", "answer"),
                 "open_ended": ("text", "answer")}.get(question_type, ("text", "answer"))
    for field in preferred:
        if field in value:
            result = value[field]
            if isinstance(result, dict):
                return _answer_field(result, question_type)
            return result, True
    return "", False


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value)
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _semantic_text(value: Any, question_type: str) -> str:
    """语义裁判保留答案与解释，不把影像标注或隐藏推理作为回答内容。"""
    value, valid = _decode(value)
    if not valid:
        return ""
    if not isinstance(value, dict):
        return _text(value)
    pieces = []
    for key in ("correct_option", "correct_options", "answer", "text", "explanation", "rationale"):
        if key in value:
            text = _semantic_text(value[key], question_type)
            if text and text not in pieces:
                pieces.append(text)
    return "\n".join(pieces)


def _options(sample: dict) -> dict[str, str]:
    """将真实 MIRA 的字符串列表和 label→文本对象都映射为标准选项。"""
    options = sample.get("options")
    if options is None and "\nOptions: " in sample.get("question", ""):
        try:
            options = json.loads(sample["question"].rsplit("\nOptions: ", 1)[1])
        except (TypeError, ValueError):
            return {}
    result = {}
    if isinstance(options, dict):
        items = list(options.items())
    elif isinstance(options, list) and len(options) <= 26:
        items = [(chr(ord("A") + index), value) for index, value in enumerate(options)]
    else:
        return {}
    for raw_label, value in items:
        if not isinstance(raw_label, str) or not isinstance(value, str) or not value.strip():
            return {}
        label = raw_label.strip().upper().strip("().、:：")
        if not re.fullmatch(r"[A-Z]", label):
            return {}
        match = _LABELLED.fullmatch(value.strip())
        if match:
            explicit, text = match.groups()
            if isinstance(options, dict) and explicit.upper() != label:
                return {}
            label, value = explicit.upper(), text
        if label in result:
            return {}
        result[label] = value.strip()
    return result


def _plain_format(value: str) -> str:
    # Markdown 加粗、行内代码和无序列表符号只是显示格式。
    return re.sub(r"(?m)^\s*[-*+]\s+", "", value.replace("**", "").replace("__", "").replace("`", "")).strip()


def _answer_candidates(value: str) -> list[str]:
    """只解析完整直接回答或明确的答案标记，不在推理正文中捡选项字母。"""
    plain = _plain_format(value)
    markers = list(_ANSWER_MARKER.finditer(plain))
    if markers:
        # 多次明确声明不同答案也不随意取最后一条，交由调用方检查一致性。
        candidates = []
        for index, match in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(plain)
            tail = plain[match.end():end].lstrip()
            first_block = re.split(r"\n\s*\n|\n\s*(?:Explanation|Rationale|解释|理由)\s*[:：]", tail,
                                   maxsplit=1, flags=re.I)[0]
            candidates.append(first_block.strip())
        return candidates
    return [plain]


def _option_label(value: Any, options: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    text = _plain_format(value).strip().strip("。")
    short = re.fullmatch(r"(?:option\s+)?\(?([A-Z])\)?[.!。]?", text, flags=re.I)
    if short:
        label = short.group(1).upper()
        return label if label in options else None
    labelled = _LABELLED.fullmatch(text)
    if labelled:
        label, description = labelled.groups()
        label = label.upper()
        # 标签后带的文字也必须吻合，不能把“A. 不是答案”误当作选择 A。
        if label in options and _clean(description) == _clean(options[label]):
            return label
        return None
    parenthesized = re.fullmatch(r"([A-Z])\s*\((.+)\)[.!。]?", text, flags=re.I | re.S)
    if parenthesized:
        label, description = parenthesized.groups()
        label = label.upper()
        return label if label in options and _clean(description) == _clean(options[label]) else None
    labels = [label for label, content in options.items() if _clean(text) == _clean(content)]
    return labels[0] if len(labels) == 1 else None


def _natural_choice_candidates(value: str, options: dict[str, str]) -> list[str]:
    """识别独立完整选项段，或明确陈述“... is A. 选项全文”；不抓取推理中的零散字母。"""
    plain = _plain_format(value)
    blocks = re.split(r"\n\s*\n", plain)
    found = []
    for index, block in enumerate(blocks):
        block = block.strip()
        if _option_label(block, options) is not None:
            previous = blocks[index - 1].strip() if index else ""
            # 候选清单与被否定/考虑过的选项不属于最终回答。
            if re.search(r"(?:candidates?|alternatives?|options?|consider(?:ed|ing)?|reject(?:ed)?|"
                         r"not\s+(?:the\s+)?answer|候选|考虑|排除)\s*[:：]?\s*$", previous, re.I):
                continue
            found.append(block)
            continue
        # 只接受陈述段的第一句/第一行末尾恰好为完整选项的形式。
        # 例如“The primary risk ... is A. Stroke.”；不将“A was discussed”视为答案。
        first = block.split("\n", 1)[0].strip()
        for match in re.finditer(r"\bis\s*(?::\s*)?", first, re.I):
            before, after = first[:match.start()], first[match.end():].strip()
            if not re.match(r"(?:the\s|based\s+on\s|therefore\b)", before, re.I):
                continue
            if re.search(r"\b(?:not|maybe|perhaps|possibly|considered|rejected|wrong|incorrect|if|whether)\b", before, re.I):
                continue
            if _option_label(after, options) is not None:
                found.append(after)
    return found


def _explicit_choice_label(value: str, options: dict[str, str]) -> str | None:
    """仅用于明确 Answer 标记后的单个标签与同行解释，不从一般推理句抽取字母。"""
    text = _plain_format(value)
    match = re.match(r"^(?:option\s+)?\(?([A-Z])\)?(?=$|[\s.,，。:：;；\-])", text, re.I)
    if match is None or match.group(1).upper() not in options:
        return None
    label = match.group(1).upper()
    rest = text[match.end():].lstrip(" \t.,，。:：;；-")
    if not rest:
        return label
    if re.search(r"\b(?:maybe|perhaps|possibly|uncertain)\b|不确定|也许|可能", rest, re.I):
        return None
    # 允许原选项全文后接解释，但不能把 C. CT 这类标签与内容矛盾的回答算作 C。
    option = options[label].strip().rstrip(".。")
    description = re.match(re.escape(option) + r"(?=$|[\s.,，。:：;；])", rest, re.I)
    if description:
        rest = rest[description.end():].lstrip(" \t.,，。:：;；-")
        if not rest:
            return label
        if re.match(r"^(?:is|was)\s+(?:not|incorrect|wrong)\b", rest, re.I):
            return None
        allowed = r"^(?:because|since|as|which|that|it|this|is|was|provides|shows|indicates|represents)\b"
    else:
        allowed = r"^(?:because|since|as|which|it|this|explanation|rationale)\b|^(?:因为|由于|解释|理由)[:：]?"
    return label if re.search(allowed, rest, re.I) else None


def _negates_selected_answer(value: str, selected: set[str]) -> bool:
    """拒绝同时肯定与明确否定同一选项的回答；只检查大写独立标签，避免误认冠词 a。"""
    plain = _plain_format(value)
    for label in selected:
        if re.search(r"\b(?i:not)\s*(?:(?i:option)\s+)?" + label + r"\b", plain):
            return True
        if re.search(r"\b" + label + r"\s+(?i:is|was)\s+(?i:not|incorrect|wrong)\b", plain):
            return True
    for match in re.finditer(
        r"\b(?i:answer|choice)\s+(?i:is|was)\s+(?:(?i:actually|instead)\s+)?([A-Z])\b"
        r"|\b(?i:actually|instead|rather|correction)\s*[:,]?\s+([A-Z])\b", plain,
    ):
        if (match.group(1) or match.group(2)) not in selected:
            return True
    # 同行解释若明确声称另一选项才正确，也不能忽略矛盾而只保留开头标签。
    for match in re.finditer(
        r"\b([A-Z])\s+(?i:is|was)\s+(?:(?i:actually)\s+)?(?:(?i:the)\s+)?(?i:correct|right|final)\b", plain,
    ):
        if match.group(1) not in selected:
            return True
    return False


def _bare_choice_set(value: str, options: dict[str, str]) -> set[str] | None:
    """完整单选标签或明确并列标签集合；不接受混入解释或 or 的文字。"""
    direct = _option_label(value, options)
    if direct is not None:
        return {direct}
    if not re.fullmatch(r"\(?[A-Z]\)?(?:\s*(?:[,，、;；&+]|\band\b|和|及)\s*\(?[A-Z]\)?)+[.。]?",
                        value, flags=re.I):
        return None
    parts = re.split(r"[,，、;；&+]|\band\b|和|及", value, flags=re.I)
    selected = {part.strip().strip("().。").upper() for part in parts}
    return selected if selected and selected.issubset(options) else None


def _explicit_first_line_choices(value: str, options: dict[str, str]) -> set[str] | None:
    """明确答案标记后的首行可给完整集合，允许 because/因为 引出同行解释。"""
    selected = _bare_choice_set(value, options)
    if selected is not None:
        return selected
    explanation = re.search(r"\s+(?:because|since|as)\b|[，,]?\s*(?:因为|由于)", value, flags=re.I)
    if explanation is None:
        return None
    answer, detail = value[:explanation.start()].strip().rstrip(".,，。"), value[explanation.start():]
    if re.search(r"\b(?:maybe|perhaps|possibly|uncertain)\b|不确定|也许|可能", detail, re.I):
        return None
    return _bare_choice_set(answer, options)


def _choice_set(value: Any, options: dict[str, str], *, multiple: bool) -> set[str] | None:
    if not options:
        return None
    if isinstance(value, list):
        if not value or any(not isinstance(item, str) for item in value):
            return None
        labels = [_option_label(item, options) for item in value]
        if any(label is None for label in labels):
            return None
        result = set(labels)
        return result if multiple or len(result) == 1 else None
    if not isinstance(value, str):
        return None
    resolved = []
    plain = _plain_format(value)
    explicit = bool(_ANSWER_MARKER.search(plain))
    candidates = _answer_candidates(value)
    marked_candidates = set(candidates) if explicit else set()
    natural = _natural_choice_candidates(value, options)
    if explicit:
        # 独立答案段与后续显式结论也必须一致，避免“先A后B”被取最后一条。
        candidates.extend(natural)
        for block in re.split(r"\n\s*\n", plain):
            if _bare_choice_set(block.strip(), options) is not None and block.strip() not in candidates:
                candidates.append(block.strip())
    elif natural:
        candidates = natural
    for block in re.split(r"\n\s*\n", plain):
        correction = re.fullmatch(r"(?:actually|instead|rather|correction)\s*[:,]?\s+(.+)", block.strip(), re.I)
        if correction and _option_label(correction.group(1), options) is not None:
            candidates.append(correction.group(1))
    for candidate in candidates:
        if multiple and candidate in marked_candidates:
            lines = [line.strip() for line in candidate.splitlines() if line.strip()]
            first_choices = _explicit_first_line_choices(lines[0], options) if lines else None
            pure_option_lines = len(lines) > 1 and all(_option_label(line, options) is not None for line in lines)
            if first_choices is not None and not pure_option_lines:
                # 解释可另起一行，不要求空行或 Explanation 标签；独立选项不能被当作解释吞掉。
                for line in lines[1:]:
                    extra = _bare_choice_set(line, options)
                    if extra is not None and extra != first_choices:
                        return None
                    if extra is None and re.fullmatch(r"[A-Z](?:\s*[,，、;；]\s*[A-Z])*[.。]?", line):
                        return None
                resolved.append(first_choices)
                continue
        direct = _option_label(candidate, options)
        if direct is None and candidate in marked_candidates:
            direct = _explicit_choice_label(candidate, options)
        if direct is None and "\n" in candidate:
            lines = [line.strip() for line in candidate.splitlines() if line.strip()]
            # 答案标记后第一行给出一个完整选项，后面另起行解释时保留该答案。
            # 如果多行都是选项，则走下方集合规则，不能默默选第一条。
            first = _option_label(lines[0], options)
            rest = [_option_label(line, options) for line in lines[1:]]
            if first is not None and not any(label is not None for label in rest):
                direct = first
        if direct is not None:
            resolved.append({direct})
            continue
        # 只允许明确并列，or/或者是含糊选择，不作答案集。
        labels = re.fullmatch(r"\(?[A-Z]\)?(?:\s*(?:[,，、;；&+]|\band\b|和|及)\s*\(?[A-Z]\)?)+[.。]?",
                              candidate, flags=re.I)
        if labels:
            parts = re.split(r"[,，、;；&+]|\band\b|和|及", candidate, flags=re.I)
            selected = {part.strip().strip("().。").upper() for part in parts}
        elif multiple and "\n" in candidate:
            values = [_option_label(line, options) for line in candidate.splitlines() if line.strip()]
            if not values or any(item is None for item in values):
                return None
            selected = set(values)
        else:
            return None
        if not selected or not selected.issubset(options) or (not multiple and len(selected) != 1):
            return None
        resolved.append(selected)
    if not resolved or any(item != resolved[0] for item in resolved):
        return None
    if _negates_selected_answer(value, resolved[0]):
        return None
    return resolved[0]


def _yes_no(value: Any, *, reference: bool = False) -> str | None:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if not isinstance(value, str):
        return None
    values = []
    candidates = _answer_candidates(value)
    plain = _plain_format(value)
    if _ANSWER_MARKER.search(plain) and re.match(r"^(yes|no|true|false|是|否|不是|不)(?=$|[\s,.，。!！:：])", plain, re.I):
        candidates.append(plain)
    # “Yes. … No.”或两个独立肯否段存在冲突时不能只取首词。
    for block in re.split(r"\n\s*\n", plain):
        if re.match(r"^(yes|no|true|false|不是|是|否|不)(?:[,.，。!！:：]|\s*$)", block.strip(), re.I):
            candidates.append(block.strip())
    for candidate in candidates:
        clean = _plain_format(candidate)
        if re.search(r"\b(?:maybe|perhaps|possibly|uncertain)\b|不确定|可能|或者|"
                     r"\b(?:yes|no)\s*(?:or|and|/)\s*(?:yes|no)\b|"
                     r"\b(?:actually|rather|instead)\s+(?:yes|no)\b", clean, re.I):
            return None
        # MIRA 的参考 text 常为“Yes/No, 解释”；只取明确首词，不从描述推断肯否。
        pattern = r"^(yes|no|true|false|不是|是|否|不)(?=$|[\s,.，。!！:：])"
        match = re.match(pattern, clean, re.I)
        if not match:
            return None
        values.append("Yes" if match.group(1).casefold() in {"yes", "true", "是"} else "No")
    return values[0] if values and len(set(values)) == 1 else None


def score_answer(sample: dict, prediction: str) -> dict:
    """解析最终答案并评分；参考可评分而模型无法解析时计 0，始终保留在分母中。

    开放题和不明确的参考不计算客观准确率，返回 None；不能把文本匹配当医学正确率。
    normalized_* 用于答案文本指标，prediction_raw 保留原始模型输出供人工审查。
    """
    question_type = sample.get("question_type")
    if question_type not in QUESTION_TYPES:
        parts = str(sample.get("id", "")).split(":")
        question_type = parts[3] if len(parts) == 5 and parts[0] == "mira" and parts[3] in QUESTION_TYPES else "open_ended"
    reference_raw = sample.get("answer", sample.get("reference", ""))
    reference_value, reference_ok = _answer_field(reference_raw, question_type)
    visible, visible_ok = _without_thinking(prediction)
    prediction_value, prediction_ok = _answer_field(visible, question_type) if visible_ok else ("", False)
    result = {
        "question_type": question_type, "prediction_raw": prediction,
        "prediction_answer": prediction_value if prediction_ok else None,
        "reference_answer": reference_value if reference_ok else None,
        "normalized_prediction": _text(prediction_value) if prediction_ok else "",
        "normalized_reference": _text(reference_value) if reference_ok else "",
        "semantic_prediction": _semantic_text(visible, question_type) if visible_ok else "",
        "semantic_reference": _semantic_text(reference_raw, question_type),
        "answer_accuracy": None, "eligible": False, "parse_status": "not_objective", "choice_f1": None,
    }
    if question_type == "open_ended":
        if not prediction_ok:
            result["parse_status"] = "prediction_unparseable"
        return result
    if question_type in {"single_choice", "multiple_choice"}:
        options = _options(sample)
        multiple = question_type == "multiple_choice"
        reference = _choice_set(reference_value, options, multiple=multiple) if reference_ok else None
        predicted = _choice_set(prediction_value, options, multiple=multiple) if prediction_ok else None
    else:
        reference = _yes_no(reference_value, reference=True) if reference_ok else None
        predicted = _yes_no(prediction_value) if prediction_ok else None
    if reference is None:
        result["parse_status"] = "invalid_reference"
        return result
    result.update(eligible=True, answer_accuracy=float(predicted == reference),
                  parse_status="ok" if predicted is not None else "prediction_unparseable")
    result["normalized_reference"] = ", ".join(sorted(reference)) if isinstance(reference, set) else reference
    if predicted is not None:
        result["normalized_prediction"] = ", ".join(sorted(predicted)) if isinstance(predicted, set) else predicted
    if question_type in {"single_choice", "multiple_choice"}:
        result["choice_f1"] = 2 * len(reference & predicted) / (len(reference) + len(predicted)) if predicted else 0.0
    return result
