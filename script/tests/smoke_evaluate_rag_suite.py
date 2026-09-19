"""用本地真实模型检查四组评估入口；每组仅一个合成问答、最多生成 8 个 token。

在 MLMtest 环境中按需运行：
    python script/tests/smoke_evaluate_rag_suite.py --device cuda

数据、图片、配置和评估结果均放在临时目录；只读加载原始模型和两个已有 LoRA。
这是运行完整性检查，不代表真实测试集上的医学回答质量。
"""

import argparse
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image
import torch
from peft import PeftModel

from script import evaluate_rag as evaluation


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def release_log_files():
    """Windows 删除临时目录前需要先关闭日志文件句柄。"""
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()


@contextmanager
def without_optional_reports():
    """只隐藏可选报告模块，保留 PyTorch 延迟导入的模块状态。"""
    absent = object()
    original = sys.modules.get("rag_reports", absent)
    sys.modules["rag_reports"] = None
    try:
        yield
    finally:
        if original is absent:
            sys.modules.pop("rag_reports", None)
        else:
            sys.modules["rag_reports"] = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA，请切换至 MLMtest 环境后再运行。")

    qwen_adapter = PROJECT_ROOT / "script" / "qwen3_vl_2b_lora_adapter"
    deepseek_adapter = PROJECT_ROOT / "script" / "deepseek_mira_lora_adapter"
    cases = [
        ("Qwen", "Qwen3-VL-2B-Instruct", None, "text_and_images"),
        ("Qwen_lora", "Qwen3-VL-2B-Instruct", qwen_adapter, "text_and_images"),
        ("Deepseek", "DeepSeek-Model", None, "text_only"),
        ("Deepseek_lora", "DeepSeek-Model", deepseek_adapter, "text_only"),
    ]
    observed_loads = []
    real_load_adapter = evaluation.load_lora_adapter

    def observed_load_adapter(model, adapter_path):
        # 使用真实 PEFT 加载，只观察载入结果；既不模拟模型，也不保存其参数。
        assert not isinstance(model, PeftModel), "每组应重新加载独立的原始基座模型。"
        wrapped = real_load_adapter(model, adapter_path)
        if adapter_path is None:
            assert wrapped is model
            assert not isinstance(wrapped, PeftModel)
            lora_parameters = 0
        else:
            assert isinstance(wrapped, PeftModel)
            assert not wrapped.training
            lora_parameters = sum(
                parameter.numel() for name, parameter in wrapped.named_parameters()
                if "lora_" in name
            )
            assert lora_parameters > 0, "真实 LoRA 参数必须已装入模型。"
            assert all(not parameter.requires_grad for parameter in wrapped.parameters())
        observed_loads.append({
            "model_type": wrapped.config.model_type,
            "lora_path": str(Path(adapter_path).resolve()) if adapter_path else None,
            "lora_parameters": lora_parameters,
            "is_peft_model": isinstance(wrapped, PeftModel),
        })
        return wrapped

    results = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="evaluate-rag-four-models-") as temporary, \
            without_optional_reports():
        root = Path(temporary)
        dataset = root / "qa.jsonl"
        image_path = root / "synthetic.png"
        with Image.new("RGB", (96, 96), color="red") as image:
            image.save(image_path)
        samples = [{
            "id": "smoke:test", "question": "What is 2 + 2? Answer with one number.",
            "images": [str(image_path)], "reference": "4",
        }, {
            "id": "smoke:train", "question": "Training-only question; must not be evaluated.",
            "images": [], "reference": "DO_NOT_EVALUATE_TRAINING_SAMPLE",
        }]
        dataset.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        manifest = root / "ids.json"
        manifest.write_text(json.dumps({
            "test_ids": ["smoke:test"], "train_ids": ["smoke:train"],
        }), encoding="utf-8")
        config = root / "evaluate_rag.json"
        config.write_text(json.dumps([{
            "name": name, "model": model, "pathLora": str(adapter) if adapter else None,
            "useRAG": False, "datasetPath": str(dataset), "datasetFilter": str(manifest),
        } for name, model, adapter, _ in cases]), encoding="utf-8")
        output = root / "results"
        try:
            with patch.object(evaluation, "load_lora_adapter", side_effect=observed_load_adapter), \
                    patch.object(evaluation, "QwenEmbedding", side_effect=AssertionError(
                        "useRAG=false 时不应加载嵌入模型。"
                    )):
                status = evaluation.main([
                    "--config", str(config), "--max_new_tokens", "8",
                    "--factscore_method", "keyword", "--device", args.device,
                    "--output_dir", str(output),
                ])
            assert status == 0, f"四组评估返回错误码：{status}"
            assert len(observed_loads) == 4, observed_loads
            suite = read_json(output / "suite_comparison.json")
            assert len(suite["summaries"]) == 4
            assert {row["run_name"] for row in suite["summaries"]} == {case[0] for case in cases}
            fingerprints = set()
            test_fingerprints = set()
            for (name, model, adapter, input_mode), observed in zip(cases, observed_loads):
                directory = output / name
                summary = read_json(directory / "no_rag" / "summary.json")
                job_manifest = read_json(directory / "run_config.json")
                test_ids = read_json(directory / "test_ids.json")
                predictions = [json.loads(line) for line in (
                    directory / "no_rag" / "predictions.jsonl"
                ).read_text(encoding="utf-8").splitlines() if line.strip()]
                assert summary["model"] == model
                assert summary["run_name"] == name
                assert summary["name"] == "no_rag"
                assert summary["lora_path"] == (str(adapter) if adapter else None)
                assert summary["dataset_filter"] == str(manifest)
                assert summary["generation_input_mode"] == input_mode
                assert summary["num_samples"] == 1 and summary["use_rag"] is False
                assert job_manifest["status"] == "complete"
                assert job_manifest["report_status"] == "skipped_optional_dependency"
                assert test_ids == ["smoke:test"]
                assert len(predictions) == 1 and predictions[0]["id"] == "smoke:test"
                assert predictions[0]["retrieved_evidence"] == []
                assert predictions[0]["retrieved_count"] == 0
                assert predictions[0]["generation_from_cache"] is False
                assert isinstance(predictions[0]["prediction"], str)
                assert observed["lora_path"] == (str(adapter) if adapter else None)
                assert observed["is_peft_model"] is (adapter is not None)
                assert observed["model_type"] == ("qwen2" if model == "DeepSeek-Model" else "qwen3_vl")
                fingerprints.add(summary["generation_fingerprint"])
                test_fingerprints.add(summary["test_fingerprint"])
                results.append({
                    "name": name, "model": model, "input_mode": input_mode,
                    "status": "passed", "answer": predictions[0]["prediction"], **observed,
                })
            assert len(fingerprints) == 4, "四组模型的答案缓存必须相互隔离。"
            assert len(test_fingerprints) == 1, "四组必须评估完全相同的测试问答。"
        finally:
            release_log_files()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({
        "smoke_results": results, "seconds": round(time.perf_counter() - started, 2),
        "temporary_files_cleaned": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
