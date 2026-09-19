"""Opt-in integration check using the actual local DeepSeek and Qwen weights.

Run in the MLMtest environment:
    python script/tests/smoke_evaluate_rag_models.py --device cuda

Each model evaluates one synthetic QA and generates at most eight tokens. All
datasets/configuration/results are temporary; model weights are read only.
This checks execution and input routing, not answer quality or performance.
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

from script import evaluate_rag as evaluation


def release_log_files():
    """Close main()'s FileHandler before TemporaryDirectory cleanup on Windows."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def without_optional_reports():
    """Hide only the optional module, keeping lazy Torch imports intact."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or fix the CUDA environment.")

    results = []
    with tempfile.TemporaryDirectory(prefix="evaluate-rag-local-models-") as temporary, \
            without_optional_reports():
        root = Path(temporary)
        dataset = root / "qa.jsonl"
        image_path = root / "synthetic.png"
        dataset.write_text(json.dumps({
            "id": "smoke:0",
            "question": "What is 2 + 2? Answer with one number.",
            "images": [str(image_path)],
            "reference": "4",
        }) + "\n", encoding="utf-8")
        # Same image path in both runs: it intentionally does not exist for
        # DeepSeek, proving that text-only inference does not attempt to open it.
        for model, input_mode in (
            ("DeepSeek-Model", "text_only"),
            ("Qwen3-VL-2B-Instruct", "text_and_images"),
        ):
            if model == "DeepSeek-Model":
                assert not image_path.exists()
            else:
                with Image.new("RGB", (96, 96), color="red") as image:
                    image.save(image_path)
            config = root / f"{model}.json"
            config.write_text(json.dumps({
                "model": model, "useRAG": False, "datasetPath": str(dataset),
            }), encoding="utf-8")
            output = root / model
            started = time.perf_counter()
            try:
                # These guards do not replace either real generation model.
                # The no-RAG path must work without embeddings or report extras.
                with patch.object(
                    evaluation, "QwenEmbedding",
                    side_effect=AssertionError("No-RAG must not load the embedding model"),
                ):
                    status = evaluation.main([
                        "--config", str(config), "--N", "1",
                        "--max_new_tokens", "8", "--factscore_method", "keyword",
                        "--device", args.device, "--output_dir", str(output),
                    ])
                assert status == 0, f"{model}: evaluation returned {status}"
                summary = read_json(output / "no_rag" / "summary.json")
                comparison = read_json(output / "comparison.json")
                manifest = read_json(output / "run_config.json")
                predictions = [json.loads(line) for line in (
                    output / "no_rag" / "predictions.jsonl"
                ).read_text(encoding="utf-8").splitlines() if line.strip()]
                assert summary["model"] == model
                assert Path(summary["model_path"]) == PROJECT_ROOT / model
                assert summary["generation_input_mode"] == input_mode
                assert summary["num_samples"] == 1 and summary["use_rag"] is False
                assert len(predictions) == 1
                assert predictions[0]["retrieved_evidence"] == []
                assert predictions[0]["retrieved_count"] == 0
                assert predictions[0]["generation_from_cache"] is False
                assert isinstance(predictions[0]["prediction"], str)
                assert comparison["baseline"] == "no_rag"
                assert comparison["summaries"] == [summary]
                assert manifest["status"] == "complete"
                assert manifest["report_status"] == "skipped_optional_dependency"
                results.append({
                    "model": model, "input_mode": input_mode, "status": "passed",
                    "generated_answer": predictions[0]["prediction"],
                    "seconds": round(time.perf_counter() - started, 2),
                })
            finally:
                release_log_files()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(json.dumps({"smoke_results": results, "temporary_files_cleaned": True},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
