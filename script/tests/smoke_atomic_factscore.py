"""用本地原版裁判检查事实拆分及二元核验；仅用合成样本，不写正式评估结果。"""

from pathlib import Path
import json
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script import evaluate_rag as evaluation
from script.lib.atomic_factscore import AtomicFactScorer


def main():
    args = evaluation.resolve_run_configs(evaluation.build_parser().parse_args([]))[0]
    args = evaluation.judge_arguments(args)
    args.device = "cuda"
    llm = evaluation.create_generator(args)
    sample = {"question": "What are the patient's heart rate and ejection fraction?",
              "reference": "The heart rate is 80 beats per minute. The ejection fraction is 60%."}
    cases = [(sample["reference"], 1.0),
             ("The heart rate is 80 beats per minute. The ejection fraction is 20%.", 0.5)]
    observed = []
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="atomic-factscore-smoke-") as directory:
            scorer = AtomicFactScorer(llm, directory, {"path": args.model_path}, max_new_tokens=1024)
            for prediction, expected in cases:
                details = scorer.score(sample, prediction)
                observed.append({"expected": expected, **details})
                print(json.dumps({"score": details["score"], "claim_count": details["claim_count"],
                                  "supported_count": details["supported_count"], "status": details["status"]}), flush=True)
            assert all(row["status"] == "ok" and row["score"] == row["expected"] for row in observed), observed
    finally:
        llm.close()
        print(json.dumps({"results": observed, "seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
