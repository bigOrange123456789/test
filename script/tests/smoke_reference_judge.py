"""对固定本地裁判做小规模真实推理校准；只验证接口及明显正确/矛盾样例。

该检查不训练模型，不写正式评估目录，不能证明小型裁判在全部医学问题上可靠。
"""

from pathlib import Path
import sys
import tempfile
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script import evaluate_rag as evaluation
from script.lib.reference_judge import ReferenceJudge


def main():
    args = evaluation.resolve_run_configs(evaluation.build_parser().parse_args([]))[0]
    args = evaluation.judge_arguments(args)
    args.device = "cuda"
    llm = evaluation.create_generator(args)
    cases = [
        ("否定一致", {"question": "Does the patient have evidence of pulmonary hypertension?",
                     "reference": "No. There is no evidence of pulmonary hypertension."},
         "There is no evidence of pulmonary hypertension.", 1.0),
        ("肯否矛盾", {"question": "Does the patient have evidence of pulmonary hypertension?",
                     "reference": "No. There is no evidence of pulmonary hypertension."},
         "Yes, the patient has pulmonary hypertension.", 0.0),
        ("数值矛盾", {"question": "What is the patient's ejection fraction?",
                     "reference": "The ejection fraction is 60%."},
         "The ejection fraction is 20%.", 0.0),
    ]
    started = time.perf_counter()
    observed = []
    try:
        with tempfile.TemporaryDirectory(prefix="reference-judge-smoke-") as directory:
            scorer = ReferenceJudge(llm, directory, {"path": args.model_path}, max_new_tokens=256, retries=1)
            for name, sample, prediction, expected in cases:
                result = scorer.score(sample, prediction)
                observed.append({"name": name, "expected": expected, **result})
            assert all(item["status"] == "ok" for item in observed), observed
            assert all(item["score"] == item["expected"] for item in observed), observed
    finally:
        llm.close()
        print(json.dumps({"cases": observed, "elapsed_seconds": time.perf_counter() - started},
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
