"""统一参考答案裁判的离线测试，不加载真实模型或占用 GPU。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.lib.reference_judge import ReferenceJudge, aggregate_scores, parse_judgement


def sample(**changes):
    return {
        "id": "mira:test:3:open_ended:0", "question": "Which diagnosis is supported?",
        "reference": "Atrial fibrillation", "question_type": "open_ended", **changes,
    }


class ParserTests(unittest.TestCase):
    def test_exact_json_and_full_fence(self):
        self.assertEqual(parse_judgement('{"correctness": 1, "reason": "一致"}'), (1.0, "一致"))
        self.assertEqual(parse_judgement('```json\n{"correctness": 0.5, "reason": "部分一致"}\n```'),
                         (0.5, "部分一致"))

    def test_rejects_malformed_or_ambiguous_output(self):
        examples = [
            "", "Analysis: {\"correctness\": 1, \"reason\": \"ok\"}",
            '{"correctness": 1, "reason": "ok"} {"correctness": 0, "reason": "no"}',
            '{"correctness": true, "reason": "ok"}', '{"correctness": "1", "reason": "ok"}',
            '{"correctness": 0.7, "reason": "ok"}', '{"correctness": NaN, "reason": "ok"}',
            '{"correctness": 1, "reason": ""}', '{"correctness": 1}',
            '{"correctness": 1, "reason": "ok", "extra": 1}',
            '{"correctness": 1, "correctness": 0, "reason": "ok"}',
            '[{"correctness": 1, "reason": "ok"}]',
            '```json\n{"correctness": 1, "reason": "ok"}\n``` extra',
        ]
        for raw in examples:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judgement(raw)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="reference-judge-")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.llm = Mock()
        self.llm.text.return_value = '{"correctness": 1, "reason": "结论一致"}'
        self.identity = {"model_path": "fixed-base", "lora": None, "weight_sha256": "123"}

    def judge(self, **kwargs):
        return ReferenceJudge(self.llm, self.cache, self.identity, **kwargs)

    def test_score_and_cache_shared_across_model_names_and_sample_ids(self):
        first = self.judge().score(sample(model="ModelA", run_name="A"), "AF")
        second = self.judge().score(sample(id="other", model="ModelB", run_name="B"), "AF")
        self.assertEqual(first["score"], 1.0)
        self.assertEqual(first["status"], "ok")
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.llm.text.assert_called_once()
        prompt = self.llm.text.call_args.args[0]
        self.assertNotIn("ModelA", prompt)
        self.assertNotIn("fixed-base", prompt)
        self.assertNotIn("mira:test:3", prompt)
        self.assertIn("Atrial fibrillation", prompt)

    def test_only_question_metadata_enters_judge_prompt(self):
        self.judge().score(sample(metadata={"options": {"A": "AF"}, "model": "DO_NOT_SEND"}), "AF")
        prompt = self.llm.text.call_args.args[0]
        self.assertIn('"options":{"A":"AF"}', prompt)
        self.assertNotIn("DO_NOT_SEND", prompt)

    def test_evaluation_reference_has_precedence(self):
        self.judge().score(sample(reference='{"text":"RAW"}', evaluation_reference="NORMALIZED"), "AF")
        prompt = self.llm.text.call_args.args[0]
        self.assertIn("NORMALIZED", prompt)
        self.assertNotIn("RAW", prompt)

    def test_one_format_retry_with_shorter_budget(self):
        self.llm.text.side_effect = ["not-json", '{"correctness": 0.5, "reason": "部分支持"}']
        result = self.judge().score(sample(), "AF")
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual([call.kwargs["max_new_tokens"] for call in self.llm.text.call_args_list], [256, 128])

    def test_failed_parse_is_none_and_not_cached(self):
        self.llm.text.return_value = "invalid"
        judge = self.judge()
        result = judge.score(sample(), "AF")
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "judge_failed")
        self.assertEqual(self.llm.text.call_count, 2)
        self.assertEqual(list(self.cache.glob("*.json")), [])
        judge.score(sample(), "AF")
        self.assertEqual(self.llm.text.call_count, 4)

    def test_inference_exception_is_none_without_format_retry(self):
        self.llm.text.side_effect = RuntimeError("CUDA out of memory")
        result = self.judge().score(sample(), "AF")
        self.assertIsNone(result["score"])
        self.assertIn("CUDA out of memory", result["reason"])
        self.llm.text.assert_called_once()

    def test_invalid_input_is_explicit_failure(self):
        result = self.judge().score(sample(reference={"text": "not-normalized"}), "AF")
        self.assertEqual(result["status"], "judge_failed")
        self.assertIsNone(result["score"])
        self.llm.text.assert_not_called()

    def test_retry_count_must_be_an_integer(self):
        for value in (True, 1.0, -1, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.judge(retries=value)

    def test_non_text_judge_output_is_serializable_failure(self):
        self.llm.text.return_value = object()
        result = self.judge().score(sample(), "AF")
        self.assertEqual(result["status"], "judge_failed")
        self.assertIsNone(result["score"])
        json.dumps(result)

    def test_cache_changes_with_answer_reference_question_options_and_judge(self):
        judge = self.judge()
        records = [
            judge.score(sample(), "AF"), judge.score(sample(), "Different"),
            judge.score(sample(reference="Different reference"), "AF"),
            judge.score(sample(question="Different question"), "AF"),
            judge.score(sample(options={"A": "AF"}), "AF"),
            ReferenceJudge(self.llm, self.cache, {"model": "new-judge"}).score(sample(), "AF"),
            self.judge(max_new_tokens=128).score(sample(), "AF"),
            self.judge(retries=0).score(sample(), "AF"),
        ]
        self.assertEqual(len({record["fingerprint"] for record in records}), len(records))
        self.assertEqual(self.llm.text.call_count, len(records))

    def test_corrupt_cache_is_recomputed(self):
        result = self.judge().score(sample(), "AF")
        path = self.cache / f"{result['fingerprint']}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["result"]["score"] = 0
        path.write_text(json.dumps(value), encoding="utf-8")
        actual = self.judge().score(sample(), "AF")
        self.assertFalse(actual["from_cache"])
        self.assertEqual(actual["score"], 1)
        self.assertEqual(self.llm.text.call_count, 2)

    def test_no_disk_cache_still_reuses_memory(self):
        judge = ReferenceJudge(self.llm, None, self.identity)
        judge.score(sample(), "AF")
        self.assertTrue(judge.score(sample(), "AF")["from_cache"])
        self.llm.text.assert_called_once()


class AggregateTests(unittest.TestCase):
    def test_failed_scores_are_not_zero_and_coverage_is_explicit(self):
        result = aggregate_scores([
            {"id": "a", "score": 1.0, "status": "ok"},
            {"id": "b", "score": None, "status": "judge_failed", "reason": "bad JSON"},
            {"id": "c", "score": 0.0, "status": "ok"},
        ])
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["std"], 0.5)
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["num_scored"], 2)
        self.assertEqual(result["num_failed"], 1)
        self.assertEqual(result["failures"][0]["id"], "b")

    def test_all_failed_and_empty_do_not_invent_scores(self):
        result = aggregate_scores([{"score": None, "status": "judge_failed"}])
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["std"])
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(aggregate_scores([])["coverage"])


if __name__ == "__main__":
    unittest.main()
