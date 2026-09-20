"""按参考答案 token 长度分配回答预算：离线测试，不读取模型权重或占用 GPU。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script import evaluate_rag as evaluation


class RecordingGenerator:
    supports_images = False

    def __init__(self, args, *, count=10, attempts=None):
        self.args = args
        self.count = count
        self.counted = []
        self.calls = []
        self.last_generation_info = {}
        self.results = attempts or [("Answer: final", {
            "hit_token_limit": False, "output_tokens": 7, "elapsed_seconds": 0.25,
        })]

    def count_tokens(self, text):
        self.counted.append(text)
        return self.count

    def chat(self, messages, max_new_tokens):
        self.calls.append((copy.deepcopy(messages), max_new_tokens))
        result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        prediction, info = result
        self.last_generation_info = copy.deepcopy(info)
        return prediction


class AdaptiveGenerationTests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "enabled": True, "multiplier": 2.5, "extraTokens": 7,
            "minNewTokens": 32, "maxNewTokens": 256, "retryOnTruncation": True,
        }
        self.args = SimpleNamespace(
            max_new_tokens=128, max_input_tokens=16384,
            reference_length_budget=copy.deepcopy(self.policy),
            evaluation=copy.deepcopy(evaluation.EVALUATION_DEFAULTS),
        )
        self.sample = {
            "id": "mira:test:0:open_ended:0", "question_type": "open_ended",
            "question": "What is the diagnosis?", "images": [],
            "reference": "PRIVATE_REFERENCE_DIAGNOSIS.",
        }

    def test_native_token_count_drives_ceiling_and_lower_upper_clamps(self):
        for count, expected in ((0, 32), (11, 35), (1000, 256)):
            with self.subTest(count=count):
                llm = RecordingGenerator(self.args, count=count)
                self.assertEqual(evaluation.generate_answer(self.sample, [], llm), "Answer: final")
                self.assertEqual([limit for _, limit in llm.calls], [expected])
                self.assertEqual(llm.counted, [self.sample["reference"]])
                audit = llm.last_generation_info["token_budget"]
                self.assertEqual(audit["reference_tokens"], count)
                self.assertEqual(audit["initial_max_new_tokens"], expected)
                self.assertEqual(audit["count_method"], "native_tokenizer_without_special_tokens")

    def test_counting_removes_hidden_thought_and_structured_visual_evidence(self):
        reference = {
            "text": "<think>PRIVATE_HIDDEN_THOUGHT</think>PRIVATE_FINAL_ANSWER",
            "explanation": "PRIVATE_MEDICAL_EXPLANATION",
            "visual_evidence": "PRIVATE_VISUAL_ANNOTATION",
        }
        sample = {**self.sample, "answer": reference, "reference": json.dumps(reference)}
        llm = RecordingGenerator(self.args)
        evaluation.generate_answer(sample, [], llm)
        self.assertEqual(len(llm.counted), 1)
        self.assertIn("PRIVATE_FINAL_ANSWER", llm.counted[0])
        self.assertIn("PRIVATE_MEDICAL_EXPLANATION", llm.counted[0])
        self.assertNotIn("PRIVATE_HIDDEN_THOUGHT", llm.counted[0])
        self.assertNotIn("PRIVATE_VISUAL_ANNOTATION", llm.counted[0])
        self.assertNotIn("visual_evidence", llm.counted[0])
        prompt = json.dumps(llm.calls[0][0])
        self.assertNotIn("PRIVATE_", prompt, "参考内容只做本地计数，不得出现在生成提示中。")

    def test_missing_reference_uses_fixed_budget_clamped_without_tokenizer_call(self):
        sample = {**self.sample, "reference": ""}
        for fixed, expected in ((8, 32), (100, 100), (1024, 256)):
            with self.subTest(fixed=fixed):
                self.args.max_new_tokens = fixed
                llm = RecordingGenerator(self.args)
                evaluation.generate_answer(sample, [], llm)
                self.assertEqual(llm.counted, [])
                self.assertEqual(llm.calls[0][1], expected)
                self.assertEqual(llm.last_generation_info["token_budget"]["source"], "fixed_missing_reference")
                self.assertIsNone(llm.last_generation_info["token_budget"]["reference_tokens"])

    def test_truncated_answer_retries_once_at_cap_without_reference_leakage(self):
        llm = RecordingGenerator(self.args, attempts=[
            ("partial", {"hit_token_limit": True, "output_tokens": 32, "elapsed_seconds": 0.4}),
            ("complete", {"hit_token_limit": False, "output_tokens": 80, "elapsed_seconds": 0.7}),
        ])
        self.assertEqual(evaluation.generate_answer(self.sample, [], llm), "complete")
        self.assertEqual([limit for _, limit in llm.calls], [32, 256])
        self.assertEqual(llm.calls[0][0], llm.calls[1][0])
        self.assertTrue(all("PRIVATE_REFERENCE" not in json.dumps(messages) for messages, _ in llm.calls))
        info = llm.last_generation_info
        self.assertFalse(info["hit_token_limit"])
        self.assertEqual(info["max_new_tokens"], 256)
        self.assertEqual(info["output_tokens"], 80)
        self.assertEqual(info["total_output_tokens"], 112)
        self.assertAlmostEqual(info["elapsed_seconds"], 1.1)
        self.assertEqual([attempt["attempt"] for attempt in info["attempts"]], [1, 2])
        self.assertEqual([attempt["max_new_tokens"] for attempt in info["attempts"]], [32, 256])
        self.assertEqual(info["token_budget"]["policy"], self.policy)

    def test_second_truncation_does_not_trigger_third_attempt(self):
        llm = RecordingGenerator(self.args, attempts=[
            ("partial", {"hit_token_limit": True, "output_tokens": 32, "elapsed_seconds": 0.1}),
            ("still partial", {"hit_token_limit": True, "output_tokens": 256, "elapsed_seconds": 0.2}),
        ])
        self.assertEqual(evaluation.generate_answer(self.sample, [], llm), "still partial")
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(llm.last_generation_info["hit_token_limit"])
        self.assertEqual(llm.last_generation_info["total_output_tokens"], 288)

    def test_no_retry_when_disabled_or_first_budget_already_at_cap(self):
        for retry, count, expected in ((False, 10, 32), (True, 1000, 256)):
            with self.subTest(retry=retry, count=count):
                self.args.reference_length_budget["retryOnTruncation"] = retry
                llm = RecordingGenerator(self.args, count=count, attempts=[
                    ("partial", {"hit_token_limit": True, "output_tokens": expected, "elapsed_seconds": 0.1}),
                ])
                evaluation.generate_answer(self.sample, [], llm)
                self.assertEqual([limit for _, limit in llm.calls], [expected])
                self.assertEqual(len(llm.last_generation_info["attempts"]), 1)

    def test_disabled_policy_preserves_original_single_call_and_generation_info(self):
        self.args.reference_length_budget["enabled"] = False
        original = {"hit_token_limit": True, "output_tokens": 128, "elapsed_seconds": 0.5}
        llm = RecordingGenerator(self.args, attempts=[("partial", original)])
        self.assertEqual(evaluation.generate_answer(self.sample, [], llm), "partial")
        self.assertEqual(llm.counted, [])
        self.assertEqual([limit for _, limit in llm.calls], [128])
        self.assertEqual(llm.last_generation_info, original)
        self.assertEqual(evaluation.generation_settings(self.args), {
            "max_new_tokens": 128, "max_input_tokens": 16384,
        })

    def test_failed_cap_retry_preserves_first_answer_and_truncation_review_audit(self):
        llm = RecordingGenerator(self.args, attempts=[
            ("first partial answer", {"hit_token_limit": True, "output_tokens": 32, "elapsed_seconds": 0.4}),
            RuntimeError("retry context exceeds available memory"),
        ])
        self.assertEqual(evaluation.generate_answer(self.sample, [], llm), "first partial answer")
        self.assertEqual([limit for _, limit in llm.calls], [32, 256])
        info = llm.last_generation_info
        self.assertTrue(info["hit_token_limit"])
        self.assertEqual(info["max_new_tokens"], 32)
        self.assertEqual(info["total_output_tokens"], 32)
        self.assertGreaterEqual(info["elapsed_seconds"], 0.4)
        self.assertEqual(len(info["attempts"]), 2)
        self.assertIn("RuntimeError", info["attempts"][1]["error"])
        self.assertEqual(info["token_budget"]["retry_error"], info["attempts"][1]["error"])
        self.assertTrue(evaluation._needs_review({
            "generation_info": info, "semantic_details": {"status": "not_required"},
            "answer_details": {"parse_status": "ok"},
        }))

    def test_eos_at_output_limit_is_complete_instead_of_a_retry_trigger(self):
        cases = (
            ([11, 12, 99], 3, 99, False),
            ([11, 12, 99], 3, [98, 99], False),
            ([11, 12, 13], 3, [98, 99], True),
            ([11, 12], 3, 99, False),
            ([], 3, 99, False),
            ([11, 12, 13], 3, None, True),
        )
        for answer_ids, limit, eos, expected in cases:
            with self.subTest(answer_ids=answer_ids, eos=eos):
                self.assertEqual(evaluation._hit_generation_limit(
                    answer_ids, limit, SimpleNamespace(eos_token_id=eos)), expected)


class NativeTokenCountInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(model_path="local-model", model_revision="main", text_only=False)
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [10, 20, 30]

    def test_qwen_loaded_processor_uses_its_tokenizer_without_chat_markers(self):
        generator = evaluation.QwenGenerator(self.args)
        generator.processor = SimpleNamespace(tokenizer=self.tokenizer)
        with patch.object(generator, "_load", side_effect=AssertionError("不应加载模型权重")):
            self.assertEqual(generator.count_tokens("医学参考 answer"), 3)
        self.tokenizer.encode.assert_called_once_with("医学参考 answer", add_special_tokens=False)
        self.assertIsNone(generator.model)

    def test_deepseek_loaded_tokenizer_uses_same_native_interface(self):
        generator = evaluation.DeepSeekGenerator(self.args)
        generator.tokenizer = self.tokenizer
        with patch.object(generator, "_load", side_effect=AssertionError("不应加载模型权重")):
            self.assertEqual(generator.count_tokens("医学参考 answer"), 3)
        self.tokenizer.encode.assert_called_once_with("医学参考 answer", add_special_tokens=False)
        self.assertIsNone(generator.model)

    def test_unloaded_generators_load_only_one_local_tokenizer_lazily(self):
        for generator_type in (evaluation.QwenGenerator, evaluation.DeepSeekGenerator):
            with self.subTest(generator_type=generator_type.__name__):
                auto = SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer))
                generator = generator_type(self.args)
                with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=auto)}):
                    with patch.object(generator, "_load", side_effect=AssertionError("不应加载模型权重")):
                        self.assertEqual(generator.count_tokens("first"), 3)
                        self.assertEqual(generator.count_tokens("second"), 3)
                auto.from_pretrained.assert_called_once_with(
                    "local-model", trust_remote_code=True, revision="main", local_files_only=True)
                self.assertIsNone(generator.model)
                self.assertIsNone(generator.torch)


class AdaptiveConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="adaptive-config-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "evaluate_rag.json"
        self.settings = {
            "name": "Qwen", "model": "Qwen3-VL-2B-Instruct", "useRAG": False,
            "generation": {"maxNewTokens": 2048, "referenceLengthBudget": {"enabled": True}},
            "evaluation": {"prima": {"enabled": True, "factScoreLengthBudget": {"enabled": True}}},
        }

    def resolve(self, *extra):
        self.config.write_text(json.dumps(self.settings), encoding="utf-8")
        args = evaluation.build_parser().parse_args(["--config", str(self.config), *extra])
        return evaluation.resolve_run_configs(args)[0]

    def fingerprint(self, args):
        versions = {name: "test-version" for name in ("torch", "torchvision", "transformers", "qwen-vl-utils", "Pillow")}
        with patch.object(evaluation, "_model_identity", return_value={"path": "model"}):
            return evaluation.generation_fingerprint(
                args, {"use_rag": False, "top_k": 0}, "same-data", {"same-id"}, versions)

    def test_default_json_policy_is_normalized_and_reported(self):
        args = self.resolve()
        self.assertEqual(args.reference_length_budget, {
            "enabled": True, "multiplier": 4, "extraTokens": 256, "minNewTokens": 512,
            "maxNewTokens": 4096, "retryOnTruncation": True,
        })
        reported = evaluation.generation_settings(args)["reference_length_budget"]
        self.assertEqual(reported, {"version": "reference-length-v1", **args.reference_length_budget})
        self.settings["generation"].pop("referenceLengthBudget")
        self.assertFalse(self.resolve().reference_length_budget["enabled"])

    def test_explicit_cli_fixed_limit_disables_answer_budget_only(self):
        args = self.resolve("--max_new_tokens", "73")
        self.assertEqual(args.max_new_tokens, 73)
        self.assertFalse(args.reference_length_budget["enabled"])
        self.assertTrue(args.evaluation["prima"]["factScoreLengthBudget"]["enabled"])
        self.assertNotIn("reference_length_budget", evaluation.generation_settings(args))

    def test_strict_json_validation_rejects_invalid_budget_policies(self):
        invalid = (
            None, False, [], {"unknown": 1}, {"enabled": "true"}, {"retryOnTruncation": 1},
            {"multiplier": True}, {"multiplier": 0}, {"multiplier": -1},
            {"multiplier": float("nan")}, {"multiplier": float("inf")},
            {"extraTokens": -1}, {"extraTokens": True}, {"extraTokens": 1.5},
            {"minNewTokens": 0}, {"minNewTokens": True}, {"minNewTokens": 4097, "maxNewTokens": 4096},
            {"maxNewTokens": 0}, {"maxNewTokens": 40000}, {"maxNewTokens": 512.5},
        )
        for field in ("answer", "judge"):
            for value in invalid:
                with self.subTest(field=field, value=value):
                    self.settings["generation"]["referenceLengthBudget"] = {"enabled": True}
                    self.settings["evaluation"]["prima"]["factScoreLengthBudget"] = {"enabled": True}
                    if field == "answer":
                        self.settings["generation"]["referenceLengthBudget"] = value
                    else:
                        self.settings["evaluation"]["prima"]["factScoreLengthBudget"] = value
                    with self.assertRaises(ValueError):
                        self.resolve()

    def test_enabled_answer_policy_changes_generation_fingerprint(self):
        args = self.resolve()
        original = self.fingerprint(args)
        for field, value in (("multiplier", 5), ("extraTokens", 300), ("minNewTokens", 600),
                             ("maxNewTokens", 5000), ("retryOnTruncation", False), ("enabled", False)):
            with self.subTest(field=field):
                changed = copy.deepcopy(args)
                changed.reference_length_budget[field] = value
                self.assertNotEqual(self.fingerprint(changed), original)

    def test_disabled_policy_keeps_old_generation_fingerprint_shape(self):
        args = self.resolve()
        args.reference_length_budget["enabled"] = False
        disabled = self.fingerprint(args)
        # 旧调用方尚无此属性，也必须得到相同缓存指纹。
        legacy = copy.deepcopy(args)
        del legacy.reference_length_budget
        self.assertEqual(self.fingerprint(legacy), disabled)
        args.reference_length_budget.update(multiplier=7, extraTokens=999, maxNewTokens=8192)
        self.assertEqual(self.fingerprint(args), disabled)

    def test_judge_budget_changes_do_not_invalidate_answer_generation(self):
        args = self.resolve()
        original = self.fingerprint(args)
        args.evaluation["prima"]["factScoreLengthBudget"].update(multiplier=3, maxNewTokens=12000)
        args.evaluation["prima"]["factScoreMaxNewTokens"] = 2048
        self.assertEqual(self.fingerprint(args), original)


if __name__ == "__main__":
    unittest.main()
