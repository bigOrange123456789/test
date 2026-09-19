"""Offline checks for model selection, text-only prompts and cache isolation."""

from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script import evaluate_rag as evaluation


@contextmanager
def changed_directory(path):
    original = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original)


def sample(sample_id="test-id", reference="SECRET_CURRENT_REFERENCE"):
    return {
        "id": sample_id,
        "question": "What diagnosis is likely?",
        "reference": reference,
        "images": [str(PROJECT_ROOT / "not-a-real-image.png")],
    }


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="evaluate-rag-models-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "evaluate_rag.json"

    def write_config(self, payload):
        self.config.write_text(json.dumps(payload), encoding="utf-8-sig")
        return self.config

    def args(self, *argv):
        with patch.object(evaluation, "DEFAULT_CONFIG_PATH", self.config):
            return evaluation.resolve_run_config(evaluation.build_parser().parse_args(argv))


class RunConfigurationTests(TemporaryCase):
    def test_object_config_selects_deepseek_and_one_rag_task(self):
        self.write_config({
            "model": "DeepSeek-Model", "useRAG": True,
            "datasetPath": str(self.root / "mira.jsonl"),
            "chromaPath": str(self.root / "MIRA-chroma"),
        })
        args = self.args("--top_k", "3")
        self.assertEqual(args.model, "DeepSeek-Model")
        self.assertEqual(Path(args.model_path), PROJECT_ROOT / "DeepSeek-Model")
        self.assertEqual(Path(args.embedding_model_path), PROJECT_ROOT / "Qwen3-VL-Embedding-2B")
        self.assertEqual(Path(args.dataset_path), self.root / "mira.jsonl")
        self.assertEqual(Path(args.chroma_db_dir), self.root / "MIRA-chroma")
        self.assertEqual(evaluation.parse_eval_config(args.eval_config, args.top_k), [
            {"name": "rag_top3", "use_rag": True, "top_k": 3},
        ])

    def test_single_element_array_config_selects_qwen_without_rag(self):
        self.write_config([{"model": "Qwen3-VL-2B-Instruct", "useRAG": False}])
        args = self.args()
        self.assertEqual(args.model, "Qwen3-VL-2B-Instruct")
        self.assertEqual(Path(args.model_path), PROJECT_ROOT / "Qwen3-VL-2B-Instruct")
        self.assertEqual(evaluation.parse_eval_config(args.eval_config, args.top_k), [
            {"name": "no_rag", "use_rag": False, "top_k": 5},
        ])

    def test_default_config_path_is_independent_of_working_directory(self):
        self.write_config({"model": "DeepSeek-Model", "useRAG": False})
        another_directory = self.root / "unrelated"
        another_directory.mkdir()
        with changed_directory(another_directory):
            args = self.args()
        self.assertEqual(args.model, "DeepSeek-Model")
        self.assertEqual(Path(args.model_path), PROJECT_ROOT / "DeepSeek-Model")

    def test_explicit_config_replaces_default_file(self):
        self.write_config({"model": "DeepSeek-Model", "useRAG": True})
        explicit = self.root / "custom.json"
        explicit.write_text(json.dumps({"model": "Qwen3-VL-2B-Instruct", "useRAG": False}),
                            encoding="utf-8")
        args = self.args("--config", str(explicit))
        self.assertEqual(args.model, "Qwen3-VL-2B-Instruct")
        self.assertFalse(evaluation.parse_eval_config(args.eval_config, args.top_k)[0]["use_rag"])

    def test_cli_model_overrides_json_and_selects_its_default_model_path(self):
        self.write_config({"model": "Qwen3-VL-2B-Instruct", "useRAG": False})
        args = self.args("--model", "DeepSeek-Model")
        self.assertEqual(args.model, "DeepSeek-Model")
        self.assertEqual(Path(args.model_path), PROJECT_ROOT / "DeepSeek-Model")

    def test_cli_paths_and_eval_tasks_override_config(self):
        self.write_config({
            "model": "DeepSeek-Model", "useRAG": False,
            "datasetPath": "ignored.jsonl", "chromaPath": "ignored-chroma",
        })
        tasks = '[{"name":"custom","use_rag":true,"top_k":2}]'
        args = self.args(
            "--model_path", str(self.root / "custom-model"),
            "--embedding_model_path", str(self.root / "custom-embedding"),
            "--dataset_path", str(self.root / "explicit.jsonl"),
            "--chroma_db_dir", str(self.root / "explicit-chroma"),
            "--eval_config", tasks,
        )
        self.assertEqual(Path(args.model_path), self.root / "custom-model")
        self.assertEqual(Path(args.embedding_model_path), self.root / "custom-embedding")
        self.assertEqual(Path(args.dataset_path), self.root / "explicit.jsonl")
        self.assertEqual(Path(args.chroma_db_dir), self.root / "explicit-chroma")
        self.assertEqual(evaluation.parse_eval_config(args.eval_config, args.top_k), [
            {"name": "custom", "use_rag": True, "top_k": 2},
        ])

    def test_missing_optional_default_config_preserves_both_eval_modes(self):
        args = self.args()
        self.assertEqual(args.model, "Qwen3-VL-2B-Instruct")
        self.assertEqual([c["use_rag"] for c in evaluation.parse_eval_config(args.eval_config, 5)],
                         [False, True])

    def test_explicit_missing_config_reports_error(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.args("--config", str(self.root / "missing.json"))

    def test_invalid_config_shape_or_values_are_rejected(self):
        cases = [
            [], [{"model": "DeepSeek-Model"}, {"model": "Qwen3-VL-2B-Instruct"}],
            "DeepSeek-Model", 42, None, ["DeepSeek-Model"],
            {"model": "unknown-model"}, {"model": 42},
            {"useRAG": "false"}, {"useRAG": 0}, {"useRAG": None},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.write_config(payload)
                with self.assertRaises(ValueError):
                    self.args()

    def test_invalid_cli_model_is_rejected_by_parser(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            evaluation.build_parser().parse_args(["--model", "unrecognized"])


class GeneratorSelectionAndPromptTests(TemporaryCase):
    def test_factory_selects_model_without_loading_weights(self):
        for choice, target, other, images in (
            ("DeepSeek-Model", "DeepSeekGenerator", "QwenGenerator", False),
            ("Qwen3-VL-2B-Instruct", "QwenGenerator", "DeepSeekGenerator", True),
        ):
            with self.subTest(choice=choice):
                args = self.args("--model", choice)
                with patch.object(evaluation, target) as selected, patch.object(evaluation, other) as unused:
                    self.assertIs(evaluation.create_generator(args), selected.return_value)
                    selected.assert_called_once_with(args)
                    unused.assert_not_called()
                generator = getattr(evaluation, target)(args)
                self.assertEqual(generator.supports_images, images)
                self.assertIsNone(generator.model)

    def test_text_only_generation_retains_evidence_but_never_images_or_target_answer(self):
        args = self.args("--model", "DeepSeek-Model")
        llm = SimpleNamespace(args=args, supports_images=False, chat=Mock(return_value="answer"))
        current = sample()
        evidence = sample("knowledge-id", "SAFE_RETRIEVED_ANSWER")
        with patch.object(evaluation, "_image_blocks", side_effect=AssertionError("image path accessed")):
            self.assertEqual(evaluation.generate_answer(current, [evidence], llm), "answer")
        messages, max_tokens = llm.chat.call_args.args
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn(current["question"], serialized)
        self.assertIn(evidence["reference"], serialized)
        self.assertNotIn(current["reference"], serialized)
        self.assertNotIn("not-a-real-image.png", serialized)
        self.assertNotIn('"type": "image"', serialized)
        self.assertEqual(max_tokens, args.max_new_tokens)

    def test_qwen_keeps_current_and_evidence_image_blocks(self):
        args = self.args("--model", "Qwen3-VL-2B-Instruct")
        llm = SimpleNamespace(args=args, supports_images=True, chat=Mock(return_value="answer"))
        current = sample()
        evidence = sample("knowledge-id", "SAFE_RETRIEVED_ANSWER")
        evidence["images"] = [str(self.root / "evidence.png")]
        evaluation.generate_answer(current, [evidence], llm)
        messages = llm.chat.call_args.args[0]
        blocks = messages[1]["content"]
        images = [block for block in blocks if block["type"] == "image"]
        self.assertEqual(len(images), 2)
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn("evidence.png", serialized)
        self.assertIn("not-a-real-image.png", serialized)
        self.assertNotIn(current["reference"], serialized)


class PredictionCacheIsolationTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.current = sample()
        self.knowledge = sample("knowledge-id", "retrieved answer")
        self.no_rag = {"name": "no_rag", "use_rag": False, "top_k": 1}
        self.rag = {"name": "rag_top1", "use_rag": True, "top_k": 1}

    def write_cache(self, config, fingerprint, evidence=None):
        directory = self.root / config["name"]
        directory.mkdir(exist_ok=True)
        record = dict(self.current, prediction="cached answer", config_name=config["name"],
                      generation_fingerprint=fingerprint, retrieved_evidence=evidence or [])
        (directory / "generations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        return record

    def load_cache(self, config, fingerprint):
        return evaluation.load_prediction_cache(self.root, config, [self.current],
                                                [self.knowledge], fingerprint)

    def test_matching_no_rag_cache_is_reused(self):
        record = self.write_cache(self.no_rag, "same-run")
        self.assertEqual(self.load_cache(self.no_rag, "same-run"), {self.current["id"]: record})

    def test_matching_rag_cache_is_reused_with_its_evidence(self):
        record = self.write_cache(self.rag, "rag-run", [dict(self.knowledge, similarity=0.7)])
        self.assertEqual(self.load_cache(self.rag, "rag-run"), {self.current["id"]: record})

    def test_no_rag_cache_cannot_fill_missing_rag_predictions(self):
        self.write_cache(self.no_rag, "no-rag-run")
        self.assertEqual(self.load_cache(self.rag, "rag-run"), {})

    def test_no_rag_cache_with_different_model_fingerprint_is_rejected(self):
        self.write_cache(self.no_rag, "qwen-run")
        self.assertEqual(self.load_cache(self.no_rag, "deepseek-run"), {})

    def test_invalid_rag_evidence_cannot_be_recovered_from_no_rag(self):
        self.write_cache(self.no_rag, "no-rag-run")
        self.write_cache(self.rag, "rag-run")
        self.assertEqual(self.load_cache(self.rag, "rag-run"), {})

    def test_fingerprint_changes_when_model_backend_or_rag_mode_changes(self):
        args = self.args("--model", "Qwen3-VL-2B-Instruct")
        other_args = copy.copy(args)
        other_args.model = "DeepSeek-Model"
        versions = {key: "test" for key in
                    ("torch", "torchvision", "transformers", "qwen-vl-utils", "Pillow")}
        with patch.object(evaluation, "_model_identity", return_value={"path": "same-path"}):
            qwen = evaluation.generation_fingerprint(args, self.no_rag, "dataset", {"test-id"}, versions)
            deepseek = evaluation.generation_fingerprint(other_args, self.no_rag, "dataset", {"test-id"}, versions)
            rag = evaluation.generation_fingerprint(args, self.rag, "dataset", {"test-id"}, versions)
        self.assertNotEqual(qwen, deepseek)
        self.assertNotEqual(qwen, rag)


class DeepSeekChatTests(TemporaryCase):
    def generator(self):
        class FakeInputs(dict):
            def to(self, device):
                return self

        args = self.args("--model", "DeepSeek-Model")
        generator = evaluation.DeepSeekGenerator(args)
        generator.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=1024),
            generate=Mock(return_value=evaluation.np.array([[1, 2, 3, 40, 41]])),
        )
        generator.tokenizer = Mock()
        generator.tokenizer.eos_token = "<EOS>"
        generator.tokenizer.apply_chat_template.return_value = "<BOS>question<ASSISTANT><EOS>"
        generator.tokenizer.return_value = FakeInputs(
            input_ids=evaluation.np.array([[1, 2, 3]]),
            attention_mask=evaluation.np.array([[1, 1, 1]]),
        )
        generator.tokenizer.decode.return_value = "  Answer only  "
        generator.torch = SimpleNamespace(inference_mode=nullcontext)
        generator.generation_config = object()
        generator.device = "cpu"
        return generator

    def test_chat_flattens_text_blocks_and_decodes_only_generated_tokens(self):
        generator = self.generator()
        messages = [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": [
                {"type": "text", "text": "Evidence"},
                {"type": "text", "text": "Question"},
            ]},
        ]
        self.assertEqual(generator.chat(messages, 10), "Answer only")
        chat_call = generator.tokenizer.apply_chat_template.call_args
        self.assertEqual(chat_call.args[0], [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Evidence\nQuestion"},
            {"role": "assistant", "content": ""},
        ])
        self.assertFalse(chat_call.kwargs["add_generation_prompt"])
        self.assertFalse(chat_call.kwargs["tokenize"])
        token_call = generator.tokenizer.call_args
        self.assertEqual(token_call.args[0], "<BOS>question<ASSISTANT>")
        self.assertFalse(token_call.kwargs["add_special_tokens"])
        self.assertFalse(token_call.kwargs["truncation"])
        self.assertEqual(generator.tokenizer.decode.call_args.args[0].tolist(), [40, 41])
        self.assertEqual(generator.model.generate.call_args.kwargs["max_new_tokens"], 10)

    def test_chat_rejects_image_blocks_before_generation(self):
        generator = self.generator()
        with self.assertRaises(ValueError):
            generator.chat([{"role": "user", "content": [
                {"type": "image", "image": "must-not-open.png"},
            ]}], 10)
        generator.model.generate.assert_not_called()
        generator.tokenizer.apply_chat_template.assert_not_called()

    def test_chat_rejects_input_over_budget_without_silent_truncation(self):
        generator = self.generator()
        generator.args.max_input_tokens = 2
        with self.assertRaises(ValueError):
            generator.chat([{"role": "user", "content": "Question"}], 10)
        generator.model.generate.assert_not_called()

    def test_chat_checks_input_and_output_against_model_context_window(self):
        generator = self.generator()
        generator.model.config.max_position_embeddings = 8
        with self.assertRaises(ValueError):
            generator.chat([{"role": "user", "content": "Question"}], 10)
        generator.model.generate.assert_not_called()

    def test_chat_rejects_template_without_expected_terminal_eos(self):
        generator = self.generator()
        generator.tokenizer.apply_chat_template.return_value = "<BOS>question<ASSISTANT>"
        with self.assertRaises(ValueError):
            generator.chat([{"role": "user", "content": "Question"}], 10)
        generator.model.generate.assert_not_called()


class MainEvaluationSmokeTests(TemporaryCase):
    def run_evaluation(self, model_name, use_rag):
        dataset = self.root / "dataset.jsonl"
        rows = [dict(sample(str(index), "This is a possible myocardial infarction."),
                     question=f"What is the likely diagnosis for patient {index}?")
                for index in range(3)]
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        # 固定 0 号题为测试集，其余两题仍可作为 RAG 知识库。
        selection = self.root / "test_selection.json"
        selection.write_text(json.dumps({"test_ids": ["0"]}), encoding="utf-8")
        self.write_config({"model": model_name, "useRAG": use_rag, "datasetPath": str(dataset),
                           "datasetFilter": str(selection)})
        output = self.root / "results"
        generated_messages = []

        def create_fake(args):
            def chat(messages, token_budget):
                generated_messages.append(messages)
                return "This is a possible myocardial infarction."
            return SimpleNamespace(args=args, supports_images=model_name != "DeepSeek-Model",
                                   chat=chat, close=Mock())

        def retrieve(args, configs, test, knowledge, caches):
            return {row["id"]: [dict(item, similarity=0.75) for item in knowledge]
                    for row in test} if use_rag else {}

        # main creates its handlers before calling basicConfig; explicitly close
        # these here so test cleanup works on Windows without changing root logging.
        def close_test_handlers(**kwargs):
            for handler in kwargs.get("handlers", []):
                handler.close()

        selected = "DeepSeekGenerator" if model_name == "DeepSeek-Model" else "QwenGenerator"
        unused = "QwenGenerator" if selected == "DeepSeekGenerator" else "DeepSeekGenerator"
        with patch.object(evaluation, "DEFAULT_CONFIG_PATH", self.config), \
                patch.object(evaluation, "validate_model_choice"), \
                patch.object(evaluation, selected, side_effect=create_fake) as selected_class, \
                patch.object(evaluation, unused) as unused_class, \
                patch.object(evaluation, "prepare_retrieval", side_effect=retrieve), \
                patch.object(evaluation, "_model_identity", return_value={"path": "mock"}), \
                patch.object(evaluation.logging, "basicConfig", side_effect=close_test_handlers), \
                patch.dict(sys.modules, {"rag_reports": None}), \
                patch.dict(os.environ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = evaluation.main([
                "--N", "1", "--top_k", "1", "--factscore_method", "keyword",
                "--output_dir", str(output),
            ])

        self.assertEqual(exit_code, 0)
        selected_class.assert_called_once()
        unused_class.assert_not_called()
        task_name = "rag_top1" if use_rag else "no_rag"
        predictions = [json.loads(line) for line in
                       (output / task_name / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["retrieved_count"], 1 if use_rag else 0)
        self.assertFalse(predictions[0]["generation_from_cache"])
        summary = json.loads((output / task_name / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["model"], model_name)
        self.assertEqual(summary["generation_input_mode"],
                         "text_only" if model_name == "DeepSeek-Model" else "text_and_images")
        self.assertEqual(set(summary["metrics"]), set(evaluation.METRICS))
        manifest = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["arguments"]["model"], model_name)
        self.assertEqual(len(generated_messages), 1)
        user_content = generated_messages[0][1]["content"]
        image_blocks = [block for block in user_content if block["type"] == "image"]
        self.assertEqual(len(image_blocks), 0 if model_name == "DeepSeek-Model" else (2 if use_rag else 1))

    def test_deepseek_without_rag_completes_and_saves_metrics_without_rag_reports(self):
        self.run_evaluation("DeepSeek-Model", False)

    def test_deepseek_with_rag_completes_and_saves_text_only_predictions(self):
        self.run_evaluation("DeepSeek-Model", True)

    def test_qwen_without_rag_completes_and_keeps_current_image(self):
        self.run_evaluation("Qwen3-VL-2B-Instruct", False)

    def test_qwen_with_rag_completes_and_keeps_evidence_and_current_images(self):
        self.run_evaluation("Qwen3-VL-2B-Instruct", True)


if __name__ == "__main__":
    unittest.main()
