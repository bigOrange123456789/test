"""Offline regression tests for text-only MIRA LoRA preparation (no model weights)."""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script.lib import finetune_deepseek_mira_lora as finetune

try:
    import torch
except ImportError:
    torch = None


def make_sample(index=0, question="What is shown?", answer="Target answer", options=None):
    return finetune.TextSample(
        sample_id=f"mira:train:{index}:open_ended:0",
        question=question,
        answer=answer,
        options=options,
    )


class ManifestAndDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def manifest(self, payload):
        path = self.root / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8-sig")
        return path

    def write_split(self, split, rows, include_image_columns=False):
        fieldnames = ["vqa_json"]
        if include_image_columns:
            fieldnames += ["image_path", "caption"]
        with (self.root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for questions in rows:
                row = {"vqa_json": json.dumps(questions, ensure_ascii=False)}
                if include_image_columns:
                    row.update(image_path="missing/image.png", caption="SECRET_CAPTION\nMore caption")
                writer.writerow(row)

    def test_manifest_preserves_order_and_metadata(self):
        payload = {
            "train_ids": [make_sample(3).sample_id, make_sample(1).sample_id],
            "test_ids": [make_sample(2).sample_id], "seed": 42,
        }
        ids, loaded = finetune.load_split_manifest(self.manifest(payload))
        self.assertEqual(ids, payload["train_ids"])
        self.assertEqual(loaded, payload)

    def test_null_manifest_uses_every_train_csv_qa_and_excludes_other_splits(self):
        self.write_split("train", [{
            "open_ended": [{"question": "Open", "answer": "A"}],
            "closed_ended": [{"question": "Closed", "answer": "B"}],
            "single_choice": [{"question": "Single", "answer": "C"}],
            "multiple_choice": [{"question": "Multiple", "answer": "D"}],
        }], include_image_columns=True)
        self.write_split("validation", [{
            "open_ended": [{"question": "Held out", "answer": "Do not train"}],
        }], include_image_columns=True)
        args = finetune.build_parser().parse_args([
            "--all-train-data", "--data-root", str(self.root),
            "--model-dir", str(self.root / "base"), "--output-dir", str(self.root / "adapter"),
        ])
        samples, manifest = finetune.prepare_samples(args)
        self.assertEqual(len(samples), 4)
        self.assertEqual([sample.sample_id.split(":")[3] for sample in samples],
                         ["open_ended", "closed_ended", "single_choice", "multiple_choice"])
        self.assertEqual(manifest["selection_mode"], "full_train_split")
        self.assertEqual(manifest["source_splits"], ["train"])
        self.assertEqual(args.data_selection_audit["manifest_train_count"], 4)
        self.assertEqual(args.data_selection_audit["selection_mode"], "full_train_split")

    def test_manifest_rejects_invalid_ids_duplicates_and_train_test_overlap(self):
        sample_id = make_sample().sample_id
        invalid_payloads = [
            [], {}, {"train_ids": []}, {"train_ids": "not a list"},
            {"train_ids": [""]}, {"train_ids": [None]},
            {"train_ids": [sample_id, sample_id]},
            {"train_ids": [sample_id], "test_ids": "not a list"},
            {"train_ids": [sample_id], "test_ids": [1]},
            {"train_ids": [sample_id], "test_ids": [sample_id]},
            {"train_ids": [sample_id], "test_ids": [make_sample(2).sample_id] * 2},
            {"train_ids": [sample_id], "test_ids": ["mira:train:-1:open_ended:0"]},
            {"train_ids": ["mira:train:-1:open_ended:0"]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                finetune.load_split_manifest(self.manifest(payload))

    def test_ids_require_exact_source_category_and_canonical_indices(self):
        for split in ("train", "validation", "test"):
            self.assertEqual(
                finetune.parse_sample_id(f"mira:{split}:143951:multiple_choice:0"),
                (split, 143951, "multiple_choice", 0),
            )
        for sample_id in (
            "mira:train:-1:open_ended:0", "mira:train:01:open_ended:0",
            "mira:train:0:open_ended:-2", "mira:train:0:open_ended:01",
            "mira:unknown:0:open_ended:0", "mira:train:0:unknown:0",
            "mira:train:0:open_ended:0:extra", "mira:train:0:open_ended:0\n",
            "mira:../train:0:open_ended:0",
        ):
            with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                finetune.parse_sample_id(sample_id)

    def test_qa_selection_crosses_source_splits_and_restores_manifest_order(self):
        self.write_split("train", [
            {"open_ended": [
                {"question": "First\nquestion", "answer": "First answer"},
                {"question": "Second QA", "answer": ["A", "C"]},
            ], "multiple_choice": [{"question": "Choose", "options": ["A", "B"], "answer": "B"}]},
            {"open_ended": [{"question": "Held out QA", "answer": "Held out answer"}]},
        ], include_image_columns=True)
        self.write_split("validation", [{"open_ended": [{
            "question": "Validation QA", "answer": {"answer": "A", "explanation": "Evidence"},
        }]}])
        ids = [
            "mira:validation:0:open_ended:0", "mira:train:0:open_ended:1",
            "mira:train:0:multiple_choice:0", "mira:train:0:open_ended:0",
        ]
        samples = finetune.load_training_samples(self.root, ids)
        self.assertEqual([sample.sample_id for sample in samples], ids)
        self.assertEqual([sample.question for sample in samples],
                         ["Validation QA", "Second QA", "Choose", "First\nquestion"])
        self.assertEqual(samples[0].answer, {"answer": "A", "explanation": "Evidence"})
        self.assertEqual(samples[1].answer, ["A", "C"])
        self.assertEqual(samples[2].options, ["A", "B"])
        self.assertFalse((self.root / "missing" / "image.png").exists())
        self.assertNotIn("SECRET_CAPTION", finetune.sample_question(samples[-1]))

    def test_reader_does_not_require_images_or_caption_columns(self):
        self.write_split("test", [{"open_ended": [{"question": "Text only", "answer": "Answer"}]}])
        sample, = finetune.load_training_samples(self.root, ["mira:test:0:open_ended:0"])
        self.assertEqual(sample.question, "Text only")

    def test_unselected_rows_are_not_decoded_as_training_questions(self):
        with (self.root / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["vqa_json"])
            writer.writeheader()
            writer.writerow({"vqa_json": "not JSON and not selected"})
            writer.writerow({"vqa_json": json.dumps({"open_ended": [{
                "question": "Selected", "answer": "Answer",
            }]})})
            writer.writerow({"vqa_json": "also not selected"})
        sample, = finetune.load_training_samples(self.root, [make_sample(1).sample_id])
        self.assertEqual(sample.question, "Selected")

    def test_missing_row_category_or_qa_is_not_silently_dropped(self):
        self.write_split("train", [{"open_ended": [{"question": "Question", "answer": "Answer"}]}])
        for sample_id in ("mira:train:10:open_ended:0", "mira:train:0:multiple_choice:0",
                          "mira:train:0:open_ended:3"):
            for policy in ("error", "skip"):
                skipped = []
                with self.subTest(sample_id=sample_id, policy=policy), self.assertRaises(ValueError):
                    finetune.load_training_samples(
                        self.root, [sample_id], incomplete_policy=policy, skipped_incomplete=skipped,
                    )
                self.assertEqual(skipped, [])

    def test_missing_csv_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            finetune.load_training_samples(self.root, [make_sample().sample_id])

    def test_invalid_questions_fail_instead_of_being_converted_to_python_repr(self):
        for question in ("", "  ", None, ["question"], {"question": "value"}, 123):
            with self.subTest(question=question):
                self.write_split("train", [{"open_ended": [{"question": question, "answer": "Answer"}]}])
                with self.assertRaises(ValueError):
                    finetune.load_training_samples(self.root, [make_sample().sample_id])

    def test_skip_records_missing_fields_and_default_api_remains_strict(self):
        cases = [
            ({"answer": "Answer"}, ["question"]),
            ({"question": None, "answer": "Answer"}, ["question"]),
            ({"question": " \n\t", "answer": "Answer"}, ["question"]),
            ({"question": "Question"}, ["answer"]),
            ({"question": "Question", "answer": None}, ["answer"]),
            ({"question": "Question", "answer": ""}, ["answer"]),
            ({"question": "Question", "answer": " \n\t"}, ["answer"]),
            ({"question": "Question", "answer": {}}, ["answer"]),
            ({"question": "Question", "answer": []}, ["answer"]),
            ({}, ["question", "answer"]),
        ]
        for qa, missing_fields in cases:
            with self.subTest(qa=qa):
                self.write_split("train", [
                    {"open_ended": [qa]},
                    {"open_ended": [{"question": "Valid", "answer": "Original answer"}]},
                ])
                ids = [make_sample().sample_id, make_sample(1).sample_id]
                for kwargs in ({}, {"incomplete_policy": "error"}):
                    with self.assertRaisesRegex(ValueError, ids[0]):
                        finetune.load_training_samples(self.root, ids, **kwargs)
                sentinel = {"id": "previous", "missing_fields": ["answer"]}
                skipped = [sentinel]
                samples = finetune.load_training_samples(
                    self.root, ids, incomplete_policy="skip", skipped_incomplete=skipped,
                )
                self.assertEqual([sample.sample_id for sample in samples], ids[1:])
                self.assertEqual(samples[0].answer, "Original answer")
                self.assertEqual(skipped, [sentinel, {"id": ids[0], "missing_fields": missing_fields}])

    def test_skip_retains_false_zero_and_structured_answers_in_manifest_order(self):
        answers = [False, 0, ["A", "B"], {"answer": "A", "explanation": "Evidence"}]
        self.write_split("train", [
            {"open_ended": [{"question": f"Question {index}", "answer": answer}]}
            for index, answer in enumerate(answers)
        ])
        ids = [make_sample(index).sample_id for index in (3, 1, 2, 0)]
        skipped = []
        samples = finetune.load_training_samples(
            self.root, ids, incomplete_policy="skip", skipped_incomplete=skipped,
        )
        self.assertEqual([sample.sample_id for sample in samples], ids)
        for sample, index in zip(samples, (3, 1, 2, 0)):
            self.assertEqual(sample.answer, answers[index])
            self.assertIs(type(sample.answer), type(answers[index]))
        self.assertEqual(skipped, [])

    def test_missing_answer_with_malformed_question_is_audited_without_inventing_qa(self):
        self.write_split("train", [
            {"open_ended": [{"question": {"text": "Explanation", "visual_evidence": "Figure"}}]},
            {"open_ended": [{"question": "Valid", "answer": "Answer"}]},
        ])
        ids = [make_sample().sample_id, make_sample(1).sample_id]
        skipped = []
        samples = finetune.load_training_samples(
            self.root, ids, incomplete_policy="skip", skipped_incomplete=skipped,
        )
        self.assertEqual([sample.sample_id for sample in samples], ids[1:])
        self.assertEqual(skipped, [{
            "id": ids[0], "missing_fields": ["question", "answer"], "invalid_fields": ["question"],
        }])

    def test_skip_does_not_hide_corrupt_json_qa_objects_or_nonstring_complete_questions(self):
        corrupt_payloads = [
            [], {"open_ended": {}}, {"open_ended": [None]}, {"open_ended": ["not a QA"]},
            *({"open_ended": [{"question": value, "answer": "Answer"}]}
              for value in ([], {}, ["Question"], {"question": "Question"}, 123, False)),
        ]
        for payload in corrupt_payloads:
            with self.subTest(payload=payload):
                self.write_split("train", [payload])
                with self.assertRaises(ValueError):
                    finetune.load_training_samples(
                        self.root, [make_sample().sample_id], incomplete_policy="skip",
                    )
        (self.root / "train.csv").write_text('vqa_json\nnot-json\n', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            finetune.load_training_samples(self.root, [make_sample().sample_id], incomplete_policy="skip")
        for malformed_csv in ('vqa_json\n"unterminated\n', 'vqa_json\n{},extra\n', 'wrong_column\n{}\n'):
            with self.subTest(csv=malformed_csv):
                (self.root / "train.csv").write_text(malformed_csv, encoding="utf-8")
                with self.assertRaises((ValueError, csv.Error)):
                    finetune.load_training_samples(
                        self.root, [make_sample().sample_id], incomplete_policy="skip",
                    )

    def test_all_incomplete_fails_after_recording_every_skipped_id(self):
        self.write_split("train", [{"open_ended": [{}]}, {"open_ended": [{"question": "Question"}]}])
        ids = [make_sample(1).sample_id, make_sample().sample_id]
        skipped = []
        with self.assertRaisesRegex(ValueError, "没有可训练"):
            finetune.load_training_samples(
                self.root, ids, incomplete_policy="skip", skipped_incomplete=skipped,
            )
        self.assertEqual([item["id"] for item in skipped], ids)

    def test_skip_console_examples_are_bounded_but_audit_is_complete(self):
        self.write_split("train", [
            *({"open_ended": [{"question": "Missing answer"}]} for _ in range(12)),
            {"open_ended": [{"question": "Valid", "answer": "Answer"}]},
        ])
        ids = [make_sample(index).sample_id for index in range(13)]
        skipped, output = [], io.StringIO()
        with redirect_stdout(output):
            finetune.load_training_samples(
                self.root, ids, incomplete_policy="skip", skipped_incomplete=skipped,
            )
        self.assertEqual(len(skipped), 12)
        self.assertEqual(sum(sample_id in output.getvalue() for sample_id in ids), 10)
        self.assertIn("12", output.getvalue())

    def selection_args(self, *extra):
        self.write_split("train", [
            {"open_ended": [{"question": "First valid", "answer": "First answer"}]},
            {"open_ended": [{"question": "Missing answer"}]},
            {"open_ended": [{"question": "Last valid", "answer": "Last answer"}]},
            {"open_ended": [{"question": "Outside limit", "answer": "Do not backfill"}]},
            {"open_ended": [{"question": "Held out", "answer": "Do not use for training"}]},
        ])
        manifest = self.manifest({
            "train_ids": [make_sample(index).sample_id for index in (2, 1, 0, 3)],
            "test_ids": [make_sample(4).sample_id], "data_root": str(self.root),
        })
        return finetune.build_parser().parse_args([
            "--split-manifest", str(manifest), "--model-dir", str(self.root / "base"),
            "--output-dir", str(self.root / "adapter"), *extra,
        ])

    def test_prepare_filters_only_selected_prefix_and_preserves_audit_and_manifest(self):
        args = self.selection_args("--limit", "3")
        original_manifest = args.split_manifest.read_bytes()
        samples, manifest = finetune.prepare_samples(args)
        self.assertEqual([sample.sample_id for sample in samples], [make_sample(2).sample_id, make_sample().sample_id])
        self.assertEqual(args.data_selection_audit, {
            "selection_mode": "split_manifest", "source_splits": ["manifest_train_ids"],
            "incomplete_samples_policy": "skip", "manifest_train_count": 4,
            "selected_train_count": 3, "actual_train_count": 2, "skipped_incomplete_count": 1,
            "skipped_incomplete_ids": [make_sample(1).sample_id],
            "skipped_incomplete_details": [{"id": make_sample(1).sample_id, "missing_fields": ["answer"]}],
        })
        self.assertEqual(manifest, json.loads(original_manifest.decode("utf-8-sig")))
        self.assertEqual(args.split_manifest.read_bytes(), original_manifest)
        self.assertFalse(args.output_dir.exists())

    def test_prepare_strict_mode_and_all_incomplete_limit_do_not_backfill(self):
        args = self.selection_args("--incomplete-samples", "error")
        with self.assertRaisesRegex(ValueError, make_sample(1).sample_id):
            finetune.prepare_samples(args)
        args.incomplete_samples, args.limit = "skip", 1
        args.split_manifest = self.manifest({
            "train_ids": [make_sample(1).sample_id, make_sample().sample_id], "data_root": str(self.root),
        })
        with self.assertRaisesRegex(ValueError, "没有可训练"):
            finetune.prepare_samples(args)

    def test_validation_modes_filter_data_without_writing_training_audit(self):
        args = self.selection_args()
        args.model_dir.mkdir()
        (args.model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 4096,
        }), encoding="utf-8")
        original_files = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for mode in ("--dry-run", "--check-data"):
            with self.subTest(mode=mode), patch.object(finetune, "prepare_tokens") as tokenize:
                status = finetune.main([
                    mode, "--split-manifest", str(args.split_manifest), "--model-dir", str(args.model_dir),
                    "--output-dir", str(args.output_dir),
                ])
                self.assertEqual(status, 0)
                self.assertEqual(tokenize.call_count, int(mode == "--check-data"))
                if tokenize.called:
                    self.assertEqual(len(tokenize.call_args.args[1]), 3)
                self.assertFalse(args.output_dir.exists())
                self.assertEqual(
                    {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, original_files,
                )

    def test_training_persists_audit_before_train_and_retains_actual_ids_in_metadata(self):
        args = self.selection_args("--limit", "3")
        model, tokenizer = Mock(), Mock(pad_token_id=99)
        linear = type("FakeLinear", (), {})
        model.named_modules.return_value = [(target, linear()) for target in finetune.TARGET_MODULES.split(",")]
        model.named_parameters.return_value = [("lora_weight", SimpleNamespace(requires_grad=True))]
        trainer = Mock(state=SimpleNamespace(global_step=1))
        audit_at_train = []

        def train_without_model():
            audit_at_train.append(json.loads((args.output_dir / "data_selection.json").read_text(encoding="utf-8")))

        trainer.train.side_effect = train_without_model
        modules = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Linear=linear)),
            "peft": SimpleNamespace(LoraConfig=Mock(), TaskType=SimpleNamespace(CAUSAL_LM="causal"),
                                    get_peft_model=Mock(return_value=model)),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
                Trainer=Mock(return_value=trainer), TrainerCallback=type("FakeCallback", (), {}),
                TrainingArguments=Mock(), set_seed=Mock(),
            ),
            "transformers.trainer_callback": SimpleNamespace(
                PrinterCallback=type("FakePrinterCallback", (), {}),
            ),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(finetune, "dependency_report", return_value=([], [])), \
                patch.object(finetune, "prepare_tokens", return_value=(tokenizer, [], {})), \
                patch.object(finetune, "device_and_dtype", return_value=("cpu", "float32", "fake_dtype")), \
                patch.object(finetune.metadata, "version", return_value="4.57.0"):
            finetune.train(args)
        trainer.train.assert_called_once_with()
        self.assertEqual(len(audit_at_train), 1)
        saved = json.loads((args.output_dir / "training_metadata.json").read_text(encoding="utf-8"))
        for key, value in args.data_selection_audit.items():
            self.assertEqual(saved[key], value)
            self.assertEqual(audit_at_train[0][key], value)
        self.assertEqual(saved["train_ids"], [make_sample(2).sample_id, make_sample().sample_id])
        self.assertEqual(audit_at_train[0]["split_manifest"], str(args.split_manifest))
        self.assertEqual(audit_at_train[0]["data_root"], str(self.root.resolve()))

    def test_question_contains_options_but_never_the_answer(self):
        sample = make_sample(question="Choose a side", options={"A": "Left", "B": "Right"},
                             answer="SECRET_ANSWER")
        prompt = finetune.sample_question(sample)
        for expected in ("Choose a side", "A", "Left", "B", "Right"):
            self.assertIn(expected, prompt)
        self.assertNotIn("SECRET_ANSWER", prompt)

    def test_structured_targets_preserve_full_answer_and_explanation(self):
        for answer in ({"answer": ["A", "C"], "explanation": "Evidence", "confidence": 0.9},
                       ["A", "C"], 0, False):
            with self.subTest(answer=answer):
                self.assertEqual(json.loads(finetune.sample_answer(make_sample(answer=answer))), answer)

    def test_output_guard_protects_base_and_disallows_overwriting_adapters(self):
        model_dir = self.root / "models" / "base"
        model_dir.mkdir(parents=True)
        for output_dir in (model_dir, model_dir / "adapter", self.root / "models", self.root):
            with self.subTest(output_dir=output_dir), self.assertRaises(ValueError):
                finetune.output_is_safe(model_dir, output_dir)
        output = self.root / "adapters" / "new"
        finetune.output_is_safe(model_dir, output)
        output.mkdir(parents=True)
        finetune.output_is_safe(model_dir, output)
        (output / "adapter_config.json").write_text("{}", encoding="ascii")
        with self.assertRaises(ValueError):
            finetune.output_is_safe(model_dir, output)
        output_file = self.root / "existing_file"
        output_file.write_text("preserve", encoding="ascii")
        with self.assertRaises(ValueError):
            finetune.output_is_safe(model_dir, output_file)
        self.assertEqual(output_file.read_text(encoding="ascii"), "preserve")


class FakeTokenizer:
    """DeepSeek-like template: generation opens thinking; assistant strips it."""

    chat_template = "DeepSeek-style test template"
    eos_token = "[EOS]"
    eos_token_id = 99
    pad_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        result = "[BOS]"
        for message in messages:
            content = message["content"]
            if message["role"] == "assistant":
                content = content.rsplit("</think>", 1)[-1]
                result += "[ASSISTANT]" + content + self.eos_token
            elif message["role"] == "user":
                result += "[USER]" + content
            else:
                result += content
        if add_generation_prompt:
            result += "[ASSISTANT]<think>\n"
        return self.encode(result) if tokenize else result

    def encode(self, text, add_special_tokens=False, **kwargs):
        result = []
        while text:
            if text.startswith(self.eos_token):
                result.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            else:
                result.append(ord(text[0]) + 100)
                text = text[1:]
        return result

    def __call__(self, text, add_special_tokens=False, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TokenPreparationTests(unittest.TestCase):
    def test_prompt_uses_empty_assistant_prefix_without_generation_thinking(self):
        tokenizer = FakeTokenizer()
        prompt = finetune.render_prompt(tokenizer, "QUESTION", "SYSTEM")
        self.assertEqual(prompt, "[BOS]SYSTEM[USER]QUESTION[ASSISTANT]")
        self.assertNotIn("<think>", prompt)
        self.assertNotIn(tokenizer.eos_token, prompt)

    def test_only_answer_and_eos_are_supervised(self):
        tokenizer = FakeTokenizer()
        sample = make_sample(question="QUESTION", answer="ANSWER")
        feature = finetune.build_feature(sample, tokenizer, 256, "SYSTEM")
        prompt = finetune.render_prompt(tokenizer, finetune.sample_question(sample), "SYSTEM")
        prefix_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode("ANSWER" + tokenizer.eos_token)
        self.assertEqual(feature["input_ids"], prefix_ids + answer_ids)
        self.assertEqual(feature["labels"], [-100] * len(prefix_ids) + answer_ids)
        self.assertEqual(feature["attention_mask"], [1] * len(feature["input_ids"]))
        self.assertEqual(feature["labels"][-1], tokenizer.eos_token_id)

    def test_template_must_not_silently_remove_original_answer_content(self):
        tokenizer = FakeTokenizer()
        answer = "Original reasoning</think>Original answer"
        with self.assertRaises(ValueError):
            finetune.build_feature(make_sample(answer=answer), tokenizer, 512, "SYSTEM")

    def test_overlength_samples_fail_without_silent_truncation(self):
        tokenizer = FakeTokenizer()
        sample = make_sample(answer="A" * 300)
        with self.assertRaises(ValueError):
            finetune.build_feature(sample, tokenizer, 64, "SYSTEM")

    def test_token_boundary_mismatch_fails_instead_of_masking_answer_tokens(self):
        class BoundaryMergingTokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                if "[ASSISTANT]ANSWER" in text:
                    boundary = len(super().encode(text.split("ANSWER", 1)[0]))
                    ids[boundary - 1:boundary + 1] = [90001]
                return ids

        with self.assertRaises(ValueError):
            finetune.build_feature(make_sample(answer="ANSWER"), BoundaryMergingTokenizer(), 256, "SYSTEM")


@unittest.skipIf(torch is None, "torch is needed only for tensor collation tests")
class CollatorTests(unittest.TestCase):
    def test_right_padding_masks_padding_but_keeps_real_eos_labels(self):
        features = [
            {"input_ids": [11, 22, 99], "attention_mask": [1, 1, 1], "labels": [-100, 22, 99]},
            {"input_ids": [11, 33, 44, 99], "attention_mask": [1, 1, 1, 1], "labels": [-100, 33, 44, 99]},
        ]
        original = json.loads(json.dumps(features))
        result = finetune.TextCollator(pad_token_id=99)(features)
        self.assertEqual(result["input_ids"].tolist(), [[11, 22, 99, 99], [11, 33, 44, 99]])
        self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(result["labels"].tolist(), [[-100, 22, 99, -100], [-100, 33, 44, 99]])
        self.assertEqual(result["input_ids"].dtype, torch.long)
        self.assertEqual(features, original)

    def test_partial_final_batch_does_not_require_configured_batch_size(self):
        feature = {"input_ids": [11, 22, 99], "attention_mask": [1, 1, 1], "labels": [-100, 22, 99]}
        result = finetune.TextCollator(pad_token_id=99)([feature])
        self.assertEqual(tuple(result["input_ids"].shape), (1, 3))
        self.assertEqual(result["labels"].tolist(), [[-100, 22, 99]])


class ProgressTests(unittest.TestCase):
    def test_each_optimizer_step_reports_speed_eta_loss_and_final_duration(self):
        callback = finetune.TrainingProgressCallback(effective_batch_size=8)
        state = SimpleNamespace(global_step=0, max_steps=4)
        output = io.StringIO()
        with redirect_stdout(output), patch.object(finetune.time, "perf_counter", side_effect=[100.0, 102.0, 105.0]):
            callback.on_train_begin(None, state, None)
            state.global_step = 1
            callback.on_step_end(None, state, None)
            callback.on_step_end(None, state, None)
            callback.on_log(None, state, None, logs={"loss": 1.25})
            callback.on_train_end(None, state, None)
        result = output.getvalue()
        self.assertGreaterEqual(result.count("[train]"), 2)
        self.assertGreaterEqual(result.count("\r"), 3)
        self.assertIn("0.500 step/s", result)
        self.assertIn("4.00 QA/s", result)
        self.assertIn("ETA 00:00:06", result)
        self.assertIn("1.250000", result)
        self.assertNotIn("[loss]", result)
        self.assertIn("00:00:05", result)


if __name__ == "__main__":
    unittest.main()
