import csv
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script import finetune_qwen3_vl_lora as finetune

try:
    import torch
except ImportError:
    torch = None


def make_sample(index=0, **changes):
    sample = finetune.Sample(
        id=f"mira:train:{index}:open_ended:0", split="train", row_index=index,
        category="open_ended", qa_index=0, question=f"Question {index}",
        answer="Target answer", options=None, caption="Source caption",
        images=[f"images/{index}.png"], next_cursor={},
    )
    return replace(sample, **changes)


class ManifestAndDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def manifest(self, payload):
        path = self.root / "split.json"
        path.write_text(json.dumps(payload), encoding="utf-8-sig")
        return path

    def write_split(self, split, rows):
        with (self.root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "caption", "vqa_json"])
            writer.writeheader()
            for index, questions in enumerate(rows):
                writer.writerow({
                    "image_path": f"images/{index}.png", "caption": "Source caption",
                    "vqa_json": json.dumps({"open_ended": questions}),
                })

    def test_manifest_preserves_training_order_and_metadata(self):
        payload = {
            "train_ids": [make_sample(3).id, make_sample(1).id],
            "test_ids": [make_sample(2).id], "seed": 42,
        }
        ids, loaded = finetune.load_split_manifest(self.manifest(payload))
        self.assertEqual(ids, payload["train_ids"])
        self.assertEqual(loaded, payload)

    def test_manifest_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            finetune.load_split_manifest(self.manifest({
                "train_ids": [make_sample().id], "test_ids": [make_sample().id],
            }))

    def test_manifest_rejects_invalid_training_and_test_arrays(self):
        invalid_payloads = [
            [], {}, {"train_ids": []}, {"train_ids": "not a list"},
            {"train_ids": [""]}, {"train_ids": [None]},
            {"train_ids": [make_sample().id, make_sample().id]},
            {"train_ids": [make_sample().id], "test_ids": "not a list"},
            {"train_ids": [make_sample().id], "test_ids": [1]},
            {"train_ids": ["mira:train:-1:open_ended:0"]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                finetune.load_split_manifest(self.manifest(payload))

    def test_ids_require_canonical_nonnegative_indices_and_known_categories(self):
        for split in ("train", "validation", "test"):
            self.assertEqual(finetune.id_split(f"mira:{split}:0:multiple_choice:12"), split)
        for sample_id in (
            "mira:train:-1:open_ended:0", "mira:train:01:open_ended:0",
            "mira:train:0:open_ended:-2", "mira:train:0:open_ended:01",
            "mira:unknown:0:open_ended:0", "mira:train:0:unknown:0",
            "mira:train:0:open_ended:0:extra", "mira:train:0:open_ended:0\n",
        ):
            with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                finetune.id_split(sample_id)

    def test_resolve_selected_qas_in_manifest_order_across_csv_splits(self):
        self.write_split("train", [
            [{"question": "First QA", "answer": "First answer"},
             {"question": "Second QA", "answer": "Second answer"}],
            [{"question": "Held out QA", "answer": "Held out answer"}],
        ])
        self.write_split("validation", [[{"question": "Validation QA", "answer": {"label": "A"}}]])
        ids = ["mira:validation:0:open_ended:0", "mira:train:0:open_ended:1"]
        samples = finetune.load_training_samples(self.root, ids)
        self.assertEqual([sample.id for sample in samples], ids)
        self.assertEqual([sample.question for sample in samples], ["Validation QA", "Second QA"])
        self.assertEqual(samples[0].answer, {"label": "A"})

    def test_missing_selected_qa_is_not_silently_dropped(self):
        self.write_split("train", [[{"question": "Question", "answer": "Answer"}]])
        with self.assertRaisesRegex(ValueError, "not found"):
            finetune.load_training_samples(self.root, [make_sample(10).id])

    def test_missing_source_split_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            finetune.load_training_samples(self.root, [make_sample().id])

    def test_user_prompt_does_not_leak_answer_rationale_or_caption(self):
        sample = make_sample(
            options={"A": "Left", "B": "Right"}, answer="SECRET_ANSWER",
            extra={"rationale": "SECRET_RATIONALE"}, caption="SECRET_CAPTION",
        )
        question = finetune.sample_question(sample)
        self.assertIn(sample.question, question)
        self.assertIn('"A":"Left"', question)
        for secret in ("SECRET_ANSWER", "SECRET_RATIONALE", "SECRET_CAPTION"):
            self.assertNotIn(secret, question)

    def test_structured_target_preserves_answer_and_explanation(self):
        answer = {"answer": ["A", "C"], "explanation": "Evidence", "confidence": 0.9}
        self.assertEqual(json.loads(finetune.sample_answer(make_sample(answer=answer))), answer)

    def test_dataset_skips_missing_question_or_answer_and_records_ids(self):
        samples = [make_sample(), make_sample(1, question=""), make_sample(2, answer=None)]
        dataset = finetune.MIRATrainingDataset(samples, self.root)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].sample.id, samples[0].id)
        self.assertEqual(dataset.skipped_ids, [samples[1].id, samples[2].id])

    def test_output_guard_protects_model_ancestors_and_descendants(self):
        model_dir = self.root / "models" / "base"
        model_dir.mkdir(parents=True)
        for output_dir in (model_dir, model_dir / "adapter", self.root / "models", self.root):
            with self.subTest(output_dir=output_dir), self.assertRaisesRegex(ValueError, "original weights"):
                finetune.output_is_safe(model_dir, output_dir, allow_existing=True)
        finetune.output_is_safe(model_dir, self.root / "adapters" / "new")

    def test_output_guard_requires_opt_in_for_existing_nonempty_adapter(self):
        output_dir = self.root / "adapter"
        output_dir.mkdir()
        (output_dir / "adapter_config.json").write_text("{}", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "not empty"):
            finetune.output_is_safe(self.root / "base", output_dir)
        finetune.output_is_safe(self.root / "base", output_dir, allow_existing=True)


class ProgressTests(unittest.TestCase):
    def test_duration_formats_elapsed_time_and_unknown_eta(self):
        self.assertEqual(finetune.format_duration(3661.9), "01:01:01")
        self.assertEqual(finetune.format_duration(0), "00:00:00")
        for seconds in (-1, float("inf"), float("nan")):
            self.assertEqual(finetune.format_duration(seconds), "--:--:--")

    def test_each_optimizer_step_reports_speed_eta_and_final_duration(self):
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
        self.assertEqual(result.count("[train]"), 1)
        self.assertIn("0.500 step/s", result)
        self.assertIn("4.00 QA/s", result)
        self.assertIn("ETA 00:00:06", result)
        self.assertIn("1.250000", result)
        self.assertIn("00:00:05 (5.00 seconds)", result)

    def test_resume_speed_uses_steps_since_resume(self):
        callback = finetune.TrainingProgressCallback(effective_batch_size=4)
        state = SimpleNamespace(global_step=80, max_steps=100)
        output = io.StringIO()
        with redirect_stdout(output), patch.object(finetune.time, "perf_counter", side_effect=[100.0, 102.0]):
            callback.on_train_begin(None, state, None)
            state.global_step = 81
            callback.on_step_end(None, state, None)
        self.assertIn("0.500 step/s", output.getvalue())
        self.assertIn("ETA 00:00:38", output.getvalue())


class FakeImage:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeProcessor:
    """Emulate visual-token expansion; a tokenizer-only prefix would be shorter."""

    def __init__(self, sequences, prefixes=None):
        self.sequences = sequences
        self.prefixes = prefixes or [[10, 20, 20, 20, 30] for _ in sequences]
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prompt" if add_generation_prompt else "full"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        index = (len(self.calls) - 1) // 2
        tokens = self.prefixes[index] if kwargs["text"] == ["prompt"] else self.sequences[index]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "pixel_values": torch.full((2, 3), float(index + 1)),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


@unittest.skipIf(torch is None, "torch is needed only for tensor collation tests")
class CollatorTests(unittest.TestCase):
    def collator(self, processor, max_length=64, pad_token_id=99):
        tokenizer = SimpleNamespace(pad_token_id=pad_token_id, eos_token_id=99)
        return finetune.QwenVLDataCollator(processor, tokenizer, max_length, 4096, 262144, "System")

    def record(self, index=0, **changes):
        return finetune.MIRARecord(make_sample(index, **changes), Path("dataset"))

    def test_expanded_image_prompt_and_batch_padding_mask_preserve_answer_eos(self):
        processor = FakeProcessor([
            [10, 20, 20, 20, 30, 40, 99],
            [10, 20, 20, 20, 30, 41, 42, 99],
        ])
        images = [FakeImage(), FakeImage()]
        with patch.object(finetune, "read_image", side_effect=images):
            batch = self.collator(processor)([self.record(), self.record(1)])
        self.assertEqual(batch["labels"].tolist(), [
            [-100, -100, -100, -100, -100, 40, 99, -100],
            [-100, -100, -100, -100, -100, 41, 42, 99],
        ])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 1, 1, 1, 1, 0])
        self.assertEqual(batch["input_ids"][0, -1].item(), 99)
        self.assertEqual(tuple(batch["pixel_values"].shape), (4, 3))
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 2, 2], [1, 2, 2]])
        self.assertEqual(len(processor.calls), 4)
        for index, image in enumerate(images):
            self.assertIs(processor.calls[index * 2]["images"][0], image)
            self.assertIs(processor.calls[index * 2 + 1]["images"][0], image)
            self.assertFalse(processor.calls[index * 2]["truncation"])
            self.assertTrue(image.closed)

    def test_eos_fallback_for_missing_pad_token(self):
        processor = FakeProcessor([[10, 20, 20, 20, 30, 40, 99]])
        with patch.object(finetune, "read_image", return_value=FakeImage()):
            batch = self.collator(processor, pad_token_id=None)([self.record()])
        self.assertEqual(batch["labels"][0, -1].item(), 99)

    def test_mismatched_processed_prefix_fails_instead_of_mislabeling(self):
        processor = FakeProcessor([[10, 20, 20, 20, 30, 40, 99]], prefixes=[[10, 21, 30]])
        image = FakeImage()
        with patch.object(finetune, "read_image", return_value=image), self.assertRaisesRegex(ValueError, "processed prompt"):
            self.collator(processor)([self.record()])
        self.assertTrue(image.closed)

    def test_overlong_sequence_fails_without_truncating_visual_or_target_tokens(self):
        processor = FakeProcessor([[10, 20, 20, 20, 30, 40, 99]])
        with patch.object(finetune, "read_image", return_value=FakeImage()), self.assertRaisesRegex(ValueError, "exceed --max-length"):
            self.collator(processor, max_length=6)([self.record()])

    def test_empty_completion_is_rejected(self):
        processor = FakeProcessor([[10, 20, 20, 20, 30]])
        with patch.object(finetune, "read_image", return_value=FakeImage()), self.assertRaisesRegex(ValueError, "no supervised answer tokens"):
            self.collator(processor)([self.record()])

    def test_missing_later_image_closes_already_loaded_images(self):
        processor = FakeProcessor([[10, 20, 20, 20, 30, 40, 99]])
        image = FakeImage()
        with patch.object(finetune, "read_image", side_effect=[image, FileNotFoundError("missing.png")]), self.assertRaises(FileNotFoundError):
            self.collator(processor)([self.record(images=["exists.png", "missing.png"])])
        self.assertTrue(image.closed)
        self.assertEqual(processor.calls, [])


if __name__ == "__main__":
    unittest.main()
