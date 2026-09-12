"""Regression tests for MIRA grouping and crash-safe Chroma ingestion."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import embed_mira_chroma as mira


class SimulatedPowerLoss(BaseException):
    """Represent an abrupt stop between database persistence and checkpointing."""


class FakeEncoder:
    """Return small deterministic test embeddings in the production dimension."""

    calls = []
    loaded = 0

    def __init__(self, args):
        """Track lazy loading so completed jobs can be checked without a model."""
        type(self).loaded += 1

    def encode(self, samples):
        """Track sample IDs and retain a valid normalized vector for each sample."""
        type(self).calls.extend(sample.id for sample in samples)
        return [[1.0] + [0.0] * (mira.DIMENSION - 1) for sample in samples]


class MiraIndexTests(unittest.TestCase):
    """Exercise the real CSV reader, Chroma writes, and checkpoint transactions."""

    def setUp(self):
        """Build a two-row CSV with multiline text, all question types, and two images."""
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.model = self.root / "model"
        self.data.mkdir()
        (self.model / "1_Pooling").mkdir(parents=True)
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "sentence_bert_config.json").write_text(json.dumps({"transformer_task": "feature-extraction"}), encoding="utf-8")
        (self.model / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode": "lasttoken", "embedding_dimension": 2048}), encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"fixture-only")
        records = [
            {"image_path": '["images/a.jpg", "images/b.jpg"]', "caption": "caption\nsecond line",
             "vqa_json": json.dumps({
                 "open_ended": [{"question": "first\nquestion", "answer": {"text": "answer", "visual_evidence": "evidence"}},
                                {"question": "second", "answer": "two"}],
                 "closed_ended": [{"question": "third", "answer": False}],
                 "single_choice": [{"question": "fourth", "options": ["A", "B"], "answer": {"correct_option": "A"}}],
                 "multiple_choice": [{"question": "fifth", "answer": {"correct_options": ["A", "B"]}}],
             })},
            {"image_path": "images/c.jpg", "caption": "", "vqa_json": json.dumps({"open_ended": [{"question": "sixth", "visual_evidence": "source evidence"}]})},
        ]
        with (self.data / "train.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "caption", "vqa_json"])
            writer.writeheader()
            writer.writerows(records)
        FakeEncoder.calls = []
        FakeEncoder.loaded = 0

    def tearDown(self):
        """Close test-only Chroma systems before removing temporary Windows files."""
        from chromadb.api.shared_system_client import SharedSystemClient

        for system in list(SharedSystemClient._identifier_to_system.values()):
            system.stop()
        SharedSystemClient.clear_system_cache()
        self.temporary.cleanup()

    def args(self, *extra):
        """Construct CLI arguments scoped exclusively to the temporary fixture."""
        return mira.parse_args(["--data-root", str(self.data), "--model-dir", str(self.model),
                                "--db-dir", str(self.root / "chroma"), "--expected-count", "6",
                                "--commit-every", "2", *extra])

    def checkpoint(self):
        """Read the exact on-disk checkpoint, not the in-memory state."""
        return json.loads((self.root / "chroma/mira_qwen3_vl_embedding.checkpoint.json").read_text(encoding="utf-8"))

    def collection(self):
        """Reconnect to the test collection as a new reader."""
        import chromadb
        from chromadb.config import Settings

        return chromadb.PersistentClient(path=str(self.root / "chroma"), settings=Settings(anonymized_telemetry=False)).get_collection("mira_qwen3_vl_embedding", embedding_function=None)

    def test_flatten_and_resume_at_every_question(self):
        """Resuming any row-internal cursor must yield exactly the remaining IDs."""
        samples = list(mira.iter_samples(self.data, "train"))
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({sample.id for sample in samples}), 6)
        self.assertEqual(samples[0].images, ["images/a.jpg", "images/b.jpg"])
        self.assertIn("first\nquestion", samples[0].document())
        self.assertIn("evidence", samples[0].document())
        self.assertIn("correct_option", samples[3].document())
        for index, sample in enumerate(samples):
            resumed = list(mira.iter_samples(self.data, "train", sample.next_cursor))
            self.assertEqual([row.id for row in resumed], [row.id for row in samples[index + 1:]])

    def test_missing_source_fields_are_preserved(self):
        """Incomplete annotations remain identifiable records and are never invented."""
        sample = list(mira.iter_samples(self.data, "train"))[-1]
        self.assertEqual(sample.missing_fields, ["answer"])
        self.assertIn("[not provided in source]", sample.document())
        self.assertIn("source evidence", sample.document())
        self.assertFalse(sample.metadata()["has_answer"])
        self.assertEqual(mira.scan_dataset(self.args()), 6)

    def test_limited_run_and_normal_resume(self):
        """A limited run resumes mid-row and a completed rerun loads no model."""
        first = mira.run_index(self.args("--limit", "3"), FakeEncoder)
        self.assertEqual(first["count"], 3)
        self.assertEqual(self.checkpoint()["progress"]["train"]["cursor"]["qa"], 3)
        second = mira.run_index(self.args(), FakeEncoder)
        self.assertEqual(second["count"], 6)
        self.assertEqual(second["encoded_now"], 3)
        loaded = FakeEncoder.loaded
        third = mira.run_index(self.args(), FakeEncoder)
        self.assertEqual(third["encoded_now"], 0)
        self.assertEqual(FakeEncoder.loaded, loaded)
        self.assertEqual(len(set(FakeEncoder.calls)), 6)
        self.assertEqual(len(self.collection().get(where={"has_answer": False})["ids"]), 1)

    def test_crash_after_upsert_before_checkpoint(self):
        """Persisted IDs must not be encoded again after a checkpoint was not saved."""
        original = mira.Checkpoint.save
        saves = 0

        def fail_second_save(checkpoint):
            """Interrupt the first post-upsert checkpoint write."""
            nonlocal saves
            saves += 1
            if saves == 2:
                raise SimulatedPowerLoss()
            original(checkpoint)

        with patch.object(mira.Checkpoint, "save", fail_second_save):
            with self.assertRaises(SimulatedPowerLoss):
                mira.run_index(self.args(), FakeEncoder)
        self.assertEqual(self.collection().count(), 2)
        self.assertEqual(self.checkpoint()["progress"]["train"]["processed"], 0)
        result = mira.run_index(self.args(), FakeEncoder)
        self.assertEqual(result["count"], 6)
        self.assertEqual(result["encoded_now"], 4)
        self.assertEqual(len(FakeEncoder.calls), 6)

    def test_encoding_failure_does_not_advance_checkpoint(self):
        """A failed batch must remain resumable with no claimed database records."""
        with patch.object(FakeEncoder, "encode", side_effect=RuntimeError("simulated image failure")):
            with self.assertRaisesRegex(RuntimeError, "simulated image failure"):
                mira.run_index(self.args(), FakeEncoder)
        self.assertEqual(self.collection().count(), 0)
        self.assertEqual(self.checkpoint()["progress"]["train"]["processed"], 0)
        self.assertEqual(mira.run_index(self.args(), FakeEncoder)["count"], 6)

    def test_changed_config_and_source_are_rejected(self):
        """Do not mix incompatible embeddings or changed source rows in the same collection."""
        mira.run_index(self.args("--limit", "1"), FakeEncoder)
        resumed = mira.run_index(self.args("--max-seq-length", "16384", "--limit", "1"), FakeEncoder)
        self.assertEqual(resumed["count"], 2)
        with self.assertRaises(ValueError):
            mira.run_index(self.args("--max-pixels", "16384"), FakeEncoder)
        with (self.data / "train.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            mira.run_index(self.args(), FakeEncoder)

    def test_multiple_image_path_formats(self):
        """Support CSV single-image values and explicit multi-image lists."""
        self.assertEqual(mira.parse_image_paths("images/a.jpg"), ["images/a.jpg"])
        self.assertEqual(mira.parse_image_paths("['a.jpg', 'b.jpg']"), ["a.jpg", "b.jpg"])
        with self.assertRaises(ValueError):
            mira.parse_image_paths('["a.jpg", ""]')

    def test_malformed_question_object_is_preserved(self):
        """Keep an original malformed question object instead of guessing a question."""
        original_question = {"text": "An answer was incorrectly placed here", "visual_evidence": "A"}
        with (self.data / "train.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "vqa_json"])
            writer.writeheader()
            writer.writerow({"image_path": "images/a.jpg", "vqa_json": json.dumps({
                "open_ended": [{"question": original_question}],
            })})
        sample = next(mira.iter_samples(self.data, "train"))
        self.assertEqual(sample.missing_fields, ["question", "answer"])
        self.assertEqual(sample.extra["original_question"], original_question)
        self.assertIn("An answer was incorrectly placed here", sample.document())

    def test_instruct_model_is_not_accepted(self):
        """A chat model directory must not silently become an embedding source."""
        (self.model / "1_Pooling/config.json").unlink()
        with self.assertRaisesRegex(ValueError, "Instruct"):
            mira.model_identity(self.model)


if __name__ == "__main__":
    unittest.main()
