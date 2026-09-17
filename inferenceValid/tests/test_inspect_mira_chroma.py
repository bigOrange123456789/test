"""Tests for bounded sampling and existing-database-only MIRA inspection."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
import numpy as np
from chromadb.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inspect_mira_chroma as inspector


class InspectorTests(unittest.TestCase):
    """Check Chroma reads, sampling, validation, and display without a model."""

    def setUp(self):
        """Create an independent temporary database with four normalized vectors."""
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temporary.name)
        self.client = chromadb.PersistentClient(path=str(self.root), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.create_collection("mira_test", embedding_function=None)
        vectors = np.eye(4, inspector.EXPECTED_DIMENSION, dtype=np.float32)
        self.collection.add(ids=[f"id-{i}" for i in range(4)], embeddings=vectors,
                            documents=[f"question {i}\nanswer {i}" for i in range(4)],
                            metadatas=[{"image_paths": '["images/a.jpg"]'} for _ in range(4)])
        self.args = inspector.parse_args(["--db-dir", str(self.root), "--collection", "mira_test", "--seed", "7"])

    def tearDown(self):
        """Clean up independent test data, tolerating native handles on Windows."""
        self.temporary.cleanup()

    def capture(self, collection=None):
        """Collect a report in memory so its text and exit status can be asserted."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = inspector.inspect_collection(self.collection if collection is None else collection, self.args)
        return status, output.getvalue()

    def test_existing_collection_samples_without_writing(self):
        """Confirm seeded sampling, vector formats, and unchanged stored records."""
        before = self.collection.get(include=["documents", "metadatas"])
        client, collection = inspector.open_collection(self.args)
        status, output = self.capture(collection)
        self.assertEqual(status, 0)
        self.assertEqual(output.count("ID: id-"), 3)
        self.assertIn("shape=(2048,)", output)
        self.assertIn("images/a.jpg", output)
        self.assertEqual(self.capture(collection)[1], output)
        self.assertEqual(self.collection.count(), 4)
        self.assertEqual(self.collection.get(include=["documents", "metadatas"]), before)

    def test_missing_database_is_not_created(self):
        """Reject a typo in the database path without creating directories."""
        self.args.db_dir = self.root / "not-created"
        with self.assertRaises(FileNotFoundError):
            inspector.open_collection(self.args)
        self.assertFalse(self.args.db_dir.exists())

    def test_missing_collection_is_not_created(self):
        """List existing names when the requested collection does not exist."""
        self.args.collection = "missing"
        with self.assertRaisesRegex(ValueError, "mira_test"):
            inspector.open_collection(self.args)
        self.assertEqual([c.name for c in self.client.list_collections()], ["mira_test"])

    def test_empty_collection_does_not_claim_success(self):
        """Report an empty database as unverified rather than successfully encoded."""
        empty = self.client.create_collection("empty", embedding_function=None)
        status, output = self.capture(empty)
        self.assertEqual(status, 1)
        self.assertIn("集合为空", output)

    def test_limit_is_clamped_and_full_vector_is_printed(self):
        """Allow complete values and long text without duplicates in a small collection."""
        self.args.sample_size = 20
        self.args.vector_values = 0
        self.args.text_limit = 0
        status, output = self.capture()
        self.assertEqual(status, 0)
        self.assertEqual(output.count("ID: id-"), 4)
        vectors = [json.loads(line) for line in output.splitlines() if line.startswith("[0.0") or line.startswith("[1.0")]
        self.assertEqual(len(vectors), 4)
        self.assertTrue(all(len(vector) == 2048 for vector in vectors))

    def test_invalid_vectors_and_missing_content_fail_checks(self):
        """Detect zero vectors, wrong dimensions, nonfinite values, and missing fields."""
        for vector in (None, np.zeros(2048), np.ones(4), np.full(2048, np.nan), np.full(2048, np.inf)):
            with self.subTest(vector_type=type(vector)):
                self.assertTrue(inspector.vector_stats(vector)["issues"])
        self.collection.add(ids=["invalid"], embeddings=[np.zeros(2048).tolist()])
        self.args.sample_size = 100
        status, output = self.capture()
        self.assertEqual(status, 1)
        self.assertIn("缺少问答原文", output)
        self.assertIn("缺少元数据", output)

    def test_large_collection_only_fetches_requested_records(self):
        """Use random offsets with limit=1 instead of retrieving all IDs or embeddings."""
        fake = Mock()
        fake.count.return_value = 1_120_031
        fake.metadata = {}
        fake.get.side_effect = lambda **kw: {"ids": [f"id-{kw['offset']}"], "embeddings": [np.eye(1, 2048)[0]],
                                            "documents": ["question/answer"], "metadatas": [{"split": "train"}]}
        self.assertEqual(self.capture(fake)[0], 0)
        self.assertEqual(fake.get.call_count, 3)
        offsets = [call.kwargs["offset"] for call in fake.get.call_args_list]
        self.assertEqual(len(set(offsets)), 3)
        self.assertTrue(all(call.kwargs["limit"] == 1 for call in fake.get.call_args_list))

    def test_concurrent_change_warns_and_missing_record_fails(self):
        """Do not claim a consistent snapshot if the collection changes while reading."""
        with patch.object(type(self.collection), "count", side_effect=[4, 5]):
            self.assertIn("不是全库一致性快照", self.capture()[1])
        with patch.object(type(self.collection), "get", return_value={"ids": []}):
            self.assertEqual(self.capture()[0], 1)

    def test_shortening_and_argument_validation(self):
        """Make truncation explicit and reject nonpositive sample counts."""
        self.assertEqual(inspector.shorten("abcdef", 0), "abcdef")
        self.assertIn("省略 3 个字符", inspector.shorten("abcdef", 3))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            inspector.parse_args(["--sample-size", "0"])


if __name__ == "__main__":
    unittest.main()
