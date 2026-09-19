"""Offline tests for the canonical MIRA CSV to evaluation-record adapter."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script.lib.mira_eval_data import load_mira_dataset


class MiraEvaluationDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write_split(self, split, groups):
        with (self.root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "caption", "vqa_json"])
            writer.writeheader()
            for group in groups:
                writer.writerow({"image_path": '["images/absent.png", "images/second.png"]',
                                 "caption": "SECRET_CAPTION\nsecond caption line",
                                 "vqa_json": json.dumps(group, ensure_ascii=False)})

    def test_preserves_ids_options_complete_answers_and_no_annotation_leak(self):
        answer = {"correct_options": ["A", "C"], "explanation": "完整解释", "visual_evidence": ["area 1"]}
        options = {"A": "first", "B": "second", "C": "third"}
        self.write_split("train", [{
            "open_ended": [{"question": "Question with\na newline?", "answer": "plain answer"}],
            "multiple_choice": [{"question": "Choose all.", "options": options, "answer": answer,
                                 "extra_hint": "SECRET_HINT"}],
        }, {"closed_ended": [{"question": "Is it present?", "answer": False}]}])
        rows = load_mira_dataset(self.root)
        self.assertEqual([row["id"] for row in rows], [
            "mira:train:0:open_ended:0", "mira:train:0:multiple_choice:0",
            "mira:train:1:closed_ended:0",
        ])
        self.assertEqual(rows[0]["question"], "Question with\na newline?")
        self.assertEqual(rows[0]["reference"], "plain answer")
        self.assertEqual(json.loads(rows[1]["reference"]), answer)
        self.assertEqual(json.loads(rows[1]["question"].split("\nOptions: ", 1)[1]), options)
        self.assertEqual(rows[2]["reference"], "false")
        for row in rows:
            self.assertEqual(set(row), {"id", "question", "reference", "images"})
            self.assertNotIn("SECRET", row["question"])
            self.assertNotIn("完整解释", row["question"])

    def test_missing_images_are_not_opened_and_paths_are_shared(self):
        self.write_split("train", [{"open_ended": [
            {"question": "First?", "answer": "a"},
            {"question": "Second?", "answer": "b"},
            {"question": "Third?", "answer": "c", "image_path": "custom\\override.png"},
            {"question": "Fourth?", "answer": "d", "image_paths": [str(self.root / "absolute.png")]},
        ]}])
        rows = load_mira_dataset(self.root)
        self.assertEqual(rows[0]["images"], [str(self.root / "images" / "absent.png"),
                                            str(self.root / "images" / "second.png")])
        self.assertIs(rows[0]["images"][0], rows[1]["images"][0])
        self.assertEqual(rows[2]["images"], [str(self.root / "custom" / "override.png")])
        self.assertEqual(rows[3]["images"], [str(self.root / "absolute.png")])
        self.assertTrue(all(Path(image).is_absolute() for row in rows for image in row["images"]))
        self.assertTrue(all(not Path(image).exists() for row in rows for image in row["images"]))

    def test_skips_incomplete_qa_with_warning_without_renumbering(self):
        self.write_split("train", [{"open_ended": [
            {"question": "Missing answer?"},
            {"answer": "Missing question"},
            {"question": "Whitespace answer?", "answer": "  "},
            {"question": "Complete?", "answer": "yes"},
        ]}])
        with self.assertLogs("rag_eval", level="WARNING") as captured:
            rows = load_mira_dataset(self.root)
        self.assertEqual([row["id"] for row in rows], ["mira:train:0:open_ended:3"])
        self.assertTrue(any("总计跳过 3" in message for message in captured.output))
        self.assertTrue(any("mira:train:0:open_ended:0" in message for message in captured.output))

    def test_default_only_train_and_explicit_sources_preserve_original_ids(self):
        for split in ("train", "validation", "test"):
            self.write_split(split, [{"open_ended": [{"question": "Question?", "answer": split}]}])
        self.assertEqual(len(load_mira_dataset(self.root)), 1)
        rows = load_mira_dataset(self.root, ("validation", "test", "train"))
        self.assertEqual([row["id"] for row in rows], [
            "mira:validation:0:open_ended:0", "mira:test:0:open_ended:0", "mira:train:0:open_ended:0",
        ])

    def test_rejects_invalid_sources_missing_csv_and_empty_dataset(self):
        for splits in ((), ("unknown",), ("train", "train"), "train"):
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                load_mira_dataset(self.root, splits)
        with self.assertRaises(FileNotFoundError):
            load_mira_dataset(self.root)
        self.write_split("train", [])
        with self.assertRaisesRegex(ValueError, "没有包含有效问题和答案"):
            load_mira_dataset(self.root)

    def test_malformed_source_fails_with_original_row_context(self):
        self.write_split("train", [{"open_ended": "must be a list"}])
        with self.assertRaisesRegex(ValueError, "train.csv 数据行 1"):
            load_mira_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
