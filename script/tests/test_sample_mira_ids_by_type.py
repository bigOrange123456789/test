"""按题型抽样和全数据题型统计的离线回归测试。"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.sample_mira_ids import (
    QUESTION_TYPES,
    _validated_config,
    allocate_type_counts,
    sample_dataset,
    validate_type_ratios,
)


class QuestionTypeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for index in range(8):
            rows.append({
                "image_path": f"image-{index}.png",
                "vqa_json": json.dumps({category: [{"question": f"q-{index}-{category}", "answer": "a"}]
                                         for category in QUESTION_TYPES}),
            })
        with (self.root / "train.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_path", "vqa_json"))
            writer.writeheader()
            writer.writerows(rows)

    def test_exact_train_and_test_quotas_and_full_dataset_counts(self):
        quotas = {category: 2 for category in QUESTION_TYPES}
        with patch("script.sample_mira_ids.load_embedded_ids", return_value=set()), \
                patch("script.sample_mira_ids.load_keywords", return_value=["not-present"]):
            result = sample_dataset(self.root, train_counts=quotas, test_counts=quotas, seed=17)
        expected = {category: 8 for category in QUESTION_TYPES}
        self.assertEqual(result["question_type_counts"]["train"], expected)
        self.assertEqual(result["sampling_pool_question_type_counts"], expected)
        self.assertEqual(result["selected_question_type_counts"], {"train": quotas, "test": quotas})
        self.assertEqual(result["requested_question_type_counts"], {"train": quotas, "test": quotas})
        self.assertEqual(len(result["train_ids"]), 8)
        self.assertEqual(len(result["test_ids"]), 8)
        self.assertTrue(set(result["train_ids"]).isdisjoint(result["test_ids"]))

    def test_zero_quota_does_not_borrow_another_question_type(self):
        train = {"open_ended": 2, "closed_ended": 0, "single_choice": 0, "multiple_choice": 0}
        test = {"open_ended": 1, "closed_ended": 0, "single_choice": 0, "multiple_choice": 0}
        with patch("script.sample_mira_ids.load_embedded_ids", return_value=set()), \
                patch("script.sample_mira_ids.load_keywords", return_value=["not-present"]):
            result = sample_dataset(self.root, train_counts=train, test_counts=test)
        self.assertEqual(result["selected_question_type_counts"], {"train": train, "test": test})
        self.assertTrue(all(":open_ended:" in sample_id for sample_id in result["train_ids"] + result["test_ids"]))

    def test_shortage_reports_the_question_type(self):
        train = {"open_ended": 9, "closed_ended": 0, "single_choice": 0, "multiple_choice": 0}
        test = {"open_ended": 1, "closed_ended": 0, "single_choice": 0, "multiple_choice": 0}
        with patch("script.sample_mira_ids.load_embedded_ids", return_value=set()), \
                patch("script.sample_mira_ids.load_keywords", return_value=["not-present"]), \
                self.assertRaisesRegex(ValueError, "open_ended"):
            sample_dataset(self.root, train_counts=train, test_counts=test)

    def test_invalid_type_maps_are_rejected(self):
        base = {category: 1 for category in QUESTION_TYPES}
        with patch("script.sample_mira_ids.load_embedded_ids", return_value=set()), \
                patch("script.sample_mira_ids.load_keywords", return_value=["not-present"]):
            for invalid in ({"open_ended": 1}, {**base, "open_ended": -1}, {**base, "open_ended": True}):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    sample_dataset(self.root, train_counts=invalid, test_counts=base)

    def test_ratio_allocation_preserves_totals_with_largest_remainder(self):
        equal = {category: 0.25 for category in QUESTION_TYPES}
        normalized = validate_type_ratios(equal, "ratios")
        train = allocate_type_counts(11, normalized)
        test = allocate_type_counts(5, normalized)
        self.assertEqual(train, {
            "open_ended": 3, "closed_ended": 3,
            "single_choice": 3, "multiple_choice": 2,
        })
        self.assertEqual(test, {
            "open_ended": 2, "closed_ended": 1,
            "single_choice": 1, "multiple_choice": 1,
        })
        self.assertEqual(sum(train.values()), 11)
        self.assertEqual(sum(test.values()), 5)
        self.assertEqual(validate_type_ratios({key: 25 for key in QUESTION_TYPES}, "ratios"), normalized)

    def test_ratio_config_derives_independent_train_and_test_quotas(self):
        ratios = {category: 1 for category in QUESTION_TYPES}
        payload = {
            "train_count": 11,
            "test_count": 5,
            "train_ratios": ratios,
            "test_ratios": ratios,
            "data_root": str(self.root),
            "keywords_file": None,
            "db_dir": str(self.root),
            "output": str(self.root / "sample.json"),
        }
        args = _validated_config(payload, self.root / "sample_mira_ids.json")
        self.assertEqual(args.train_count, 11)
        self.assertEqual(args.test_count, 5)
        self.assertEqual(sum(args.train_counts.values()), 11)
        self.assertEqual(sum(args.test_counts.values()), 5)
        self.assertEqual(args.train_counts, {
            "open_ended": 3, "closed_ended": 3,
            "single_choice": 3, "multiple_choice": 2,
        })
        self.assertEqual(args.test_counts, {
            "open_ended": 2, "closed_ended": 1,
            "single_choice": 1, "multiple_choice": 1,
        })

    def test_invalid_or_conflicting_ratios_are_rejected(self):
        for invalid in (
            {"open_ended": 0, "closed_ended": 0, "single_choice": 0, "multiple_choice": 0},
            {"open_ended": -1, "closed_ended": 1, "single_choice": 1, "multiple_choice": 1},
            {"open_ended": True, "closed_ended": 1, "single_choice": 1, "multiple_choice": 1},
            {"open_ended": float("nan"), "closed_ended": 1, "single_choice": 1, "multiple_choice": 1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_type_ratios(invalid, "ratios")


if __name__ == "__main__":
    unittest.main()
