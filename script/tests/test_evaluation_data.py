"""固定测试集的数据选择回归测试，不需要模型、CUDA 或真实图片。"""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from script.lib.evaluation_data import load_test_ids, prepare_evaluation_data


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "split.json"

    def config(self, **kwargs):
        values = dict(dataset_path=self.root, dataset_filter=self.manifest, N=None,
                      seed=42, knowledge_size=None, exclude_shared_images=False,
                      source_splits=["train"])
        values.update(kwargs)
        return argparse.Namespace(**values)

    def filter(self, values):
        self.manifest.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8-sig")

    def split(self, name, groups):
        with (self.root / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_path", "caption", "vqa_json"])
            writer.writeheader()
            for index, group in enumerate(groups):
                writer.writerow({"image_path": f"missing/{index}.png", "caption": "不可输入的答案提示",
                                 "vqa_json": group if isinstance(group, str) else json.dumps(group)})

    def qa(self, question="问题？", answer="回答", **kwargs):
        return {"open_ended": [{"question": question, "answer": answer, **kwargs}]}

    def jsonl(self, rows):
        path = self.root / "dataset.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_selected_mira_preserves_order_and_qa_fields(self):
        answer = {"correct_options": ["A", "C"], "explanation": "解释"}
        self.split("train", [self.qa("首题\n换行", answer),
                             {"multiple_choice": [{"question": "选择？", "options": {"A": "a", "C": "c"},
                                                    "answer": answer, "image_paths": ["other.png"]}]}])
        identifiers = ["mira:train:1:multiple_choice:0", "mira:train:0:open_ended:0"]
        self.filter({"test_ids": identifiers, "train_ids": []})
        with patch("script.lib.evaluation_data.load_mira_dataset", side_effect=AssertionError("不得全量读取")):
            test, knowledge, meta = prepare_evaluation_data(self.config(), [{"use_rag": False}])
        self.assertEqual([row["id"] for row in test], identifiers)
        self.assertEqual(json.loads(test[0]["reference"]), answer)
        self.assertIn('\nOptions: {"A":"a","C":"c"}', test[0]["question"])
        self.assertEqual(test[1]["question"], "首题\n换行")
        self.assertEqual(test[0]["images"], [str(self.root / "other.png")])
        self.assertNotIn("不可输入", test[0]["question"])
        self.assertEqual(knowledge, [])
        self.assertEqual(meta["loaded_samples"], 2)
        self.assertEqual(meta["reserved_test_ids"], identifiers)
        self.assertEqual(meta["test_selection"], "filter")

    def test_only_selected_rows_are_decoded(self):
        self.split("train", ["不是JSON", self.qa()])
        self.filter(["mira:train:1:open_ended:0"])
        test, _, _ = prepare_evaluation_data(self.config(), [{"use_rag": False}])
        self.assertEqual(len(test), 1)

    def test_filtered_ids_can_reference_multiple_official_splits(self):
        for split in ("test", "validation", "train"):
            self.split(split, [self.qa(answer=split)])
        ids = [f"mira:{split}:0:open_ended:0" for split in ("test", "train", "validation")]
        self.filter(ids)
        test, _, _ = prepare_evaluation_data(self.config(), [{"use_rag": False}])
        self.assertEqual([row["id"] for row in test], ids)

    def test_invalid_filters_fail_before_reading_source(self):
        bad_values = [[], {}, {"test_ids": []}, [True], [None], [" "], [1, "1"],
                      ["mira:train:-1:open_ended:0"], ["mira:train:01:open_ended:0"],
                      ["mira:train:0:unknown:0"], ["mira:train:0:open_ended:x"],
                      {"test_ids": [1], "train_ids": ["1"]},
                      {"test_ids": ["a"], "train_ids": ["b", "b"]}]
        for value in bad_values:
            with self.subTest(value=value):
                self.filter(value)
                with self.assertRaises(ValueError):
                    load_test_ids(self.manifest)

    def test_missing_target_row_category_or_qa_index_fails(self):
        self.split("train", [self.qa()])
        for identifier in ("mira:train:4:open_ended:0", "mira:train:0:multiple_choice:0",
                           "mira:train:0:open_ended:3", "not-a-mira-id"):
            with self.subTest(identifier=identifier):
                self.filter([identifier])
                with self.assertRaises(ValueError):
                    prepare_evaluation_data(self.config(), [{"use_rag": False}])

    def test_invalid_selected_qa_fails_instead_of_skipping(self):
        for group in (self.qa(question=" "), self.qa(answer=" "), self.qa(answer=None),
                      {"open_ended": [False]}, "broken json"):
            with self.subTest(group=group):
                self.split("train", [group])
                self.filter(["mira:train:0:open_ended:0"])
                with self.assertRaises(ValueError):
                    prepare_evaluation_data(self.config(), [{"use_rag": False}])

    def test_null_filter_uses_all_official_test_only(self):
        self.split("train", ["训练集损坏不影响无RAG测试"])
        self.split("test", [self.qa(answer=False), self.qa(answer=0)])
        test, knowledge, meta = prepare_evaluation_data(
            self.config(dataset_filter=None), [{"use_rag": False}])
        self.assertEqual([row["id"] for row in test],
                         ["mira:test:0:open_ended:0", "mira:test:1:open_ended:0"])
        self.assertEqual([row["reference"] for row in test], ["false", "0"])
        self.assertEqual(knowledge, [])
        self.assertEqual(meta["test_selection"], "official_test")
        self.assertEqual(meta["requested_test_count"], 2)

    def test_null_filter_missing_or_invalid_official_test_fails(self):
        with self.assertRaises(FileNotFoundError):
            prepare_evaluation_data(self.config(dataset_filter=None), [{"use_rag": False}])
        self.split("test", [self.qa(answer=None)])
        with self.assertRaisesRegex(ValueError, "answer"):
            prepare_evaluation_data(self.config(dataset_filter=None), [{"use_rag": False}])

    def test_N_is_prefix_not_random_and_reserves_all_filtered_ids(self):
        self.split("train", [self.qa(answer=str(index)) for index in range(4)])
        ids = ["mira:train:2:open_ended:0", "mira:train:0:open_ended:0"]
        self.filter(ids)
        test, knowledge, meta = prepare_evaluation_data(self.config(N=1), [{"use_rag": True}])
        self.assertEqual([row["id"] for row in test], ids[:1])
        self.assertEqual([row["id"] for row in knowledge],
                         ["mira:train:1:open_ended:0", "mira:train:3:open_ended:0"])
        self.assertEqual(meta["requested_test_count"], 2)
        self.assertEqual(meta["reserved_test_ids"], ids)

    def test_rag_uses_train_source_and_optional_image_exclusion(self):
        self.split("test", [self.qa()])
        self.split("train", [self.qa(), self.qa(), self.qa(), self.qa()])
        test, knowledge, meta = prepare_evaluation_data(
            self.config(dataset_filter=None, exclude_shared_images=True, knowledge_size=2),
            [{"use_rag": True}])
        self.assertEqual(len(test), 1)
        self.assertEqual(len(knowledge), 2)
        self.assertEqual(meta["excluded_shared_image_samples"], 1)
        self.assertEqual(meta["knowledge_candidates"], 3)
        self.assertEqual(meta["excluded_by_knowledge_limit"], 1)
        again = prepare_evaluation_data(
            self.config(dataset_filter=None, exclude_shared_images=True, knowledge_size=2),
            [{"use_rag": True}])[1]
        self.assertEqual(knowledge, again)

    def test_jsonl_null_filter_uses_all_records_and_has_no_knowledge(self):
        path = self.jsonl([{"id": index, "question": "q", "reference": "a", "images": []}
                           for index in range(3)])
        test, knowledge, meta = prepare_evaluation_data(
            self.config(dataset_path=path, dataset_filter=None, N=1), [{"use_rag": True}])
        self.assertEqual([row["id"] for row in test], [0])
        self.assertEqual(knowledge, [])
        self.assertEqual(meta["test_selection"], "jsonl_all")
        self.assertEqual(meta["requested_test_count"], 3)

    def test_jsonl_filter_preserves_requested_order_and_knowledge_complement(self):
        path = self.jsonl([{"id": index, "question": "q", "reference": "a", "images": ["x.png"]}
                           for index in range(4)])
        self.filter(["2", 0])
        test, knowledge, _ = prepare_evaluation_data(self.config(dataset_path=path), [{"use_rag": True}])
        self.assertEqual([row["id"] for row in test], [2, 0])
        self.assertEqual([row["id"] for row in knowledge], [1, 3])
        self.assertEqual(test[0]["images"], [str(self.root / "x.png")])

    def test_jsonl_missing_filter_id_or_duplicate_source_id_fails(self):
        row = {"id": "a", "question": "q", "reference": "a", "images": []}
        path = self.jsonl([row])
        self.filter(["missing"])
        with self.assertRaisesRegex(ValueError, "不在 JSONL"):
            prepare_evaluation_data(self.config(dataset_path=path), [{"use_rag": False}])
        self.jsonl([row, row])
        with self.assertRaisesRegex(ValueError, "重复"):
            prepare_evaluation_data(self.config(dataset_path=path, dataset_filter=None), [{"use_rag": False}])

    def test_invalid_limits_fail(self):
        for field in ("N", "knowledge_size"):
            for value in (0, -1, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    prepare_evaluation_data(self.config(**{field: value}), [{"use_rag": False}])


if __name__ == "__main__":
    unittest.main()
