import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inferenceValid.embed_mira_chroma import CATEGORIES, iter_samples
from script import sample_mira_ids
from script.sample_mira_ids import sample_dataset


class SampleMiraIdsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.write_split("train", [self.row(index) for index in range(5)])
        embedded_patch = patch.object(sample_mira_ids, "load_embedded_ids", return_value=set())
        self.embedded_ids = embedded_patch.start()
        self.addCleanup(embedded_patch.stop)
        keywords_patch = patch.object(sample_mira_ids, "load_keywords", return_value=["hypertension"])
        self.keywords = keywords_patch.start()
        self.addCleanup(keywords_patch.stop)
        self.console = io.StringIO()
        console_patch = redirect_stdout(self.console)
        console_patch.__enter__()
        self.addCleanup(console_patch.__exit__, None, None, None)

    def row(self, index):
        return {
            "image_path": f"images/{index}.png",
            "caption": "A caption",
            "vqa_json": {
                category: [{"question": f"Question {index}\nSecond line", "answer": "Answer"}]
                for category in CATEGORIES
            },
        }

    def write_split(self, split, rows):
        with (self.root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_path", "caption", "vqa_json"))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "vqa_json": json.dumps(row["vqa_json"])})

    def images_for(self, ids):
        samples = {sample.id: sample for sample in iter_samples(self.root, "train")}
        return {
            os.path.normcase(os.path.realpath(self.root / image))
            for sample_id in ids
            for image in samples[sample_id].images
        }

    def ids_for(self, split="train"):
        return [sample.id for sample in iter_samples(self.root, split)]

    def test_counts_disjointness_and_determinism(self):
        result = sample_dataset(self.root, 7, 4, seed=123)
        repeated = sample_dataset(self.root, 7, 4, seed=123)
        self.assertEqual(len(result["train_ids"]), 7)
        self.assertEqual(len(result["test_ids"]), 4)
        self.assertEqual(len(set(result["train_ids"])), 7)
        self.assertEqual(len(set(result["test_ids"])), 4)
        self.assertTrue(set(result["train_ids"]).isdisjoint(result["test_ids"]))
        self.assertEqual(result["train_ids"], repeated["train_ids"])
        self.assertEqual(result["test_ids"], repeated["test_ids"])
        self.assertEqual(result["total_samples"], 20)

    def test_full_population_matches_embedding_ids(self):
        rows = [self.row(index) for index in range(5)]
        rows[0]["vqa_json"]["open_ended"].append({"question": "Second QA", "answer": "Answer"})
        self.write_split("train", rows)
        result = sample_dataset(self.root, 20, 1)
        expected = {sample.id for sample in iter_samples(self.root, "train")}
        self.assertEqual(set(result["train_ids"] + result["test_ids"]), expected)
        self.assertIn("mira:train:0:open_ended:0", expected)
        self.assertIn("mira:train:0:open_ended:1", expected)
        self.assertIn("mira:train:4:multiple_choice:0", expected)

    def test_explicit_multiple_source_splits(self):
        self.write_split("test", [self.row(99)])
        result = sample_dataset(self.root, 23, 1, splits=["train", "test"])
        expected = {
            sample.id for split in ("train", "test") for sample in iter_samples(self.root, split)
        }
        self.assertEqual(set(result["train_ids"] + result["test_ids"]), expected)
        self.assertEqual(result["total_samples"], 24)

    def test_shared_images_are_excluded(self):
        result = sample_dataset(self.root, 5, 1, seed=4, exclude_shared_images=True)
        self.assertEqual(len(result["train_ids"]), 5)
        self.assertTrue(self.images_for(result["train_ids"]).isdisjoint(self.images_for(result["test_ids"])))
        self.assertEqual(result["train_candidates"], 16)

    def test_qa_overrides_and_path_aliases_share_images(self):
        rows = [self.row(index) for index in range(3)]
        aliases = ["images/common.png", "images/../images/common.png", "images\\common.png"]
        for index, row in enumerate(rows):
            row["vqa_json"] = {
                "open_ended": [{
                    "question": "A question", "answer": "An answer",
                    "image_paths": [f"images/extra-{index}.png", aliases[index]],
                }]
            }
        self.write_split("train", rows)
        sample_dataset(self.root, 1, 1, exclude_shared_images=False)
        with self.assertRaises(ValueError):
            sample_dataset(self.root, 1, 1, exclude_shared_images=True)

    def test_rejects_insufficient_population(self):
        with self.assertRaises(ValueError):
            sample_dataset(self.root, 20, 1)

    def test_rejects_invalid_counts(self):
        for train_count, test_count in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.subTest(train_count=train_count, test_count=test_count):
                with self.assertRaises(ValueError):
                    sample_dataset(self.root, train_count, test_count)

    def test_rejects_duplicate_source_splits(self):
        with self.assertRaises(ValueError):
            sample_dataset(self.root, 1, 1, splits=["train", "train"])

    def test_complete_embedding_coverage(self):
        self.embedded_ids.return_value = set(self.ids_for())
        result = sample_dataset(self.root, 7, 4)
        coverage = result["embedding_coverage"]
        self.assertEqual(coverage["total_samples"], 20)
        self.assertEqual(coverage["embedded_samples"], 20)
        self.assertEqual(coverage["missing_samples"], 0)
        self.assertTrue(coverage["all_embedded"])
        self.assertEqual(result["selected_embedding_counts"], {"train": 7, "test": 4})

    def test_zero_embedding_coverage_uses_fallback(self):
        result = sample_dataset(self.root, 7, 4)
        coverage = result["embedding_coverage"]
        self.assertEqual(coverage["total_samples"], 20)
        self.assertEqual(coverage["embedded_samples"], 0)
        self.assertEqual(coverage["missing_samples"], 20)
        self.assertFalse(coverage["all_embedded"])
        self.assertEqual(result["selected_embedding_counts"], {"train": 0, "test": 0})

    def test_prioritizes_embeddings_for_both_sets(self):
        embedded = set(self.ids_for()[::2])
        self.embedded_ids.return_value = embedded
        result = sample_dataset(self.root, 6, 2, seed=18)
        self.assertTrue(set(result["train_ids"]).issubset(embedded))
        self.assertTrue(set(result["test_ids"]).issubset(embedded))
        self.assertEqual(result["embedded_samples"], len(embedded))
        self.assertEqual(result["selected_embedding_counts"], {"train": 6, "test": 2})

    def test_shortage_prioritizes_test_then_fills_training(self):
        embedded = set(self.ids_for()[:5])
        self.embedded_ids.return_value = embedded
        result = sample_dataset(self.root, 8, 3)
        selected = set(result["train_ids"] + result["test_ids"])
        self.assertTrue(embedded.issubset(selected))
        self.assertTrue(set(result["test_ids"]).issubset(embedded))
        self.assertEqual(len(result["train_ids"]), 8)
        self.assertEqual(len(result["test_ids"]), 3)
        self.assertEqual(result["selected_embedding_counts"], {"train": 2, "test": 3})

    def test_test_shortage_uses_all_embeddings_then_falls_back(self):
        embedded = set(self.ids_for()[:2])
        self.embedded_ids.return_value = embedded
        result = sample_dataset(self.root, 4, 5)
        self.assertTrue(embedded.issubset(result["test_ids"]))
        self.assertEqual(result["selected_embedding_counts"], {"train": 0, "test": 2})
        self.assertEqual(len(result["train_ids"]), 4)
        self.assertEqual(len(result["test_ids"]), 5)

    def test_coverage_audits_all_splits_without_expanding_default_pool(self):
        self.write_split("validation", [self.row(98)])
        self.write_split("test", [self.row(99), self.row(100)])
        embedded = set(self.ids_for()[:3] + self.ids_for("validation")[:2] + self.ids_for("test")[:1])
        self.embedded_ids.return_value = embedded | {"unrelated-id", "mira:train:999:open_ended:0"}
        result = sample_dataset(self.root, 4, 2)
        coverage = result["embedding_coverage"]
        self.assertEqual(coverage["total_samples"], 32)
        self.assertEqual(coverage["embedded_samples"], 6)
        self.assertEqual(coverage["missing_samples"], 26)
        self.assertEqual(coverage["collection_count"], 8)
        self.assertEqual(coverage["unmatched_chroma_ids"], 2)
        self.assertFalse(coverage["all_embedded"])
        self.assertEqual(coverage["splits"], {
            "train": {"total_samples": 20, "embedded_samples": 3, "missing_samples": 17},
            "validation": {"total_samples": 4, "embedded_samples": 2, "missing_samples": 2},
            "test": {"total_samples": 8, "embedded_samples": 1, "missing_samples": 7},
        })
        self.assertEqual(result["total_samples"], 20)
        self.assertEqual(result["embedded_samples"], 3)
        self.assertTrue(all(sample_id.startswith("mira:train:")
                            for sample_id in result["train_ids"] + result["test_ids"]))
        self.assertIn("Dataset groups (all existing CSV splits): 32", self.console.getvalue())
        self.assertIn("Groups with stored embeddings: 6", self.console.getvalue())
        self.assertIn("Groups without embeddings: 26", self.console.getvalue())

    def test_shared_image_exclusion_overrides_embedding_priority(self):
        embedded = set(self.ids_for()[:len(CATEGORIES)])
        self.embedded_ids.return_value = embedded
        result = sample_dataset(self.root, 3, 1, exclude_shared_images=True)
        self.assertTrue(set(result["test_ids"]).issubset(embedded))
        self.assertTrue(set(result["train_ids"]).isdisjoint(embedded))
        self.assertTrue(self.images_for(result["train_ids"]).isdisjoint(self.images_for(result["test_ids"])))
        self.assertEqual(result["train_candidates"], 16)
        self.assertEqual(result["selected_embedding_counts"], {"train": 0, "test": 1})

    def test_embedded_sampling_is_deterministic(self):
        self.embedded_ids.return_value = set(self.ids_for()[::3])
        first = sample_dataset(self.root, 7, 4, seed=13)
        second = sample_dataset(self.root, 7, 4, seed=13)
        self.assertEqual(first["train_ids"], second["train_ids"])
        self.assertEqual(first["test_ids"], second["test_ids"])

    def test_explicit_database_and_collection(self):
        database = self.root / "vectors"
        result = sample_dataset(self.root, 2, 1, db_dir=database, collection="custom-mira")
        self.embedded_ids.assert_called_once_with(database.resolve(), "custom-mira")
        self.assertEqual(result["db_dir"], str(database.resolve()))
        self.assertEqual(result["chroma_collection"], "custom-mira")

    def test_default_database_is_dataset_sibling(self):
        sample_dataset(self.root, 2, 1)
        self.embedded_ids.assert_called_once_with(self.root.parent / "MIRA-chroma", "mira_qwen3_vl_embedding")

    def keyword_rows(self, texts):
        return [{
            "image_path": f"images/{index}.png",
            "caption": "A caption",
            "vqa_json": {"open_ended": [{"question": text, "answer": "Answer"}]},
        } for index, text in enumerate(texts)]

    def test_default_keywords_file_is_configured_dataset_sibling(self):
        result = sample_dataset(self.root, 2, 1)
        expected = self.root.parent / "MIRA_myConfig" / "cardiovascular_ai_keywords2.txt"
        self.keywords.assert_called_once_with(expected.resolve())
        self.assertEqual(result["keywords_file"], str(expected.resolve()))
        self.assertEqual(result["keywords"], ["hypertension"])

    def test_both_conditions_then_embedded_then_keyword_then_neither(self):
        self.write_split("train", self.keyword_rows(
            ["Hypertension"] * 3 + ["Neutral"] * 3 + ["Hypertension"] * 3 + ["Neutral"] * 3))
        ids = self.ids_for()
        both, embedded_only, keyword_only, neither = map(set, (ids[:3], ids[3:6], ids[6:9], ids[9:]))
        self.embedded_ids.return_value = both | embedded_only
        result = sample_dataset(self.root, 5, 2, seed=21)
        training, testing = set(result["train_ids"]), set(result["test_ids"])
        self.assertTrue(testing.issubset(both))
        self.assertEqual(training & both, both - testing)
        self.assertTrue(embedded_only.issubset(training))
        self.assertEqual(len(training & keyword_only), 1)
        self.assertTrue((training | testing).isdisjoint(neither))
        self.assertEqual(result["keyword_samples"], 6)
        self.assertEqual(result["keyword_embedded_samples"], 3)
        self.assertEqual(result["selected_keyword_counts"], {"train": 2, "test": 2})
        self.assertEqual(result["selected_keyword_embedding_counts"], {"train": 1, "test": 2})
        self.assertEqual(result["selected_embedding_counts"], {"train": 4, "test": 2})
        repeated = sample_dataset(self.root, 5, 2, seed=21)
        self.assertEqual(result["train_ids"], repeated["train_ids"])
        self.assertEqual(result["test_ids"], repeated["test_ids"])

    def test_keyword_shortage_falls_back_without_duplicate_ids(self):
        self.write_split("train", self.keyword_rows(["hypertension", "Neutral", "hypertension", "Neutral"]))
        ids = self.ids_for()
        self.embedded_ids.return_value = set(ids[:2])
        result = sample_dataset(self.root, 2, 2)
        self.assertEqual(set(result["test_ids"]), set(ids[:2]))
        self.assertEqual(set(result["train_ids"]), set(ids[2:]))
        self.assertEqual(result["selected_keyword_counts"], {"train": 1, "test": 1})
        self.assertEqual(result["selected_keyword_embedding_counts"], {"train": 0, "test": 1})

    def test_keyword_coverage_counts_qas_once_and_audits_all_splits(self):
        rows = self.keyword_rows(["Hypertension hypertension", "Neutral"])
        rows[0]["caption"] = "hypertension"
        rows[0]["vqa_json"]["open_ended"][0]["answer"] = "hypertension"
        rows[0]["vqa_json"]["open_ended"].append({"question": "Neutral sibling", "answer": "Answer"})
        self.write_split("train", rows)
        self.write_split("validation", self.keyword_rows(["hypertension", "Neutral"]))
        self.write_split("test", self.keyword_rows(["hypertension"]))
        self.embedded_ids.return_value = {self.ids_for()[0], self.ids_for("validation")[0]}
        result = sample_dataset(self.root, 1, 1)
        self.assertEqual(result["keyword_coverage"], {
            "keyword_count": 1,
            "matched_samples": 3,
            "matched_embedded_samples": 2,
            "splits": {
                "train": {"matched_samples": 1, "matched_embedded_samples": 1},
                "validation": {"matched_samples": 1, "matched_embedded_samples": 1},
                "test": {"matched_samples": 1, "matched_embedded_samples": 0},
            },
        })
        self.assertEqual(result["total_samples"], 3)
        self.assertEqual(result["embedding_coverage"]["total_samples"], 6)
        self.assertEqual(result["keyword_samples"], 1)
        self.assertEqual(result["keyword_embedded_samples"], 1)
        self.assertEqual(result["sample_unit"], "question_answer")
        self.assertIn("Groups matching at least one keyword: 3", self.console.getvalue())
        self.assertIn("Keyword-matching groups with stored embeddings: 2", self.console.getvalue())

    def test_shared_image_exclusion_overrides_keyword_embedding_priority(self):
        rows = self.keyword_rows(["hypertension", "hypertension", "Neutral", "Neutral"])
        rows[1]["image_path"] = rows[0]["image_path"]
        self.write_split("train", rows)
        self.embedded_ids.return_value = set(self.ids_for())
        result = sample_dataset(self.root, 2, 1, exclude_shared_images=True)
        self.assertTrue(self.images_for(result["train_ids"]).isdisjoint(self.images_for(result["test_ids"])))
        self.assertEqual(result["selected_keyword_embedding_counts"], {"train": 0, "test": 1})
        self.assertEqual(result["excluded_shared_image_samples"], 1)

    def run_cli(self, output, train_count, test_count):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returncode = sample_mira_ids.main([
                "--data-root", str(self.root), "--train-count", str(train_count),
                "--test-count", str(test_count), "--output", str(output),
            ])
        return SimpleNamespace(returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def test_cli_writes_json(self):
        output = self.root / "split.json"
        process = self.run_cli(output, 3, 2)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(len(result["train_ids"]), 3)
        self.assertEqual(len(result["test_ids"]), 2)
        self.assertIn("embedding_coverage", result)
        self.assertIn("20", process.stdout)

    def test_cli_invalid_counts_do_not_create_output(self):
        output = self.root / "invalid.json"
        process = self.run_cli(output, 0, 2)
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(output.exists())

    def test_cli_accepts_explicit_keywords_file(self):
        output = self.root / "split.json"
        keywords_file = self.root / "custom-keywords.txt"
        returncode = sample_mira_ids.main([
            "--data-root", str(self.root), "--train-count", "2", "--test-count", "1",
            "--keywords-file", str(keywords_file), "--output", str(output),
        ])
        self.assertEqual(returncode, 0)
        self.keywords.assert_called_once_with(keywords_file.resolve())
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["keywords_file"], str(keywords_file.resolve()))
        self.assertIn("keyword_coverage", result)


class KeywordMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def sample(self, **overrides):
        fields = {"question": "Neutral", "options": None, "answer": "Answer", "extra": {},
                  "caption": "Hypertension", "images": ["images/Hypertension.png"]}
        return SimpleNamespace(**(fields | overrides))

    def test_load_keywords_normalizes_whitespace_deduplicates_and_skips_comments(self):
        path = self.root / "keywords.txt"
        path.write_text("\ufeff# Cardiovascular terms\n  Hypertension  \n\n  # More terms\n"
                        "HYPERTENSION\nHeart\t failure\n", encoding="utf-8")
        self.assertEqual(sample_mira_ids.load_keywords(path), ["Hypertension", "Heart failure"])

    def test_missing_and_empty_keyword_files_are_rejected(self):
        path = self.root / "keywords.txt"
        with self.assertRaises(FileNotFoundError):
            sample_mira_ids.load_keywords(path)
        path.write_text("\n# Comments only\n \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            sample_mira_ids.load_keywords(path)

    def test_keyword_and_abbreviation_boundaries_are_case_insensitive(self):
        pattern = sample_mira_ids.compile_keywords(["myocardial infarction (MI)", "hypertension"])
        for text in ("MI", "Prior mi.", "MYOCARDIAL\n  INFARCTION", "Hypertension."):
            with self.subTest(text=text):
                self.assertTrue(sample_mira_ids.matches_keywords(self.sample(question=text), pattern))
        for text in ("similar", "mild pain", "BMI", "prehypertension", "hypertensions", "infarctions"):
            with self.subTest(text=text):
                self.assertFalse(sample_mira_ids.matches_keywords(self.sample(question=text), pattern))

    def test_nested_qa_values_match_without_metadata_or_dictionary_key_leakage(self):
        pattern = sample_mira_ids.compile_keywords(["hypertension"])
        for fields in (
            {"question": "History of hypertension"},
            {"options": {"A": "Hypertension", "B": "Other"}},
            {"answer": {"explanations": [None, 3, {"finding": "hypertension"}]}},
            {"extra": {"rationale": [{"text": "hypertension"}]}},
        ):
            with self.subTest(fields=fields):
                self.assertTrue(sample_mira_ids.matches_keywords(self.sample(**fields), pattern))
        self.assertFalse(sample_mira_ids.matches_keywords(self.sample(), pattern))
        self.assertFalse(sample_mira_ids.matches_keywords(
            self.sample(answer={"hypertension": "Absent"}, extra={"hypertension": False}), pattern))

    def test_phrase_cannot_span_unrelated_qa_fields(self):
        pattern = sample_mira_ids.compile_keywords(["heart failure"])
        self.assertFalse(sample_mira_ids.matches_keywords(
            self.sample(question="heart", answer="failure"), pattern))
        self.assertTrue(sample_mira_ids.matches_keywords(
            self.sample(question="heart\t\n failure"), pattern))


class LoadEmbeddedIdsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "vectors #1"
        self.database.mkdir()
        self.sqlite_path = self.database / "chroma.sqlite3"
        self.connection = sqlite3.connect(self.sqlite_path)
        self.addCleanup(self.connection.close)
        self.connection.executescript("""
            CREATE TABLE databases (id TEXT PRIMARY KEY, name TEXT, tenant_id TEXT);
            CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, database_id TEXT);
            CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT);
            CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
            INSERT INTO databases VALUES ('db-main', 'default_database', 'default_tenant');
            INSERT INTO collections VALUES ('collection-main', 'mira', 'db-main');
            INSERT INTO segments VALUES
                ('metadata-main', 'urn:chroma:segment/metadata/sqlite', 'METADATA', 'collection-main');
            INSERT INTO embeddings (segment_id, embedding_id) VALUES
                ('metadata-main', 'first-id'), ('metadata-main', 'second-id');
        """)
        self.connection.commit()

    def test_missing_database_does_not_create_it(self):
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            sample_mira_ids.load_embedded_ids(missing, "mira")
        self.assertFalse(missing.exists())

    def test_existing_directory_without_database_does_not_create_sqlite(self):
        missing = self.root / "empty"
        missing.mkdir()
        with self.assertRaises(FileNotFoundError):
            sample_mira_ids.load_embedded_ids(missing, "mira")
        self.assertEqual(list(missing.iterdir()), [])

    def test_missing_collection_does_not_create_it(self):
        before = self.sqlite_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "(?i)collection"):
            sample_mira_ids.load_embedded_ids(self.database, "missing")
        self.assertEqual(self.sqlite_path.read_bytes(), before)

    def test_reads_stored_ids_without_document_or_vector_tables(self):
        self.assertEqual(sample_mira_ids.load_embedded_ids(self.database, "mira"), {"first-id", "second-id"})

    def test_excludes_other_segments_collections_databases_and_tenants(self):
        self.connection.executescript("""
            INSERT INTO databases VALUES
                ('db-other', 'other_database', 'default_tenant'),
                ('tenant-other', 'default_database', 'other_tenant');
            INSERT INTO collections VALUES
                ('collection-other', 'other-name', 'db-main'),
                ('collection-other-db', 'mira', 'db-other'),
                ('collection-other-tenant', 'mira', 'tenant-other');
            INSERT INTO segments VALUES
                ('vector-main', 'urn:chroma:segment/vector/hnsw-local-persisted', 'VECTOR', 'collection-main'),
                ('metadata-other', 'urn:chroma:segment/metadata/sqlite', 'METADATA', 'collection-other'),
                ('metadata-other-db', 'urn:chroma:segment/metadata/sqlite', 'METADATA', 'collection-other-db'),
                ('metadata-other-tenant', 'urn:chroma:segment/metadata/sqlite', 'METADATA', 'collection-other-tenant');
            INSERT INTO embeddings (segment_id, embedding_id) VALUES
                ('vector-main', 'vector-only-id'),
                ('metadata-other', 'other-collection-id'),
                ('metadata-other-db', 'other-database-id'),
                ('metadata-other-tenant', 'other-tenant-id');
        """)
        self.connection.commit()
        self.assertEqual(sample_mira_ids.load_embedded_ids(self.database, "mira"), {"first-id", "second-id"})

    def test_read_leaves_database_and_directory_unchanged(self):
        contents = self.sqlite_path.read_bytes()
        modified = self.sqlite_path.stat().st_mtime_ns
        files = set(self.database.iterdir())
        sample_mira_ids.load_embedded_ids(self.database, "mira")
        self.assertEqual(self.sqlite_path.read_bytes(), contents)
        self.assertEqual(self.sqlite_path.stat().st_mtime_ns, modified)
        self.assertEqual(set(self.database.iterdir()), files)

    def test_empty_collection_has_no_embedded_ids(self):
        with self.connection:
            self.connection.execute("DELETE FROM embeddings")
        self.assertEqual(sample_mira_ids.load_embedded_ids(self.database, "mira"), set())

    def test_missing_metadata_segment_has_actionable_error(self):
        with self.connection:
            self.connection.execute("DELETE FROM segments")
        with self.assertRaisesRegex(ValueError, "(?i)(segment|metadata)"):
            sample_mira_ids.load_embedded_ids(self.database, "mira")

    def test_unsupported_metadata_type_is_rejected(self):
        with self.connection:
            self.connection.execute("UPDATE segments SET type = 'unsupported-metadata-type'")
        with self.assertRaisesRegex(ValueError, "(?i)(segment|metadata|unsupported)"):
            sample_mira_ids.load_embedded_ids(self.database, "mira")

    def test_unsupported_schema_has_actionable_error(self):
        with self.connection:
            self.connection.execute("DROP TABLE databases")
        with self.assertRaisesRegex(ValueError, "(?i)(schema|sqlite|chroma)"):
            sample_mira_ids.load_embedded_ids(self.database, "mira")


if __name__ == "__main__":
    unittest.main()
