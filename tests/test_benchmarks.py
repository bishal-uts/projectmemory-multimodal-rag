"""Small synthetic fixtures exercise official schemas; these are not benchmarks."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from projectmemory.datasets import daily_omni, mmqa
from projectmemory.io import read_jsonl


def synthetic_mmqa(qid: str = "synthetic-question") -> dict:
    return {
        "qid": qid, "question": "SYNTHETIC_QUERY_SENTINEL",
        "answers": [{"answer": "SYNTHETIC_GOLD_SENTINEL"}],
        "supporting_context": [{"doc_id": "support", "doc_part": "text"},
                               {"doc_id": "picture", "doc_part": "image"}],
        "metadata": {"type": "Compose(TextQ,ImageQ)", "modalities": ["text", "image"],
                     "text_doc_ids": ["support", "distractor"], "image_doc_ids": ["picture"],
                     "table_id": "table", "intermediate_answers": ["SYNTHETIC_INTERMEDIATE_SENTINEL"]},
    }


def synthetic_daily(index: int = 0) -> dict:
    return {"Question": f"SYNTHETIC_QUERY_SENTINEL {index}",
            "Choice": ["A. Red", "B. SYNTHETIC_GOLD_SENTINEL", "C. Green", "D. Blue"],
            "Answer": "B", "video_id": f"synthetic-video-{index}",
            "Type": "AV Event Alignment", "video_duration": "30s"}


class MultiModalQATests(unittest.TestCase):
    def test_sampling_is_repeatable_and_independent_of_file_order(self):
        rows = [synthetic_mmqa(str(index)) for index in range(20)]
        first = mmqa.sample_questions(rows, 5, 42)
        self.assertEqual(first, mmqa.sample_questions(list(reversed(rows)), 5, 42))
        self.assertNotEqual(first, mmqa.sample_questions(rows, 5, 43))
        with self.assertRaises(ValueError):
            mmqa.sample_questions(rows, 0, 42)

    def test_list_answers_are_required_members_not_alternatives(self):
        raw = synthetic_mmqa()
        raw["answers"].append({"answer": "second answer"})
        parsed = mmqa.parse_question(raw)
        self.assertEqual(parsed["metadata"]["answer_semantics"], "unordered_list")
        self.assertEqual(len(parsed["reference_answers"]), 2)
        self.assertEqual(parsed["metadata"]["required_modalities"], ["text", "image"])

    def test_missing_support_in_candidate_pool_is_rejected(self):
        raw = synthetic_mmqa()
        raw["metadata"]["text_doc_ids"] = ["distractor"]
        with self.assertRaisesRegex(ValueError, "Supporting sources absent"):
            mmqa.parse_question(raw)

    def test_image_path_cannot_escape_media_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                mmqa.parse_context({"id": "bad", "path": "../outside.png"}, "image", Path(temporary))

    def test_prepare_keeps_distractors_and_does_not_index_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            fixtures = {
                "dev": [synthetic_mmqa()],
                "text": [{"id": "support", "title": "Source", "text": "Evidence prose"},
                         {"id": "distractor", "title": "Distractor", "text": "Distractor prose"},
                         {"id": "unrelated", "text": "Not in the supplied candidates"}],
                "image": [{"id": "picture", "title": "A source picture", "path": "picture.jpg"}],
                "table": [{"id": "table", "title": "Data", "table": {
                    "header": [{"column_name": "Name"}, {"column_name": "Value"}],
                    "table_rows": [[{"text": "All rows"}, {"text": "17"}]],
                }}],
            }
            for kind, rows in fixtures.items():
                with gzip.open(raw / mmqa.FILES[kind], "wt", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
            with patch.object(mmqa, "download", side_effect=AssertionError("Unexpected network access")):
                manifest = mmqa.prepare(root / "out", raw_dir=raw, media_dir=root / "images")
            corpus = read_jsonl(root / "out/corpus.jsonl")
            queries = read_jsonl(root / "out/queries.jsonl")
            serialized = json.dumps(corpus)
            for sentinel in ("SYNTHETIC_QUERY_SENTINEL", "SYNTHETIC_GOLD_SENTINEL", "SYNTHETIC_INTERMEDIATE_SENTINEL"):
                self.assertNotIn(sentinel, serialized)
            self.assertIn("mmqa:text:distractor", {row["id"] for row in corpus})
            self.assertNotIn("mmqa:text:unrelated", {row["id"] for row in corpus})
            self.assertEqual(len(corpus), 4)
            self.assertEqual(len(queries[0]["metadata"]["candidate_ids"]), 4)
            image_row = next(row for row in corpus if row["modality"] == "image")
            self.assertEqual(image_row["text"], "")
            self.assertEqual(image_row["metadata"]["representation"], "metadata_only")
            self.assertEqual(manifest["missing_image_files"], 1)
            self.assertFalse(manifest["native_multimodal_ready"])
            table_row = next(row for row in corpus if row["metadata"]["original_modality"] == "table")
            self.assertIn("Name | Value", table_row["text"])
            self.assertIn("All rows | 17", table_row["text"])


class DailyOmniTests(unittest.TestCase):
    def test_known_media_is_not_fake_relevance_annotation(self):
        parsed = daily_omni.parse_question(synthetic_daily())
        self.assertEqual(parsed["relevant_ids"], [])
        self.assertEqual(parsed["metadata"]["evaluation_protocol"], "known_video_qa")
        self.assertEqual(parsed["answer_label"], "B")
        self.assertEqual(len(parsed["options"]), 4)

    def test_sampling_is_repeatable_and_distinguishes_multiple_questions_per_video(self):
        rows = [synthetic_daily(index) for index in range(20)]
        rows[1]["video_id"] = rows[0]["video_id"]
        all_queries = daily_omni.sample_questions(rows, 100, 42)
        self.assertEqual(len({row["id"] for row in all_queries}), 20)
        self.assertEqual([row["id"] for row in daily_omni.sample_questions(rows, 5, 42)],
                         [row["id"] for row in daily_omni.sample_questions(list(reversed(rows)), 5, 42)])

    def test_official_duplicate_pattern_is_preserved_with_distinct_ids(self):
        rows = [synthetic_daily(), synthetic_daily()]
        selected = daily_omni.sample_questions(rows, 100, 42)
        self.assertEqual(len(selected), 2)
        self.assertNotEqual(selected[0]["id"], selected[1]["id"])
        self.assertEqual(selected[0]["question"], selected[1]["question"])
        rows[1]["Answer"] = "C"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            daily_omni.sample_questions(rows, 100, 42)

    def test_malformed_choice_and_unsafe_video_id_are_rejected(self):
        malformed = synthetic_daily()
        malformed["Choice"][0] = "D. Wrong label"
        with self.assertRaises(ValueError):
            daily_omni.parse_question(malformed)
        malformed = synthetic_daily()
        malformed["video_id"] = "../outside"
        with self.assertRaises(ValueError):
            daily_omni.parse_question(malformed)

    def test_prepare_does_not_download_media_or_index_qa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "qa.json").write_text(json.dumps([synthetic_daily()]), encoding="utf-8")
            with patch.object(daily_omni, "download", side_effect=AssertionError("Unexpected network access")):
                manifest = daily_omni.prepare(root / "out", raw_dir=raw, media_dir=root / "Videos")
            corpus = read_jsonl(root / "out/corpus.jsonl")
            queries = read_jsonl(root / "out/queries.jsonl")
            serialized = json.dumps(corpus)
            self.assertNotIn("SYNTHETIC_QUERY_SENTINEL", serialized)
            self.assertNotIn("SYNTHETIC_GOLD_SENTINEL", serialized)
            self.assertEqual(manifest["missing_media_by_modality"], {"video": 1, "audio": 1})
            self.assertFalse(manifest["native_multimodal_ready"])
            with self.assertRaises(FileNotFoundError):
                daily_omni.require_media(queries, corpus)
            # Existence validation is deliberately separate from media decoding.
            for row in corpus:
                path = Path(row["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic path-validation fixture, not playable media")
            daily_omni.require_media(queries, corpus)
            self.assertTrue(all(Path(row["path"]).is_absolute() for row in corpus))


if __name__ == "__main__":
    unittest.main()
