import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projectmemory.datasets.ami import MEETINGS, convert_meeting, media_manifest, prepare
from projectmemory.io import read_jsonl


def sample_meeting():
    return {
        "meeting_transcripts": [
            {"speaker": "Engineer", "content": f"Recorded source utterance {index}."}
            for index in range(25)
        ],
        "specific_query_list": [
            {
                "query": "What did the team decide?",
                "answer": "REFERENCE ANSWER MUST STAY OUT OF THE INDEX",
                "relevant_text_span": [["9", "10"], ["24", "24"]],
            }
        ],
        "general_query_list": [
            {"query": "Summarize the whole meeting.", "answer": "GENERAL GOLD SUMMARY"}
        ],
        "topic_list": [{"topic": "ANNOTATED TOPIC MUST STAY OUT OF THE INDEX"}],
    }


class AmiConversionTests(unittest.TestCase):
    def test_inclusive_support_spans_cross_chunk_boundaries(self):
        corpus, queries, _ = convert_meeting(sample_meeting(), "ES2006a", "dev")
        self.assertEqual(len(corpus), 3)
        self.assertEqual(queries[0]["relevant_ids"], [row["id"] for row in corpus])
        self.assertEqual(corpus[-1]["metadata"]["utterance_end"], 24)
        self.assertEqual(queries[0]["metadata"]["gold_utterance_spans"], [[9, 10], [24, 24]])

    def test_support_does_not_include_adjacent_nonoverlapping_chunks(self):
        payload = sample_meeting()
        payload["specific_query_list"][0]["relevant_text_span"] = [["10", "19"]]
        corpus, queries, _ = convert_meeting(payload, "ES2006a", "dev")
        self.assertEqual(queries[0]["relevant_ids"], [corpus[1]["id"]])

    def test_source_chunks_do_not_depend_on_questions_or_answers(self):
        payload = sample_meeting()
        corpus, queries, general = convert_meeting(payload, "ES2006a", "dev")
        payload["specific_query_list"] = []
        payload["general_query_list"] = []
        payload["topic_list"] = []
        unchanged, _, _ = convert_meeting(payload, "ES2006a", "dev")
        self.assertEqual(corpus, unchanged)
        serialized = json.dumps(corpus)
        self.assertNotIn("REFERENCE ANSWER", serialized)
        self.assertNotIn("GENERAL GOLD", serialized)
        self.assertNotIn("ANNOTATED TOPIC", serialized)
        self.assertEqual(queries[0]["metadata"]["candidate_source_ids"], ["ES2006a"])
        self.assertEqual(general[0]["relevant_ids"], [])
        self.assertFalse(general[0]["metadata"]["retrieval_evaluable"])

    def test_invalid_gold_spans_fail_instead_of_silently_mislabeling(self):
        for span in [["0", "25"], ["-1", "2"], ["3", "2"]]:
            payload = sample_meeting()
            payload["specific_query_list"][0]["relevant_text_span"] = [span]
            with self.subTest(span=span), self.assertRaises(ValueError):
                convert_meeting(payload, "ES2006a", "dev")

    def test_teams_are_disjoint_between_development_and_test(self):
        dev = {meeting[:-1] for meeting, _, split in MEETINGS if split == "dev"}
        test = {meeting[:-1] for meeting, _, split in MEETINGS if split == "test"}
        self.assertEqual(dev, {"ES2006"})
        self.assertEqual(test, {"ES2011"})
        self.assertFalse(dev & test)

    def test_manifest_preserves_observed_missing_media(self):
        manifest = media_manifest()
        meetings = {row["meeting_id"]: row for row in manifest["meetings"]}
        self.assertIsNone(meetings["ES2011b"]["whiteboard_preview"])
        self.assertIsNone(meetings["ES2011a"]["document_directory"])
        self.assertTrue(meetings["ES2006a"]["video"]["url"].endswith(".Corner.avi"))
        self.assertFalse(meetings["ES2006a"]["local_media_downloaded"])

    def test_prepare_keeps_source_corpus_when_query_limit_changes(self):
        def fixture_download(url, path, **kwargs):
            self.assertIn("/data/Product/", url)
            path.write_text(json.dumps(sample_meeting()), encoding="utf-8")
            return path

        with tempfile.TemporaryDirectory() as temp, patch(
            "projectmemory.datasets.ami.download", side_effect=fixture_download
        ) as downloader:
            full, subset = Path(temp) / "full", Path(temp) / "subset"
            prepare(full)
            report = prepare(subset, limit=3, seed=17)
            self.assertEqual(downloader.call_count, 16)
            self.assertEqual(read_jsonl(full / "corpus.jsonl"), read_jsonl(subset / "corpus.jsonl"))
            self.assertEqual(report["queries"], 3)
            self.assertEqual(report["general_queries_excluded_from_retrieval"], 8)
            expected_queries = read_jsonl(subset / "queries.jsonl")
            prepare(subset, limit=3, seed=17)
            self.assertEqual(expected_queries, read_jsonl(subset / "queries.jsonl"))
            self.assertEqual(report["modalities_prepared"], ["text"])


if __name__ == "__main__":
    unittest.main()
