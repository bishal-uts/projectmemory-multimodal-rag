"""Retrieval correctness tests; no model downloads or ML packages are required."""

import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projectmemory.embeddings import EmbeddingRetriever
from projectmemory.retrieval import BM25Retriever, create_retriever


class BM25Tests(unittest.TestCase):
    def test_scores_match_bm25_formula_and_length_normalization(self):
        corpus = [{"id": "short", "text": "cat"},
                  {"id": "long", "text": "cat dog dog"},
                  {"id": "irrelevant", "text": "bird"}]
        results = BM25Retriever(corpus).search("CAT")
        self.assertEqual([item["id"] for item in results], ["short", "long"])
        idf = math.log1p((3 - 2 + 0.5) / (2 + 0.5))
        expected = idf * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 1 / (5 / 3)))
        self.assertAlmostEqual(results[0]["score"], expected, places=12)

    def test_no_overlap_empty_and_zero_k_do_not_return_arbitrary_records(self):
        retriever = BM25Retriever([{"id": "a", "text": "cat"}, {"id": "empty"}])
        for query in ("", " !!! ", "unseen"):
            self.assertEqual(retriever.search(query), [])
        self.assertEqual(retriever.search("cat", top_k=0), [])
        self.assertEqual(BM25Retriever([]).search("anything"), [])
        self.assertEqual(BM25Retriever([{"id": "empty"}]).search("anything"), [])

    def test_ties_are_deterministic_across_input_order(self):
        corpus = [{"id": "z", "text": "same"}, {"id": "a", "text": "same"}]
        expected = ["a", "z"]
        for records in (corpus, list(reversed(corpus))):
            self.assertEqual([row["id"] for row in BM25Retriever(records).search("same")], expected)

    def test_title_and_non_ascii_text_are_searchable(self):
        retriever = BM25Retriever([{"id": "diagram", "title": "Über architecture", "text": "東京"}])
        self.assertEqual(retriever.search("ÜBER")[0]["id"], "diagram")
        self.assertEqual(retriever.search("東京")[0]["id"], "diagram")

    def test_candidate_scope_uses_record_ids_and_retains_global_statistics(self):
        corpus = [{"id": "first", "source_id": "meeting-a", "text": "cat"},
                  {"id": "second", "source_id": "meeting-b", "text": "cat cat"},
                  {"id": "third", "source_id": "meeting-a", "text": "cat dog"}]
        retriever = BM25Retriever(corpus)
        unfiltered = {row["id"]: row["score"] for row in retriever.search("cat")}
        scoped = retriever.search("cat", candidate_ids=["first", "third"])
        self.assertEqual({row["id"] for row in scoped}, {"first", "third"})
        for row in scoped:
            self.assertEqual(row["score"], unfiltered[row["id"]])
        self.assertEqual(retriever.search("cat", candidate_ids={"second"})[0]["id"], "second")
        self.assertEqual(retriever.search("cat", candidate_ids=[]), [])
        self.assertEqual(retriever.search("cat", candidate_ids=["unknown"]), [])
        self.assertEqual(retriever.search("cat", candidate_ids=["meeting-a"]), [])

    def test_invalid_parameters_and_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            BM25Retriever([{"id": "x"}, {"id": "x"}])
        for kwargs in ({"k1": 0}, {"b": -1}, {"b": float("nan")}):
            with self.assertRaises(ValueError):
                BM25Retriever([], **kwargs)
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                BM25Retriever([]).search("query", value)


class RecordingEncoder:
    """Deliberately unnormalized outputs exercise cosine normalization."""

    def __init__(self):
        self.calls = []

    def encode(self, values, **kwargs):
        self.calls.append((values, kwargs))
        if kwargs.get("prompt_name") == "SearchQuery":
            return [[5, 0] for _ in values]
        return [[4, 0] if isinstance(value, dict) or "match" in value else [0, 2]
                for value in values]


class EmbeddingTests(unittest.TestCase):
    def test_native_media_and_text_have_separate_formats_and_correct_cosine(self):
        with tempfile.TemporaryDirectory() as directory:
            media_paths = []
            for modality in ("image", "audio", "video"):
                path = Path(directory) / modality
                path.write_bytes(b"fake media; never decoded by this mock")
                media_paths.append(path)
            corpus = [{"id": kind, "modality": kind, "path": str(path),
                       "text": "caption must not enter the native encoder"}
                      for kind, path in zip(("image", "audio", "video"), media_paths)]
            corpus += [{"id": "plain", "modality": "text", "text": "other"},
                       {"id": "titled", "modality": "text", "title": "Diagram", "text": "match"}]
            encoder = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder) as loader:
                retriever = EmbeddingRetriever(corpus, mode="multimodal")
                loader.assert_not_called()
                results = retriever.search("find evidence", top_k=5)
                loader.assert_called_once()
            self.assertEqual([row["id"] for row in results], ["audio", "image", "titled", "video", "plain"])
            self.assertEqual([row["score"] for row in results], [1.0, 1.0, 1.0, 1.0, 0.0])
            for index, modality in enumerate(("image", "audio", "video")):
                values, kwargs = encoder.calls[index]
                self.assertEqual(values, [{modality: str(media_paths[index].resolve())}])
                self.assertEqual(kwargs["prompt"], "")
                self.assertNotIn("prompt_name", kwargs)
            self.assertEqual(encoder.calls[3][0], ["title: none | text: other",
                                                 "title: Diagram | text: match"])
            self.assertEqual(encoder.calls[3][1]["prompt"], "")
            self.assertEqual(encoder.calls[-1][1]["prompt_name"], "SearchQuery")
            self.assertEqual(len(encoder.calls), 5)  # One call per modality, plus query.

    def test_text_baseline_uses_media_transcript_without_opening_media(self):
        corpus = [{"id": "audio", "modality": "audio", "text": "match transcript", "title": "Meeting"}]
        encoder = RecordingEncoder()
        with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder):
            self.assertEqual(EmbeddingRetriever(corpus, mode="text").search("query")[0]["score"], 1.0)
        self.assertEqual(encoder.calls[0][0], ["title: Meeting | text: match transcript"])
        self.assertEqual(encoder.calls[0][1]["prompt"], "")

    def test_dense_candidate_scope_applies_before_top_k(self):
        corpus = [{"id": "a", "source_id": "meeting-a", "text": "match"},
                  {"id": "b", "source_id": "meeting-b", "text": "other"}]
        encoder = RecordingEncoder()
        with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder) as loader:
            retriever = EmbeddingRetriever(corpus)
            self.assertEqual(retriever.search("query", candidate_ids=[]), [])
            self.assertEqual(retriever.search("query", candidate_ids=["unknown"]), [])
            loader.assert_not_called()
            self.assertEqual(retriever.search("query", 1, {"b"}), [{"id": "b", "score": 0.0}])
            self.assertEqual(retriever.search("query", 1, ["a"]), [{"id": "a", "score": 1.0}])

    def test_native_missing_media_never_falls_back_to_captions(self):
        for item in ({"id": "a", "modality": "audio", "text": "available transcript"},
                     {"id": "v", "modality": "video", "path": str(Path("missing.mp4").resolve())}):
            with self.assertRaises(FileNotFoundError), patch.object(EmbeddingRetriever, "_load_model") as loader:
                EmbeddingRetriever([item], mode="multimodal")
            loader.assert_not_called()

    def test_empty_search_is_lazy_and_precision_is_validated(self):
        with patch.object(EmbeddingRetriever, "_load_model") as loader:
            retriever = EmbeddingRetriever([{"id": "a", "text": "match"}])
            self.assertEqual(retriever.search("  "), [])
            self.assertEqual(retriever.search("query", 0), [])
            self.assertEqual(EmbeddingRetriever([]).search("query"), [])
            loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "float16"):
            EmbeddingRetriever([], dtype="float16")

    def test_cache_reuses_index_but_media_content_changes_invalidate_it(self):
        # Workspace-local temporary files permit atomic replace under Windows
        # AppContainer sandboxes, where the redirected system temp may not.
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            media = Path(directory) / "asset.wav"
            media.write_bytes(b"first")
            corpus = [{"id": "audio", "modality": "audio", "path": str(media)}]
            options = {"mode": "multimodal", "cache_dir": Path(directory) / "cache", "revision": "fixed"}
            first = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=first):
                EmbeddingRetriever(corpus, **options).search("query")
            self.assertEqual(len(first.calls), 2)
            second = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=second):
                EmbeddingRetriever(corpus, **options).search("query")
            self.assertEqual(len(second.calls), 1)  # Only the query is encoded.
            media.write_bytes(b"other")  # Same byte count, different content.
            third = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=third):
                EmbeddingRetriever(corpus, **options).search("query")
            self.assertEqual(len(third.calls), 2)

    def test_cache_key_changes_for_text_model_and_mode(self):
        corpus = [{"id": "a", "text": "match"}]
        signatures = {
            EmbeddingRetriever(corpus)._fingerprint(),
            EmbeddingRetriever(corpus, model_name="different-model")._fingerprint(),
            EmbeddingRetriever(corpus, mode="multimodal")._fingerprint(),
            EmbeddingRetriever([{"id": "a", "text": "changed"}])._fingerprint(),
            EmbeddingRetriever(corpus, revision="commit")._fingerprint(),
        }
        self.assertEqual(len(signatures), 5)

    def test_build_index_initializes_query_model_even_when_index_is_cached(self):
        corpus = [{"id": "a", "title": "First", "text": "match"},
                  {"id": "b", "title": "Second", "text": "other"}]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            first = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=first) as loader:
                retriever = EmbeddingRetriever(corpus, cache_dir=Path(directory))
                retriever.build_index()
                loader.assert_called_once()
                self.assertEqual(len(first.calls), 1)  # Different titles share a batch.
                self.assertEqual(len(first.calls[0][0]), 2)
                self.assertEqual(retriever.search("query", 1), [{"id": "a", "score": 1.0}])
                self.assertEqual(len(first.calls), 2)
            second = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=second) as loader:
                cached = EmbeddingRetriever(corpus, cache_dir=Path(directory))
                cached.build_index()
                loader.assert_called_once()
                self.assertEqual(second.calls, [])
                self.assertEqual(cached.search("query", 1), [{"id": "a", "score": 1.0}])
                self.assertEqual(len(second.calls), 1)

    def test_nonfinite_encoder_values_are_rejected(self):
        encoder = RecordingEncoder()
        encoder.encode = lambda *args, **kwargs: [[float("nan"), 1]]
        with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                EmbeddingRetriever([{"id": "a", "text": "match"}]).search("query")

    def test_interrupted_index_resumes_completed_batches_with_new_batch_size(self):
        corpus = [{"id": str(index), "text": "match"} for index in range(5)]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            first = RecordingEncoder()
            original_encode = first.encode

            def interrupted(values, **kwargs):
                if len(first.calls) == 1:
                    raise RuntimeError("interrupted")
                return original_encode(values, **kwargs)

            first.encode = interrupted
            with patch.object(EmbeddingRetriever, "_load_model", return_value=first):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    EmbeddingRetriever(corpus, cache_dir=Path(directory), batch_size=2).build_index()
            second = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=second):
                resumed = EmbeddingRetriever(corpus, cache_dir=Path(directory), batch_size=3)
                resumed.build_index()
                self.assertEqual(len(second.calls), 1)
                self.assertEqual(len(second.calls[0][0]), 3)
                self.assertEqual([row["id"] for row in resumed.search("query")], list("01234"))

    def test_damaged_partial_cache_is_reencoded_without_reusing_invalid_indices(self):
        corpus = [{"id": str(index), "text": "match"} for index in range(3)]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            retriever = EmbeddingRetriever(corpus, cache_dir=Path(directory), batch_size=2)
            fingerprint = retriever._fingerprint()
            partial = Path(directory) / f"partial-{fingerprint}"
            partial.mkdir()
            (partial / "batch-damaged.json").write_text("{", encoding="utf-8")
            (partial / "batch-invalid.json").write_text(json.dumps({
                "fingerprint": fingerprint, "indices": [-1], "vectors": [[1, 0]],
            }), encoding="utf-8")
            encoder = RecordingEncoder()
            with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder):
                retriever.build_index()
            self.assertEqual(sum(len(values) for values, _ in encoder.calls), 3)

    def test_length_batched_index_restores_original_record_alignment(self):
        corpus = [{"id": "short-match", "text": "match"},
                  {"id": "long-other", "text": "other " * 30},
                  {"id": "medium-match", "text": "match " * 10},
                  {"id": "tiny-other", "text": "x"}]
        encoder = RecordingEncoder()
        with patch.object(EmbeddingRetriever, "_load_model", return_value=encoder):
            retriever = EmbeddingRetriever(corpus, batch_size=2)
            results = retriever.search("query", 4)
        self.assertEqual([row["id"] for row in results[:2]], ["medium-match", "short-match"])
        self.assertTrue(all(row["score"] == 1.0 for row in results[:2]))
        self.assertTrue(all(row["score"] == 0.0 for row in results[2:]))

    def test_factory_keeps_bm25_dependency_free(self):
        self.assertIsInstance(create_retriever([], "bm25"), BM25Retriever)
        self.assertIsInstance(create_retriever([], "dense-text"), EmbeddingRetriever)
        with self.assertRaises(ValueError):
            create_retriever([], "unknown")

    def test_unload_preserves_index_and_reloads_only_query_encoder(self):
        corpus = [{"id": "match", "text": "match"}, {"id": "other", "text": "other"}]
        first, second = RecordingEncoder(), RecordingEncoder()
        retriever = EmbeddingRetriever(corpus)
        with patch.object(EmbeddingRetriever, "_load_model", side_effect=[first, second]) as load:
            before = retriever.search("query")
            vectors = retriever._vectors
            retriever.unload()
            self.assertIsNone(retriever._model)
            self.assertIs(retriever._vectors, vectors)
            after = retriever.search("query")
            self.assertEqual(load.call_count, 2)
        self.assertEqual(before, after)
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.calls[0][1]["prompt_name"], "SearchQuery")


if __name__ == "__main__":
    unittest.main()
