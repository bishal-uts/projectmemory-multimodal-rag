import math
import pytest

from projectmemory.cli import score_choices
from projectmemory.evaluation import comparison_markdown, corpus_fingerprint, evaluate_retrieval, retrieval_metrics, validate_dataset


def test_recall_is_not_hit_rate_and_duplicate_hits_do_not_inflate():
    result = retrieval_metrics(["a", "a", "x"], ["a", "b"], 3)
    assert result["recall@3"] == .5
    assert result["hit_rate@3"] == 1
    assert result["all_support@3"] == 0
    assert result["ndcg@3"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_rank_discount_and_reciprocal_rank():
    result = retrieval_metrics(["irrelevant", "gold"], ["gold"], 2)
    assert result["mrr@2"] == .5
    assert result["ndcg@2"] == pytest.approx(1 / math.log2(3))


def test_duplicates_cannot_promote_evidence_from_beyond_top_k():
    result = retrieval_metrics(["irrelevant", "irrelevant", "gold"], ["gold"], 2)
    assert result["recall@2"] == 0
    assert result["hit_rate@2"] == 0
    assert result["mrr@2"] == 0
    assert result["ndcg@2"] == 0


def test_duplicates_preserve_later_rank_and_get_credit_only_once():
    result = retrieval_metrics(["first", "first", "second"], ["first", "second"], 3)
    assert result["recall@3"] == 1
    assert result["ndcg@3"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    later = retrieval_metrics(["irrelevant", "irrelevant", "gold"], ["gold"], 3)
    assert later["mrr@3"] == pytest.approx(1 / 3)


def test_media_content_changes_invalidate_comparison_at_the_same_path(tmp_path):
    media = tmp_path / "evidence.png"
    corpus = [{"id": "image", "source_id": "s", "modality": "image", "path": str(media), "text": "descriptor"}]
    queries = [{"id": "q", "question": "Find evidence", "relevant_ids": ["image"]}]

    class Retriever:
        def search(self, question, top_k):
            return [{"id": "image", "score": 1.0}]

    missing = evaluate_retrieval(corpus, queries, Retriever())
    media.write_bytes(b"first")
    first = evaluate_retrieval(corpus, queries, Retriever())
    media.write_bytes(b"other")  # Same filename and byte count; actual content differs.
    second = evaluate_retrieval(corpus, queries, Retriever())
    assert len({report["corpus_fingerprint"] for report in (missing, first, second)}) == 3
    assert second["corpus_fingerprint"] == corpus_fingerprint(corpus)
    for previous in (missing, first):
        with pytest.raises(ValueError, match="identical"):
            comparison_markdown([previous, second])


def test_media_fingerprint_accepts_unassigned_files_and_rejects_bad_paths(tmp_path):
    assert corpus_fingerprint([{"id": "audio", "modality": "audio"}])
    for path in ("relative.wav", 123, str(tmp_path)):
        with pytest.raises(ValueError, match="path"):
            corpus_fingerprint([{"id": "audio", "modality": "audio", "path": path}])


def test_no_gold_is_not_perfect_recall():
    with pytest.raises(ValueError):
        retrieval_metrics([], [], 5)


def test_missing_gold_and_answer_leakage_rejected():
    corpus = [{"id": "a", "modality": "text", "text": "source"}]
    queries = [{"id": "q", "question": "Which source?", "relevant_ids": ["missing"]}]
    with pytest.raises(ValueError, match="absent"):
        validate_dataset(corpus, queries)
    corpus[0]["reference_answers"] = ["secret"]
    with pytest.raises(ValueError, match="labels"):
        validate_dataset(corpus, [])


def test_retriever_gets_question_only_and_test_split_stays_separate():
    class Spy:
        def search(self, query, top_k):
            assert query == "test question"
            return [{"id": "a", "score": 1.0}]
    corpus = [{"id": "a", "modality": "text", "text": "source"}]
    queries = [{"id": "q1", "question": "dev question", "split": "dev", "relevant_ids": ["a"]}, {"id": "q2", "question": "test question", "split": "test", "reference_answers": ["secret"], "relevant_ids": ["a"]}]
    report = evaluate_retrieval(corpus, queries, Spy(), split="test")
    assert report["query_count"] == 1
    assert report["metrics"]["recall@5"] == 1
    assert "secret" not in str(report)
    other = dict(report, corpus_fingerprint="different")
    with pytest.raises(ValueError, match="identical"):
        comparison_markdown([report, other])


def test_choice_scoring_counts_missing_predictions_as_wrong():
    queries = [{"id": "a", "answer_label": "B"}, {"id": "b", "answer_label": "A"}]
    result = score_choices(queries, [{"id": "a", "answer_label": "B"}])
    assert result["accuracy"] == .5
    assert result["coverage"] == .5
    with pytest.raises(ValueError, match="Unknown"):
        score_choices(queries, [{"id": "unrelated", "answer_label": "A"}])


def test_choice_scoring_rejects_invalid_labels_and_duplicates():
    queries = [{"id": "a", "answer_label": "B"}]
    with pytest.raises(ValueError, match="A/B/C/D"):
        score_choices(queries, [{"id": "a", "answer_label": "probably B"}])
    with pytest.raises(ValueError, match="Duplicate"):
        score_choices(queries, [{"id": "a", "answer_label": "B"}, {"id": "a", "answer_label": "B"}])


def test_supplied_meeting_scope_is_not_cross_meeting_retrieval():
    corpus = [{"id": "a", "source_id": "meeting1", "modality": "text"}, {"id": "b", "source_id": "meeting2", "modality": "text"}]
    queries = [{"id": "q", "question": "What was decided?", "relevant_ids": ["a"], "metadata": {"candidate_source_ids": ["meeting1"]}}]
    class Scoped:
        def search(self, query, top_k, candidate_ids):
            assert candidate_ids == ["a"]
            return [{"id": "a", "score": 1.0}]
    result = evaluate_retrieval(corpus, queries, Scoped())
    assert result["supplied_source_scoped_queries"] == 1
    assert result["queries"][0]["candidate_count"] == 1
