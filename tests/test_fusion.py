import numpy as np
import pytest

from projectmemory.fusion import PrecomputedImageRanker, SeparateEncoderRetriever, reciprocal_rank_fusion


def test_rrf_sums_reciprocal_ranks_across_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert [row["id"] for row in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_with_disjoint_lists_alternates_and_ties_go_to_the_first_list():
    fused = reciprocal_rank_fusion([["t1", "t2", "t3"], ["i1", "i2", "i3"]])
    assert [row["id"] for row in fused[:5]] == ["t1", "i1", "t2", "i2", "t3"]


def test_rrf_rejects_duplicates_and_bad_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a", "a"]])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def _ranker():
    images = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    questions = np.array([[0.0, 2.0]])
    return PrecomputedImageRanker(["img:a", "img:b", "img:c"], images, ["Which one?"], questions)


def test_precomputed_ranker_orders_by_cosine_and_respects_scope():
    ranker = _ranker()
    assert [row["id"] for row in ranker.rank("Which one?")] == ["img:b", "img:c", "img:a"]
    assert [row["id"] for row in ranker.rank("Which one?", {"img:a", "img:c"})] == ["img:c", "img:a"]
    with pytest.raises(ValueError):
        ranker.rank("Another question?")


def test_precomputed_ranker_validates_inputs():
    with pytest.raises(ValueError):
        PrecomputedImageRanker(["a", "a"], np.eye(2), ["q"], np.ones((1, 2)))
    with pytest.raises(ValueError):
        PrecomputedImageRanker(["a", "b"], np.eye(2), ["q"], np.ones((1, 3)))


class _TextRetriever:
    def __init__(self, order):
        self.order = order
        self.calls = []

    def search(self, query, top_k=5, candidate_ids=None):
        self.calls.append((query, top_k, list(candidate_ids)))
        allowed = set(candidate_ids)
        return [{"id": identity, "score": 1.0 - index / 10} for index, identity in enumerate(self.order)
                if identity in allowed][:top_k]


def test_separate_encoder_retriever_fuses_full_text_and_image_lists():
    text = _TextRetriever(["txt:2", "txt:1", "tab:1"])
    retriever = SeparateEncoderRetriever(text, ["txt:1", "txt:2", "tab:1"], _ranker())
    hits = retriever.search("Which one?", top_k=5)
    assert [hit["id"] for hit in hits] == ["txt:2", "img:b", "txt:1", "img:c", "tab:1"]
    assert text.calls == [("Which one?", 3, ["txt:1", "txt:2", "tab:1"])]


def test_separate_encoder_retriever_scopes_both_lists():
    text = _TextRetriever(["txt:2", "txt:1"])
    retriever = SeparateEncoderRetriever(text, ["txt:1", "txt:2"], _ranker())
    hits = retriever.search("Which one?", top_k=5, candidate_ids=["txt:1", "img:a"])
    assert [hit["id"] for hit in hits] == ["txt:1", "img:a"]


def test_separate_encoder_retriever_rejects_overlapping_indexes():
    with pytest.raises(ValueError):
        SeparateEncoderRetriever(_TextRetriever([]), ["img:a"], _ranker())
