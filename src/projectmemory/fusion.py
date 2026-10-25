"""Separate-encoder retrieval: one encoder per modality, merged by reciprocal rank fusion.

This is the conventional multimodal RAG baseline compared with native EmbeddingGemma 2
retrieval. Text and table records are ranked by a text retriever; images are ranked by a
dedicated image-text encoder whose vectors were computed elsewhere (SigLIP 2 on the
cluster). Scores from different encoders are not comparable, so the lists are merged by
rank only. With one list per modality, equal-weight fusion alternates between the lists;
ties go to the earlier list, so the text list fills positions 1, 3 and 5 of a top five.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .retrieval import validate_top_k

RRF_K = 60  # Cormack et al. (2009); the usual default, fixed before any scoring.


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[dict[str, Any]]:
    """Merge ranked ID lists; ties keep list order, then ID order, for determinism."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("RRF k must be a positive integer.")
    scores: dict[str, float] = {}
    first_list: dict[str, int] = {}
    for list_index, ranking in enumerate(rankings):
        if len(set(ranking)) != len(ranking):
            raise ValueError("Each ranking must list an ID at most once.")
        for rank, identity in enumerate(ranking, start=1):
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            first_list.setdefault(identity, list_index)
    ordered = sorted(scores, key=lambda identity: (-scores[identity], first_list[identity], identity))
    return [{"id": identity, "score": scores[identity]} for identity in ordered]


class PrecomputedImageRanker:
    """Cosine ranking of images from stored question and image vectors (no model loaded)."""

    def __init__(self, image_ids: Sequence[str], image_vectors: np.ndarray,
                 questions: Sequence[str], question_vectors: np.ndarray) -> None:
        image_vectors = np.asarray(image_vectors, dtype=np.float64)
        question_vectors = np.asarray(question_vectors, dtype=np.float64)
        if image_vectors.ndim != 2 or question_vectors.ndim != 2 or image_vectors.shape[1] != question_vectors.shape[1]:
            raise ValueError("Image and question vectors must be 2-D with the same width.")
        if len(image_ids) != len(image_vectors) or len(set(image_ids)) != len(image_ids):
            raise ValueError("Image IDs must be unique and match the image vectors.")
        if len(questions) != len(question_vectors) or len(set(questions)) != len(questions):
            raise ValueError("Question texts must be unique and match the question vectors.")
        if not np.isfinite(image_vectors).all() or not np.isfinite(question_vectors).all():
            raise ValueError("Stored vectors must be finite.")
        self.image_ids = list(image_ids)
        self._images = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)
        self._questions = {text: vector / np.linalg.norm(vector) for text, vector in zip(questions, question_vectors)}

    def rank(self, question: str, candidate_ids: set[str] | None = None) -> list[dict[str, Any]]:
        if question not in self._questions:
            raise ValueError("No stored image-encoder vector for this exact question text.")
        scores = self._images @ self._questions[question]
        rows = [{"id": identity, "score": float(score)} for identity, score in zip(self.image_ids, scores)
                if candidate_ids is None or identity in candidate_ids]
        return sorted(rows, key=lambda row: (-row["score"], row["id"]))


class SeparateEncoderRetriever:
    """Text retriever for text and tables plus a precomputed image ranker, fused by rank."""

    def __init__(self, text_retriever, text_ids: Sequence[str], image_ranker: PrecomputedImageRanker,
                 rrf_k: int = RRF_K) -> None:
        if set(text_ids) & set(image_ranker.image_ids):
            raise ValueError("A record cannot be in both the text and the image index.")
        self.text_retriever = text_retriever
        self.text_ids = list(text_ids)
        self.image_ranker = image_ranker
        self.rrf_k = rrf_k

    def ranked_lists(self, query: str, candidate_ids=None) -> tuple[list[dict], list[dict]]:
        allowed = None if candidate_ids is None else set(candidate_ids)
        text_scope = self.text_ids if allowed is None else [identity for identity in self.text_ids if identity in allowed]
        text_hits = (self.text_retriever.search(query, top_k=len(text_scope), candidate_ids=text_scope)
                     if text_scope else [])
        return text_hits, self.image_ranker.rank(query, allowed)

    def search(self, query: str, top_k: int = 5, candidate_ids=None) -> list[dict[str, Any]]:
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not top_k or not query.strip():
            return []
        text_hits, image_hits = self.ranked_lists(query, candidate_ids)
        fused = reciprocal_rank_fusion([[hit["id"] for hit in text_hits], [hit["id"] for hit in image_hits]],
                                       k=self.rrf_k)
        return fused[:top_k]
