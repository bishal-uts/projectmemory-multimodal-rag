"""Deterministic retrieval baselines with no required third-party packages."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any


def tokenize(text: str) -> list[str]:
    """Unicode word tokens, with case folding and no stemming or stopword removal."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def validate_corpus(corpus: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy records and reject IDs that would make evaluation ambiguous."""
    records = [dict(item) for item in corpus]
    seen: set[str] = set()
    for item in records:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every corpus record needs a nonempty string id.")
        if identifier in seen:
            raise ValueError(f"Duplicate corpus id: {identifier}")
        seen.add(identifier)
        for field in ("text", "title"):
            if item.get(field) is not None and not isinstance(item[field], str):
                raise ValueError(f"Record {identifier!r}: {field} must be a string.")
    return records


def record_text(item: dict[str, Any]) -> str:
    """Use supplied text, OCR, or transcript plus title; never use gold answers."""
    return "\n".join(value.strip() for field in ("title", "text")
                     if (value := item.get(field)) and value.strip())


def validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer.")


def candidate_indices(corpus: list[dict[str, Any]],
                      candidate_ids: list[str] | set[str] | None) -> set[int]:
    """Resolve an optional scope using retrieval-unit IDs.

    Unknown IDs are ignored. ``None`` means all records, while an empty list
    means no candidates. Callers must expand parent source IDs to record IDs.
    """
    if candidate_ids is None:
        return set(range(len(corpus)))
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a collection of IDs, not a string.")
    allowed = set(candidate_ids)
    return {index for index, item in enumerate(corpus) if item["id"] in allowed}


class BM25Retriever:
    """Okapi BM25 using positive Robertson IDF, k1=1.5, and b=0.75.

    Search returns only positive-score matches. An unknown or empty query returns
    no results, rather than arbitrary zero-score documents. Exact score ties are
    ordered lexicographically by stable corpus ID, independently of input order.
    Each input record is one retrieval unit; chunk long sources upstream and keep
    their source_id/page/timestamp metadata for attribution.
    """

    def __init__(self, corpus: list[dict[str, Any]], *, k1: float = 1.5,
                 b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero.")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between zero and one.")
        self.corpus = validate_corpus(corpus)
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for index, item in enumerate(self.corpus):
            counts = Counter(tokenize(record_text(item)))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term].append((index, frequency))
        self._average_length = (sum(self._lengths) / len(self.corpus)
                                if self.corpus else 0.0)

    def search(self, query: str, top_k: int = 5,
               candidate_ids: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
        """Rank within candidate scope, retaining the full index's BM25 IDF."""
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not top_k or not self._average_length:
            return []
        candidates = candidate_indices(self.corpus, candidate_ids)
        if not candidates:
            return []
        scores: dict[int, float] = defaultdict(float)
        document_count = len(self.corpus)
        for term, query_frequency in Counter(tokenize(query)).items():
            postings = self._postings.get(term, ())
            if not postings:
                continue
            frequency = len(postings)
            idf = math.log1p((document_count - frequency + 0.5) / (frequency + 0.5))
            for index, count in postings:
                if index not in candidates:
                    continue
                normalization = 1 - self.b + self.b * self._lengths[index] / self._average_length
                scores[index] += query_frequency * idf * count * (self.k1 + 1) / (
                    count + self.k1 * normalization
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], self.corpus[index]["id"]))
        return [{"id": self.corpus[index]["id"], "score": scores[index]}
                for index in ranked[:top_k] if scores[index] > 0]


def create_retriever(corpus: list[dict[str, Any]], method: str = "bm25",
                     **kwargs: Any) -> Any:
    """Construct a baseline without importing ML libraries for BM25 users."""
    if method == "bm25":
        return BM25Retriever(corpus, **kwargs)
    if method in {"dense-text", "multimodal"}:
        from .embeddings import EmbeddingRetriever
        return EmbeddingRetriever(corpus, mode="text" if method == "dense-text" else "multimodal",
                                  **kwargs)
    raise ValueError(f"Unknown retrieval method {method!r}; use bm25, dense-text, or multimodal.")
