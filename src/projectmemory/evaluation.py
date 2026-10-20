"""Retrieval metrics only: no generation, judge calls, or hidden answer access."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from statistics import mean
import time

from .io import fingerprint


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Retrieval metrics require at least one gold source")
    # Duplicates consume their original rank positions and retrieval budget.
    # Removing them before truncation would promote evidence from beyond k.
    seen = set()
    hits = []
    for index, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in relevant:
            hits.append(index)
    dcg = sum(1 / math.log2(index + 1) for index in hits)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(k, len(relevant)) + 1))
    return {
        f"recall@{k}": len(hits) / len(relevant),
        f"hit_rate@{k}": float(bool(hits)),
        f"mrr@{k}": 1 / hits[0] if hits else 0.0,
        f"ndcg@{k}": dcg / ideal,
        f"all_support@{k}": float(len(hits) == len(relevant)),
    }


def corpus_fingerprint(corpus: list[dict]) -> str:
    """Fingerprint records and local native-media content, including missing files.

    Text baselines may use descriptors before media is downloaded. A missing or
    unassigned asset is therefore recorded rather than rejected; downloading or
    replacing it changes the signature used to guard comparisons. Directories
    and relative paths are invalid native-media references.
    """
    assets = []
    path_hashes = {}
    for record in corpus:
        if record.get("modality") == "text":
            continue
        asset = {"id": record["id"]}
        raw_path = record.get("path")
        if raw_path is None or raw_path == "":
            asset["status"] = "unassigned"
        else:
            if not isinstance(raw_path, str):
                raise ValueError(f"Media path for {record['id']} must be a string")
            path = Path(raw_path)
            if not path.is_absolute():
                raise ValueError(f"Media path for {record['id']} must be absolute: {raw_path}")
            if not path.exists():
                asset["status"] = "missing"
            else:
                if not path.is_file():
                    raise ValueError(f"Media path for {record['id']} must refer to a file: {raw_path}")
                resolved = path.resolve()
                if resolved not in path_hashes:
                    digest = hashlib.sha256()
                    with resolved.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    path_hashes[resolved] = digest.hexdigest()
                asset.update(status="present", sha256=path_hashes[resolved])
        assets.append(asset)
    return fingerprint({"records": corpus, "native_media": assets})


def validate_dataset(corpus: list[dict], queries: list[dict]) -> None:
    for label, records in (("corpus", corpus), ("queries", queries)):
        ids = [record.get("id") for record in records]
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError(f"Every {label} record must have a nonempty string id")
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} IDs: {duplicates[:5]}")
    corpus_ids = {record["id"] for record in corpus}
    if not corpus_ids:
        raise ValueError("The corpus is empty")
    for record in corpus:
        if "reference_answers" in record or "answer_label" in record:
            raise ValueError(f"Evaluation labels found in corpus record {record['id']}")
        if record.get("modality") not in {"text", "image", "audio", "video"}:
            raise ValueError(f"Unknown modality on {record['id']}")
    for query in queries:
        if not isinstance(query.get("question"), str) or not query["question"].strip():
            raise ValueError(f"Empty question: {query['id']}")
        missing = set(query.get("relevant_ids", [])) - corpus_ids
        if missing:
            raise ValueError(f"Gold evidence absent from corpus for {query['id']}: {sorted(missing)[:5]}")


def evaluate_retrieval(corpus: list[dict], queries: list[dict], retriever, *, k: int = 5, method: str = "unknown", split: str | None = None, config: dict | None = None) -> dict:
    validate_dataset(corpus, queries)
    if k < 1:
        raise ValueError("k must be positive")
    selected = [q for q in queries if split is None or q.get("split") == split]
    if not selected:
        raise ValueError(f"No queries match split {split!r}")
    scored, skipped = [], []
    corpus_ids = {row["id"] for row in corpus}
    sources = {}
    for row in corpus:
        sources.setdefault(row.get("source_id"), []).append(row["id"])
    for query in selected:
        if not query.get("relevant_ids"):
            skipped.append({"id": query["id"], "reason": "No retrieval relevance labels"})
            continue
        source_scope = query.get("metadata", {}).get("candidate_source_ids")
        candidate_ids = None
        if source_scope is not None:
            candidate_ids = sorted({item for source in source_scope for item in sources.get(source, [])})
            if not candidate_ids or not set(query["relevant_ids"]).issubset(candidate_ids):
                raise ValueError(f"Invalid supplied-source scope for {query['id']}")
        started = time.perf_counter()
        # Only question text reaches the retriever, never reference answers or gold IDs.
        hits = (retriever.search(query["question"], top_k=k, candidate_ids=candidate_ids)
                if candidate_ids is not None else retriever.search(query["question"], top_k=k))
        elapsed = time.perf_counter() - started
        ids = [hit["id"] for hit in hits]
        if set(ids) - corpus_ids:
            raise ValueError("Retriever returned unknown source IDs")
        if candidate_ids is not None and set(ids) - set(candidate_ids):
            raise ValueError("Retriever returned sources outside the supplied-source scope")
        scored.append({
            "id": query["id"], "question": query["question"], "hits": hits,
            "relevant_ids": query["relevant_ids"], "latency_seconds": elapsed,
            "candidate_count": len(candidate_ids) if candidate_ids is not None else len(corpus),
            "candidate_source_ids": source_scope,
            "metrics": retrieval_metrics(ids, query["relevant_ids"], k),
        })
    if not scored:
        raise ValueError("No labelled retrieval queries. Supplied-media QA datasets cannot be scored as corpus retrieval.")
    metrics = {name: mean(row["metrics"][name] for row in scored) for name in scored[0]["metrics"]}
    times = sorted(row["latency_seconds"] for row in scored)
    return {
        "schema_version": 1, "task": "retrieval", "method": method, "k": k,
        "created_at": datetime.now(timezone.utc).isoformat(), "split": split,
        "config": config or {}, "corpus_fingerprint": corpus_fingerprint(corpus),
        "query_fingerprint": fingerprint(selected), "corpus_size": len(corpus),
        "query_count": len(scored), "skipped_count": len(skipped),
        "supplied_source_scoped_queries": sum(row["candidate_source_ids"] is not None for row in scored),
        "metrics": metrics,
        "latency_seconds": {"mean": mean(times), "p95": times[max(0, math.ceil(.95 * len(times)) - 1)]},
        "queries": scored, "skipped": skipped,
        "limitations": ["Retrieval only; answer generation and answer quality were not evaluated.", "Subset/custom corpus results are not directly comparable to published full-benchmark scores."],
    }


def comparison_markdown(reports: list[dict]) -> str:
    if not reports:
        raise ValueError("No reports to compare")
    signatures = {(r["corpus_fingerprint"], r["query_fingerprint"], r["k"]) for r in reports}
    if len(signatures) != 1:
        raise ValueError("Comparison requires identical corpus, questions, and k")
    k = reports[0]["k"]
    lines = ["# Retrieval comparison", "", f"Corpus: {reports[0]['corpus_size']} items. Questions: {reports[0]['query_count']}. k={k}.", "", "| Method | Recall | Hit rate | MRR | nDCG | All support | Mean search (ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        m = report["metrics"]
        scores = [m[f"{name}@{k}"] for name in ("recall", "hit_rate", "mrr", "ndcg", "all_support")]
        lines.append("| " + report["method"] + " | " + " | ".join(f"{value:.4f}" for value in scores) + f" | {report['latency_seconds']['mean'] * 1000:.2f} |")
    lines.extend(["", "Search latency excludes model loading and index construction. This evaluates retrieval, not generated answers.", "Do not compare this custom subset directly with published full-benchmark leaderboards.", ""])
    return "\n".join(lines)
