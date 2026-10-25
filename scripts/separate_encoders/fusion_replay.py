"""Exact replay of the saved separate-encoder rankings into the answer evaluator.

Kept apart from run_separate_encoders.py so that its hash, which every answer run
records as replay provenance, changes only when the replay itself changes. The checks
mirror evaluate_saved_rankings.py, which accepts only the three original methods.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from projectmemory.evaluation import corpus_fingerprint
from projectmemory.io import fingerprint

METHOD = "separate-encoders"


class FusionReplay:
    model_name = "saved-retrieval-rankings"
    mode = "verified-ranking-replay"

    def __init__(self, corpus: list[dict], queries: list[dict], report_path: Path):
        self.report_path = Path(report_path).resolve()
        raw = self.report_path.read_bytes()
        report = json.loads(raw)
        if report.get("task") != "retrieval" or report.get("schema_version") != 1 or report.get("method") != METHOD:
            raise ValueError("Expected the saved separate-encoder retrieval report")
        self.method, self.k, self.split = METHOD, report["k"], report.get("split")
        if self.split is not None:
            raise ValueError("The comparison uses every prepared question")
        self.corpus_fingerprint, self.query_fingerprint = corpus_fingerprint(corpus), fingerprint(queries)
        if (report.get("corpus_fingerprint"), report.get("query_fingerprint")) != (self.corpus_fingerprint, self.query_fingerprint):
            raise ValueError("Saved rankings do not match the current corpus, media or questions")
        if report.get("corpus_size") != len(corpus) or report.get("query_count") != len(queries) or report.get("skipped") != []:
            raise ValueError("Replay requires complete rankings for every question")
        rows = {row["id"]: row for row in report["queries"]}
        lookup = {record["id"] for record in corpus}
        if set(rows) != {query["id"] for query in queries} or len(rows) != len(report["queries"]):
            raise ValueError("Saved rankings must cover each question exactly once")
        self._rankings = {}
        for query in queries:
            row = rows[query["id"]]
            if row.get("question") != query["question"] or row.get("relevant_ids") != query["relevant_ids"]:
                raise ValueError("Saved ranking text or labels differ from the prepared query")
            if query.get("metadata", {}).get("candidate_source_ids") is not None or row.get("candidate_source_ids") is not None:
                raise ValueError("MultiModalQA questions have no supplied-source scope")
            hits = row["hits"]
            ids = [hit["id"] for hit in hits]
            if not 0 < len(hits) <= self.k or len(set(ids)) != len(ids) or not set(ids) <= lookup:
                raise ValueError("Saved hits must be unique corpus IDs within k")
            if any(isinstance(hit["score"], bool) or not math.isfinite(hit["score"]) for hit in hits):
                raise ValueError("Saved scores must be finite numbers")
            self._rankings[query["question"]] = [{"id": hit["id"], "score": hit["score"]} for hit in hits]
        self.query_ids = [query["id"] for query in queries]
        self.report_sha256 = hashlib.sha256(raw).hexdigest()
        self.runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # evaluate_answers records revision in its resumable run manifest.
        self.revision = f"saved-report-sha256:{self.report_sha256};replay-script-sha256:{self.runner_sha256}"

    def search(self, query: str, top_k: int = 5, candidate_ids=None) -> list[dict]:
        if top_k != self.k or candidate_ids is not None or query not in self._rankings:
            raise ValueError("Replay serves exactly the saved question, k and scope")
        return [dict(hit) for hit in self._rankings[query]]

    def provenance(self, dataset: Path, limit: int) -> dict:
        return {"schema_version": 1, "protocol": "exact_saved_retrieval_ranking_replay", "dataset": str(Path(dataset).resolve()),
                "retrieval_report": str(self.report_path), "retrieval_report_sha256": self.report_sha256,
                "replay_script_sha256": self.runner_sha256, "corpus_fingerprint": self.corpus_fingerprint,
                "full_query_fingerprint": self.query_fingerprint, "method": self.method, "k": self.k, "split": self.split,
                "complete_report_query_ids": self.query_ids, "evaluated_query_ids": self.query_ids[:limit],
                "embedding_model_loaded": False, "record_format_failures": True,
                "evidence_selection": "Exact stored hit order and scores; no reranking, media pairing, or evidence truncation."}
