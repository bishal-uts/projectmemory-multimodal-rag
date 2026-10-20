"""Evaluate answers against verified saved rankings, without loading an encoder.

This replays the exact evidence from a completed retrieval experiment. It does
not rerank, pair media, trim evidence, or use relevance labels to select sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.evaluation import corpus_fingerprint, validate_dataset
from projectmemory.io import fingerprint, read_jsonl
from projectmemory.rag import evaluate_answers
from projectmemory.settings import DEFAULT_MODELS, create_generator


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SavedRankingsRetriever:
    """Strict replay of a full retrieval report for the current prepared data."""

    model_name = "saved-retrieval-rankings"
    mode = "verified-ranking-replay"

    def __init__(self, corpus: list[dict], queries: list[dict], report_path: Path):
        validate_dataset(corpus, queries)
        self.report_path = Path(report_path).resolve()
        raw = self.report_path.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get("task") != "retrieval" or report.get("schema_version") != 1:
            raise ValueError("Expected a completed schema-version-1 retrieval report")
        self.method = report.get("method")
        if self.method not in {"bm25", "dense-text", "multimodal"}:
            raise ValueError("Saved report has an unsupported retrieval method")
        self.k = _positive_int(report.get("k"), "Saved retrieval k")
        self.split = report.get("split")
        if self.split is not None and (not isinstance(self.split, str) or not self.split):
            raise ValueError("Saved report split must be a nonempty string or null")
        selected = [q for q in queries if self.split is None or q.get("split") == self.split]
        if not selected:
            raise ValueError("Saved report split selects no questions in this dataset")
        self.corpus_fingerprint = corpus_fingerprint(corpus)
        self.query_fingerprint = fingerprint(selected)
        if report.get("corpus_fingerprint") != self.corpus_fingerprint:
            raise ValueError("Saved rankings do not match the current corpus/media fingerprint")
        if report.get("query_fingerprint") != self.query_fingerprint:
            raise ValueError("Saved rankings do not match the complete selected query fingerprint")
        if report.get("corpus_size") != len(corpus):
            raise ValueError("Saved report corpus size differs from this dataset")
        if (report.get("query_count") != len(selected) or report.get("skipped_count") != 0
                or report.get("skipped") != []):
            raise ValueError("Replay requires complete rankings for every selected question, with no skipped queries")
        rows = report.get("queries")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Saved report must contain per-question ranking objects")
        row_ids = [row.get("id") for row in rows]
        if any(not isinstance(identity, str) or not identity for identity in row_ids):
            raise ValueError("Saved ranking query IDs must be nonempty strings")
        expected_ids = {q["id"] for q in selected}
        if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
            raise ValueError("Saved ranking query IDs must uniquely cover the complete selected question set")
        by_id = {row["id"]: row for row in rows}
        texts = [q["question"] for q in selected]
        if len(texts) != len(set(texts)):
            raise ValueError("Ambiguous duplicate question text cannot be replayed through a question-only retriever")
        lookup = {record["id"]: record for record in corpus}
        self._rankings, self._candidates = {}, {}
        for query in selected:
            row = by_id[query["id"]]
            question = query["question"]
            if row.get("question") != question:
                raise ValueError("Saved ranking question text differs from the prepared query")
            if query.get("options"):
                raise ValueError("Supplied-media multiple-choice QA is not a saved-ranking retrieval experiment")
            relevant = query.get("relevant_ids")
            if not relevant or row.get("relevant_ids") != relevant:
                raise ValueError("Saved ranking relevance annotations differ from the prepared query")
            scope = query.get("metadata", {}).get("candidate_source_ids")
            if scope is not None and (not isinstance(scope, list) or not scope
                                      or any(not isinstance(source, str) or not source for source in scope)
                                      or len(scope) != len(set(scope))):
                raise ValueError("Query supplied-source scope must contain unique nonempty source IDs")
            if row.get("candidate_source_ids") != scope:
                raise ValueError("Saved ranking supplied-source scope differs from the prepared query")
            candidates = ({record["id"] for record in corpus if record.get("source_id") in scope}
                          if scope is not None else None)
            if candidates is not None and (not candidates or not set(relevant).issubset(candidates)):
                raise ValueError("Prepared query has an invalid supplied-source scope")
            if row.get("candidate_count") != (len(candidates) if candidates is not None else len(corpus)):
                raise ValueError("Saved ranking candidate count differs from the supplied-source scope")
            hits = row.get("hits")
            if not isinstance(hits, list) or len(hits) > self.k:
                raise ValueError("Saved hits must be a list within the report's retrieval k")
            clean, seen = [], set()
            for hit in hits:
                if not isinstance(hit, dict):
                    raise ValueError("Each saved hit must be an object")
                identity, score = hit.get("id"), hit.get("score")
                if not isinstance(identity, str) or identity not in lookup or identity in seen:
                    raise ValueError("Saved hits contain a duplicate or unknown evidence ID")
                if candidates is not None and identity not in candidates:
                    raise ValueError("Saved evidence falls outside the query's supplied-source scope")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("Saved retrieval scores must be finite numbers")
                seen.add(identity)
                clean.append({"id": identity, "score": score})
            self._rankings[question] = clean
            self._candidates[question] = candidates
        self.selected_query_ids = [q["id"] for q in selected]
        self.report_sha256 = hashlib.sha256(raw).hexdigest()
        self.runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # evaluate_answers records revision in its resumable run manifest.
        self.revision = (f"saved-report-sha256:{self.report_sha256};"
                         f"replay-script-sha256:{self.runner_sha256}")

    def search(self, query: str, top_k: int = 5, candidate_ids=None) -> list[dict]:
        if _positive_int(top_k, "Requested k") != self.k:
            raise ValueError("Replay must use exactly the saved report's retrieval k")
        if not isinstance(query, str) or query not in self._rankings:
            raise ValueError("Question text does not match a verified saved ranking")
        expected = self._candidates[query]
        if expected is None:
            if candidate_ids is not None:
                raise ValueError("Requested source scope differs from the saved ranking")
        elif candidate_ids is None or set(candidate_ids) != expected:
            raise ValueError("Requested source scope differs from the saved ranking")
        return [dict(hit) for hit in self._rankings[query]]

    def provenance(self, dataset: Path, limit: int) -> dict:
        return {
            "schema_version": 1, "protocol": "exact_saved_retrieval_ranking_replay",
            "dataset": str(Path(dataset).resolve()), "retrieval_report": str(self.report_path),
            "retrieval_report_sha256": self.report_sha256, "replay_script_sha256": self.runner_sha256,
            "corpus_fingerprint": self.corpus_fingerprint, "full_query_fingerprint": self.query_fingerprint,
            "method": self.method, "k": self.k, "split": self.split,
            "complete_report_query_ids": self.selected_query_ids,
            "evaluated_query_ids": self.selected_query_ids[:limit],
            "embedding_model_loaded": False,
            "evidence_selection": "Exact stored hit order and scores; no reranking, media pairing, or evidence truncation.",
        }


def _save_provenance(output: Path, provenance: dict) -> None:
    path = output / "retrieval_source.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Output contains different replay provenance; choose a new output directory")
        return
    if any((output / name).exists() for name in ("run.json", "predictions.jsonl", "report.json")):
        raise ValueError("Existing answer results have no replay provenance; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
            temporary = stream.name
            json.dump(provenance, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def run(dataset: Path, retrieval_report: Path, output: Path, *, provider: str = "local",
        limit: int = 100, max_output_tokens: int = 4096, record_format_failures: bool = False,
        generator=None) -> dict:
    """Validate provenance first, then use the existing grounded answer evaluator."""
    _positive_int(limit, "limit")
    _positive_int(max_output_tokens, "max_output_tokens")
    if provider not in DEFAULT_MODELS:
        raise ValueError("Choose local, nvidia or gemini")
    if not isinstance(record_format_failures, bool):
        raise ValueError("record_format_failures must be a boolean")
    if record_format_failures and provider != "local":
        raise ValueError("Recording format failures is supported only for the local Gemma provider")
    dataset, output = Path(dataset), Path(output)
    corpus, queries = read_jsonl(dataset / "corpus.jsonl"), read_jsonl(dataset / "queries.jsonl")
    retriever = SavedRankingsRetriever(corpus, queries, retrieval_report)
    provenance = retriever.provenance(dataset, limit)
    provenance["record_format_failures"] = record_format_failures
    _save_provenance(output, provenance)
    if generator is None:
        generator = create_generator(provider=provider, max_output_tokens=max_output_tokens)
    elif generator.provider != provider or generator.max_output_tokens != max_output_tokens:
        raise ValueError("Injected generator must match the requested provider and output budget")
    if record_format_failures:
        from projectmemory.evaluation_policy import FormatFailureRecorder
        generator = FormatFailureRecorder(generator)
    report = evaluate_answers(corpus, queries, retriever, generator, output,
                              k=retriever.k, split=retriever.split, limit=limit, method=retriever.method)
    if record_format_failures:
        from projectmemory.evaluation_policy import summarize_evaluation_failures
        report = summarize_evaluation_failures(report, output)
    return report


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--retrieval-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), default="local")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--record-format-failures", action="store_true",
                        help="Count exhausted local JSON/citation-format failures as unanswered and incorrect; all other errors still stop")
    args = parser.parse_args(argv)
    try:
        report = run(args.dataset, args.retrieval_report, args.output, provider=args.provider,
                     limit=args.limit, max_output_tokens=args.max_output_tokens,
                     record_format_failures=args.record_format_failures)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Saved-ranking evaluation stopped: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"provider": report["provider"], "model": report["model"],
                      "method": report["method"], "completed": report["completed"],
                      "metrics": report["metrics"], "evaluation_failures": report.get("evaluation_failures"),
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
