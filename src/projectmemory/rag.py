"""Grounded retrieval-to-answer flow and resumable reference-based evaluation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
import time

from .evaluation import corpus_fingerprint, validate_dataset
from .io import fingerprint, read_jsonl, write_json


def _before_search(retriever):
    if str(getattr(retriever, "device", "")).startswith("cuda"):
        from .localruntime import stop
        stop()


def _answer(generator, question, evidence, retriever, options=None):
    if getattr(generator, "provider", None) == "local" and hasattr(retriever, "unload"):
        retriever.unload()
    return generator.answer(question, evidence, **({"options": options} if options else {}))


def ask_question(corpus, question, retriever, generator=None, top_k=5, source_id=None):
    lookup = {row["id"]: row for row in corpus}
    candidate_ids = [row["id"] for row in corpus if row.get("source_id") == source_id] if source_id else None
    _before_search(retriever)
    hits = retriever.search(question, top_k=top_k, candidate_ids=candidate_ids)
    if any(hit["id"] not in lookup for hit in hits):
        raise ValueError("Retriever returned an unknown evidence ID")
    evidence = [dict(lookup[hit["id"]], retrieval_score=hit["score"], score=hit["score"]) for hit in hits]
    answer = _answer(generator, question, evidence, retriever) if generator else None
    return {"question": question, "evidence": evidence, "generated_answer": answer}


def _tokens(value):
    return re.findall(r"\w+", str(value).casefold(), flags=re.UNICODE)


def _normal(value):
    return " ".join(_tokens(value))


def _token_f1(prediction, reference):
    predicted, gold = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((predicted & gold).values())
    if not overlap:
        return float(not predicted and not gold)
    precision, recall = overlap / sum(predicted.values()), overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction, reference):
    predicted, gold = _tokens(prediction), _tokens(reference)
    previous = [0] * (len(gold) + 1)
    for word in predicted:
        current = [0]
        for index, other in enumerate(gold, 1):
            current.append(previous[index - 1] + 1 if word == other else max(previous[index], current[-1]))
        previous = current
    if not predicted or not gold:
        return float(not predicted and not gold)
    return 2 * previous[-1] / (len(predicted) + len(gold))


def answer_metrics(query, answer):
    references = query.get("reference_answers", [])
    if not references:
        raise ValueError("Answer evaluation requires reference answers")
    abstained = bool(answer.get("abstained"))
    citations = set(answer.get("citations", []))
    support = set(query.get("relevant_ids", []))
    metrics = {"answer_coverage": float(not abstained)}
    if support:
        metrics.update({"citation_gold_precision": len(citations & support) / len(citations) if citations else 0.0,
                        "citation_gold_recall": len(citations & support) / len(support)})
    if query.get("options"):
        metrics["choice_accuracy"] = float(not abstained and answer.get("answer_label") == query.get("answer_label"))
        return metrics
    if query.get("metadata", {}).get("dataset") == "ami_qmsum":
        prediction = answer.get("answer", "") if not abstained else ""
        metrics["token_f1"] = max(_token_f1(prediction, ref) for ref in references)
        metrics["rouge_l_f1"] = max(_rouge_l(prediction, ref) for ref in references)
    else:
        predicted = {_normal(item) for item in answer.get("answer_items", []) if _normal(item)} if not abstained else set()
        gold = {_normal(item) for item in references if _normal(item)}
        metrics["answer_set_exact_match"] = float(predicted == gold)
        metrics["answer_set_precision"] = len(predicted & gold) / len(predicted) if predicted else 0.0
        metrics["answer_set_recall"] = len(predicted & gold) / len(gold) if gold else 0.0
        metrics["token_f1"] = _token_f1(" ".join(sorted(predicted)), " ".join(sorted(gold)))
    return metrics


def evaluate_answers(corpus, queries, retriever, generator, output: Path, *, k=5,
                     split=None, limit=5, method="bm25", media_mode="combined",
                     prediction_only=False):
    """Resume only an identical run; persist each completed answer immediately.

    Supplied-media QA routes its known clip directly to the generator. RAG uses
    retrieved evidence. Gold answers/supports never enter a generation request.
    Failures stop the run with completed predictions retained, not invented scores.
    """
    validate_dataset(corpus, queries)
    if limit < 1 or k < 1:
        raise ValueError("limit and k must be positive")
    selected = [row for row in queries if split is None or row.get("split") == split][:limit]
    if not selected:
        raise ValueError("No matching questions")
    if media_mode not in {"audio", "video", "combined"}:
        raise ValueError("media_mode must be audio, video or combined")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    retriever_config = {name: getattr(retriever, name) for name in
                        ("model_name", "revision", "mode", "dtype", "device", "processing_kwargs", "k1", "b")
                        if hasattr(retriever, name)}
    # Configuration and implementation hashes make resumed scoring sensitive to
    # changes in prompts, inference settings, retrieval, and metric definitions.
    implementation_files = ["rag.py", "generation.py", "nvidia.py", "retrieval.py", "embeddings.py"]
    if generator.provider == "local":
        implementation_files += ["local_generation.py", "localruntime.py"]
    implementation = {name: fingerprint(Path(__file__).with_name(name).read_text(encoding="utf-8"))
                      for name in implementation_files}
    manifest = {"schema_version": 2, "corpus_fingerprint": corpus_fingerprint(corpus),
                "query_fingerprint": fingerprint(selected), "questions": len(selected),
                "method": method, "k": k, "split": split, "provider": generator.provider,
                "model": generator.model, "max_output_tokens": generator.max_output_tokens,
                "reasoning_budget": getattr(generator, "reasoning_budget", None),
                "retriever_config": retriever_config, "implementation": implementation,
                "media_mode": media_mode, "prediction_only": prediction_only}
    if hasattr(generator, "evaluation_config"):
        manifest["generator_config"] = generator.evaluation_config()
    config_path, prediction_path = output / "run.json", output / "predictions.jsonl"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Output contains a different evaluation configuration; choose a new output directory")
    else:
        if prediction_path.exists():
            raise ValueError("Predictions exist without a run manifest; choose a new output directory")
        write_json(config_path, manifest)
    completed = read_jsonl(prediction_path) if prediction_path.exists() else []
    by_id = {row["id"]: row for row in completed}
    selected_ids = {q["id"] for q in selected}
    if len(by_id) != len(completed) or set(by_id) - selected_ids:
        raise ValueError("Saved prediction IDs do not match this run")
    lookup = {row["id"]: row for row in corpus}
    for query in selected:
        if query["id"] in by_id:
            continue
        started = time.perf_counter()
        if query.get("options"):
            from .datasets.daily_omni import require_media
            require_media([query], corpus, mode=media_mode)
            kinds = {"audio", "video"} if media_mode == "combined" else {media_mode}
            evidence = [lookup[identity] for identity in query["metadata"]["known_media_ids"]
                        if lookup[identity]["modality"] in kinds]
            # Video-only records must actually be silent, not merely labelled so.
            from .av import probe_media
            for row in evidence:
                if row["modality"] == "video" and probe_media(Path(row["path"]))["audio_streams"]:
                    raise ValueError("Daily-Omni visual input still has audio; prepare silent visual files before scoring ablations")
            answer = _answer(generator, query["question"], evidence, retriever, options=query["options"])
        else:
            scope = query.get("metadata", {}).get("candidate_source_ids")
            candidate_ids = [row["id"] for row in corpus if row.get("source_id") in scope] if scope is not None else None
            _before_search(retriever)
            hits = retriever.search(query["question"], top_k=k, candidate_ids=candidate_ids)
            if any(hit["id"] not in lookup or (candidate_ids is not None and hit["id"] not in candidate_ids) for hit in hits):
                raise ValueError("Retrieved evidence violates the query's supplied source scope")
            evidence = [lookup[hit["id"]] for hit in hits]
            answer = _answer(generator, query["question"], evidence, retriever)
        if set(answer.get("citations", [])) - {row["id"] for row in evidence}:
            raise ValueError("Answer contains citations outside the provided evidence")
        row = {"id": query["id"], "question": query["question"], "evidence_ids": [item["id"] for item in evidence],
               "answer_label": answer.get("answer_label", ""), "result": answer,
               "elapsed_seconds": time.perf_counter() - started,
               "metrics": {} if prediction_only else answer_metrics(query, answer)}
        with prediction_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        by_id[row["id"]] = row
        print(f"Answered {len(by_id)}/{len(selected)} questions", flush=True)
    ordered = [by_id[q["id"]] for q in selected]
    metric_names = {key for row in ordered for key in row["metrics"]}
    report = {**manifest, "task": "answer_evaluation", "created_at": datetime.now(timezone.utc).isoformat(),
              "completed": len(ordered), "query_count": len(ordered),
              "config": {"dataset_manifest": {"dataset": selected[0].get("metadata", {}).get("dataset", "unknown")}},
              "metrics": {key: mean(row["metrics"][key] for row in ordered if key in row["metrics"]) for key in sorted(metric_names)},
              "model_versions": sorted({str(row["result"].get("model_version") or "unspecified") for row in ordered}),
              "limitations": ["Lexical reference overlap is not a complete measure of factual correctness.",
                              "Gold-source citation overlap does not prove that each claim is supported.",
                              "Public benchmarks may overlap with model training/evaluation data.",
                              "This is a supplied-media QA experiment when options are present, not retrieval evaluation."]}
    write_json(output / "report.json", report)
    return report
