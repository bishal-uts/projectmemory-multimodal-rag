"""Separate-encoder baseline for the MultiModalQA comparison, added after the main runs.

Text passages and serialized tables keep their EmbeddingGemma 2 text ranking; images are
ranked by SigLIP 2, whose vectors are computed on the Cetus cluster by
separate_encoders/siglip_embed.py; the two lists are merged by reciprocal rank fusion
(k=60, fixed before scoring). Everything else matches the three existing methods: the
same 100 questions and 2,015 candidates, top-five evidence, and the frozen local Gemma 4
generator replaying the saved rankings.

  export    write the cluster bundle: the corpus images and the question text only
  retrieve  build the fused rankings, score them like the other methods, image diagnostics
  answers   replay the saved fused rankings into the fixed local generator (resumable)
  control   re-run 20 original native questions with today's generator code (see DECODER_NOTE)
  analyze   audit all four answer runs and compare them (paired bootstrap, seed 42)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from projectmemory.evaluation import comparison_markdown, corpus_fingerprint, evaluate_retrieval, validate_dataset  # noqa: E402
from projectmemory.io import fingerprint, read_jsonl, write_json, write_jsonl  # noqa: E402

DATASET = ROOT / "data/prepared/mmqa"
RESULTS = ROOT / "results/separate-encoders"
ANSWERS = ROOT / "results/final-mmqa-local-separate-encoders"
BUNDLE = ROOT / ".cache/separate-encoders/bundle"
VECTORS = RESULTS / "siglip2-mmqa.npz"
METHOD = "separate-encoders"
EXISTING = ("bm25", "dense-text", "multimodal")
LABELS = {"bm25": "BM25", "dense-text": "Dense text", "multimodal": "Native multimodal",
          METHOD: "Separate encoders (SigLIP 2 + text, RRF)"}
TEXT_MODEL, TEXT_REVISION = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
ADDED_NOTE = ("Added on 24 October 2026, after the three-method comparison had been run and reported, "
              "on the same development questions. Encoders, fusion rule and k were fixed before scoring.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[list[dict], list[dict]]:
    corpus, queries = read_jsonl(DATASET / "corpus.jsonl"), read_jsonl(DATASET / "queries.jsonl")
    validate_dataset(corpus, queries)
    return corpus, queries


def is_image(identity: str) -> bool:
    return identity.startswith("mmqa:image:")


def export() -> None:
    corpus, queries = load()
    images = [row for row in corpus if row["modality"] == "image"]
    paths = [Path(row["path"]) for row in images]
    if len({path.name for path in paths}) != len(paths):
        raise SystemExit("Image file names must be unique to share one folder")
    (BUNDLE / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for row, path in zip(images, paths):
        shutil.copy2(path, BUNDLE / "images" / path.name)
        rows.append({"id": row["id"], "file": path.name, "sha256": sha256(path)})
    write_jsonl(BUNDLE / "images.jsonl", rows)
    # Question text only: answers and relevance labels never leave this machine.
    write_jsonl(BUNDLE / "questions.jsonl", [{"id": query["id"], "question": query["question"]} for query in queries])
    print(f"bundle: {len(rows)} images, {len(queries)} questions -> {BUNDLE}")


def image_ranker(corpus: list[dict], queries: list[dict]):
    import numpy as np
    from projectmemory.fusion import PrecomputedImageRanker
    record = json.loads(VECTORS.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(VECTORS) != record["output_sha256"]:
        raise SystemExit("SigLIP 2 vectors do not match their record")
    data = np.load(VECTORS)
    image_ids = [str(item) for item in data["image_ids"]]
    if sorted(image_ids) != sorted(row["id"] for row in corpus if row["modality"] == "image"):
        raise SystemExit("SigLIP 2 vectors do not cover exactly the corpus images")
    stored = dict(zip((str(item) for item in data["question_ids"]), (str(item) for item in data["questions"])))
    if stored != {query["id"]: query["question"] for query in queries}:
        raise SystemExit("SigLIP 2 question vectors do not match the prepared questions")
    ranker = PrecomputedImageRanker(image_ids, data["image_vectors"], [str(item) for item in data["questions"]],
                                    data["question_vectors"])
    return ranker, record


def gold_image_recall(report: dict) -> dict:
    """Same diagnostic as plot_results.py: labelled image sources in the original top five."""
    recalls, any_hit = [], []
    for row in report["queries"]:
        gold = {identity for identity in row["relevant_ids"] if is_image(identity)}
        if gold:
            top = {hit["id"] for hit in row["hits"][:report["k"]]}
            recalls.append(len(top & gold) / len(gold))
            any_hit.append(float(bool(top & gold)))
    return {"questions": len(recalls), "image_support_recall@5": mean(recalls), "any_gold_image@5": mean(any_hit)}


def recall_by_need(report: dict) -> dict:
    groups = {"needs_image": [], "no_image": []}
    for row in report["queries"]:
        key = "needs_image" if any(is_image(identity) for identity in row["relevant_ids"]) else "no_image"
        groups[key].append(row["metrics"][f"recall@{report['k']}"])
    return {key: {"questions": len(values), "recall@5": mean(values)} for key, values in groups.items()}


def image_only(ranking_by_question, queries: list[dict]) -> dict:
    """Rank only the 920 images: how well does each encoder place the labelled image?"""
    rows = []
    for query in queries:
        gold = {identity for identity in query["relevant_ids"] if is_image(identity)}
        if not gold:
            continue
        ranked = [hit["id"] for hit in ranking_by_question(query["question"])]
        positions = sorted(ranked.index(identity) + 1 for identity in gold)
        rows.append({"id": query["id"], "gold_image_ranks": positions,
                     "recall@5": sum(position <= 5 for position in positions) / len(positions),
                     "hit@5": float(positions[0] <= 5), "reciprocal_rank": 1 / positions[0]})
    return {"questions": len(rows), "image_recall@5": mean(row["recall@5"] for row in rows),
            "image_hit@5": mean(row["hit@5"] for row in rows), "mrr": mean(row["reciprocal_rank"] for row in rows),
            "median_first_gold_rank": median(row["gold_image_ranks"][0] for row in rows), "per_question": rows}


def embedding_retriever(corpus: list[dict], mode: str, device: str):
    from projectmemory.embeddings import EmbeddingRetriever
    return EmbeddingRetriever(corpus, mode=mode, model_name=TEXT_MODEL, device=device, cache_dir=ROOT / ".cache/embeddings",
                              revision=TEXT_REVISION, batch_size=8, dtype="float32")


def retrieve(device: str) -> None:
    from projectmemory.fusion import RRF_K, SeparateEncoderRetriever
    corpus, queries = load()
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    dense = json.loads((ROOT / "results/mmqa-comparison/dense-text.json").read_text(encoding="utf-8"))
    if (dense["config"]["model_revision"], dense["config"]["dtype"]) != (TEXT_REVISION, "float32"):
        raise SystemExit("Text encoder settings must match the dense-text run")
    ranker, siglip = image_ranker(corpus, queries)
    text_ids = [row["id"] for row in corpus if row["modality"] == "text"]
    image_ids = [row["id"] for row in corpus if row["modality"] == "image"]
    started = time.perf_counter()
    text = embedding_retriever(corpus, "text", device)
    text.build_index()
    retriever = SeparateEncoderRetriever(text, text_ids, ranker, rrf_k=RRF_K)
    build_seconds = time.perf_counter() - started
    config = {
        "dataset": str(DATASET.resolve()), "dataset_manifest": manifest,
        "text_encoder": {"model": TEXT_MODEL, "model_revision": TEXT_REVISION, "mode": "text", "device": device,
                         "dtype": "float32", "batch_size": 8, "records": len(text_ids),
                         "representation": "997 passages and 98 serialized tables, title-aware Document prefix (identical to dense text)"},
        "image_encoder": {key: siglip[key] for key in ("model", "revision", "dtype", "device", "dimension", "text_max_tokens",
                                                       "questions_truncated", "image_processing", "text_processing",
                                                       "transformers", "torch", "host")} | {"records": len(image_ids),
                                                       "vectors_sha256": siglip["output_sha256"]},
        "fusion": {"method": "reciprocal_rank_fusion", "k": RRF_K, "lists": ["text and tables (EmbeddingGemma 2)", "images (SigLIP 2)"],
                   "tie_break": "earlier list (text), then ID", "consequence": "the top five always holds three text/table items and two images"},
        "added_after_main_comparison": ADDED_NOTE,
    }
    report = evaluate_retrieval(corpus, queries, retriever, k=5, method=METHOD, config=config)
    report["index_build_seconds"] = build_seconds
    report["limitations"] = report["limitations"] + [
        ADDED_NOTE,
        "Search latency excludes SigLIP 2 encoding of the question, which ran with the image vectors on the cluster."]
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / f"{METHOD}.json", report)

    reports = {method: json.loads((ROOT / f"results/mmqa-comparison/{method}.json").read_text(encoding="utf-8"))
               for method in EXISTING} | {METHOD: report}
    gold = {method: gold_image_recall(item) for method, item in reports.items()}
    by_need = {method: recall_by_need(item) for method, item in reports.items()}
    diagnostics = {"image_only_rankings": {
        "siglip2_pixels": image_only(ranker.rank, queries),
        "embeddinggemma_titles": image_only(lambda question: text.search(question, top_k=len(image_ids), candidate_ids=image_ids), queries)}}
    text.unload()
    native = embedding_retriever(corpus, "multimodal", device)
    native.build_index()
    diagnostics["image_only_rankings"]["embeddinggemma_pixels"] = image_only(
        lambda question: native.search(question, top_k=len(image_ids), candidate_ids=image_ids), queries)
    native.unload()
    summary = {"schema_version": 1, "method": METHOD, "note": ADDED_NOTE,
               "metrics": {method: item["metrics"] for method, item in reports.items()},
               "gold_image_recall": gold, "recall_by_image_need": by_need, "diagnostics": diagnostics,
               "image_only_note": "Ranks only the 920 images for the questions with labelled images, ignoring text and fusion. "
                                  "EmbeddingGemma titles is the dense-text representation of images; pixels is the native one."}
    write_json(RESULTS / "retrieval-summary.json", summary)
    (RESULTS / "comparison.md").write_text(comparison_markdown(list(reports.values())), encoding="utf-8")
    print(json.dumps({"metrics": summary["metrics"], "gold_image_recall": gold,
                      "image_only": {name: {key: value for key, value in item.items() if key != "per_question"}
                                     for name, item in diagnostics["image_only_rankings"].items()}}, indent=2))


def answers() -> None:
    from evaluate_saved_rankings import _save_provenance
    from projectmemory.evaluation_policy import FormatFailureRecorder, summarize_evaluation_failures
    from projectmemory.localruntime import stop
    from projectmemory.rag import evaluate_answers
    from projectmemory.settings import create_generator
    from separate_encoders.fusion_replay import FusionReplay
    corpus, queries = load()
    retriever = FusionReplay(corpus, queries, RESULTS / f"{METHOD}.json")
    _save_provenance(ANSWERS, retriever.provenance(DATASET, len(queries)))
    # Exactly the generator of run_final_evaluations.py: local Gemma 4, 4,096 output tokens, failures recorded.
    generator = FormatFailureRecorder(create_generator(provider="local", max_output_tokens=4096))
    try:
        report = evaluate_answers(corpus, queries, retriever, generator, ANSWERS, k=retriever.k, split=None,
                                  limit=len(queries), method=METHOD)
        report = summarize_evaluation_failures(report, ANSWERS)
    finally:
        stop(root=ROOT)
    print(json.dumps({"completed": report["completed"], "metrics": report["metrics"],
                      "failures": report.get("evaluation_failures")}, indent=2))


CONTROL = ROOT / "results/control-mmqa-local-multimodal-v2"
DECODER_NOTE = ("The three original MMQA runs used local_generation.py before the video-only decoder repair "
                "(media protocol v1); the new run uses the repaired file (v2). The repair changed video frame "
                "sampling only, which MMQA evidence (text, tables, images) never uses. generator-control.json "
                "re-runs original native questions with the repaired file to check that answers are unchanged.")


def control(limit: int = 20) -> None:
    """Re-run the first questions of the original native condition with the current generator code."""
    from evaluate_saved_rankings import run as replay
    from projectmemory.localruntime import stop
    try:
        replay(DATASET, ROOT / "results/mmqa-comparison/multimodal.json", CONTROL, limit=limit, record_format_failures=True)
    finally:
        stop(root=ROOT)
    original = read_jsonl(ROOT / "results/final-mmqa-local-multimodal/predictions.jsonl")[:limit]
    rerun = read_jsonl(CONTROL / "predictions.jsonl")
    fields = ("answer", "answer_items", "citations", "abstained", "answer_label")
    rows = [{"id": old["id"], "same_evidence": old["evidence_ids"] == new["evidence_ids"],
             "same_answer": all(old["result"].get(name) == new["result"].get(name) for name in fields),
             "same_metrics": old["metrics"] == new["metrics"]} for old, new in zip(original, rerun)]
    old_run = json.loads((ROOT / "results/final-mmqa-local-multimodal/run.json").read_text(encoding="utf-8"))
    new_run = json.loads((CONTROL / "run.json").read_text(encoding="utf-8"))
    summary = {"questions": len(rows), "identical_answers": sum(row["same_answer"] for row in rows),
               "identical_metrics": sum(row["same_metrics"] for row in rows),
               "identical_evidence": sum(row["same_evidence"] for row in rows),
               "original_local_generation": old_run["implementation"]["local_generation.py"],
               "rerun_local_generation": new_run["implementation"]["local_generation.py"],
               "note": DECODER_NOTE, "rows": rows}
    write_json(RESULTS / "generator-control.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


def compatible_except_decoder(runs: dict[str, dict]) -> dict:
    """compatible_runs, allowing only the documented video-decoder difference in the generator code."""
    first = runs[EXISTING[0]]["manifest"]
    allowed = {}
    for condition, run in runs.items():
        manifest = run["manifest"]
        for key in ("schema_version", "corpus_fingerprint", "query_fingerprint", "questions", "k", "split", "provider",
                    "model", "max_output_tokens", "reasoning_budget", "prediction_only", "media_mode"):
            if manifest.get(key) != first.get(key):
                raise SystemExit(f"Unpaired run: {key} differs for {condition}")
        ours, theirs = dict(manifest["generator_config"]), dict(first["generator_config"])
        code_ours, code_theirs = dict(manifest["implementation"]), dict(first["implementation"])
        for config in (ours, theirs):
            config.pop("adapter_sha256"), config.pop("media_protocol")
        for code in (code_ours, code_theirs):
            code.pop("local_generation.py")
        if ours != theirs or code_ours != code_theirs:
            raise SystemExit(f"Generator settings or other code differ for {condition}")
        protocol = manifest["generator_config"]["media_protocol"]
        allowed[condition] = {"media_protocol_version": protocol["version"],
                              "local_generation_fingerprint": manifest["implementation"]["local_generation.py"]}
    return allowed


def analyze() -> None:
    import analyze_final_results as audit
    corpus, queries = load()
    expected = corpus_fingerprint(corpus)
    runs = {method: audit.load_run(ROOT / f"results/final-mmqa-local-{method}", corpus, queries, task="mmqa",
                                   condition=method, expected_count=len(queries), expected_corpus_fingerprint=expected)
            for method in (*EXISTING, METHOD)}
    audit.compatible_runs({method: runs[method] for method in EXISTING}, task="mmqa")
    versions = compatible_except_decoder(runs)
    control_summary = json.loads((RESULTS / "generator-control.json").read_text(encoding="utf-8"))
    summary = audit.summarize_task(runs, corpus, queries, task="mmqa")
    summary.update({"note": ADDED_NOTE, "generator_code_versions": versions, "generator_code_note": DECODER_NOTE,
                    "generator_control": {key: value for key, value in control_summary.items() if key != "rows"},
                    "analysis_script_sha256": sha256(Path(__file__)),
                    "audit_functions": "scripts/analyze_final_results.py load_run, compatible_runs, summarize_task",
                    "audit_script_sha256": sha256(ROOT / "scripts/analyze_final_results.py")})
    write_json(RESULTS / "answer-analysis.json", summary)
    lines = ["| Method | Exact match | Token F1 | Answered | Invalid outputs |", "|---|---:|---:|---:|---:|"]
    for method, item in summary["conditions"].items():
        metrics = item["metrics"]
        lines.append(f"| {LABELS[method]} | {metrics['answer_set_exact_match']:.0%} | {metrics.get('token_f1', float('nan')):.2%} "
                     f"| {item['answered']} | {item['invalid_model_response_count']} |")
    lines += ["", "| Comparison | Difference | 95% interval |", "|---|---:|---:|"]
    for pair in summary["paired_comparisons"]:
        if METHOD in (pair["baseline"], pair["comparison"]):
            low, high = pair["ci95"]
            lines.append(f"| {LABELS[pair['comparison']]} minus {LABELS[pair['baseline']]} | {pair['difference'] * 100:+.0f} points "
                         f"| {low * 100:+.0f} to {high * 100:+.0f} |")
    (RESULTS / "answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["export", "retrieve", "answers", "control", "analyze"])
    parser.add_argument("--device", default="cuda", help="Device for EmbeddingGemma 2 (retrieve step)")
    args = parser.parse_args()
    if args.step == "export":
        export()
    elif args.step == "retrieve":
        retrieve(args.device)
    elif args.step == "answers":
        answers()
    elif args.step == "control":
        control()
    else:
        analyze()


if __name__ == "__main__":
    main()
