"""Dataset preparation, retrieval, grounded local/API answers and evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import time

from .evaluation import comparison_markdown, evaluate_retrieval, validate_dataset
from .io import read_jsonl, write_json
from .settings import load_project_env, create_generator, provider_status, configured_provider, DEFAULT_MODELS

DEFAULT_MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def _load_dataset(folder: Path) -> tuple[list[dict], list[dict]]:
    return read_jsonl(folder / "corpus.jsonl"), read_jsonl(folder / "queries.jsonl")


def _retriever(corpus, args, method):
    from .retrieval import BM25Retriever
    if method == "bm25":
        return BM25Retriever(corpus)
    from .embeddings import EmbeddingRetriever
    revision = args.revision or (DEFAULT_MODEL_REVISION if args.model == "google/embeddinggemma-2" else None)
    return EmbeddingRetriever(corpus, mode="text" if method == "dense-text" else "multimodal", model_name=args.model, device=args.device, cache_dir=args.cache_dir, revision=revision, batch_size=args.batch_size, dtype=args.dtype)


def _retrieval_options(parser):
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision", help="Model commit; defaults to the verified EmbeddingGemma 2 revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")


def _generation_options(parser):
    parser.add_argument("--provider", choices=list(DEFAULT_MODELS), help="Defaults to local Gemma 4 E2B; API providers remain available")
    parser.add_argument("--generator-model", help="Generation model ID; never changes the embedding model")
    parser.add_argument("--requests-per-minute", type=float, default=5, help="API pacing; ignored by the local provider")
    parser.add_argument("--max-output-tokens", type=int, default=4096)


def _generator(args):
    return create_generator(args.provider, args.generator_model,
                            requests_per_minute=args.requests_per_minute,
                            max_output_tokens=args.max_output_tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ProjectMemory: dataset preparation, retrieval, and reproducible evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check the local runtime without downloading models")
    doctor.add_argument("--check-model", action="store_true", help="Also query public Hugging Face model metadata")
    prepare = commands.add_parser("prepare", help="Download small annotations and create a reproducible dataset; no bulk media")
    prepare.add_argument("dataset", choices=["ami", "mmqa", "daily-omni"])
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--limit", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--raw-dir", type=Path)
    prepare.add_argument("--media-dir", type=Path)
    validate = commands.add_parser("validate", help="Check IDs, support coverage, modality coverage and media availability")
    validate.add_argument("--dataset", type=Path, required=True)
    media = commands.add_parser("fetch-media", help="Fetch original MMQA images or Daily-Omni media with a transfer budget")
    media.add_argument("--dataset", type=Path, required=True)
    media.add_argument("--max-mb", type=float, default=150, help="Decimal MB budget; Daily-Omni requires its whole 3.91 GB archive")
    media.add_argument("--limit", type=int, help="Number of missing media files to fetch, not number of questions")
    media.add_argument("--media-root", type=Path)
    silent = commands.add_parser("prepare-video-only", help="Make full-duration silent visual copies for audio/video ablations")
    silent.add_argument("--dataset", type=Path, required=True)
    silent.add_argument("--output", type=Path, default=Path("data/media/silent-video"))
    evaluate = commands.add_parser("evaluate", help="Evaluate one retrieval approach against gold supporting source IDs")
    _retrieval_options(evaluate)
    evaluate.add_argument("--method", choices=["bm25", "dense-text", "multimodal"], default="bm25")
    evaluate.add_argument("--split")
    evaluate.add_argument("--output", type=Path, default=Path("results/retrieval.json"))
    compare = commands.add_parser("compare", help="Run chosen methods against exactly the same corpus and queries")
    _retrieval_options(compare)
    compare.add_argument("--methods", nargs="+", choices=["bm25", "dense-text", "multimodal"], default=["bm25"])
    compare.add_argument("--split")
    compare.add_argument("--output", type=Path, default=Path("results/comparison"))
    search = commands.add_parser("search", help="Search prepared sources and return source citations (no generated answer)")
    _retrieval_options(search)
    search.add_argument("question")
    search.add_argument("--method", choices=["bm25", "dense-text", "multimodal"], default="bm25")
    search.add_argument("--source-id")
    ask = commands.add_parser("ask", help="Retrieve evidence and generate a grounded answer locally or through an API")
    _retrieval_options(ask)
    _generation_options(ask)
    ask.add_argument("question")
    ask.add_argument("--method", choices=["bm25", "dense-text", "multimodal"], default="bm25")
    ask.add_argument("--source-id")
    ask.add_argument("--output", type=Path)
    answer_eval = commands.add_parser("evaluate-answers", help="Run or resume local/API answer evaluation; starts with five questions")
    _retrieval_options(answer_eval)
    _generation_options(answer_eval)
    answer_eval.add_argument("--method", choices=["bm25", "dense-text", "multimodal"], default="bm25")
    answer_eval.add_argument("--split")
    answer_eval.add_argument("--limit", type=int, default=5)
    answer_eval.add_argument("--media-mode", choices=["audio", "video", "combined"], default="combined")
    answer_eval.add_argument("--output", type=Path, default=Path("results/answers"))
    ingest = commands.add_parser("ingest", help="Prepare local text, PDF, Office, image, audio and video source files")
    ingest.add_argument("paths", type=Path, nargs="+")
    ingest.add_argument("--output", type=Path, default=Path("data/prepared/uploads"))
    ingest.add_argument("--chunk-seconds", type=float, default=20)
    ingest.add_argument("--max-duration-seconds", type=float)
    serve = commands.add_parser("serve", help="Open the local project application at http://127.0.0.1:8765")
    serve.add_argument("--port", type=int, default=8765)
    score = commands.add_parser("score-choices", help="Score externally generated Daily-Omni choices; never calls a model")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("results/choice_accuracy.json"))
    return parser


def score_choices(queries: list[dict], predictions: list[dict]) -> dict:
    gold = {row["id"]: row for row in queries}
    if len(gold) != len(queries) or not gold:
        raise ValueError("Choice scoring requires nonempty, unique question IDs")
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    unknown = set(by_id) - set(gold)
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:5]}")
    results = []
    for query in queries:
        label = query.get("answer_label")
        if label not in {"A", "B", "C", "D"}:
            raise ValueError(f"Missing valid gold answer_label on {query['id']}")
        predicted = by_id.get(query["id"], {}).get("answer_label", "").strip().upper()
        if predicted and predicted not in {"A", "B", "C", "D"}:
            raise ValueError(f"Prediction for {query['id']} must use answer_label A/B/C/D")
        results.append({"id": query["id"], "correct": predicted == label, "answered": bool(predicted)})
    return {"task": "supplied_media_multiple_choice_qa", "questions": len(results), "predictions_supplied": len(by_id), "accuracy": sum(row["correct"] for row in results) / len(results), "coverage": sum(row["answered"] for row in results) / len(results), "missing_predictions_count_as_incorrect": True, "results": results, "limitations": ["This measures answer-choice accuracy, not corpus retrieval or timestamp grounding."]}


def main(argv=None) -> int:
    # Windows redirected consoles may default to cp1252; model answers and
    # source filenames can contain any Unicode character.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        load_project_env()
        if args.command == "doctor":
            packages = {}
            for name in ["torch", "numpy", "sentence-transformers", "transformers", "pillow", "soundfile", "torchcodec", "pytest"]:
                try:
                    packages[name] = metadata.version(name)
                except metadata.PackageNotFoundError:
                    packages[name] = None
            report = {"python": sys.version.split()[0], "ffmpeg": bool(shutil.which("ffmpeg")), "packages": packages, "generators": {provider: provider_status(provider) for provider in DEFAULT_MODELS}, "default_provider": configured_provider(), "api_calls_made": False, "default_runtime": "BM25 uses Python standard library only"}
            if args.check_model:
                from urllib.request import urlopen
                with urlopen("https://huggingface.co/api/models/google/embeddinggemma-2", timeout=30) as response:
                    model = json.load(response)
                report["model"] = {key: model.get(key) for key in ["id", "sha", "gated", "library_name"]}
            print(json.dumps(report, indent=2))
        elif args.command == "prepare":
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            module = importlib.import_module(f"projectmemory.datasets.{args.dataset.replace('-', '_')}")
            kwargs = {name: getattr(args, name) for name in ("raw_dir", "media_dir") if getattr(args, name) is not None}
            output = args.output or Path("data/prepared") / args.dataset
            report = module.prepare(output, limit=args.limit, seed=args.seed, **kwargs)
            print(json.dumps(report, indent=2, default=str))
        elif args.command == "fetch-media":
            from .media import fetch_media
            report = fetch_media(args.dataset, max_bytes=int(args.max_mb * 1_000_000), limit=args.limit, media_root=args.media_root)
            print(json.dumps(report, indent=2))
            if report["selected_media_missing"]:
                return 1
        elif args.command == "validate":
            corpus, queries = _load_dataset(args.dataset)
            validate_dataset(corpus, queries)
            media = [row for row in corpus if row["modality"] != "text"]
            missing = [row["id"] for row in media if not row.get("path") or not Path(row["path"]).is_file()]
            print(json.dumps({"valid_schema": True, "corpus_size": len(corpus), "queries": len(queries), "modalities": dict(Counter(row["modality"] for row in corpus)), "query_splits": dict(Counter(row.get("split", "unspecified") for row in queries)), "retrieval_labelled_queries": sum(bool(row.get("relevant_ids")) for row in queries), "media_items": len(media), "media_missing": len(missing), "missing_media_example_ids": missing[:10], "native_media_ready": (not missing) if media else None, "coverage_note": "This reflects prepared corpus records only; consult download_manifest.json for pending streams."}, indent=2))
        elif args.command == "prepare-video-only":
            from .av import prepare_video_only_corpus
            print(json.dumps(prepare_video_only_corpus(args.dataset, args.output), indent=2))
        elif args.command in {"evaluate", "compare"}:
            corpus, queries = _load_dataset(args.dataset)
            manifest_path = args.dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            if "daily" in str(manifest.get("dataset", "")).lower() or (queries and all(q.get("options") for q in queries)):
                raise ValueError("Daily-Omni is supplied-video QA. Use score-choices; it has no corpus-retrieval relevance labels.")
            methods = args.methods if args.command == "compare" else [args.method]
            reports = []
            for method in methods:
                print(f"Building {method} index over {len(corpus)} source items...", file=sys.stderr)
                start = time.perf_counter()
                retriever = _retriever(corpus, args, method)
                if hasattr(retriever, "build_index"):
                    retriever.build_index()
                build_time = time.perf_counter() - start
                report = evaluate_retrieval(corpus, queries, retriever, k=args.top_k, method=method, split=args.split, config={"model": args.model if method != "bm25" else None, "dataset": str(args.dataset.resolve()), "dataset_manifest": manifest, "device": args.device if method != "bm25" else None, "dtype": args.dtype if method != "bm25" else None, "batch_size": args.batch_size if method != "bm25" else None})
                report["index_build_seconds"] = build_time
                if method != "bm25":
                    report["config"]["model_revision"] = args.revision or (DEFAULT_MODEL_REVISION if args.model == "google/embeddinggemma-2" else None)
                output = args.output / f"{method}.json" if args.command == "compare" else args.output
                write_json(output, report)
                reports.append(report)
                print(json.dumps({"method": method, "questions": report["query_count"], "metrics": report["metrics"], "report": str(output)}, indent=2))
            if args.command == "compare":
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "comparison.md").write_text(comparison_markdown(reports), encoding="utf-8")
        elif args.command in {"search", "ask"}:
            from .application import ask_application_question
            corpus, _ = _load_dataset(args.dataset)
            result = ask_application_question(corpus, args.question, _retriever(corpus, args, args.method),
                                  _generator(args) if args.command == "ask" else None,
                                  top_k=args.top_k, source_id=args.source_id)
            if getattr(args, "output", None):
                write_json(args.output, result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "evaluate-answers":
            from .rag import evaluate_answers
            corpus, queries = _load_dataset(args.dataset)
            supplied_media = bool(queries and all(row.get("options") for row in queries))
            retriever = None if supplied_media else _retriever(corpus, args, args.method)
            report = evaluate_answers(corpus, queries, retriever, _generator(args), args.output,
                                      k=args.top_k, split=args.split, limit=args.limit,
                                      method="supplied-media" if supplied_media else args.method,
                                      media_mode=args.media_mode)
            print(json.dumps(report, indent=2))
        elif args.command == "ingest":
            from .ingestion import ingest_paths
            report = ingest_paths(args.paths, args.output, chunk_seconds=args.chunk_seconds,
                                  max_duration_seconds=args.max_duration_seconds)
            print(json.dumps(report, indent=2))
        elif args.command == "serve":
            from .server import serve
            serve(root=Path.cwd(), port=args.port)
        elif args.command == "score-choices":
            queries = read_jsonl(args.dataset / "queries.jsonl")
            report = score_choices(queries, read_jsonl(args.predictions))
            write_json(args.output, report)
            print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
