"""Run/resume the fixed local MMQA and normalized Daily-Omni comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from projectmemory.io import read_jsonl, write_json, fingerprint
from projectmemory.local_generation import LocalGemmaGenerator
from projectmemory.localruntime import stop
from projectmemory.rag import evaluate_answers
from projectmemory.evaluation_policy import FormatFailureRecorder, summarize_evaluation_failures
from evaluate_saved_rankings import run as replay


def run_av(mode, output_root):
    dataset = ROOT / "data/prepared/daily-omni-local-normalized"
    output = output_root / f"final-local-av-{mode}"
    source_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    corpus = read_jsonl(dataset / "corpus.jsonl")
    queries = read_jsonl(dataset / "queries.jsonl")
    if len(queries) != 20 or not all(q.get("options") for q in queries):
        raise ValueError("Final AV protocol requires the fixed 20-question supplied-media subset")

    class ProtocolGenerator(FormatFailureRecorder):
        def evaluation_config(self):
            return {**super().evaluation_config(), "subset_manifest_fingerprint": fingerprint(source_manifest),
                    "experiment_runner_fingerprint": fingerprint(Path(__file__).read_text(encoding="utf-8"))}

    generator = ProtocolGenerator(LocalGemmaGenerator(max_output_tokens=4096))
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source_manifest.json"
    if source_path.exists() and json.loads(source_path.read_text(encoding="utf-8")) != source_manifest:
        raise ValueError("Saved AV subset provenance differs; choose a new experiment output")
    write_json(source_path, source_manifest)
    report = evaluate_answers(corpus, queries, None, generator, output, k=5, limit=20,
                              method="supplied-media", media_mode=mode)
    return summarize_evaluation_failures(report, output)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "mmqa", "av"], default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results",
                        help="Use a new directory for reproduction with changed code or dataset paths")
    parser.add_argument("--retrieval-root", type=Path, default=ROOT / "results/mmqa-comparison",
                        help="Directory holding matching fresh or original BM25/dense/native retrieval reports")
    args = parser.parse_args()
    try:
        if args.suite in {"all", "mmqa"}:
            for method in ("bm25", "dense-text", "multimodal"):
                print(f"Starting MMQA / {method}", flush=True)
                report = replay(ROOT / "data/prepared/mmqa", args.retrieval_root / f"{method}.json",
                                args.output_root / f"final-mmqa-local-{method}", limit=100,
                                record_format_failures=True)
                print(json.dumps({"condition": method, "completed": report["completed"], "metrics": report["metrics"],
                                  "failures": report.get("evaluation_failures")}), flush=True)
        if args.suite in {"all", "av"}:
            for mode in ("audio", "video", "combined"):
                print(f"Starting normalized Daily-Omni / {mode}", flush=True)
                report = run_av(mode, args.output_root)
                print(json.dumps({"condition": mode, "completed": report["completed"], "metrics": report["metrics"],
                                  "failures": report.get("evaluation_failures")}), flush=True)
    finally:
        stop(root=ROOT)


if __name__ == "__main__":
    main()
