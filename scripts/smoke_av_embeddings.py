"""Test native audio/video embeddings on a real Daily-Omni prefix, not accuracy."""
from pathlib import Path
import argparse
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.av import segment_media, configure_media_runtime
from projectmemory.embeddings import EmbeddingRetriever
from projectmemory.io import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/prepared/daily-omni")
    parser.add_argument("--output", type=Path, default=ROOT / "results/av-embedding-smoke.json")
    parser.add_argument("--media-output", type=Path, default=ROOT / "data/media/av-smoke-segments")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.seconds <= 5:
        parser.error("--seconds must be greater than zero and at most five for this bounded smoke test")
    if args.threads < 1:
        parser.error("--threads must be positive")
    # The model was downloaded previously. Avoid touching user-wide caches or
    # fetching new revisions during a capability check.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    configure_media_runtime()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    corpus = read_jsonl(args.dataset / "corpus.jsonl")
    video = next((r for r in corpus if r["modality"] == "video" and Path(r["path"]).is_file()), None)
    if video is None:
        raise SystemExit("Fetch the Daily-Omni media first.")
    audio = next((r for r in corpus if r["source_id"] == video["source_id"] and r["modality"] == "audio"), None)
    if audio is None or not Path(audio["path"]).is_file():
        raise SystemExit("The selected clip needs its real separate audio file.")
    records = []
    started = time.perf_counter()
    for original in [video, audio]:
        for row in segment_media(Path(original["path"]), args.media_output,
                                 chunk_seconds=args.seconds, max_duration_seconds=args.seconds):
            if row["modality"] == original["modality"]:
                row["metadata"]["dataset"] = "daily_omni"
                row["metadata"]["benchmark_source_id"] = original["source_id"]
                records.append(row)
    if len(records) != 2:
        raise RuntimeError("Expected one real audio segment and one real silent video segment")
    revision = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    retriever = EmbeddingRetriever(
        records, mode="multimodal", device=args.device, revision=revision,
        batch_size=1, cache_dir=ROOT / ".cache/embeddings",
        processing_kwargs={"video": {"fps": 1, "max_frames": 3,
                           "overflow_strategy": "uniform", "add_timestamps": True}},
    )
    retriever.build_index()
    question = next(q["question"] for q in read_jsonl(args.dataset / "queries.jsonl")
                    if q["metadata"]["video_id"] == video["source_id"])
    hits = retriever.search(question, top_k=2)
    if len(hits) != 2:
        raise RuntimeError("Native audio/video smoke did not return both evidence records")
    report = {
        "task": "native_audio_video_runtime_smoke", "passed": True,
        "model": retriever.model_name, "revision": revision,
        "device": args.device, "dtype": retriever.dtype, "torch_version": torch.__version__,
        "threads": args.threads, "source_dataset": "daily_omni", "source_id": video["source_id"],
        "modalities_tested": ["audio", "video"], "prefix_seconds": args.seconds,
        "question": question, "hits": hits, "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "limitation": "Runtime smoke only, not a retrieval or QA quality benchmark. Only the first few seconds of one real clip were encoded, with at most three sampled visual frames. No generated answer or relevance label is inferred from this test.",
    }
    write_json(args.output, report)
    write_jsonl(args.output.with_suffix(".corpus.jsonl"), records)
    print(f"Native audio/video runtime smoke passed. Report: {args.output}")


if __name__ == "__main__":
    main()
