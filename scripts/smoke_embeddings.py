"""Verify native text/image encoding on real prepared sources; not a benchmark."""
from pathlib import Path
import argparse
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from projectmemory.cli import DEFAULT_MODEL_REVISION
from projectmemory.embeddings import EmbeddingRetriever
from projectmemory.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/prepared/mmqa"))
    parser.add_argument("--output", type=Path, default=Path("results/embedding-smoke.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    corpus = read_jsonl(args.dataset / "corpus.jsonl")
    texts = [row for row in corpus if row["modality"] == "text"][:2]
    images = [row for row in corpus if row["modality"] == "image"
              and row.get("path") and Path(row["path"]).is_file()][:2]
    if len(texts) != 2 or len(images) != 2:
        raise SystemExit("Prepare the dataset and fetch at least two images first.")
    started = time.perf_counter()
    retriever = EmbeddingRetriever(texts + images, mode="multimodal", device=args.device,
                                   revision=DEFAULT_MODEL_REVISION, batch_size=1,
                                   cache_dir=Path(".cache/embeddings"))
    retriever.build_index()
    question = read_jsonl(args.dataset / "queries.jsonl")[0]["question"]
    hits = retriever.search(question, top_k=4)
    if len(hits) != 4:
        raise RuntimeError("Expected a ranking containing all four smoke-test sources")
    report = {"task": "native_text_image_runtime_smoke", "passed": True,
              "model": retriever.model_name, "revision": DEFAULT_MODEL_REVISION,
              "device": args.device, "dtype": retriever.dtype,
              "source_ids": [row["id"] for row in texts + images],
              "modalities_tested": ["text", "image"], "question": question,
              "hits": hits, "elapsed_seconds": time.perf_counter() - started,
              "limitation": "Runtime validation only. This is not a retrieval quality benchmark; audio and video were not tested."}
    write_json(args.output, report)
    print(f"Native text/image encoding passed. Report: {args.output}")


if __name__ == "__main__":
    main()
