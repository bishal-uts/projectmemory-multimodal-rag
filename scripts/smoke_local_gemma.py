"""Exercise local generation on public evidence; not a QA accuracy benchmark."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.io import read_jsonl, write_json
from projectmemory.av import segment_media
from projectmemory.local_generation import LocalGemmaGenerator
from projectmemory.localruntime import stop
from projectmemory.retrieval import BM25Retriever
from projectmemory.rag import ask_question


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = ROOT / "results/local-gemma-smoke.json"
    corpus = read_jsonl(ROOT / "data/prepared/ami/corpus.jsonl")
    mmqa = read_jsonl(ROOT / "data/prepared/mmqa/corpus.jsonl")
    image = next(row for row in mmqa if row["modality"] == "image" and Path(row["path"]).is_file())
    image = {**image, "title": "Image evidence", "text": ""}  # Do not let a title answer the visual question.
    original = read_jsonl(ROOT / "results/av-embedding-smoke.corpus.jsonl")
    media = []
    for row in original:
        segments = segment_media(Path(row["metadata"]["source_file"]), ROOT / "data/media/local-gemma-smoke-segments",
                                 chunk_seconds=20, max_duration_seconds=20)
        media.extend(item for item in segments if item["modality"] == row["modality"])
    audio = [row for row in media if row["modality"] == "audio"]
    video = [row for row in media if row["modality"] == "video"]
    generator = LocalGemmaGenerator(max_output_tokens=1024)
    retriever = BM25Retriever(corpus)
    cases = [
        ("text", "What features should the remote control have?", None),
        ("image", "Briefly describe what is visibly shown in the image.", [image]),
        ("audio", "Transcribe the speech in the attached audio evidence.", audio),
        ("video", "Briefly describe what is visible in the supplied video frames.", video),
        ("audio_video", "Describe the visible scene and the words or sounds heard in these synchronized twenty-second clips.", video + audio),
    ]
    report = {"task": "local_generation_runtime_smoke", "created_at": datetime.now(timezone.utc).isoformat(),
              "model": generator.model, "configuration": generator.evaluation_config(), "cases": [],
              "limitations": ["Integration checks only; no answer accuracy is inferred.",
                              "Audio/video cases use a twenty-second prefix, not a full benchmark clip.",
                              "Audio check uses an explicit transcription prompt; earlier generic audio questions sometimes failed validation or claimed missing audio.",
                              "Video is processed as timestamped frames at one frame per second."]}
    try:
        for name, question, evidence in cases:
            started = time.perf_counter()
            try:
                if evidence is None:
                    response = ask_question(corpus, question, retriever, generator, top_k=3, source_id="ES2006a")
                    evidence, answer = response["evidence"], response["generated_answer"]
                else:
                    answer = generator.answer(question, evidence)
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                row = {"condition": name, "completed": True, "question": question,
                       "evidence_ids": [item["id"] for item in evidence], "result": answer,
                       "elapsed_seconds": time.perf_counter() - started, "gpu_after_request": gpu}
                print(f"{name}: {answer['answer']} ({row['elapsed_seconds']:.1f}s)", flush=True)
            except Exception as exc:
                row = {"condition": name, "completed": False, "error_type": type(exc).__name__,
                       "error": str(exc), "elapsed_seconds": time.perf_counter() - started}
                print(f"{name}: {type(exc).__name__}: {exc}", flush=True)
            report["cases"].append(row)
            write_json(output, report)
    finally:
        stop()
    report["completed"] = sum(row["completed"] for row in report["cases"])
    report["passed"] = report["completed"] == len(cases)
    write_json(output, report)
    print(f"Completed {report['completed']}/{len(cases)} local runtime checks. Report: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
