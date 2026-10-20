"""Prepare a 20-second, unlabelled application demo from one original AV file.

This command performs local media ingestion only: no downloads, inference, GPU
work, or API requests. The deliberate prefix is unsuitable for benchmark scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.av import probe_media
from projectmemory.ingestion import ingest_paths
from projectmemory.io import fingerprint, read_jsonl, write_json


VIDEO_ID = "-jR3C_yA_G0"
DEMO_SECONDS = 20
DEMO_QUESTION = "Describe the visible scene and the spoken explanation."


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def prepare(source_dir: Path, output_dir: Path) -> dict:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or source_dir.is_relative_to(output_dir):
        raise ValueError("The demo output must not replace the source dataset or its ancestor")
    corpus_path = source_dir / "corpus.jsonl"
    matches = [row for row in read_jsonl(corpus_path)
               if row.get("id") == f"daily_omni:video:{VIDEO_ID}"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one prepared Daily-Omni video record for {VIDEO_ID}")
    original = matches[0].get("metadata", {}).get("original_path")
    if not original or not Path(original).is_absolute():
        raise ValueError("Daily-Omni original_path must identify the original video with its soundtrack")
    source = Path(original).resolve(strict=True)
    source_identity = identity(source)
    expected_hash = matches[0].get("metadata", {}).get("original_sha256")
    if expected_hash and expected_hash != source_identity["sha256"]:
        raise ValueError("Original video no longer matches its prepared SHA-256")
    info = probe_media(source)
    if not info["video_streams"] or not info["audio_streams"]:
        raise ValueError("The original demo file must contain both video and audio")
    if info["duration_seconds"] < DEMO_SECONDS:
        raise ValueError("The demo requires at least 20 seconds of original media")
    source_id = f"file:{source_identity['sha256']}"
    existing = output_dir / "corpus.jsonl"
    if existing.is_file() and any(row.get("source_id") != source_id for row in read_jsonl(existing)):
        raise ValueError("The demo directory contains unrelated evidence; choose a separate output")
    queries = output_dir / "queries.jsonl"
    if queries.is_file() and read_jsonl(queries):
        raise ValueError("The demo output must not contain benchmark questions or answer labels")

    ingestion = ingest_paths([source], output_dir, chunk_seconds=DEMO_SECONDS,
                             max_duration_seconds=DEMO_SECONDS)
    corpus = read_jsonl(output_dir / "corpus.jsonl")
    if len(corpus) != 2 or {row["modality"] for row in corpus} != {"audio", "video"}:
        raise RuntimeError("Expected exactly one audio and one silent video demo segment")
    if {row["source_id"] for row in corpus} != {source_id}:
        raise RuntimeError("Demo audio and video must share their original source ID")
    for row in corpus:
        metadata = row.get("metadata", {})
        if metadata.get("start_seconds") != 0 or metadata.get("end_seconds") != DEMO_SECONDS:
            raise RuntimeError("Demo audio and video must cover the same original 0–20 second interval")
        if metadata.get("max_duration_seconds") != DEMO_SECONDS:
            raise RuntimeError("The deliberate demo prefix must be recorded in every segment")
        if row.get("text") or "answer_label" in row or "reference_answers" in row:
            raise RuntimeError("Demo evidence must contain native media without answer labels")
        clip = probe_media(Path(row["path"]))
        if row["modality"] == "video" and (len(clip["video_streams"]) != 1 or clip["audio_streams"]):
            raise RuntimeError("Demo video must have one video stream and no audio")
        if row["modality"] == "audio" and (len(clip["audio_streams"]) != 1 or clip["video_streams"]):
            raise RuntimeError("Demo audio must have one audio stream and no video")
        if abs(clip["duration_seconds"] - DEMO_SECONDS) > .1:
            raise RuntimeError("Encoded demo segment differs from 20 seconds by more than one video frame")
    if identity(source) != source_identity:
        raise RuntimeError("Original demo video changed during preparation")
    manifest = {
        **ingestion, "dataset": "demo_av_20_second_prefix", "demo_only": True,
        "protocol": "unlabelled_application_demo_not_benchmark",
        "source_video_id": VIDEO_ID, "source": source_identity,
        "source_corpus": identity(corpus_path), "source_duration_seconds": info["duration_seconds"],
        "preparation_script": identity(Path(__file__)),
        "interval": {"start_seconds": 0, "end_seconds": DEMO_SECONDS},
        "max_duration_seconds": DEMO_SECONDS, "shared_source_id": source_id,
        "corpus_records_fingerprint": fingerprint(corpus),
        "assets": [{"id": row["id"], "modality": row["modality"],
                    **identity(Path(row["path"]))} for row in corpus],
        "demo_question": DEMO_QUESTION, "observed_generated_answer": None,
        "inference_performed": False, "queries": 0,
        "notes": ingestion["notes"] + [
            "Only the first 20 seconds of one original video with its soundtrack are ingested.",
            "Audio and silent video share the original source hash and 0–20 second interval.",
            "This deliberately cropped prefix is an application demo, not Daily-Omni evaluation data.",
            "No official question, answer, transcript, or generated description enters the demo corpus.",
            "Top 3 can return only the two available evidence records in this collection.",
            "Media preparation does not establish retrieval or generated-answer correctness.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/prepared/daily-omni")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared/demo-av")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "corpus_records": manifest["corpus_records"],
                      "interval": manifest["interval"], "shared_source_id": manifest["shared_source_id"],
                      "demo_question": manifest["demo_question"], "inference_performed": False}, indent=2))


if __name__ == "__main__":
    main()
