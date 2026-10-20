"""Prepare a small, reproducible AMI/QMSum transcript retrieval collection.

Only source utterances enter the corpus. QMSum summaries and query annotations
remain in separate evaluation files. Media is recorded in a manifest, not fetched.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from projectmemory.io import download, write_json, write_jsonl


QMSUM_REVISION = "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e"
QMSUM_BASE = f"https://raw.githubusercontent.com/Yale-LILY/QMSum/{QMSUM_REVISION}/data/Product"
AMI_BASE = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
MEETINGS = tuple(
    [(f"ES2006{part}", "val", "dev") for part in "abcd"]
    + [(f"ES2011{part}", "test", "test") for part in "abcd"]
)
# Read-only HTTP HEAD checks on 2026-10-15: every WAV and Corner AVI returned 200.
VERIFIED_MEDIA_BYTES = {
    "ES2006a": (41_098_968, 43_821_724),
    "ES2006b": (69_860_056, 52_673_404),
    "ES2006c": (69_813_976, 78_040_206),
    "ES2006d": (62_958_296, 67_734_590),
    "ES2011a": (35_643_096, 35_354_166),
    "ES2011b": (50_600_664, 49_333_800),
    "ES2011c": (51_714_092, 56_573_386),
    "ES2011d": (63_434_456, 57_252_204),
}


def convert_meeting(
    payload: dict,
    meeting_id: str,
    split: str,
    chunk_utterances: int = 10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Map inclusive, zero-based gold utterance spans onto fixed source chunks.

    Chunks are created without looking at query annotations. A relevant chunk has
    at least one utterance in a gold span; this is derived retrieval supervision,
    not an official QMSum metric or audiovisual timestamp annotation.
    """
    if chunk_utterances < 1:
        raise ValueError("chunk_utterances must be positive")
    utterances = payload["meeting_transcripts"]
    if not utterances:
        raise ValueError(f"Meeting {meeting_id} has no transcript utterances")
    corpus = []
    for start in range(0, len(utterances), chunk_utterances):
        end = min(start + chunk_utterances, len(utterances)) - 1
        corpus.append(
            {
                "id": f"ami:{meeting_id}:utterances:{start:05d}-{end:05d}",
                "modality": "text",
                "text": "\n".join(
                    f"{row['speaker']}: {row['content']}"
                    for row in utterances[start : end + 1]
                ),
                "title": f"AMI {meeting_id}, utterances {start}-{end}",
                "source_id": meeting_id,
                "metadata": {
                    "dataset": "ami_qmsum",
                    "split": split,
                    "team_id": meeting_id[:-1],
                    "utterance_start": start,
                    "utterance_end": end,
                    "utterance_index_base": 0,
                    "span_end_inclusive": True,
                    "representation": "human_transcript",
                },
            }
        )

    queries = []
    for index, item in enumerate(payload.get("specific_query_list", [])):
        spans = []
        for span in item.get("relevant_text_span", []):
            if len(span) != 2:
                raise ValueError(f"Invalid support span in {meeting_id}: {span}")
            start, end = map(int, span)
            if not 0 <= start <= end < len(utterances):
                raise ValueError(f"Out-of-range support span in {meeting_id}: {span}")
            spans.append([start, end])
        if not spans:
            raise ValueError(f"Specific query {index} in {meeting_id} has no support spans")
        relevant_ids = [
            row["id"]
            for row in corpus
            if any(
                row["metadata"]["utterance_start"] <= end
                and row["metadata"]["utterance_end"] >= start
                for start, end in spans
            )
        ]
        queries.append(
            {
                "id": f"ami:{meeting_id}:specific:{index:03d}",
                "question": item["query"],
                "reference_answers": [item["answer"]],
                "relevant_ids": relevant_ids,
                "split": split,
                "metadata": {
                    "dataset": "ami_qmsum",
                    "meeting_id": meeting_id,
                    "team_id": meeting_id[:-1],
                    "candidate_source_ids": [meeting_id],
                    "gold_utterance_spans": spans,
                    "utterance_index_base": 0,
                    "span_end_inclusive": True,
                    "relevance": "chunk_overlaps_gold_utterance_span",
                    "gold_modalities": ["text"],
                },
            }
        )

    general_queries = [
        {
            "id": f"ami:{meeting_id}:general:{index:03d}",
            "question": item["query"],
            "reference_answers": [item["answer"]],
            "relevant_ids": [],
            "split": split,
            "metadata": {
                "dataset": "ami_qmsum",
                "meeting_id": meeting_id,
                "candidate_source_ids": [meeting_id],
                "retrieval_evaluable": False,
                "reason": "Whole-meeting summary has no annotated evidence spans",
            },
        }
        for index, item in enumerate(payload.get("general_query_list", []))
    ]
    return corpus, queries, general_queries


def media_manifest() -> dict:
    """Official media locations; preparation does not download or index media."""
    entries = []
    for meeting_id, _, split in MEETINGS:
        root = f"{AMI_BASE}/{meeting_id}"
        audio_bytes, video_bytes = VERIFIED_MEDIA_BYTES[meeting_id]
        has_whiteboard_image = meeting_id.startswith("ES2006") or meeting_id == "ES2011a"
        entries.append(
            {
                "meeting_id": meeting_id,
                "split": split,
                "audio": {
                    "url": f"{root}/audio/{meeting_id}.Mix-Headset.wav",
                    "format": "wav",
                    "stream": "mixed_headsets",
                    "bytes": audio_bytes,
                    "verified_http_status": 200,
                },
                "video": {
                    "url": f"{root}/video/{meeting_id}.Corner.avi",
                    "format": "avi",
                    "stream": "corner_camera_compressed",
                    "bytes": video_bytes,
                    "verified_http_status": 200,
                },
                "image_directories": {
                    "slides": f"{root}/slides/",
                    "whiteboard": f"{root}/whiteboard/",
                },
                "whiteboard_preview": (
                    f"{root}/whiteboard/{meeting_id}.strokes.avi.jpg"
                    if has_whiteboard_image else None
                ),
                "whiteboard_status": (
                    "strokes video and preview listed" if has_whiteboard_image
                    else "NoUseOfWhiteboard.txt listed; no preview image"
                ),
                "document_directory": (
                    f"{root}/shared-doc/" if meeting_id.startswith("ES2006") else None
                ),
                "document_status": (
                    "shared document files listed" if meeting_id.startswith("ES2006")
                    else "shared-doc directory returned HTTP 404"
                ),
                "local_media_downloaded": False,
            }
        )
    return {
        "dataset": "AMI Meeting Corpus",
        "verified_date": "2026-10-15",
        "verification": "HTTP HEAD for audio/video; GET directory/browser listings for other media",
        "audio_video_total_bytes": sum(sum(sizes) for sizes in VERIFIED_MEDIA_BYTES.values()),
        "license": "CC-BY-4.0 (signals and transcription; some annotations)",
        "license_url": "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml",
        "download_page": "https://groups.inf.ed.ac.uk/ami/download/",
        "availability_page": "https://groups.inf.ed.ac.uk/ami/corpus/datapresent.shtml",
        "alignment_note": (
            "QMSum spans are utterance indices, not media timestamps. Align to AMI "
            "timed annotations before claiming audio/video evidence timestamps."
        ),
        "annotation_archive": (
            "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
            "ami_public_manual_1.6.2.zip"
        ),
        "meetings": entries,
    }


def prepare(output_dir: Path, limit: int = 0, seed: int = 42, **kwargs) -> dict:
    """Download eight small JSONs and write source chunks and evaluation queries.

    ``limit`` caps specific evaluation queries, using a seeded sample. It never
    shrinks the source corpus according to gold evidence. Zero includes all queries.
    ``chunk_utterances`` defaults to ten. No embedding/answer model is needed.
    """
    if limit < 0:
        raise ValueError("limit cannot be negative")
    chunk_utterances = int(kwargs.get("chunk_utterances", 10))
    if chunk_utterances < 1:
        raise ValueError("chunk_utterances must be positive")
    output_dir = Path(output_dir)
    raw_dir = Path(kwargs.get("raw_dir") or output_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus, queries, general_queries, sources = [], [], [], []
    for meeting_id, upstream_split, split in MEETINGS:
        url = f"{QMSUM_BASE}/{upstream_split}/{meeting_id}.json"
        raw_path = raw_dir / f"{meeting_id}.json"
        download(url, raw_path, max_bytes=5_000_000)
        source_bytes = raw_path.read_bytes()
        payload = json.loads(source_bytes)
        rows, meeting_queries, meeting_general = convert_meeting(
            payload, meeting_id, split, chunk_utterances
        )
        corpus.extend(rows)
        queries.extend(meeting_queries)
        general_queries.extend(meeting_general)
        sources.append(
            {
                "meeting_id": meeting_id,
                "url": url,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
                "upstream_split": upstream_split,
                "split": split,
            }
        )

    available_queries = len(queries)
    if limit and limit < len(queries):
        indices = sorted(random.Random(seed).sample(range(len(queries)), limit))
        queries = [queries[index] for index in indices]

    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "generation_queries.jsonl", general_queries)
    write_json(output_dir / "download_manifest.json", media_manifest())
    manifest = {
        "dataset": "ami_qmsum",
        "qmsum_revision": QMSUM_REVISION,
        "seed": seed,
        "limit": limit,
        "chunk_utterances": chunk_utterances,
        "corpus_records": len(corpus),
        "queries": len(queries),
        "available_specific_queries": available_queries,
        "general_queries_excluded_from_retrieval": len(general_queries),
        "query_splits": dict(Counter(row["split"] for row in queries)),
        "sources": sources,
        "modalities_prepared": ["text"],
        "evaluation_scope": "transcript_retrieval_with_gold_utterance_spans",
        "candidate_scope": "Within the meeting named by metadata.candidate_source_ids",
        "media_status": "Manifest only; media not downloaded or indexed",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
