"""Daily-Omni known-video QA preparation, deliberately not a retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from projectmemory.io import download, write_json, write_jsonl


QA_URL = "https://huggingface.co/datasets/liarliar/Daily-Omni/resolve/main/qa.json"
VIDEO_URL = "https://huggingface.co/datasets/liarliar/Daily-Omni/resolve/main/Videos.tar"
SOURCE_URL = "https://github.com/Lliar-liar/Daily-Omni"
MAX_ANNOTATION_BYTES = 150_000_000


def parse_question(row: dict) -> dict:
    """Read the public schema and retain gold choices outside all media records."""
    question = row["Question"].strip()
    options = row["Choice"]
    answer = row["Answer"].strip().upper()
    video_id = str(row["video_id"])
    if not question or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("Daily-Omni question or video_id is invalid")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("Daily-Omni requires four labelled answer options")
    cleaned_options = []
    for expected, option in zip("ABCD", options):
        match = re.fullmatch(r"([A-D])[.)]\s*(.+)", str(option).strip(), flags=re.DOTALL)
        if not match or match.group(1) != expected:
            raise ValueError("Daily-Omni choices must be labelled A, B, C, D in order")
        cleaned_options.append(str(option).strip())
    if answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"Invalid Daily-Omni answer label: {answer}")
    identity = json.dumps([video_id, question, cleaned_options], ensure_ascii=False)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return {
        "id": f"daily_omni:{digest}", "question": question,
        "options": cleaned_options, "answer_label": answer,
        "reference_answers": [answer],
        # There are no collection-retrieval relevance annotations in Daily-Omni.
        "relevant_ids": [], "split": "benchmark",
        "metadata": {
            "dataset": "daily_omni", "video_id": video_id,
            "question_type": row.get("Type", "unknown"),
            "video_duration": row.get("video_duration", "unknown"),
            "content_parent_category": row.get("content_parent_category", "unknown"),
            "content_fine_category": row.get("content_fine_category", "unknown"),
            "video_category": row.get("video_category", "unknown"),
            "required_modalities": ["audio", "video"],
            "evaluation_protocol": "known_video_qa",
            "known_media_ids": [f"daily_omni:video:{video_id}", f"daily_omni:audio:{video_id}"],
            "answer_semantics": "multiple_choice",
        },
    }


def sample_questions(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    # The official 1,197-row file contains one exact duplicate. Keep its original
    # benchmark weighting, but give each occurrence a unique prediction key.
    seen: dict[str, dict] = {}
    occurrences: Counter = Counter()
    parsed = []
    for index, row in enumerate(rows):
        query = parse_question(row)
        identity = query["id"]
        if identity in seen and seen[identity] != query:
            raise ValueError("Conflicting Daily-Omni labels for the same question and choices")
        seen[identity] = query
        occurrences[identity] += 1
        query = dict(query, metadata=dict(query["metadata"]))
        query["metadata"]["source_row_index"] = index
        query["metadata"]["duplicate_occurrence"] = occurrences[identity]
        if occurrences[identity] > 1:
            query["id"] = f"{identity}:duplicate:{occurrences[identity]}"
        parsed.append(query)
    parsed.sort(key=lambda row: row["id"])
    return sorted(random.Random(seed).sample(parsed, min(limit, len(parsed))), key=lambda row: row["id"])


def media_records(video_id: str, media_dir: Path) -> list[dict]:
    """Use official paths: Videos/<id>/<id>_video.mp4 and <id>_audio.wav."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("Invalid video_id")
    root = media_dir.resolve()
    records = []
    for modality, suffix in (("video", "_video.mp4"), ("audio", "_audio.wav")):
        path = root / video_id / f"{video_id}{suffix}"
        records.append({
            "id": f"daily_omni:{modality}:{video_id}", "modality": modality,
            "text": "", "title": f"Daily-Omni clip {video_id}",
            "path": str(path), "source_id": video_id,
            "metadata": {"dataset": "daily_omni", "media_available": path.is_file(),
                         "representation": "native_media" if path.is_file() else "missing_media",
                         "archive_member": f"Videos/{video_id}/{video_id}{suffix}",
                         "evaluation_protocol": "known_video_qa"},
        })
    return records


def require_media(queries: list[dict], corpus: list[dict], mode: str = "combined") -> None:
    """Fail before evaluating any missing native media; do not silently drop it."""
    kinds = {"audio": {"audio"}, "video": {"video"}, "combined": {"audio", "video"}}
    if mode not in kinds:
        raise ValueError("mode must be audio, video, or combined")
    by_id = {row["id"]: row for row in corpus}
    missing = []
    for query in queries:
        for kind in kinds[mode]:
            identity = f"daily_omni:{kind}:{query['metadata']['video_id']}"
            record = by_id.get(identity)
            if not record or not record.get("path") or not Path(record["path"]).is_file():
                missing.append(identity)
    if missing:
        raise FileNotFoundError(f"Daily-Omni {mode} evaluation needs {len(set(missing))} missing media files. "
                                "See download_manifest.json; audio-only WAVs may need extraction from MP4s.")


def prepare(output_dir: Path, limit: int = 100, seed: int = 42, **kwargs) -> dict:
    """Download only QA annotations. raw_dir and media_dir may be overridden."""
    output_dir = Path(output_dir)
    raw_dir = Path(kwargs.get("raw_dir") or "data/raw/daily_omni")
    media_dir = Path(kwargs.get("media_dir") or "data/media/daily_omni/Videos").resolve()
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    qa_path = raw_dir / "qa.json"
    if not qa_path.is_file():
        download(QA_URL, qa_path, max_bytes=MAX_ANNOTATION_BYTES)
    if qa_path.stat().st_size > MAX_ANNOTATION_BYTES:
        raise ValueError("Cached Daily-Omni annotations exceed size limit")
    raw = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected the official Daily-Omni list of QA annotations")
    queries = sample_questions(raw, limit, seed)
    videos = sorted({row["metadata"]["video_id"] for row in queries})
    corpus = [record for video_id in videos for record in media_records(video_id, media_dir)]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    missing = Counter(row["modality"] for row in corpus if not row["metadata"]["media_available"])
    manifest = {
        "dataset": "daily_omni", "source": SOURCE_URL,
        "seed": seed, "requested_limit": limit, "queries": len(queries),
        "available_questions": len(raw), "unique_videos": len(videos),
        "exact_duplicate_rows_in_source": len(raw) - len({json.dumps(row, sort_keys=True) for row in raw}),
        "corpus_records": len(corpus), "protocol": "known_video_qa",
        "missing_media_by_modality": dict(missing), "native_multimodal_ready": not missing,
        "question_type_counts": dict(Counter(row["metadata"]["question_type"] for row in queries)),
        "selected_ids": [row["id"] for row in queries],
        "sources": [{"url": QA_URL, "path": str(qa_path.resolve()),
                     "sha256": hashlib.sha256(qa_path.read_bytes()).hexdigest(),
                     "bytes": qa_path.stat().st_size}],
        "license": "CC-BY-NC-SA-4.0 (dataset card)",
        "notes": [
            "Known-video multiple-choice QA; do not report collection-retrieval Recall or MRR.",
            "The video is provided to the answering model under the original protocol.",
            "Audio-only, visual-only, and combined modes must use the same selected queries.",
            "Questions and answer choices are never indexed as source evidence.",
            "Exact duplicate source rows retain benchmark weighting with distinct occurrence IDs.",
            "Visual-only evaluation must remove the audio track; audio-only must exclude video frames.",
            "Native media is pending until the archive is acquired; missing media must fail evaluation.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "download_manifest.json", {
        "dataset": "daily_omni", "automatically_downloaded": False,
        "archives": [{"url": VIDEO_URL, "approximate_bytes": 3_910_000_000,
                      "destination": str(media_dir.parent), "status": "manual_large_download",
                      "note": "Official archive contains Videos/. Audio WAVs can be extracted from MP4s if absent."}],
        "required_media": [{"id": row["id"], "path": row["path"],
                            "archive_member": row["metadata"]["archive_member"],
                            "available": row["metadata"]["media_available"]} for row in corpus],
    })
    return manifest
