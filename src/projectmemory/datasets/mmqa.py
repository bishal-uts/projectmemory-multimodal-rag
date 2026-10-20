"""Prepare official MultiModalQA annotations without leaking QA into the corpus.

This adapter uses the union of every supplied candidate context for the sampled
questions, including distractors. It does not claim full-corpus or leaderboard
evaluation. Image metadata is available immediately; image pixels are a separate
download and are never replaced with invented captions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from projectmemory.io import download, write_json, write_jsonl


BASE_URL = "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset/"
IMAGE_ARCHIVE_URL = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"
SOURCE_URL = "https://github.com/allenai/multimodalqa"
FILES = {
    "dev": "MMQA_dev.jsonl.gz",
    "text": "MMQA_texts.jsonl.gz",
    "table": "MMQA_tables.jsonl.gz",
    "image": "MMQA_images.jsonl.gz",
}
MAX_ANNOTATION_BYTES = 150_000_000


def _read_rows(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sample_questions(rows: Iterable[dict], limit: int, seed: int) -> list[dict]:
    """Sample stably, independent of source row order; never select by answer."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    ordered = sorted(rows, key=lambda row: row["qid"])
    if len({row["qid"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate MultiModalQA question IDs")
    selected = random.Random(seed).sample(ordered, min(limit, len(ordered)))
    return sorted(selected, key=lambda row: row["qid"])


def context_id(modality: str, doc_id: str) -> str:
    if modality not in {"text", "table", "image"}:
        raise ValueError(f"Unknown MultiModalQA context modality: {modality}")
    return f"mmqa:{modality}:{doc_id}"


def candidate_contexts(question: dict) -> set[tuple[str, str]]:
    """Return full supplied context, not just the supporting (gold) sources."""
    metadata = question["metadata"]
    contexts = {("text", str(value)) for value in metadata.get("text_doc_ids", [])}
    contexts.update(("image", str(value)) for value in metadata.get("image_doc_ids", []))
    if metadata.get("table_id"):
        contexts.add(("table", str(metadata["table_id"])))
    supporting = {
        (item["doc_part"], str(item["doc_id"]))
        for item in question.get("supporting_context", [])
    }
    if not supporting.issubset(contexts):
        raise ValueError(f"Supporting sources absent from supplied context: {question['qid']}")
    return contexts


def parse_question(row: dict) -> dict:
    """Keep labels and support IDs exclusively in the query/evaluation file."""
    if not row.get("answers") or not row.get("supporting_context"):
        raise ValueError(f"Question {row.get('qid')} lacks public evaluation labels")
    answers = [str(answer["answer"] if isinstance(answer, dict) else answer)
               for answer in row["answers"]]
    metadata = row["metadata"]
    contexts = candidate_contexts(row)
    return {
        "id": f"mmqa:{row['qid']}",
        "question": row["question"],
        "reference_answers": answers,
        "relevant_ids": sorted({context_id(item["doc_part"], str(item["doc_id"]))
                                for item in row["supporting_context"]}),
        "split": "dev",
        "metadata": {
            "dataset": "multimodalqa",
            "original_id": row["qid"],
            "question_type": metadata.get("type", "unknown"),
            "required_modalities": metadata.get("modalities", []),
            "candidate_ids": sorted(context_id(kind, doc_id) for kind, doc_id in contexts),
            "answer_semantics": "unordered_list" if len(answers) > 1 else "single",
            "evaluation_protocol": "sampled_dev_candidate_union",
        },
    }


def serialize_table(table: dict) -> str:
    """Preserve all header/cell values; no answer-dependent row filtering."""
    header = [str(column.get("column_name", "")) for column in table.get("header", [])]
    lines = [str(table.get("table_name", "")), " | ".join(header)]
    for row in table.get("table_rows", []):
        lines.append(" | ".join(str(cell.get("text", "")) if isinstance(cell, dict)
                                else str(cell) for cell in row))
    return "\n".join(line for line in lines if line)


def parse_context(row: dict, original_modality: str, media_dir: Path) -> dict:
    doc_id = str(row["id"])
    metadata = {
        "dataset": "multimodalqa",
        "original_id": doc_id,
        "original_modality": original_modality,
        "url": row.get("url", ""),
    }
    result = {
        "id": context_id(original_modality, doc_id),
        "modality": "image" if original_modality == "image" else "text",
        "title": row.get("title", ""),
        "source_id": doc_id,
        "metadata": metadata,
    }
    if original_modality == "table":
        result["text"] = serialize_table(row["table"])
        metadata["representation"] = "serialized_table"
    elif original_modality == "text":
        result["text"] = row["text"]
        metadata["representation"] = "source_text"
    else:
        # The official image annotations provide a title, not OCR or a caption.
        result["text"] = ""
        image_root = media_dir.resolve()
        image_path = (image_root / str(row["path"])).resolve()
        if not image_path.is_relative_to(image_root):
            raise ValueError(f"Image path leaves the configured media directory: {row['path']}")
        result["path"] = str(image_path)
        metadata["media_available"] = image_path.is_file()
        metadata["representation"] = "native_image" if image_path.is_file() else "metadata_only"
        metadata["archive_member"] = str(row["path"])
    return result


def prepare(output_dir: Path, limit: int = 100, seed: int = 42, **kwargs) -> dict:
    """Download bounded annotations, cache them, and prepare reproducible files.

    Keyword options: raw_dir, media_dir. The latter points directly at the
    extracted images directory. No image archive or model weights are fetched.
    """
    output_dir = Path(output_dir)
    raw_dir = Path(kwargs.get("raw_dir") or "data/raw/mmqa")
    media_dir = Path(kwargs.get("media_dir") or "data/media/mmqa/images").resolve()
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    sources = []
    for filename in FILES.values():
        path = raw_dir / filename
        if not path.is_file():
            download(BASE_URL + filename, path, max_bytes=MAX_ANNOTATION_BYTES)
        if path.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError(f"Cached annotation exceeds size limit: {path}")
        sources.append({"url": BASE_URL + filename, "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    all_questions = list(_read_rows(raw_dir / FILES["dev"]))
    selected = sample_questions(all_questions, limit, seed)
    queries = [parse_question(row) for row in selected]
    wanted: dict[str, set[str]] = {kind: set() for kind in ("text", "table", "image")}
    for row in selected:
        for kind, doc_id in candidate_contexts(row):
            wanted[kind].add(doc_id)
    corpus = []
    for kind, ids in wanted.items():
        found = set()
        for row in _read_rows(raw_dir / FILES[kind]):
            if str(row["id"]) in ids:
                if str(row["id"]) in found:
                    raise ValueError(f"Duplicate context ID in {kind}: {row['id']}")
                corpus.append(parse_context(row, kind, media_dir))
                found.add(str(row["id"]))
        if ids - found:
            raise ValueError(f"Missing {kind} source annotations: {sorted(ids - found)[:5]}")
    corpus.sort(key=lambda row: row["id"])
    image_rows = [row for row in corpus if row["modality"] == "image"]
    missing_images = [row for row in image_rows if not row["metadata"]["media_available"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "queries.jsonl", queries)
    manifest = {
        "dataset": "multimodalqa", "source": SOURCE_URL,
        "seed": seed, "requested_limit": limit, "queries": len(queries),
        "available_dev_questions": len(all_questions), "corpus_records": len(corpus),
        "counts_by_original_modality": dict(Counter(row["metadata"]["original_modality"] for row in corpus)),
        "missing_image_files": len(missing_images),
        "native_multimodal_ready": not missing_images,
        "protocol": "sampled_dev_candidate_union",
        "selected_ids": [row["id"] for row in queries], "sources": sources,
        "notes": [
            "All supplied distractor contexts are retained for each sampled question.",
            "This sampled candidate union is not the published full benchmark protocol.",
            "Tables are serialized as text. Image titles are metadata, not visual evidence.",
            "Reference answers, intermediate answers, and support labels are never added to the corpus.",
            "Multiple reference answers can be required list members, not interchangeable alternatives.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "download_manifest.json", {
        "dataset": "multimodalqa", "automatically_downloaded": False,
        "archives": [{"url": IMAGE_ARCHIVE_URL, "destination": str(media_dir),
                      "status": "required_for_native_image_evaluation" if missing_images else "available",
                      "note": "Extract original image files into destination; preparation does not download the archive."}],
        "required_media": [{"id": row["id"], "path": row["path"],
                            "archive_member": row["metadata"]["archive_member"],
                            "available": row["metadata"]["media_available"]} for row in image_rows],
    })
    return manifest
