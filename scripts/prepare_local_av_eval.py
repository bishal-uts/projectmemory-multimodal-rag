"""Prepare an identical, full-clip Daily-Omni subset for local AV ablations.

By default this command only reads/probes/hashes existing media. The explicit
--normalize-audio-rounding experiment removes at most 1 ms of terminal PCM WAV
samples when a nominal 30-second source slightly exceeds the local limit.
Neither mode downloads, loads a model, calls APIs, or selects using answers.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.av import probe_media
from projectmemory.evaluation import validate_dataset
from projectmemory.io import fingerprint, read_jsonl, write_json, write_jsonl


MAX_AUDIO_SECONDS = 30
MAX_VIDEO_SECONDS = 60
MAX_MEDIA_BYTES = 32_000_000
ALL_ELIGIBLE_THRESHOLD = 50
LARGE_SUBSET_LIMIT = 20
MAX_ROUNDING_SECONDS = 0.001
FORBIDDEN_EVIDENCE_FIELDS = {
    "answer", "answer_label", "reference_answers", "question", "questions",
    "options", "choices", "choice", "q&a",
}


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def check_evidence(record: dict) -> None:
    """Reject labels, question content, and derived descriptions in the corpus."""
    def visit(value):
        if isinstance(value, dict):
            if FORBIDDEN_EVIDENCE_FIELDS.intersection(str(key).lower() for key in value):
                raise ValueError(f"Question/answer fields found in evidence {record['id']}")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(record)
    if record.get("text", ""):
        raise ValueError(f"Native AV evidence must have empty text: {record['id']}")


def rounding_plan(path: Path) -> dict:
    """Accept only uncompressed PCM WAVs with at most 1 ms of excess samples."""
    if path.suffix.lower() != ".wav":
        raise ValueError("Boundary normalization requires a PCM WAV file")
    try:
        with wave.open(str(path), "rb") as source:
            parameters = source.getparams()
    except (wave.Error, EOFError) as exc:
        raise ValueError("Boundary normalization requires a supported PCM WAV header") from exc
    if parameters.comptype != "NONE" or parameters.framerate <= 0:
        raise ValueError("Boundary normalization requires uncompressed PCM audio")
    kept = MAX_AUDIO_SECONDS * parameters.framerate
    removed = parameters.nframes - kept
    if removed <= 0 or removed > parameters.framerate * MAX_ROUNDING_SECONDS:
        raise ValueError("Boundary normalization must remove more than zero and at most 1 ms")
    return {"original_frames": parameters.nframes, "retained_frames": kept,
            "removed_frames": removed, "removed_samples_per_channel": removed,
            "removed_samples_all_channels": removed * parameters.nchannels,
            "sample_rate": parameters.framerate, "channels": parameters.nchannels,
            "sample_width_bytes": parameters.sampwidth, "compression": parameters.comptype,
            "original_duration_seconds": parameters.nframes / parameters.framerate,
            "normalized_duration_seconds": kept / parameters.framerate,
            "removed_duration_seconds": removed / parameters.framerate,
            "operation": "retain unchanged PCM prefix; remove terminal frames only"}


def normalize_boundary_audio(record: dict, audit: dict, output_dir: Path) -> tuple[dict, dict]:
    """Copy the exact retained PCM prefix without resampling or modifying input."""
    source = Path(record["path"])
    plan = rounding_plan(source)
    if plan != audit["boundary_normalization_plan"] or file_identity(source)["sha256"] != audit["sha256"]:
        raise ValueError("Source audio changed after the eligibility decision")
    root = (output_dir / "media").resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep Windows temp/workspace paths below MAX_PATH; the manifest stores the
    # complete SHA-256 and byte-level source/output provenance.
    target = root / f"{audit['sha256'][:20]}_30s.wav"
    with wave.open(str(source), "rb") as original:
        pcm = original.readframes(plan["original_frames"])
    frame_bytes = plan["channels"] * plan["sample_width_bytes"]
    if len(pcm) != plan["original_frames"] * frame_bytes:
        raise ValueError("PCM file is truncated relative to its declared frame count")
    retained_pcm = pcm[:plan["retained_frames"] * frame_bytes]
    descriptor, temporary_name = tempfile.mkstemp(prefix="normalize_", suffix=".wav", dir=root)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(plan["channels"])
            output.setsampwidth(plan["sample_width_bytes"])
            output.setframerate(plan["sample_rate"])
            output.setcomptype("NONE", "not compressed")
            output.writeframes(retained_pcm)
        with wave.open(str(temporary), "rb") as check:
            if (check.getnframes() != plan["retained_frames"]
                    or check.readframes(check.getnframes()) != retained_pcm):
                raise RuntimeError("Normalized audio did not preserve the exact PCM prefix")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    final_identity = file_identity(target)
    evidence = copy.deepcopy(record)
    normalization = {"id": record["id"], "source": {key: audit[key] for key in ("path", "bytes", "sha256")},
                     "output": final_identity, **plan}
    evidence["path"] = str(target)
    evidence.setdefault("metadata", {}).update({
        "representation": "boundary_normalized_pcm_audio", "media_available": True,
        "original_path": str(source), "original_sha256": audit["sha256"],
        "full_duration": False, "boundary_normalization": plan,
    })
    return evidence, normalization


def inspect_media(record: dict, normalize_audio_rounding: bool = False) -> dict:
    """Return an auditable eligibility decision without altering source media."""
    result = {"id": record["id"], "modality": record["modality"], "reasons": []}
    path = Path(record.get("path") or "")
    if not path.is_absolute() or not path.is_file():
        result["reasons"].append("missing_or_nonabsolute_media_path")
        return result
    result.update(file_identity(path))
    try:
        info = probe_media(path)
    except (OSError, ValueError, RuntimeError) as exc:
        result["reasons"].append("media_probe_failed")
        result["probe_error"] = str(exc)
        return result
    duration = info["duration_seconds"]
    result.update(duration_seconds=duration, audio_streams=len(info["audio_streams"]),
                  video_streams=len(info["video_streams"]))
    kind = record["modality"]
    metadata = record.get("metadata", {})
    limit = MAX_AUDIO_SECONDS if kind == "audio" else MAX_VIDEO_SECONDS
    if not math.isfinite(duration) or duration <= 0:
        result["reasons"].append("invalid_duration")
    elif duration > limit:
        if (normalize_audio_rounding and kind == "audio"
                and duration <= MAX_AUDIO_SECONDS + MAX_ROUNDING_SECONDS):
            try:
                result["boundary_normalization_plan"] = rounding_plan(path)
            except ValueError as exc:
                result["reasons"].append("unsupported_audio_boundary_normalization")
                result["normalization_error"] = str(exc)
        else:
            result["reasons"].append(f"{kind}_duration_exceeds_{limit}s")
    if kind == "audio":
        if len(info["audio_streams"]) != 1 or info["video_streams"]:
            result["reasons"].append("audio_requires_one_audio_stream_only")
        if metadata.get("representation") != "native_media":
            result["reasons"].append("audio_not_original_native_media")
    else:
        if len(info["video_streams"]) != 1 or info["audio_streams"]:
            result["reasons"].append("video_requires_one_silent_video_stream")
        if metadata.get("full_duration") is not True:
            result["reasons"].append("full_video_duration_not_confirmed")
    # Physical crop records cannot be mistaken for complete benchmark clips.
    if metadata.get("max_duration_seconds") is not None or metadata.get("start_seconds", 0) != 0:
        result["reasons"].append("cropped_media_not_allowed")
    if result["bytes"] > MAX_MEDIA_BYTES:
        result["reasons"].append("media_file_exceeds_32000000_bytes")
    return result


def prepare(source_dir: Path, output_dir: Path, *, normalize_audio_rounding: bool = False,
            seed: int = 42) -> dict:
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or source_dir.is_relative_to(output_dir):
        raise ValueError("Output must not be the source dataset or its ancestor")
    corpus_path, query_path = source_dir / "corpus.jsonl", source_dir / "queries.jsonl"
    corpus, queries = read_jsonl(corpus_path), read_jsonl(query_path)
    validate_dataset(corpus, queries)
    lookup = {row["id"]: row for row in corpus}
    for row in corpus:
        if row["modality"] not in {"audio", "video"}:
            raise ValueError("Local AV subset requires an audio/video-only source corpus")
        check_evidence(row)
    # Only IDs and declared known-media links affect selection; labels are copied
    # verbatim to queries.jsonl for the evaluator, never read for ranking/filtering.
    for query in queries:
        metadata = query.get("metadata", {})
        video_id = metadata.get("video_id")
        expected = {f"daily_omni:audio:{video_id}", f"daily_omni:video:{video_id}"}
        links = metadata.get("known_media_ids", [])
        if (metadata.get("evaluation_protocol") != "known_video_qa" or query.get("relevant_ids")
                or len(links) != 2 or set(links) != expected):
            raise ValueError(f"Invalid known-video AV protocol: {query['id']}")
        if set(metadata.get("required_modalities", [])) != {"audio", "video"}:
            raise ValueError(f"Both modalities must be required: {query['id']}")
    media_audit = {row["id"]: inspect_media(row, normalize_audio_rounding)
                   for row in sorted(corpus, key=lambda r: r["id"])}
    eligible, exclusions = [], []
    for query in sorted(queries, key=lambda row: row["id"]):
        reasons = []
        linked = query["metadata"]["known_media_ids"]
        for identity in linked:
            if identity not in lookup:
                reasons.append(f"missing_corpus_record:{identity}")
                continue
            record = lookup[identity]
            if record.get("source_id") != query["metadata"]["video_id"]:
                raise ValueError(f"Media belongs to the wrong video: {identity}")
            if identity != f"daily_omni:{record['modality']}:{record['source_id']}":
                raise ValueError(f"Media ID and modality disagree: {identity}")
            reasons.extend(f"{identity}:{reason}" for reason in media_audit[identity]["reasons"])
        combined_bytes = sum(media_audit.get(identity, {}).get("bytes", 0) for identity in linked)
        if combined_bytes > MAX_MEDIA_BYTES:
            reasons.append("combined_media_exceeds_32000000_bytes")
        if reasons:
            exclusions.append({"id": query["id"], "video_id": query["metadata"]["video_id"],
                               "reasons": reasons})
        else:
            eligible.append(query)
    if normalize_audio_rounding:
        selected = sorted(random.Random(seed).sample(eligible, min(LARGE_SUBSET_LIMIT, len(eligible))),
                          key=lambda row: row["id"])
    else:
        selected = eligible if len(eligible) <= ALL_ELIGIBLE_THRESHOLD else eligible[:LARGE_SUBSET_LIMIT]
    if not selected:
        raise ValueError("No full clips satisfy the joint audio/video eligibility constraints")
    selected_ids = {row["id"] for row in selected}
    for query in eligible:
        if query["id"] not in selected_ids:
            exclusions.append({"id": query["id"], "video_id": query["metadata"]["video_id"],
                               "reasons": ["deterministic_20_question_limit"]})
    media_ids = {identity for row in selected for identity in row["metadata"]["known_media_ids"]}
    selected_corpus, normalizations = [], []
    for identity in sorted(media_ids):
        evidence = lookup[identity]
        if "boundary_normalization_plan" in media_audit[identity]:
            evidence, normalization = normalize_boundary_audio(evidence, media_audit[identity], output_dir)
            normalizations.append(normalization)
        selected_corpus.append(evidence)
    source_files = [corpus_path, query_path]
    if (source_dir / "manifest.json").is_file():
        source_files.append(source_dir / "manifest.json")
    manifest = {
        "schema_version": 1, "dataset": "daily_omni_local_full_clip_subset",
        "protocol": "known_video_qa", "source_dataset_dir": str(source_dir),
        "source_files": [file_identity(path) for path in source_files],
        "preparation_script": file_identity(Path(__file__)),
        "selection": {"order": "ascending query ID (lexicographic)",
                      "rule": "all eligible if count <= 50; otherwise first 20 eligible",
                      "scope": "existing prepared source queries only; not the full Daily-Omni benchmark",
                      "answer_or_prediction_based_selection": False,
                      "source_query_count": len(queries), "eligible_query_count": len(eligible),
                      "selected_query_count": len(selected), "excluded_query_count": len(exclusions)},
        "eligibility": {"max_audio_seconds": MAX_AUDIO_SECONDS,
                        "max_silent_video_seconds": MAX_VIDEO_SECONDS,
                        "max_combined_media_bytes": MAX_MEDIA_BYTES,
                        "both_modalities_required": True, "full_clips_only": True,
                        "cropping_or_reencoding": False},
        "queries": len(selected), "corpus_records": len(selected_corpus),
        "unique_videos": len({row["metadata"]["video_id"] for row in selected}),
        "native_multimodal_ready": True, "missing_media_by_modality": {},
        "selected_ids": [row["id"] for row in selected],
        "eligible_ids": [row["id"] for row in eligible],
        "excluded_queries": sorted(exclusions, key=lambda row: row["id"]),
        "media_audit": list(media_audit.values()),
        "selected_media_ids": sorted(media_ids),
        "question_type_counts": dict(Counter(row["metadata"].get("question_type", "unknown") for row in selected)),
        "corpus_records_fingerprint": fingerprint(selected_corpus),
        "query_fingerprint": fingerprint(selected),
        "native_media_fingerprint": fingerprint([media_audit[identity] for identity in sorted(media_ids)]),
        "notes": [
            "Use the exact same queries for audio-only, visual-only, and combined evaluation.",
            "Known-video multiple-choice QA is not collection retrieval; Recall/MRR do not apply.",
            "All corpus records and query labels are preserved from the source without edits.",
            "Audio and video are required jointly even for single-modality ablations.",
            "Short-clip filtering creates a duration-biased subset, not a full-benchmark score.",
            "Video is represented by the local adapter as timestamped frames at 1 FPS.",
            "Passing duration/byte checks cannot guarantee model context fit or answer quality.",
            "No model inference or API requests were made during subset preparation.",
        ],
    }
    if normalize_audio_rounding:
        manifest["dataset"] = "daily_omni_local_boundary_normalized_subset"
        manifest["experiment"] = "known-video QA with explicit terminal audio rounding normalization"
        manifest["selection"].update({
            "rule": "sample up to 20 eligible queries without replacement from ascending query IDs using Python random.Random(seed); publish in ascending ID order",
            "seed": seed,
        })
        manifest["eligibility"].update({
            "max_source_audio_seconds": MAX_AUDIO_SECONDS + MAX_ROUNDING_SECONDS,
            "max_terminal_audio_trim_seconds": MAX_ROUNDING_SECONDS,
            "max_effective_audio_seconds": MAX_AUDIO_SECONDS,
            "full_clips_only": False, "full_video_clips_only": True,
            "cropping_or_reencoding": "terminal PCM frame removal only; no resampling or video changes",
        })
        manifest["audio_normalizations"] = normalizations
        manifest["normalized_audio_files"] = len(normalizations)
        manifest["selected_questions_with_untouched_audio"] = [
            row["id"] for row in selected
            if not any("boundary_normalization_plan" in media_audit[identity]
                       for identity in row["metadata"]["known_media_ids"])
        ]
        manifest["source_native_media_fingerprint"] = manifest["native_media_fingerprint"]
        manifest["native_media_fingerprint"] = fingerprint([
            {"id": row["id"], **file_identity(Path(row["path"]))} for row in selected_corpus
        ])
        manifest["notes"] = [
            note for note in manifest["notes"]
            if note != "All corpus records and query labels are preserved from the source without edits."
        ] + [
            "Query IDs, question text, choices, and labels are unchanged; corpus audio paths/metadata record normalization.",
            "This is a boundary-normalized experiment, not an untouched full-clip benchmark.",
            "Only PCM WAV sources in (30,30.001] seconds may lose at most 1 ms of terminal samples.",
            "All retained PCM bytes, channel counts, sample widths, and sample rates are unchanged.",
            "Original audio files and all full silent video files are unchanged.",
        ]
    write_jsonl(output_dir / "corpus.jsonl", selected_corpus)
    write_jsonl(output_dir / "queries.jsonl", selected)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/prepared/daily-omni")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalize-audio-rounding", action="store_true",
                        help="Explicit experiment: remove at most 1 ms of terminal PCM WAV samples above 30s")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the optional boundary-normalized 20-question subset")
    args = parser.parse_args()
    output = args.output or ROOT / ("data/prepared/daily-omni-local-normalized"
                                    if args.normalize_audio_rounding else "data/prepared/daily-omni-local")
    if args.normalize_audio_rounding and output.resolve() == (ROOT / "data/prepared/daily-omni-local").resolve():
        parser.error("Boundary normalization must not overwrite the strict local subset")
    manifest = prepare(args.source, output, normalize_audio_rounding=args.normalize_audio_rounding, seed=args.seed)
    print(json.dumps({"output": str(output.resolve()), "queries": manifest["queries"],
                      "unique_videos": manifest["unique_videos"],
                      "eligible_queries": manifest["selection"]["eligible_query_count"],
                      "excluded_queries": manifest["selection"]["excluded_query_count"]}, indent=2))


if __name__ == "__main__":
    main()
