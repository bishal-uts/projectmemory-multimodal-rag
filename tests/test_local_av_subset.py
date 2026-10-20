"""Integrity checks for a shared, uncropped AV evaluation population."""
import importlib.util
from pathlib import Path
import random
import wave

import pytest

from projectmemory.io import read_jsonl, write_jsonl


SPEC = importlib.util.spec_from_file_location(
    "prepare_local_av_eval", Path(__file__).resolve().parents[1] / "scripts/prepare_local_av_eval.py")
subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subset)


def source_dataset(tmp_path, monkeypatch, count=3):
    source = tmp_path / "source"
    source.mkdir()
    corpus, queries, probes = [], [], {}
    for index in range(count):
        video_id = f"clip{index:03d}"
        links = []
        for modality, suffix in (("video", ".mp4"), ("audio", ".wav")):
            path = source / (video_id + suffix)
            path.write_bytes(f"source-{video_id}-{modality}".encode())
            identity = f"daily_omni:{modality}:{video_id}"
            links.append(identity)
            metadata = {"representation": "native_silent_video" if modality == "video" else "native_media"}
            if modality == "video":
                metadata["full_duration"] = True
            corpus.append({"id": identity, "modality": modality, "path": str(path),
                           "source_id": video_id, "title": f"Clip {video_id}", "text": "",
                           "metadata": metadata})
            probes[str(path)] = {"duration_seconds": 30 if modality == "audio" else 60,
                                 "audio_streams": [{}] if modality == "audio" else [],
                                 "video_streams": [{}] if modality == "video" else []}
        queries.append({"id": f"q:{index:03d}", "question": f"Question {index}",
                        "options": ["A. one", "B. two", "C. three", "D. four"],
                        "answer_label": "C", "reference_answers": ["C"],
                        "relevant_ids": [], "split": "benchmark",
                        "metadata": {"video_id": video_id, "known_media_ids": links,
                                     "required_modalities": ["audio", "video"],
                                     "evaluation_protocol": "known_video_qa"}})
    # Ordering in the input must not affect the deterministic query order.
    write_jsonl(source / "corpus.jsonl", reversed(corpus))
    write_jsonl(source / "queries.jsonl", reversed(queries))
    monkeypatch.setattr(subset, "probe_media", lambda path: probes[str(path)])
    return source, corpus, queries, probes


def test_joint_duration_and_silence_eligibility_preserves_full_sources(tmp_path, monkeypatch):
    source, corpus, queries, probes = source_dataset(tmp_path, monkeypatch, count=4)
    probes[str(source / "clip001.wav")]["duration_seconds"] = 30.000001
    probes[str(source / "clip002.mp4")]["duration_seconds"] = 60.000001
    probes[str(source / "clip003.mp4")]["audio_streams"] = [{}]
    source_before = {path: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "output"
    manifest = subset.prepare(source, output)
    assert manifest["selected_ids"] == ["q:000"]
    assert manifest["selection"]["eligible_query_count"] == 1
    assert len(manifest["excluded_queries"]) == 3
    assert read_jsonl(output / "queries.jsonl") == queries[:1]
    expected = sorted(corpus[:2], key=lambda row: row["id"])
    assert read_jsonl(output / "corpus.jsonl") == expected
    assert source_before == {path: path.read_bytes() for path in source.iterdir()}
    assert all("answer_label" not in row for row in read_jsonl(output / "corpus.jsonl"))
    assert len(manifest["media_audit"]) == 8
    assert all(len(row["sha256"]) == 64 for row in manifest["media_audit"])
    assert any("audio_duration_exceeds_30s" in reason
               for row in manifest["excluded_queries"] for reason in row["reasons"])


@pytest.mark.parametrize("count, expected", [(50, 50), (51, 20)])
def test_selection_rule_is_deterministic_and_repeatable(tmp_path, monkeypatch, count, expected):
    source, _, _, _ = source_dataset(tmp_path, monkeypatch, count=count)
    output = tmp_path / "output"
    manifest = subset.prepare(source, output)
    assert manifest["queries"] == expected
    assert manifest["selected_ids"] == [f"q:{index:03d}" for index in range(expected)]
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    assert subset.prepare(source, output) == manifest
    assert first == {path.name: path.read_bytes() for path in output.iterdir()}


def test_missing_audio_excludes_question_even_though_video_exists(tmp_path, monkeypatch):
    source, _, _, _ = source_dataset(tmp_path, monkeypatch)
    (source / "clip001.wav").unlink()
    manifest = subset.prepare(source, tmp_path / "output")
    assert manifest["selected_ids"] == ["q:000", "q:002"]
    assert "missing_or_nonabsolute_media_path" in manifest["excluded_queries"][0]["reasons"][0]


def test_nested_labels_fail_without_publishing_dataset(tmp_path, monkeypatch):
    source, corpus, _, _ = source_dataset(tmp_path, monkeypatch)
    corpus[0]["metadata"]["answer"] = "GOLD_SENTINEL"
    write_jsonl(source / "corpus.jsonl", corpus)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Question/answer fields"):
        subset.prepare(source, output)
    assert not output.exists()


def test_cropped_audio_is_excluded_and_source_overwrite_refused(tmp_path, monkeypatch):
    source, corpus, _, _ = source_dataset(tmp_path, monkeypatch)
    corpus[1]["metadata"].update(representation="physical_media_segment", max_duration_seconds=20)
    write_jsonl(source / "corpus.jsonl", corpus)
    manifest = subset.prepare(source, tmp_path / "output")
    assert "q:000" not in manifest["selected_ids"]
    with pytest.raises(ValueError, match="Output must not"):
        subset.prepare(source, source)


def test_combined_byte_limit_is_applied_to_all_modes(tmp_path, monkeypatch):
    source, _, _, _ = source_dataset(tmp_path, monkeypatch)
    original = subset.file_identity

    def identity(path):
        result = original(path)
        if path.stem == "clip000":
            result["bytes"] = 16_000_001
        return result

    monkeypatch.setattr(subset, "file_identity", identity)
    manifest = subset.prepare(source, tmp_path / "output")
    assert manifest["selected_ids"] == ["q:001", "q:002"]
    assert manifest["excluded_queries"][0]["reasons"] == ["combined_media_exceeds_32000000_bytes"]


def write_boundary_pcm(path, excess_frames=3, sample_rate=16000):
    frames = 30 * sample_rate + excess_frames
    pcm = (bytes(range(256)) * ((frames * 4 + 255) // 256))[:frames * 4]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    return pcm


def test_opt_in_boundary_normalization_keeps_pcm_prefix_and_original(tmp_path, monkeypatch):
    source, _, queries, probes = source_dataset(tmp_path, monkeypatch)
    original = source / "clip000.wav"
    pcm = write_boundary_pcm(original)
    probes[str(original)]["duration_seconds"] = 30.000188
    before = {path: path.read_bytes() for path in source.iterdir()}
    output = tmp_path / "normalized"
    manifest = subset.prepare(source, output, normalize_audio_rounding=True)
    normalization = manifest["audio_normalizations"][0]
    path = Path(normalization["output"]["path"])
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == 16000
        assert stream.getcomptype() == "NONE"
        assert stream.getnframes() == 480000
        assert stream.readframes(stream.getnframes()) == pcm[:480000 * 4]
    assert normalization["removed_samples_per_channel"] == 3
    assert normalization["removed_samples_all_channels"] == 6
    assert normalization["removed_duration_seconds"] == 3 / 16000
    assert normalization["original_duration_seconds"] == 480003 / 16000
    assert normalization["normalized_duration_seconds"] == 30
    assert normalization["source"]["sha256"] != normalization["output"]["sha256"]
    assert manifest["eligibility"]["full_clips_only"] is False
    assert manifest["selected_questions_with_untouched_audio"] == ["q:001", "q:002"]
    assert read_jsonl(output / "queries.jsonl") == queries
    assert before == {path: path.read_bytes() for path in source.iterdir()}
    first = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert subset.prepare(source, output, normalize_audio_rounding=True) == manifest
    assert first == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_normalization_rejects_over_one_ms_and_unsupported_wav(tmp_path, monkeypatch):
    source, _, _, probes = source_dataset(tmp_path, monkeypatch, count=4)
    write_boundary_pcm(source / "clip000.wav", excess_frames=17)
    probes[str(source / "clip000.wav")]["duration_seconds"] = 30.001063
    # It is named .wav but deliberately contains the fixture's invalid bytes.
    probes[str(source / "clip001.wav")]["duration_seconds"] = 30.000188
    write_boundary_pcm(source / "clip002.wav", excess_frames=16)
    probes[str(source / "clip002.wav")]["duration_seconds"] = 30.001
    manifest = subset.prepare(source, tmp_path / "normalized", normalize_audio_rounding=True)
    assert manifest["selected_ids"] == ["q:002", "q:003"]
    assert manifest["audio_normalizations"][0]["removed_duration_seconds"] == .001
    assert "audio_duration_exceeds_30s" in manifest["excluded_queries"][0]["reasons"][0]
    assert "unsupported_audio_boundary_normalization" in manifest["excluded_queries"][1]["reasons"][0]
    nonwav = source / "unsupported.mp3"
    nonwav.write_bytes(b"not a WAV")
    with pytest.raises(ValueError, match="PCM WAV"):
        subset.rounding_plan(nonwav)


def test_normalized_selection_uses_fixed_seed_before_answers(tmp_path, monkeypatch):
    source, _, queries, _ = source_dataset(tmp_path, monkeypatch, count=52)
    expected = sorted(random.Random(42).sample([row["id"] for row in queries], 20))
    first = subset.prepare(source, tmp_path / "normalized", normalize_audio_rounding=True)
    assert first["selection"]["eligible_query_count"] == 52
    assert first["selection"]["seed"] == 42
    assert first["selected_ids"] == expected
    for query in queries:
        query["answer_label"] = "A"
        query["reference_answers"] = ["A"]
    write_jsonl(source / "queries.jsonl", reversed(queries))
    second = subset.prepare(source, tmp_path / "relabeled", normalize_audio_rounding=True)
    assert second["selected_ids"] == expected
