"""Synthetic media here are test fixtures, never benchmark evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest

from projectmemory.av import probe_media, segment_media, make_silent_video, prepare_video_only_corpus, _inside
from projectmemory.io import read_jsonl, write_jsonl


@pytest.fixture
def media(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are needed for media fixture tests")
    paths = {}
    for kind in ["audio", "video", "av"]:
        path = tmp_path / (kind + (".wav" if kind == "audio" else ".mp4"))
        command = [shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-y"]
        if kind != "audio":
            command += ["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=2.4"]
        if kind != "video":
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2.4"]
        if kind != "audio":
            command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        command += [str(path)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        paths[kind] = path
    return paths


def test_audio_fixed_boundaries_and_final_partial_segment(media, tmp_path):
    rows = segment_media(media["audio"], tmp_path / "out", chunk_seconds=1)
    assert len(rows) == 3
    assert [r["metadata"]["start_seconds"] for r in rows] == [0, 1, 2]
    assert rows[-1]["metadata"]["end_seconds"] == pytest.approx(2.4)
    for row in rows:
        assert row["modality"] == "audio"
        probe = probe_media(Path(row["path"]))
        assert not probe["video_streams"]
        assert probe["audio_streams"][0]["sample_rate"] == "16000"
        assert probe["audio_streams"][0]["channels"] == 1
        assert probe["duration_seconds"] == pytest.approx(row["metadata"]["end_seconds"] - row["metadata"]["start_seconds"], abs=1/16000)


def test_video_without_audio_does_not_invent_soundtrack(media, tmp_path):
    rows = segment_media(media["video"], tmp_path / "silent", chunk_seconds=1, max_duration_seconds=1.5)
    assert len(rows) == 2
    assert {r["modality"] for r in rows} == {"video"}
    assert rows[-1]["metadata"]["end_seconds"] == 1.5
    for row in rows:
        assert not probe_media(Path(row["path"]))["audio_streams"]
        assert row["metadata"]["duration_seconds"] <= 1.01


def test_video_soundtrack_is_split_into_separate_native_evidence(media, tmp_path):
    rows = segment_media(media["av"], tmp_path / "avout", max_duration_seconds=1)
    assert {r["modality"] for r in rows} == {"audio", "video"}
    assert len({r["source_id"] for r in rows}) == 1
    assert all(r["metadata"]["end_seconds"] == 1 for r in rows)
    assert not probe_media(Path(next(r["path"] for r in rows if r["modality"] == "video")))["audio_streams"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_segmentation_limits_fail_before_file_access(tmp_path, value):
    with pytest.raises(ValueError, match="finite positive"):
        segment_media(tmp_path / "missing", tmp_path / "out", chunk_seconds=value)
    with pytest.raises(ValueError, match="finite positive"):
        segment_media(tmp_path / "missing", tmp_path / "out", max_duration_seconds=value)


def test_output_path_escape_rejected(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(ValueError, match="leaves output"):
        _inside(root / ".." / "escape.mp4", root.resolve())


def test_content_hash_prevents_same_name_source_collisions(media, tmp_path):
    first = segment_media(media["audio"], tmp_path / "out", max_duration_seconds=0.3)
    second = segment_media(media["audio"], tmp_path / "out", max_duration_seconds=0.6)
    assert first[0]["id"] != second[0]["id"]
    assert first[0]["source_id"] == second[0]["source_id"]
    assert first[0]["path"] != second[0]["path"]


def test_delayed_audio_keeps_its_real_source_start(media, tmp_path):
    delayed = tmp_path / "delayed.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-y",
                    "-i", str(media["video"]), "-itsoffset", "0.7", "-i", str(media["audio"]),
                    "-c:v", "copy", "-c:a", "aac", str(delayed)],
                   check=True, capture_output=True, timeout=30)
    original = probe_media(delayed)
    offset = float(original["audio_streams"][0]["start_time"]) - original["start_seconds"]
    assert offset > 0.5
    rows = segment_media(delayed, tmp_path / "offset", chunk_seconds=1, max_duration_seconds=1)
    audio = next(r for r in rows if r["modality"] == "audio")
    assert audio["metadata"]["start_seconds"] == pytest.approx(offset)
    assert audio["metadata"]["end_seconds"] == 1
    assert audio["metadata"]["duration_seconds"] == pytest.approx(1 - offset, abs=0.002)


def test_full_video_copy_removes_audio_without_cropping_frames(media, tmp_path):
    original = probe_media(media["av"])
    output = make_silent_video(media["av"], tmp_path / "full-silent")
    copied = probe_media(output)
    assert not copied["audio_streams"]
    assert copied["video_streams"][0]["nb_frames"] == original["video_streams"][0]["nb_frames"]
    assert copied["duration_seconds"] == pytest.approx(2.4, abs=0.1)


def test_silent_corpus_preparation_preserves_ids_answers_and_originals(media, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    query = dataset / "queries.jsonl"
    query.write_text('{"id":"q1","answer_label":"A"}\n', encoding="utf-8")
    before = query.read_bytes()
    original_size = media["av"].stat().st_size
    write_jsonl(dataset / "corpus.jsonl", [{"id": "video1", "source_id": "clip1", "modality": "video", "path": str(media["av"]), "text": ""}])
    report = prepare_video_only_corpus(dataset, tmp_path / "silent")
    assert report["silent_full_video_copies_created"] == 1
    rows = read_jsonl(dataset / "corpus.jsonl")
    assert rows[0]["id"] == "video1" and rows[0]["source_id"] == "clip1"
    assert rows[0]["metadata"]["original_path"] == str(media["av"].resolve())
    assert rows[0]["metadata"]["full_duration"] is True
    assert query.read_bytes() == before
    assert media["av"].stat().st_size == original_size
    again = prepare_video_only_corpus(dataset, tmp_path / "silent")
    assert again["silent_full_video_copies_created"] == 0
    assert read_jsonl(dataset / "corpus.jsonl") == rows
