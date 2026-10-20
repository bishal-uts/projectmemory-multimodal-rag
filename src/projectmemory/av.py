"""Physically segment local media before retrieval; never infer evidence labels.

Times are source-relative seconds. Video outputs contain visual frames only;
audio outputs are separate mono 16 kHz PCM WAV files. FFmpeg and ffprobe must
be on PATH. Source files are read only and generated paths stay below output_dir.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def configure_media_runtime() -> Path | None:
    """Expose optional shared FFmpeg only to this process, before TorchCodec import.

    PROJECTMEMORY_FFMPEG_BIN may point to a user-supplied shared build. The
    reproducible workspace setup also recognizes its downloaded 8.1.2 build.
    Nothing is installed automatically and system/user PATH values are not edited.
    """
    configured = os.environ.get("PROJECTMEMORY_FFMPEG_BIN")
    bundled = Path(__file__).resolve().parents[2] / ".cache/ffmpeg/ffmpeg-8.1.2-full_build-shared/bin"
    directory = Path(configured).resolve() if configured else bundled
    if not directory.is_dir():
        if configured:
            raise ValueError("PROJECTMEMORY_FFMPEG_BIN must name an existing FFmpeg bin directory")
        return None
    if os.name == "nt" and not (directory / "ffmpeg.exe").is_file():
        raise ValueError("The configured FFmpeg directory does not contain ffmpeg.exe")
    key = str(directory)
    if key not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt" and key not in _DLL_DIRECTORY_HANDLES:
        _DLL_DIRECTORY_HANDLES[key] = os.add_dll_directory(key)
    return directory


def _tool(name: str) -> str:
    configure_media_runtime()
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is required for audio/video segmentation; install FFmpeg and add its bin directory to PATH.")
    return path


def _run(command: list[str], timeout: float) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{Path(command[0]).name} timed out after {timeout:g} seconds") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{Path(command[0]).name} failed: {(exc.stderr or '')[-2000:].strip()}") from exc
    return completed.stdout


def probe_media(path: Path) -> dict:
    """Inspect an existing local source without invoking a shell or fetching URLs."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Media source must be a file: {path}")
    result = json.loads(_run([
        _tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_format", "-show_streams", "-of", "json", str(path)
    ], 60))
    try:
        duration = float(result.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Media has no finite positive duration: {path}")
    streams = result.get("streams", [])
    # Album art is not a temporal video stream.
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if not videos and not audios:
        raise ValueError(f"Source has no supported audio/video streams: {path}")
    try:
        start_time = float(result.get("format", {}).get("start_time", 0))
    except (TypeError, ValueError):
        start_time = 0.0
    if not math.isfinite(start_time):
        raise ValueError(f"Media has an invalid start time: {path}")
    return {"duration_seconds": duration, "start_seconds": start_time,
            "video_streams": videos, "audio_streams": audios}


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Generated media path leaves output directory: {path}")
    return resolved


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_duration(stream: dict, fallback: float) -> float:
    try:
        value = float(stream.get("duration", fallback))
    except (TypeError, ValueError):
        value = fallback
    return min(value, fallback) if math.isfinite(value) and value > 0 else fallback


def _stream_offset(stream: dict, format_start: float) -> float:
    try:
        value = float(stream.get("start_time", format_start)) - format_start
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        raise ValueError("Source stream has a non-finite start time")
    return max(0.0, value)


def segment_media(path: Path, output_dir: Path, chunk_seconds: float = 20,
                  max_duration_seconds: float | None = None) -> list[dict]:
    """Return corpus records for fixed, physically cropped source intervals.

    A video produces a silent MP4 record and, if present, a WAV record for each
    interval. An audio-only file produces WAV records. The first temporal video
    stream and first audio stream are used; their original stream indices are
    recorded. max_duration_seconds deliberately limits the source prefix, useful
    for bounded smoke tests, and its use is explicit in every record's metadata.

    start_seconds/end_seconds describe the requested source interval. Encoded
    video duration can differ by one frame; measured duration_seconds is also
    recorded. Outputs use a content/configuration hash and are atomically replaced.
    """
    chunk_seconds = _number(chunk_seconds, "chunk_seconds")
    if max_duration_seconds is not None:
        max_duration_seconds = _number(max_duration_seconds, "max_duration_seconds")
    path = Path(path).resolve(strict=True)
    info = probe_media(path)
    ffmpeg = _tool("ffmpeg")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = _hash_file(path)
    source_id = f"file:{source_hash}"
    config = json.dumps({"version": 1, "source": source_hash, "chunk_seconds": chunk_seconds,
                         "max_duration_seconds": max_duration_seconds}, sort_keys=True)
    config_hash = hashlib.sha256(config.encode()).hexdigest()[:20]
    folder = _inside(root / config_hash, root)
    folder.mkdir(exist_ok=True)
    duration = info["duration_seconds"]
    selected_duration = min(duration, max_duration_seconds) if max_duration_seconds else duration
    streams = []
    if info["video_streams"]:
        streams.append(("video", info["video_streams"][0]))
    if info["audio_streams"]:
        streams.append(("audio", info["audio_streams"][0]))
    records = []
    for modality, stream in streams:
        offset = _stream_offset(stream, info["start_seconds"])
        stream_end = min(selected_duration, offset + _stream_duration(stream, duration))
        for index in range(math.floor(offset / chunk_seconds), math.ceil(stream_end / chunk_seconds)):
            start = max(index * chunk_seconds, offset)
            end = min((index + 1) * chunk_seconds, stream_end)
            if end - start < 0.001:
                continue
            suffix = ".mp4" if modality == "video" else ".wav"
            target = _inside(folder / f"{modality}_{index:06d}{suffix}", root)
            if target == path:
                raise ValueError("Generated media must not replace the input source")
            fd, name = tempfile.mkstemp(prefix="segment_", suffix=suffix, dir=folder)
            os.close(fd)
            temporary = Path(name)
            try:
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-protocol_whitelist", "file,pipe", "-ss", f"{start:.9f}",
                           "-i", str(path), "-t", f"{end-start:.9f}",
                           "-map", f"0:{stream['index']}", "-map_metadata", "-1"]
                if modality == "video":
                    command += ["-an", "-vf", "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                                "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart"]
                else:
                    command += ["-vn", "-af", "asetpts=PTS-STARTPTS", "-ac", "1", "-ar", "16000",
                                "-c:a", "pcm_s16le", "-threads", "2"]
                _run(command + [str(temporary)], max(120, 10 * (end - start)))
                clip_info = probe_media(temporary)
                if modality == "video" and clip_info["audio_streams"]:
                    raise RuntimeError("A visual segment unexpectedly contains an audio stream")
                if modality == "audio" and clip_info["video_streams"]:
                    raise RuntimeError("An audio segment unexpectedly contains a video stream")
                os.replace(temporary, _inside(target, root))
            finally:
                temporary.unlink(missing_ok=True)
            records.append({
                "id": f"av:{config_hash}:{modality}:{index:06d}", "source_id": source_id,
                "modality": modality, "path": str(target), "text": "",
                "title": f"{path.name} ({start:.3f}-{end:.3f}s, {modality})",
                "metadata": {"source_file": str(path), "source_sha256": source_hash,
                             "source_duration": duration, "source_stream_index": stream["index"],
                             "source_stream_start_seconds": offset,
                             "start_seconds": start, "end_seconds": end,
                             "duration_seconds": clip_info["duration_seconds"],
                             "chunk_seconds": chunk_seconds, "max_duration_seconds": max_duration_seconds,
                             "representation": "physical_media_segment", "media_available": True,
                             "audio_included": modality == "audio",
                             "timing_precision": "video_frame" if modality == "video" else "audio_sample"},
            })
    return sorted(records, key=lambda r: (r["metadata"]["start_seconds"], r["modality"]))


def make_silent_video(path: Path, output_dir: Path) -> Path:
    """Copy the full first video stream into MP4 without audio or re-encoding."""
    path = Path(path).resolve(strict=True)
    original = probe_media(path)
    if not original["video_streams"]:
        raise ValueError("A video stream is required to prepare visual-only evidence")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = _inside(root / f"{_hash_file(path)}_silent.mp4", root)
    fd, name = tempfile.mkstemp(prefix="silent_", suffix=".mp4", dir=root)
    os.close(fd)
    temporary = Path(name)
    try:
        stream = original["video_streams"][0]
        _run([_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
              "-protocol_whitelist", "file,pipe", "-i", str(path), "-map", f"0:{stream['index']}",
              "-an", "-c:v", "copy", "-map_metadata", "-1", "-movflags", "+faststart", str(temporary)], 180)
        output = probe_media(temporary)
        if output["audio_streams"] or len(output["video_streams"]) != 1:
            raise RuntimeError("Visual-only media must contain one video stream and no audio")
        copied = output["video_streams"][0]
        for key in ["codec_name", "width", "height", "nb_frames"]:
            if key in stream and str(stream[key]) != str(copied.get(key)):
                raise RuntimeError(f"Silent video copy changed {key}")
        if abs(_stream_duration(stream, original["duration_seconds"]) -
               _stream_duration(copied, output["duration_seconds"])) > 0.1:
            raise RuntimeError("Silent video copy changed source duration")
        os.replace(temporary, _inside(target, root))
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare_video_only_corpus(dataset_dir: Path, output_dir: Path) -> dict:
    """Replace video corpus paths with full silent copies; keep IDs/queries intact.

    This is necessary for modality ablations when a benchmark supplies a video
    file with its soundtrack as well as a separate audio file. Original media is
    retained and recorded in metadata. Corpus publication occurs after all copies
    have passed validation. Dataset preparation from scratch must rerun this step.
    """
    from .io import read_jsonl, write_jsonl

    dataset_dir = Path(dataset_dir)
    corpus_path = dataset_dir / "corpus.jsonl"
    rows = read_jsonl(corpus_path)
    converted = 0
    checked = 0
    for row in rows:
        if row.get("modality") != "video":
            continue
        original_path = Path(row["path"]).resolve(strict=True)
        info = probe_media(original_path)
        checked += 1
        metadata = row.setdefault("metadata", {})
        if info["audio_streams"]:
            silent = make_silent_video(original_path, output_dir)
            metadata["original_path"] = str(original_path)
            metadata["original_sha256"] = _hash_file(original_path)
            row["path"] = str(silent)
            converted += 1
        metadata.update({"audio_included": False, "full_duration": True,
                         "representation": "native_silent_video", "media_available": True})
    fd, name = tempfile.mkstemp(prefix="corpus_", suffix=".jsonl", dir=dataset_dir)
    os.close(fd)
    temporary = Path(name)
    try:
        write_jsonl(temporary, rows)
        os.replace(temporary, corpus_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"video_records_checked": checked, "silent_full_video_copies_created": converted,
            "query_labels_changed": False, "corpus_path": str(corpus_path.resolve())}
