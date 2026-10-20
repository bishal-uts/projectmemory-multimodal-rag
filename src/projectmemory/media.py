"""Explicit, bounded official-archive acquisition with selective safe extraction.

``limit`` limits media files, not questions. MultiModalQA uses selective ZIP
range requests. Daily-Omni downloads its whole TAR before extracting selected
files; limit does not reduce that download. ``max_bytes`` caps transfer and
selected uncompressed bytes. Media is fetched only when explicitly called.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
import zipfile

from projectmemory.datasets.daily_omni import VIDEO_URL
from projectmemory.datasets.mmqa import IMAGE_ARCHIVE_URL
from projectmemory.io import download, read_jsonl, write_json, write_jsonl


DEFAULT_MAX_BYTES = 150_000_000
OFFICIAL_ARCHIVES = {
    "multimodalqa": {IMAGE_ARCHIVE_URL},
    "daily_omni": {VIDEO_URL},
}


def _member_name(name: str) -> str:
    """Normalize harmless ./ prefixes; reject paths unsafe on POSIX or Windows."""
    if "\x00" in name or "\\" in name or name.startswith("/"):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    parts = name.split("/")
    if any(part == ".." or ":" in part for part in parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    normalized = str(PurePosixPath(name))
    if normalized in {"", "."}:
        return ""
    for part in PurePosixPath(normalized).parts:
        if part.endswith((".", " ")) or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part):
            raise ValueError(f"Unsafe Windows archive member path: {name!r}")
    return normalized


def _destination(path: str | Path, media_root: Path) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError("Media manifest destinations must be absolute paths")
    resolved = supplied.resolve()
    root = media_root.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Media destination leaves the configured media root: {path}")
    # Explicitly check Windows alternate streams and reserved names, even when
    # running the tests on another operating system.
    for part in resolved.relative_to(root).parts:
        _member_name(part)
    return resolved


def _available(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _selection(required_media: list[dict], media_root: Path) -> dict[str, tuple[dict, Path]]:
    selected = {}
    targets = set()
    for record in required_media:
        member = _member_name(str(record["archive_member"]))
        if not member:
            raise ValueError("A required media member cannot be an archive root")
        target = _destination(record["path"], media_root)
        # Case-folding also prevents unsafe collisions on a Windows destination
        # when the manifest is prepared on a case-sensitive host.
        target_key = str(target).casefold()
        if member in selected or target_key in targets:
            raise ValueError(f"Duplicate media member or destination: {member}")
        selected[member] = (record, target)
        targets.add(target_key)
    return selected


def _copy_member(stream, target: Path, expected_size: int, media_root: Path) -> None:
    if expected_size <= 0:
        raise ValueError(f"Media archive member is empty: {target.name}")
    target = _destination(target, media_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Resolve again after mkdir, before creating a temporary file or replacing a
    # target, so existing symlink parents cannot redirect writes outside root.
    target = _destination(target, media_root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            total = 0
            while chunk := stream.read(min(1024 * 1024, expected_size - total + 1)):
                total += len(chunk)
                if total > expected_size:
                    raise ValueError(f"Archive member exceeds its declared size: {target.name}")
                output.write(chunk)
        if total != expected_size:
            raise ValueError(f"Truncated archive member: {target.name}")
        os.replace(temporary, _destination(target, media_root))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def extract_selected(archive_path: Path, required_media: list[dict], media_root: Path,
                     max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    """Extract only exact manifest members from ZIP/TAR, never extractall().

    Every archive header is checked before writing any selected payload. Links,
    devices, path traversal, duplicate paths and expansion over budget fail.
    Existing nonempty media files are retained. Missing members are reported.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    archive_path = Path(archive_path)
    media_root = Path(media_root).resolve()
    selected = _selection(required_media, media_root)
    extracted = []
    existing = []
    uncompressed_bytes = 0
    found = set()

    def plan(entries):
        nonlocal uncompressed_bytes
        seen = set()
        jobs = []
        for name, size, entry in entries:
            if name in seen:
                raise ValueError(f"Duplicate archive member: {name}")
            seen.add(name)
            if name not in selected:
                continue
            found.add(name)
            record, target = selected[name]
            if _available(target):
                existing.append(record["id"])
                continue
            if size <= 0:
                raise ValueError(f"Media archive member is empty: {name}")
            uncompressed_bytes += size
            if uncompressed_bytes > max_bytes:
                raise ValueError(f"Selected uncompressed media exceeds {max_bytes:,}-byte limit")
            jobs.append((entry, record, target, size))
        return jobs

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            entries = []
            for info in archive.infolist():
                # ZipInfo normalizes backslashes on Windows; validate the raw
                # stored name as well so validation is consistent across hosts.
                _member_name(info.orig_filename)
                name = _member_name(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Archive symlinks are not allowed: {info.filename}")
                kind = stat.S_IFMT(mode)
                if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError(f"Special archive entries are not allowed: {info.filename}")
                if info.is_dir():
                    continue
                if not name:
                    raise ValueError("Invalid empty ZIP member name")
                entries.append((name, info.file_size, info))
            for info, record, target, size in plan(entries):
                with archive.open(info) as stream:
                    _copy_member(stream, target, size, media_root)
                extracted.append(record["id"])
    elif tarfile.is_tarfile(archive_path):
        # The official Daily-Omni archive is an uncompressed TAR. Do not scan
        # arbitrary compressed TAR payloads whose expansion is not yet bounded.
        with tarfile.open(archive_path, "r:") as archive:
            entries = []
            for info in archive.getmembers():
                name = _member_name(info.name)
                if info.issym() or info.islnk():
                    raise ValueError(f"Archive links are not allowed: {info.name}")
                if info.isdir():
                    continue
                if not info.isreg() or not name:
                    raise ValueError(f"Special archive entries are not allowed: {info.name}")
                entries.append((name, info.size, info))
            for info, record, target, size in plan(entries):
                stream = archive.extractfile(info)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {info.name}")
                with stream:
                    _copy_member(stream, target, size, media_root)
                extracted.append(record["id"])
    else:
        raise ValueError(f"Expected a ZIP or TAR archive: {archive_path}")
    return {"extracted_ids": extracted, "already_available_ids": existing,
            "missing_member_ids": [record["id"] for name, (record, _) in selected.items() if name not in found],
            "uncompressed_bytes": uncompressed_bytes}


def _refresh_prepared_files(dataset_dir: Path, manifest: dict, media_root: Path, report: dict) -> None:
    for record in manifest["required_media"]:
        record["available"] = _available(_destination(record["path"], media_root))
    manifest["last_fetch"] = report
    write_json(dataset_dir / "download_manifest.json", manifest)
    corpus_path = dataset_dir / "corpus.jsonl"
    if corpus_path.is_file():
        corpus = read_jsonl(corpus_path)
        for row in corpus:
            if row.get("modality") not in {"image", "audio", "video"} or not row.get("path"):
                continue
            available = _available(_destination(row["path"], media_root))
            metadata = row.setdefault("metadata", {})
            metadata["media_available"] = available
            metadata["representation"] = (("native_image" if row["modality"] == "image" else "native_media")
                                          if available else ("metadata_only" if row["modality"] == "image" else "missing_media"))
        write_jsonl(corpus_path, corpus)
    prepared_manifest_path = dataset_dir / "manifest.json"
    if prepared_manifest_path.is_file():
        prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        missing = [row for row in manifest["required_media"] if not row["available"]]
        prepared["native_multimodal_ready"] = not missing
        if prepared.get("dataset") == "multimodalqa":
            prepared["missing_image_files"] = len(missing)
        else:
            prepared["missing_media_by_modality"] = dict(Counter(row["id"].split(":")[1] for row in missing))
        prepared["last_media_fetch"] = report
        write_json(prepared_manifest_path, prepared)


def fetch_media(dataset_dir: Path, max_bytes: int = DEFAULT_MAX_BYTES, limit: int | None = None,
                *, media_root: Path | None = None) -> dict:
    """Fetch declared official archives and safely extract missing selected media.

    ``dataset_dir`` is e.g. data/prepared/mmqa. The default writable media root
    is data/media; a different root must be explicitly passed. Archives cache
    under data/raw/archives/<dataset>. A 3.91 GB Daily-Omni archive needs an
    explicit budget larger than that even when ``limit=1``. MultiModalQA uses
    bounded HTTP ranges instead of fetching the entire ZIP. No model/API calls
    occur. HTTP failures surface without synthetic substitutes.
    """
    if max_bytes <= 0 or (limit is not None and limit <= 0):
        raise ValueError("max_bytes and limit must be greater than zero")
    dataset_dir = Path(dataset_dir).resolve()
    media_root = Path(media_root).resolve() if media_root else (dataset_dir.parent.parent / "media").resolve()
    manifest = json.loads((dataset_dir / "download_manifest.json").read_text(encoding="utf-8"))
    dataset = manifest.get("dataset")
    if dataset not in OFFICIAL_ARCHIVES:
        raise ValueError(f"Unsupported media dataset: {dataset}")
    required = manifest.get("required_media", [])
    selection = _selection(required, media_root)
    pending = [record for record, target in selection.values() if not _available(target)]
    selected = pending[:limit] if limit is not None else pending
    report = {"dataset": dataset, "requested_media": len(required),
              "selected_missing_media": len(selected), "extracted_media": 0,
              "downloaded_bytes": 0, "uncompressed_bytes": 0, "archives": [],
              "note": ("limit bounds media files; MultiModalQA uses selective ZIP ranges, "
                       "Daily-Omni downloads the whole TAR archive.")}
    archive_budget = max_bytes
    extraction_budget = max_bytes
    remaining = list(selected)
    for archive_info in manifest.get("archives", []):
        if not remaining:
            break
        url = archive_info["url"]
        if url not in OFFICIAL_ARCHIVES[dataset]:
            raise ValueError("Only this dataset's declared official archive URLs are supported")
        destination = Path(archive_info["destination"]).resolve()
        if destination != media_root and not destination.is_relative_to(media_root):
            raise ValueError("Archive destination leaves the configured media root")
        if dataset == "multimodalqa":
            from projectmemory.remote_zip import fetch_selected_zip

            result = fetch_selected_zip(
                url, {row["archive_member"]: _destination(row["path"], media_root) for row in remaining},
                max_bytes=archive_budget,
            )
            archive_budget -= result["bytes_downloaded"]
            extraction_budget -= result["bytes_written"]
            report["downloaded_bytes"] += result["bytes_downloaded"]
            report["uncompressed_bytes"] += result["bytes_written"]
            report["extracted_media"] += sum(_available(_destination(row["path"], media_root)) for row in remaining)
            report["archives"].append({"url": url, "strategy": "selective_zip_ranges",
                                       "archive_bytes": result["archive_bytes"],
                                       "bytes_downloaded": result["bytes_downloaded"],
                                       "downloaded_files": result["downloaded"], "etag": result.get("etag")})
            remaining = [row for row in remaining if not _available(_destination(row["path"], media_root))]
            continue
        filename = PurePosixPath(urlparse(url).path).name
        data_root = dataset_dir.parent.parent.resolve()
        cache = (data_root / "raw" / "archives" / dataset / filename).resolve()
        if not cache.is_relative_to(data_root):
            raise ValueError("Archive cache path leaves the data directory")
        was_cached = _available(cache)
        if archive_budget <= 0:
            raise ValueError("Archive download budget exhausted")
        try:
            download(url, cache, max_bytes=archive_budget)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Official media archive unavailable: {url}. {error}. "
                               "No substitute media was downloaded; annotations remain usable.") from error
        size = cache.stat().st_size
        if size > archive_budget:
            raise ValueError("Archive exceeds the remaining byte budget")
        archive_budget -= size
        extracted = extract_selected(cache, remaining, media_root, max_bytes=extraction_budget)
        extraction_budget -= extracted["uncompressed_bytes"]
        report["extracted_media"] += len(extracted["extracted_ids"])
        report["uncompressed_bytes"] += extracted["uncompressed_bytes"]
        report["downloaded_bytes"] += 0 if was_cached else size
        report["archives"].append({"url": url, "cache_path": str(cache), "cache_hit": was_cached,
                                   "bytes": size, "missing_member_ids": extracted["missing_member_ids"]})
        remaining = [row for row in remaining if not _available(_destination(row["path"], media_root))]
    report["selected_media_missing"] = len(remaining)
    report["media_available"] = sum(_available(_destination(row["path"], media_root)) for row in required)
    report["media_missing"] = len(required) - report["media_available"]
    report["complete"] = report["media_missing"] == 0
    _refresh_prepared_files(dataset_dir, manifest, media_root, report)
    return report
