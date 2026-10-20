"""Download selected original ZIP members using strictly bounded HTTP ranges.

No whole-archive fallback is allowed. The archive directory and selected members
are read from the same ETag when the server provides one. Only stored/DEFLATE,
single-disk, non-ZIP64 archives are supported. Extracted member names never decide
destination paths: callers supply explicit absolute paths and must enforce their
own output-root boundary.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_MEMBER_BYTES = 50_000_000
MAX_DIRECTORY_BYTES = 50_000_000
_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_LOCAL = struct.Struct("<4s5H3L2H")


class RemoteZipError(RuntimeError):
    """Archive, transfer, integrity, or byte-budget validation failed."""


class _ByteBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.reserved = 0
        self.downloaded = 0
        self.lock = threading.Lock()

    def reserve(self, count: int) -> None:
        with self.lock:
            if self.reserved + count > self.limit:
                raise RemoteZipError(
                    f"Download byte budget exceeded: {self.reserved + count:,} requested "
                    f"bytes would exceed max_bytes={self.limit:,}."
                )
            self.reserved += count

    def received(self, count: int) -> None:
        with self.lock:
            self.downloaded += count


class _RangeReader:
    def __init__(self, url: str, budget: _ByteBudget) -> None:
        self.url, self.budget = url, budget
        try:
            with urlopen(Request(url, method="HEAD", headers={"Accept-Encoding": "identity"}),
                         timeout=30) as response:
                self.size = int(response.headers.get("Content-Length", "0"))
                raw_etag = response.headers.get("ETag")
                self.etag = raw_etag if raw_etag and not raw_etag.startswith("W/") else None
                self.modified = response.headers.get("Last-Modified")
        except (OSError, ValueError) as exc:
            raise RemoteZipError("Could not read archive size using HTTP HEAD.") from exc
        if self.size < _EOCD.size:
            raise RemoteZipError("Remote object is too small to be a ZIP archive.")

    def read(self, start: int, count: int) -> bytes:
        if count == 0:
            return b""
        if start < 0 or count < 0 or start + count > self.size:
            raise RemoteZipError("Archive requested a byte range outside the remote object.")
        self.budget.reserve(count)
        end = start + count - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        elif self.modified:
            headers["If-Unmodified-Since"] = self.modified
        with urlopen(Request(self.url, headers=headers), timeout=60) as response:
            # Close before reading any body if a server ignores the Range header.
            if response.status != 206:
                raise RemoteZipError("Server does not honor HTTP Range requests; whole-archive download refused.")
            expected = f"bytes {start}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected:
                raise RemoteZipError("Server returned an unexpected Content-Range.")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RemoteZipError("HTTP content encoding would invalidate ZIP byte offsets.")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != count:
                raise RemoteZipError("Range Content-Length does not match its requested length.")
            if self.etag and response.headers.get("ETag") != self.etag:
                raise RemoteZipError("Archive ETag changed during the download.")
            payload = response.read(count)
            self.budget.received(len(payload))
            if len(payload) != count:
                raise RemoteZipError("Range response ended before its requested length.")
            return payload


def _safe_name(name: str) -> str:
    parts = (name[:-1] if name.endswith("/") else name).split("/")
    if (not name or "\\" in name or "\x00" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)):
        raise RemoteZipError(f"Unsafe ZIP member path: {name!r}")
    return name


def _decode_name(raw: bytes, flags: int) -> str:
    try:
        return _safe_name(raw.decode("utf-8" if flags & 0x800 else "cp437"))
    except UnicodeDecodeError as exc:
        raise RemoteZipError("ZIP member filename has invalid encoding.") from exc


@dataclass(frozen=True)
class _Member:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    size: int
    offset: int


def _read_directory(reader: _RangeReader) -> list[_Member]:
    tail_size = min(reader.size, _EOCD.size + 65535)
    tail_start = reader.size - tail_size
    tail = reader.read(tail_start, tail_size)
    position = len(tail)
    while True:
        position = tail.rfind(b"PK\x05\x06", 0, position)
        if position < 0:
            raise RemoteZipError("ZIP end-of-central-directory record was not found.")
        if len(tail) - position >= _EOCD.size:
            fields = _EOCD.unpack_from(tail, position)
            if position + _EOCD.size + fields[-1] == len(tail):
                break
    _, disk, directory_disk, disk_entries, entries, directory_size, directory_offset, _ = fields
    if disk or directory_disk or disk_entries != entries:
        raise RemoteZipError("Multi-disk ZIP archives are not supported.")
    if entries == 0xFFFF or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        raise RemoteZipError("ZIP64 archives are not supported by this bounded downloader.")
    if directory_size > MAX_DIRECTORY_BYTES:
        raise RemoteZipError("ZIP central directory exceeds the metadata size limit.")
    if directory_offset + directory_size > tail_start + position:
        raise RemoteZipError("ZIP central directory overlaps or exceeds its end record.")
    central = reader.read(directory_offset, directory_size)
    members: list[_Member] = []
    seen: set[str] = set()
    position = 0
    for _ in range(entries):
        if len(central) - position < _CENTRAL.size:
            raise RemoteZipError("Truncated ZIP central directory.")
        fields = _CENTRAL.unpack_from(central, position)
        if fields[0] != b"PK\x01\x02":
            raise RemoteZipError("Invalid ZIP central-directory signature.")
        name_len, extra_len, comment_len = fields[10:13]
        end = position + _CENTRAL.size + name_len + extra_len + comment_len
        if end > len(central):
            raise RemoteZipError("Truncated ZIP member metadata.")
        name = _decode_name(central[position + _CENTRAL.size:position + _CENTRAL.size + name_len], fields[3])
        if stat.S_ISLNK(fields[15] >> 16):
            raise RemoteZipError(f"ZIP symlink entries are not allowed: {name!r}")
        if name in seen:
            raise RemoteZipError(f"Duplicate ZIP member name: {name!r}")
        seen.add(name)
        if fields[13] or any(value == 0xFFFFFFFF for value in (fields[8], fields[9], fields[16])):
            raise RemoteZipError("Multi-disk or ZIP64 member metadata is unsupported.")
        members.append(_Member(name, fields[3], fields[4], fields[7], fields[8], fields[9], fields[16]))
        position = end
    if position != len(central):
        raise RemoteZipError("ZIP central-directory size and entry count disagree.")
    return members


def _extract(reader: _RangeReader, member: _Member) -> bytes:
    local = _LOCAL.unpack(reader.read(member.offset, _LOCAL.size))
    if local[0] != b"PK\x03\x04" or local[2] != member.flags or local[3] != member.method:
        raise RemoteZipError(f"ZIP local header differs from directory for {member.name!r}.")
    name_size, extra_size = local[-2:]
    metadata = reader.read(member.offset + _LOCAL.size, name_size + extra_size)
    if _decode_name(metadata[:name_size], member.flags) != member.name:
        raise RemoteZipError("ZIP local filename does not match its central-directory entry.")
    data_offset = member.offset + _LOCAL.size + name_size + extra_size
    compressed = reader.read(data_offset, member.compressed_size)
    if member.method == 0:
        payload = compressed
    else:
        try:
            decoder = zlib.decompressobj(-15)
            payload = decoder.decompress(compressed, member.size + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise RemoteZipError("Compressed ZIP member is truncated or exceeds its declared size.")
        except zlib.error as exc:
            raise RemoteZipError(f"Cannot decompress ZIP member {member.name!r}.") from exc
    if len(payload) != member.size or zlib.crc32(payload) & 0xFFFFFFFF != member.crc32:
        raise RemoteZipError(f"ZIP size/CRC validation failed for {member.name!r}.")
    return payload


def _existing_digest(path: Path, member: _Member) -> str | None:
    if not path.is_file() or path.stat().st_size != member.size:
        return None
    checksum, digest = 0, hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
            digest.update(chunk)
    return digest.hexdigest() if checksum & 0xFFFFFFFF == member.crc32 else None


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_selected_zip(url: str, members: dict[str, Path], max_bytes: int = 150_000_000,
                       workers: int = 6) -> dict[str, Any]:
    """Fetch exact original ZIP members, with no full-archive fallback.

    Keys are exact filenames, or complete archive member names. A filename may
    resolve under one archive prefix; ambiguous basenames fail. Values must be
    absolute output paths already checked against the caller's allowed root.
    ``max_bytes`` bounds both requested HTTP body bytes (metadata included) and
    total selected uncompressed bytes, independently. Each file is also limited
    to 50 MB. Existing files are skipped only after their size and CRC match.

    Returns counts ``downloaded``/``skipped``, HTTP payload ``bytes_downloaded``,
    new-file ``bytes_written``, ``archive_bytes``, ``etag``, and per-file receipts.
    No annotation, question, answer, or other archive member is extracted.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("url must be an HTTP(S) URL without embedded credentials.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32.")
    destinations: dict[str, Path] = {}
    seen_destinations: set[str] = set()
    for name, destination in members.items():
        _safe_name(name)
        path = Path(destination)
        if not path.is_absolute():
            raise ValueError("ZIP destination paths must be absolute.")
        if path.is_symlink() or path.is_dir():
            raise RemoteZipError(f"Destination is a symlink or directory: {path}")
        key = os.path.normcase(str(path.resolve()))
        if key in seen_destinations:
            raise RemoteZipError("Two archive members cannot share the same output path.")
        seen_destinations.add(key)
        destinations[name] = path
    report: dict[str, Any] = {"downloaded": 0, "skipped": 0, "bytes_downloaded": 0,
                              "bytes_written": 0, "archive_bytes": 0, "etag": None, "files": []}
    if not destinations:
        return report
    budget = _ByteBudget(max_bytes)
    reader = _RangeReader(url, budget)
    entries = _read_directory(reader)
    full_names = {entry.name: entry for entry in entries}
    basenames: dict[str, list[_Member]] = {}
    for entry in entries:
        if not entry.name.endswith("/"):
            basenames.setdefault(PurePosixPath(entry.name).name, []).append(entry)
    selected: list[tuple[_Member, Path]] = []
    for requested, path in destinations.items():
        candidates = ([full_names[requested]] if "/" in requested and requested in full_names
                      else basenames.get(requested, []))
        if len(candidates) != 1:
            raise RemoteZipError(f"ZIP member {requested!r} is missing or its basename is ambiguous.")
        member = candidates[0]
        if member.name.endswith("/") or member.flags & 0x1 or member.method not in {0, 8}:
            raise RemoteZipError(f"ZIP member {requested!r} is a directory, encrypted, or uses unsupported compression.")
        if max(member.size, member.compressed_size) > min(max_bytes, MAX_MEMBER_BYTES):
            raise RemoteZipError(f"ZIP member {requested!r} exceeds the per-file byte limit.")
        selected.append((member, path))
    if sum(member.size for member, _ in selected) > max_bytes:
        raise RemoteZipError("Selected uncompressed files exceed max_bytes; refusing oversized extraction.")
    cancelled = threading.Event()

    def fetch_one(member: _Member, path: Path) -> dict[str, Any] | None:
        if cancelled.is_set():
            return None
        try:
            digest = _existing_digest(path, member)
            status = "skipped"
            if digest is None:
                payload = _extract(reader, member)
                digest = hashlib.sha256(payload).hexdigest()
                _atomic_write(path, payload)
                status = "downloaded"
            return {"member": member.name, "path": str(path), "size": member.size,
                    "sha256": digest, "status": status}
        except Exception:
            cancelled.set()
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, member, path) for member, path in selected]
        for future in as_completed(futures):
            try:
                receipt = future.result()
            except Exception:
                cancelled.set()
                for pending in futures:
                    pending.cancel()
                raise
            if receipt is not None:
                report["files"].append(receipt)
    report["files"].sort(key=lambda item: item["member"])
    report["downloaded"] = sum(item["status"] == "downloaded" for item in report["files"])
    report["skipped"] = sum(item["status"] == "skipped" for item in report["files"])
    report["bytes_written"] = sum(item["size"] for item in report["files"] if item["status"] == "downloaded")
    report.update(bytes_downloaded=budget.downloaded, archive_bytes=reader.size, etag=reader.etag)
    return report
