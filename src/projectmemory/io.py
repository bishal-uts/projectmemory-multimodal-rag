"""Bounded, cached downloads and deterministic UTF-8 data files."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path | str, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fingerprint(data: object) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def download(url: str, path: Path | str, max_bytes: int = 150_000_000, *, timeout: int = 60) -> Path:
    """Reuse existing files; never leave a partial file at the final destination."""
    path = Path(path)
    if urlparse(url).scheme not in {"https", "http"}:
        raise ValueError("Only HTTP(S) download URLs are supported")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if path.is_file() and path.stat().st_size:
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Cached file exceeds download limit: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        request = Request(url, headers={"User-Agent": "ProjectMemory/0.1 academic research"})
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"Download exceeds {max_bytes:,}-byte limit: {url}")
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as target:
                temp_path = Path(target.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Download exceeds {max_bytes:,}-byte limit: {url}")
                    target.write(chunk)
        if not total:
            raise ValueError(f"Empty download: {url}")
        os.replace(temp_path, path)
        return path
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
