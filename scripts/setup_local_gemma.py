"""Explicit, pinned download of Gemma 4 E2B QAT and the Windows CUDA runtime.

Run with the project's Python. No API keys or system installation are needed.
Every download is verified against its publisher's SHA-256 before use.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".cache/local-gemma4"
MODEL = "google/gemma-4-E2B-it-qat-q4_0-gguf"
REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"
RUNTIME = "b11429"
ARTIFACTS = [
    ("models/gemma-4-E2B_q4_0-it.gguf", 3349516256,
     "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
     f"https://huggingface.co/{MODEL}/resolve/{REVISION}/gemma-4-E2B_q4_0-it.gguf"),
    ("models/gemma-4-E2B-it-mmproj.gguf", 986833664,
     "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14",
     f"https://huggingface.co/{MODEL}/resolve/{REVISION}/gemma-4-E2B-it-mmproj.gguf"),
    ("downloads/llama-b11429-bin-win-cuda-12.4-x64.zip", 264478423,
     "dd6df685c1024e6aa55ca55ad35038692d5e284d4dc895e7a646c3245d3d14ff",
     f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME}/llama-b11429-bin-win-cuda-12.4-x64.zip"),
    ("downloads/cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
     "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
     f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME}/cudart-llama-bin-win-cuda-12.4-x64.zip"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(artifact):
    relative, size, digest, url = artifact
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == size and sha256(path) == digest:
        print(f"Verified existing {path.name}", flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"Unexpected partial size for {path.name}")
        if offset == size:
            break
        headers = {"User-Agent": "ProjectMemory-local-setup/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected resume range")
                progress_at = time.monotonic()
                with partial.open("ab" if offset else "wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > size:
                            raise RuntimeError(f"Download exceeded expected size: {path.name}")
                        if time.monotonic() - progress_at > 20:
                            print(f"{path.name}: {offset / size:.0%}", flush=True)
                            progress_at = time.monotonic()
            if offset == size:
                break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            print(f"Resuming {path.name} after interrupted transfer", flush=True)
    if partial.stat().st_size != size or sha256(partial) != digest:
        raise RuntimeError(f"SHA-256 verification failed for {path.name}; no model was installed")
    os.replace(partial, path)
    print(f"Verified {path.name}", flush=True)
    return path


def main():
    if os.name != "nt":
        raise SystemExit("This pinned setup script installs the Windows x64 CUDA runtime only.")
    BASE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, ARTIFACTS))
    runtime_dir = BASE / "runtime" / RUNTIME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for archive in paths[2:]:
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = (runtime_dir / member.filename).resolve()
                if not target.is_relative_to(runtime_dir.resolve()):
                    raise RuntimeError("Runtime archive contains an unsafe path")
            source.extractall(runtime_dir)
    executables = list(runtime_dir.rglob("llama-server.exe"))
    if len(executables) != 1:
        raise RuntimeError("Expected exactly one llama-server executable in the release")
    executable = executables[0]
    # CUDA dependency ZIPs can use their own top-level directory.
    dll_dirs = sorted({str(p.parent.relative_to(BASE)) for p in runtime_dir.rglob("*.dll")})
    manifest = {
        "schema_version": 1, "model": MODEL, "model_alias": "gemma-4-e2b",
        "model_revision": REVISION, "model_sha256": ARTIFACTS[0][2],
        "mmproj_sha256": ARTIFACTS[1][2], "runtime_version": RUNTIME,
        "model_file": ARTIFACTS[0][0], "mmproj_file": ARTIFACTS[1][0],
        "server_file": str(executable.relative_to(BASE)),
        "server_sha256": sha256(executable), "dll_directories": dll_dirs,
        "base_url": "http://127.0.0.1:8766", "context_size": 16384,
        "gpu_layers": 99, "batch_size": 256, "ubatch_size": 128,
        "artifacts": [{"path": a[0], "bytes": a[1], "sha256": a[2], "url": a[3]} for a in ARTIFACTS],
    }
    temporary = BASE / "runtime.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, BASE / "runtime.json")
    print("Local Gemma 4 E2B setup complete; no inference server was left running.", flush=True)


if __name__ == "__main__":
    main()
