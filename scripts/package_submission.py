"""Build a bounded, credential-free submission snapshot from explicit allowlists.

The default is a dry run. Use --write after reports and documentation are final.
This command never downloads data, loads models, calls APIs, or reads .env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md", "pyproject.toml", "run.py", ".gitignore", ".env.example", "LICENSE", "LICENSE.md"}
TREE_SUFFIXES = {
    "src": {".py", ".html", ".css", ".js", ".svg"},
    "tests": {".py"},
    "scripts": {".py"},
    "docs": {".md", ".txt", ".svg", ".png"},
}
RESULT_ROOT_FILES = {
    "ami-dev-bm25.json", "embedding-smoke.json", "av-embedding-smoke.json",
    "av-data-integrity.json", "av-media-probe.json", "av-media-silent-probe.json",
    "av-silent-preparation.json", "local-gemma-smoke.json", "nvidia-text-smoke.json",
    "gemini-text-smoke.json", "demo-av-live.json", "demo-av-browser.jpg",
}
RESULT_DIRECTORY = re.compile(
    r"(?:ami-(?:dev|test)-comparison|mmqa-(?:comparison|pilot)|"
    r"answers-(?:ami|mmqa)-(?:(?:gemini|nvidia|local)-)?(?:bm25|dense-text|multimodal)-[0-9]+|"
    r"answers-daily-(?:(?:gemini|nvidia|local)-)?(?:audio|video|combined)-[0-9]+|"
    r"final-mmqa-local-(?:bm25|dense-text|multimodal)|"
    r"final-local-av-(?:audio|video|combined)|final-analysis)"
)
RESULT_FILES = {"report.json", "run.json", "retrieval_source.json", "source_manifest.json", "predictions.jsonl", "analysis.json", "comparison.md",
                "bm25.json", "dense-text.json", "multimodal.json", "retrieval-comparison.png", "retrieval-comparison.svg"}
SKIP_DIRECTORIES = {"__pycache__", "node_modules", "dist", "build", "uploads", "downloads", "media", "models", "raw"}
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 25_000_000
SECRET_PATTERNS = (
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    re.compile(rb"nvapi-[0-9A-Za-z_-]{20,}"),
    re.compile(rb"(?:sk-proj-|sk-ant-|sk-)[0-9A-Za-z_-]{24,}"),
    re.compile(rb"(?:gh[pousr]_|github_pat_)[0-9A-Za-z_]{20,}"),
    re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)
SENSITIVE_JSON_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "secret_key", "client_secret", "authorization"}


class PackageError(ValueError):
    """A concise packaging failure that never includes credential values."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_link(path: Path) -> bool:
    # Junctions are reparse points on Windows, but not always is_symlink().
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _safe_file(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise PackageError("A submission input leaves the project directory") from None
    current = root
    for part in relative.parts:
        current /= part
        if _is_link(current):
            raise PackageError(f"Refusing linked submission input: {relative.as_posix()}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise PackageError(f"Invalid submission input: {relative.as_posix()}")
    archive_name = relative.as_posix()
    if PurePosixPath(archive_name).is_absolute() or ".." in PurePosixPath(archive_name).parts:
        raise PackageError("Unsafe archive member path")
    return archive_name


def _excluded_name(name: str) -> bool:
    lower = name.lower()
    return (lower.startswith(".") or lower in {"credentials.json", "secrets.json", "token.json", "tokens.json"}
            or lower.startswith("id_rsa") or lower.startswith("id_ed25519")
            or lower.endswith((".pem", ".key", ".p12", ".pfx")))


def selected_files(root: Path) -> list[Path]:
    """Choose source and known public reports without traversing data/caches."""
    root = root.resolve()
    selected = []
    for name in sorted(ROOT_FILES):
        path = root / name
        if path.exists() or path.is_symlink():
            _safe_file(root, path)
            selected.append(path)
    for directory, suffixes in TREE_SUFFIXES.items():
        start = root / directory
        if not start.exists():
            continue
        if _is_link(start):
            raise PackageError(f"Refusing linked source directory: {directory}")
        for folder, directories, files in os.walk(start, followlinks=False):
            kept = []
            for name in sorted(directories):
                if name.startswith(".") or name in SKIP_DIRECTORIES or name.endswith(".egg-info"):
                    continue
                candidate = Path(folder) / name
                if _is_link(candidate):
                    raise PackageError(f"Refusing linked source directory: {candidate.relative_to(root).as_posix()}")
                kept.append(name)
            directories[:] = kept
            for name in sorted(files):
                path = Path(folder) / name
                tested_environment = directory == "docs" and path.relative_to(start).as_posix() == "TESTED_ENVIRONMENT.json"
                if not _excluded_name(name) and (path.suffix.lower() in suffixes or tested_environment):
                    _safe_file(root, path)
                    selected.append(path)
    results = root / "results"
    if results.exists():
        if _is_link(results):
            raise PackageError("Refusing linked results directory")
        for entry in sorted(results.iterdir()):
            if entry.name in RESULT_ROOT_FILES and (entry.is_file() or entry.is_symlink()):
                _safe_file(root, entry)
                selected.append(entry)
            elif RESULT_DIRECTORY.fullmatch(entry.name) and entry.is_dir():
                if _is_link(entry):
                    raise PackageError(f"Refusing linked results directory: results/{entry.name}")
                for path in sorted(entry.iterdir()):
                    final_analysis = (entry.name == "final-analysis" and path.suffix.lower() in {".md", ".json", ".png", ".svg", ".csv"}
                                      and not re.search(r"(?:^|[._-])(?:raw|corpus|queries|credentials|secrets|tokens)(?:[._-]|$)", path.name.lower()))
                    if (path.name in RESULT_FILES or final_analysis) and not _excluded_name(path.name) and (path.is_file() or path.is_symlink()):
                        _safe_file(root, path)
                        selected.append(path)
    return sorted(set(selected), key=lambda path: path.relative_to(root).as_posix())


def _scan_json_secrets(value, name: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in SENSITIVE_JSON_KEYS and child not in (None, "", False):
                raise PackageError(f"Credential field found in {name}; no package was written")
            _scan_json_secrets(child, name)
    elif isinstance(value, list):
        for child in value:
            _scan_json_secrets(child, name)


def _scan_secrets(data: bytes, name: str) -> None:
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise PackageError(f"Possible credential found in {name}; no package was written")
    if name == ".env.example":
        for line in data.decode("utf-8-sig").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and (key.endswith("_KEY") or key.endswith("_TOKEN") or key.endswith("_SECRET")) and value.strip().strip("\"'"):
                raise PackageError(".env.example contains a populated credential field; no package was written")


def _portable_json(value, root: Path):
    """Rebase only local file references, retaining measurements and identities."""
    if isinstance(value, dict):
        return {key: _portable_json(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [_portable_json(child, root) for child in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        prefix = root.as_posix().rstrip("/")
        if normalized.casefold() == prefix.casefold():
            return "."
        if normalized.casefold().startswith(prefix.casefold() + "/"):
            return normalized[len(prefix) + 1:]
        if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith(("/home/", "/Users/", "//")):
            return "<external local path omitted>"
    return value


def _payload(root: Path, path: Path) -> tuple[bytes, dict]:
    name = _safe_file(root, path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise PackageError(f"Submission input exceeds {MAX_FILE_BYTES:,} bytes: {name}")
    source = path.read_bytes()
    if len(source) > MAX_FILE_BYTES:
        raise PackageError(f"Submission input grew beyond the file-size limit: {name}")
    _scan_secrets(source, name)
    payload, changes = source, []
    if (name.startswith("results/") or name == "docs/TESTED_ENVIRONMENT.json") and path.suffix.lower() in {".json", ".jsonl"}:
        try:
            if path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in source.decode("utf-8-sig").splitlines() if line.strip()]
                _scan_json_secrets(records, name)
                portable = _portable_json(records, root)
                if portable != records:
                    payload = ("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in portable) + "\n").encode()
                    changes.append("local paths rebased or omitted")
            else:
                record = json.loads(source.decode("utf-8-sig"))
                _scan_json_secrets(record, name)
                portable = _portable_json(record, root)
                if portable != record:
                    payload = (json.dumps(portable, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
                    changes.append("local paths rebased or omitted")
        except (ValueError, UnicodeDecodeError) as exc:
            if isinstance(exc, PackageError):
                raise
            raise PackageError(f"Invalid measured-result JSON: {name}") from None
    return payload, {"path": name, "bytes": len(payload), "sha256": _hash(payload),
                     "source_bytes": len(source), "source_sha256": _hash(source), "transformations": changes}


def submission_readme() -> bytes:
    return br"""# ProjectMemory submission snapshot

This archive contains application source, tests, documentation, and saved measured
results. It is a source snapshot, not a preinstalled environment. Start with
README.md, docs/PROJECT_PROPOSAL.md, docs/EVALUATION.md, docs/MMQA_RESULTS.md, and
docs/GENERATION_RESULTS.md. SUBMISSION_MANIFEST.json records every other member's
SHA-256 and size. SHA256SUMS.txt covers those members plus the manifest.

No API keys, .env, datasets, uploaded files, model weights, downloaded media,
virtual environment, or model caches are included. Data must be acquired from its
official source under its original terms. Documentation links into data/ require
that local preparation and will not open in the archive alone. Result JSON retains measured values;
absolute workspace file references are rebased, and external local paths are
omitted. The inventory retains both original-source and packaged-file hashes.
Use a NEW results directory for reproduction; historic run manifests are evidence
of the original run and may not resume against a new machine or code snapshot.

## Fresh Windows installation (Python 3.11+)

Run commands from the extracted project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[embeddings,ingestion,analysis,dev]"
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest-submission
```

Configure a compatible PyTorch/CUDA build for GPU retrieval. FFmpeg and ffprobe
are needed for audio/video ingestion; compatible shared FFmpeg is also required
by TorchCodec. See docs/AV_STATUS.md for the measured environment. CPU text/image
retrieval can be used without the local CUDA generator. The pinned local answer
runtime setup targets Windows x64 with an NVIDIA GPU.

## Prepare data, then the answer model

The following commands explicitly download public data and model files:

```powershell
.\.venv\Scripts\python.exe run.py prepare ami --limit 100
.\.venv\Scripts\python.exe run.py prepare mmqa --limit 100 --seed 42
.\.venv\Scripts\python.exe run.py fetch-media --dataset data/prepared/mmqa --max-mb 150
.\.venv\Scripts\python.exe scripts/setup_local_gemma.py
.\.venv\Scripts\python.exe run.py serve
```

Open http://127.0.0.1:8765. Local Gemma starts only when an answer is requested.
Its setup downloads approximately 5 GB, plus extraction space; it is not bundled.
EmbeddingGemma weights download separately on first dense retrieval. Follow
README.md for Daily-Omni's larger optional download, silent-video preparation,
retrieval comparisons, and answer evaluation. See docs/LOCAL_MODEL.md for audio
and video limits. No cloud API or paid service is needed for the local path.
To use an optional hosted comparison, create .env locally from .env.example and
add your own key there. Never place a key in source, reports, or this archive.

See docs/DEMO_GUIDE.md for the public 20-second AV demonstration. Its optional
observed response snapshot is results/demo-av-live.json, the single public demo
answer allowlisted by this packager. It is an integration demonstration, not a
scored benchmark condition; source media must still be prepared locally.

## Reproduce all six measured answer conditions in a fresh directory

Prepare the matching public AV pool and the explicitly boundary-normalized
subset. This downloads several GB of public media. The last command may remove
at most 1 ms of terminal PCM WAV samples from clips just above the 30-second
audio limit; its manifest records every change and preserves original files.

```powershell
.\.venv\Scripts\python.exe run.py prepare daily-omni --limit 100 --seed 42
.\.venv\Scripts\python.exe run.py fetch-media --dataset data/prepared/daily-omni --max-mb 4500
.\.venv\Scripts\python.exe run.py prepare-video-only --dataset data/prepared/daily-omni --output data/media/daily_omni/silent
.\.venv\Scripts\python.exe scripts/prepare_local_av_eval.py --normalize-audio-rounding --seed 42
```

Rebuild retrieval rankings against this machine's prepared corpus, then run the
fixed local answer comparisons and analyze their completed predictions:

```powershell
.\.venv\Scripts\python.exe run.py compare --dataset data/prepared/mmqa --methods bm25 dense-text multimodal --device cuda --batch-size 1 --output results/reproduced/mmqa-comparison
.\.venv\Scripts\python.exe scripts/run_final_evaluations.py --output-root results/reproduced --retrieval-root results/reproduced/mmqa-comparison
.\.venv\Scripts\python.exe scripts/analyze_final_results.py --runs-root results/reproduced --output results/reproduced/final-analysis
```

Archived reports are measured snapshots, not resumable runs on new paths.
Absolute source paths contribute to corpus fingerprints; file path rebasing in
this archive does not change the original recorded fingerprint or measured
values. Fresh preparation and ranking generation create new matching manifests.
The analysis checks each experiment's paired runs, recomputes saved metrics, and
requires all six conditions to complete before publishing comparisons. Separate
historical MMQA and AV implementations are retained in their own run manifests;
no cross-task accuracy difference is estimated. Existing measured files remain
unchanged under their original results folders.

## Verify or regenerate the package

Verify a received ZIP without extracting it:

```powershell
python scripts/package_submission.py --verify path/to/ProjectMemory-submission.zip
```

Preview the current allowlisted snapshot without writing anything:

```powershell
python scripts/package_submission.py
```

After finalizing source, reports, and documentation, create a fresh snapshot:

```powershell
python scripts/package_submission.py --write
```

The package inventory establishes file integrity, not correctness of model
answers or completion of unfinished evaluation conditions. Read the reports'
sample sizes, failures, abstentions, and limitations before comparing scores.
"""


def snapshot(root: Path) -> tuple[dict[str, bytes], dict]:
    root = root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "run.py").is_file():
        raise PackageError("Project root must contain pyproject.toml and run.py")
    payloads, inventory = {}, []
    total = 0
    for path in selected_files(root):
        payload, record = _payload(root, path)
        total += len(payload)
        if total > MAX_TOTAL_BYTES:
            raise PackageError(f"Submission inputs exceed the {MAX_TOTAL_BYTES:,}-byte total limit")
        payloads[record["path"]] = payload
        inventory.append(record)
    readme = submission_readme()
    payloads["SUBMISSION_README.md"] = readme
    inventory.append({"path": "SUBMISSION_README.md", "bytes": len(readme), "sha256": _hash(readme),
                      "source_bytes": None, "source_sha256": None, "transformations": ["generated packaging instructions"]})
    manifest = {"schema_version": 1, "package": "ProjectMemory submission", "files": sorted(inventory, key=lambda item: item["path"]),
                "policy": {"selection": "explicit source/docs/config and public-result allowlists", "max_file_bytes": MAX_FILE_BYTES,
                           "max_total_input_bytes": MAX_TOTAL_BYTES, "result_paths": "workspace-relative; external local paths omitted",
                           "excluded": [".env", "credentials", "data", "media", "model weights", "caches", "virtual environments", "unlisted files"]}}
    payloads["SUBMISSION_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    payloads["SHA256SUMS.txt"] = "".join(f"{_hash(data)}  {name}\n" for name, data in sorted(payloads.items())).encode()
    return payloads, manifest


def verify_archive(path: Path) -> dict:
    """Check integrity and member containment without extracting any file."""
    try:
        if path.stat().st_size > MAX_TOTAL_BYTES + MAX_FILE_BYTES:
            raise PackageError("Archive exceeds the compressed size limit")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise PackageError("Archive contains duplicate member names")
            for info in infos:
                member = PurePosixPath(info.filename)
                if (member.is_absolute() or ".." in member.parts or "\\" in info.filename or ":" in info.filename
                        or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)):
                    raise PackageError("Archive contains an unsafe member")
                if info.file_size > MAX_FILE_BYTES:
                    raise PackageError("Archive member exceeds the size limit")
            if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES + MAX_FILE_BYTES:
                raise PackageError("Archive exceeds the total size limit")
            checks = {}
            for line in archive.read("SHA256SUMS.txt").decode().splitlines():
                digest, name = line.split("  ", 1)
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in checks:
                    raise PackageError("Archive checksum inventory is invalid")
                checks[name] = digest
            if set(checks) != set(names) - {"SHA256SUMS.txt"}:
                raise PackageError("Archive inventory does not match its members")
            for name, expected in checks.items():
                if _hash(archive.read(name)) != expected:
                    raise PackageError(f"Archive checksum mismatch: {name}")
            manifest = json.loads(archive.read("SUBMISSION_MANIFEST.json"))
            records = manifest["files"]
            if len(records) != len({record["path"] for record in records}) or {record["path"] for record in records} != set(names) - {"SUBMISSION_MANIFEST.json", "SHA256SUMS.txt"}:
                raise PackageError("Archive manifest does not match its members")
            for record in records:
                if record["sha256"] != checks[record["path"]] or record["bytes"] != archive.getinfo(record["path"]).file_size:
                    raise PackageError("Archive manifest checksum or size mismatch")
            return {"verified": True, "files": len(names), "uncompressed_bytes": sum(info.file_size for info in infos), "sha256": _hash(path.read_bytes())}
    except (OSError, KeyError, TypeError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError("Could not verify the submission archive") from None


def write_archive(root: Path, payloads: dict[str, bytes], name: str = "ProjectMemory-submission.zip") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.zip", name):
        raise PackageError("Archive name must be a simple .zip filename")
    root = root.resolve()
    dist = root / "dist"
    if dist.exists() and _is_link(dist):
        raise PackageError("Refusing a linked dist directory")
    dist.mkdir(exist_ok=True)
    if dist.resolve().parent != root:
        raise PackageError("Archive output leaves the project directory")
    output = dist / name
    if output.is_symlink():
        raise PackageError("Refusing a linked archive output")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dist, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for member, payload in sorted(payloads.items()):
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        verify_archive(temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write and verify a submission ZIP; default is dry run")
    parser.add_argument("--name", default="ProjectMemory-submission.zip", help="filename inside the project's dist directory")
    parser.add_argument("--verify", type=Path, help="verify an existing ZIP without extracting it")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if args.write:
                parser.error("--verify and --write cannot be combined")
            print(json.dumps(verify_archive(args.verify), indent=2))
            return 0
        payloads, manifest = snapshot(PROJECT_ROOT)
        print(f"Selected {len(manifest['files'])} files plus 2 inventory files; {sum(map(len, payloads.values())):,} uncompressed bytes.")
        for member in payloads:
            print(member)
        if args.write:
            output = write_archive(PROJECT_ROOT, payloads, args.name)
            print(f"Written and verified: {output}")
            print(f"SHA256: {_hash(output.read_bytes())}")
        else:
            print("Dry run only. Use --write after the reports and documentation are final.")
        return 0
    except PackageError as exc:
        print(f"Packaging stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
