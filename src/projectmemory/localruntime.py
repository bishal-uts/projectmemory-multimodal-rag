"""Own a project-local llama.cpp process; never download or stop unrelated apps."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

MODEL = "google/gemma-4-E2B-it-qat-q4_0-gguf"
ALIAS = "gemma-4-e2b"
BASE_URL = "http://127.0.0.1:8766"
_LOCK = threading.RLock()
_PROCESSES: dict[Path, subprocess.Popen] = {}
_LOADED_MANIFESTS: dict[Path, str] = {}
_VERIFIED: set[tuple] = set()


def _root(root=None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def _path(base: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("Local runtime manifest has an invalid file path")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or path == base.resolve():
        raise ValueError("Local runtime files must stay inside .cache/local-gemma4")
    return path


def _manifest(root=None):
    base = _root(root) / ".cache/local-gemma4"
    manifest_path = base / "runtime.json"
    if not manifest_path.is_file():
        raise ValueError("Local Gemma is not installed. Run .\\.venv\\Scripts\\python.exe scripts/setup_local_gemma.py first.")
    config = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("model") != MODEL
            or config.get("model_alias") != ALIAS or config.get("base_url") != BASE_URL):
        raise ValueError("Unexpected local Gemma runtime manifest; rerun scripts/setup_local_gemma.py")
    for key in ("model_file", "mmproj_file", "server_file"):
        if not _path(base, config.get(key)).is_file():
            raise ValueError("Local Gemma runtime files are missing; rerun scripts/setup_local_gemma.py")
    return base, config


def _healthy() -> bool:
    try:
        with build_opener(ProxyHandler({})).open(Request(BASE_URL + "/health"), timeout=1) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


def status(root=None) -> dict:
    project = _root(root)
    try:
        _manifest(project)
    except (OSError, ValueError, KeyError, TypeError):
        return {"installed": False, "running": False, "available": False, "model": MODEL,
                "detail": "Run scripts/setup_local_gemma.py to install the local model and runtime."}
    with _LOCK:
        process = _PROCESSES.get(project)
        running = process is not None and process.poll() is None
    return {"installed": True, "running": running, "available": True, "model": MODEL,
            "detail": "Local CUDA runtime is running." if running else "Installed; starts automatically when generating an answer."}


def _verify(path: Path, expected: str) -> None:
    stat = path.stat()
    stamp = (str(path), stat.st_size, stat.st_mtime_ns, expected)
    if stamp in _VERIFIED:
        return
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Missing local model integrity hash")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Local model/runtime hash mismatch; rerun scripts/setup_local_gemma.py")
    _VERIFIED.add(stamp)


def ensure_running(root=None) -> str:
    """Start the verified files on loopback, lazily, without touching API keys."""
    project = _root(root)
    with _LOCK:
        base, config = _manifest(project)
        identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        paths = {key: _path(base, config[key]) for key in ("model_file", "mmproj_file", "server_file")}
        for key, digest in (("model_file", "model_sha256"), ("mmproj_file", "mmproj_sha256"), ("server_file", "server_sha256")):
            _verify(paths[key], config[digest])
        process = _PROCESSES.get(project)
        if process is not None and _LOADED_MANIFESTS.get(project) != identity:
            stop(project)
            process = None
        if process is not None and process.poll() is None and _healthy():
            return BASE_URL
        if process is not None and process.poll() is None:
            raise RuntimeError("The managed local model is still starting or is unresponsive; retry after it finishes.")
        # Do not attach to or kill an unknown process already using the port.
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", 8766)) == 0:
                raise RuntimeError("Local model port 8766 is already in use by another process. Stop that process before starting this project's model.")
        command = [str(paths["server_file"]), "--model", str(paths["model_file"]),
                   "--mmproj", str(paths["mmproj_file"]), "--alias", ALIAS,
                   "--host", "127.0.0.1", "--port", "8766", "--ctx-size", str(config["context_size"]),
                   "--n-gpu-layers", str(config["gpu_layers"]), "--parallel", "1",
                   "--batch-size", str(config["batch_size"]), "--ubatch-size", str(config["ubatch_size"]),
                   "--flash-attn", "on", "--no-context-shift", "--jinja", "--no-webui", "--offline",
                   "--chat-template-kwargs", '{"enable_thinking":false}', "--log-verbosity", "1"]
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("LLAMA_ARG_"):
                environment.pop(name)
        for name in ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            environment.pop(name, None)
        dll_dirs = [str(_path(base, item)) for item in config.get("dll_directories", [])]
        environment["PATH"] = os.pathsep.join([str(paths["server_file"].parent), *dll_dirs, environment.get("PATH", "")])
        logs = base / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "llama-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=project, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        _PROCESSES[project] = process
        _LOADED_MANIFESTS[project] = identity
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("The local CUDA runtime could not start. See .cache/local-gemma4/logs/llama-server.log.")
            if _healthy():
                return BASE_URL
            time.sleep(0.25)
        stop(project)
        raise RuntimeError("Local model startup exceeded 120 seconds. See .cache/local-gemma4/logs/llama-server.log.")


def stop(root=None) -> None:
    """Release GPU memory by stopping only the subprocess owned by this module."""
    with _LOCK:
        process = _PROCESSES.pop(_root(root), None)
        _LOADED_MANIFESTS.pop(_root(root), None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _cleanup():
    for project in list(_PROCESSES):
        stop(project)


atexit.register(_cleanup)
