"""Read only known project settings; never execute or display .env contents."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODELS = {
    "local": "google/gemma-4-E2B-it-qat-q4_0-gguf",
    "nvidia": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "gemini": "gemini-3.8-flash",
}
ALLOWED_SETTINGS = {"NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY",
                    "GEMINI_API_KEY", "GOOGLE_API_KEY", "PROJECTMEMORY_PROVIDER",
                    "PROJECTMEMORY_MODEL"}


def load_project_env(path: Path | str = ".env") -> None:
    path = Path(path)
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in ALLOWED_SETTINGS:
            continue
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed quoted project setting on .env line {number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if value and not os.environ.get(key):
            os.environ[key] = value


def configured_provider() -> str:
    provider = os.environ.get("PROJECTMEMORY_PROVIDER", "local").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError("PROJECTMEMORY_PROVIDER must be local, nvidia or gemini")
    return provider


def key_available(provider: str) -> bool:
    if provider == "local":
        return True  # Local inference never requires an API key.
    names = ({"nvidia": ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY"),
              "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")})[provider]
    return any(bool(os.environ.get(name)) for name in names)


def provider_status(provider: str, root: Path | None = None) -> dict:
    """Report setup without starting a model, downloading files, or exposing keys."""
    if provider == "local":
        from .localruntime import status
        runtime = status(root=root)
        return {"model": DEFAULT_MODELS[provider], "requires_api_key": False,
                "configured": bool(runtime.get("available")), **runtime}
    return {"model": DEFAULT_MODELS[provider], "requires_api_key": True,
            "configured": key_available(provider)}


def create_generator(provider: str | None = None, model: str | None = None, **kwargs):
    load_project_env()
    provider = provider or configured_provider()
    if provider not in DEFAULT_MODELS:
        raise ValueError("Choose local, nvidia or gemini")
    # A default model override belongs to its configured provider. Selecting a
    # comparison provider must not send that model name to the wrong service.
    model = model or (os.environ.get("PROJECTMEMORY_MODEL") if provider == configured_provider() else None) or DEFAULT_MODELS[provider]
    if provider == "local":
        from .local_generation import LocalGemmaGenerator
        return LocalGemmaGenerator(model=model, **kwargs)
    if provider == "nvidia":
        from .nvidia import NvidiaGenerator
        return NvidiaGenerator(model=model, **kwargs)
    from .generation import GeminiGenerator
    return GeminiGenerator(model=model, **kwargs)
