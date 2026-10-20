import os
import sys
from types import SimpleNamespace
import pytest
from projectmemory.settings import load_project_env


def test_env_only_reads_known_settings_without_evaluating(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "existing")
    original_path = os.environ.get("PATH")
    path = tmp_path / ".env"
    path.write_text('NVIDIA_API_KEY="literal$(no_command)"\nGEMINI_API_KEY=other\nPATH=bad\n', encoding="utf-8")
    load_project_env(path)
    assert os.environ["NVIDIA_API_KEY"] == "literal$(no_command)"
    assert os.environ["GEMINI_API_KEY"] == "existing"
    assert os.environ.get("PATH") == original_path


def test_invalid_env_does_not_echo_secret(tmp_path):
    path = tmp_path / ".env"
    path.write_text('NVIDIA_API_KEY="secret-secret', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_project_env(path)
    assert "secret-secret" not in str(error.value)


def test_local_default_needs_no_key_but_reports_missing_install(monkeypatch):
    from projectmemory.settings import configured_provider, key_available, provider_status
    monkeypatch.delenv("PROJECTMEMORY_PROVIDER", raising=False)
    monkeypatch.setitem(sys.modules, "projectmemory.localruntime", SimpleNamespace(
        status=lambda root=None: {"installed": False, "available": False, "running": False,
                                  "detail": "Local model files are missing"}))
    assert configured_provider() == "local"
    assert key_available("local") is True
    status = provider_status("local")
    assert status["requires_api_key"] is False
    assert status["configured"] is False
    assert status["installed"] is False


def test_factory_keeps_model_override_with_its_provider(monkeypatch):
    from projectmemory.settings import create_generator, DEFAULT_MODELS
    monkeypatch.setattr("projectmemory.settings.load_project_env", lambda: None)
    monkeypatch.setenv("PROJECTMEMORY_PROVIDER", "local")
    monkeypatch.setenv("PROJECTMEMORY_MODEL", "local-custom-model")
    monkeypatch.setitem(sys.modules, "projectmemory.local_generation", SimpleNamespace(
        LocalGemmaGenerator=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "projectmemory.generation", SimpleNamespace(
        GeminiGenerator=lambda **kwargs: kwargs))
    assert create_generator()["model"] == "local-custom-model"
    assert create_generator("gemini")["model"] == DEFAULT_MODELS["gemini"]
    assert create_generator("gemini", "explicit-model")["model"] == "explicit-model"


def test_cli_accepts_local_generation_provider():
    from projectmemory.cli import build_parser
    args = build_parser().parse_args(["ask", "--dataset", "sample", "--provider", "local", "What happened?"])
    assert args.provider == "local"
