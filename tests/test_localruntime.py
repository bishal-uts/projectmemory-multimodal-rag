"""Runtime ownership and integrity checks without starting a model or GPU."""
import hashlib
import json
from unittest.mock import Mock

import pytest

from projectmemory import localruntime


@pytest.fixture
def installed(tmp_path, monkeypatch):
    monkeypatch.setattr(localruntime, "_PROCESSES", {})
    monkeypatch.setattr(localruntime, "_LOADED_MANIFESTS", {})
    monkeypatch.setattr(localruntime, "_VERIFIED", set())
    base = tmp_path / ".cache" / "local-gemma4"
    base.mkdir(parents=True)
    config = {"schema_version": 1, "model": localruntime.MODEL,
              "model_alias": localruntime.ALIAS, "base_url": localruntime.BASE_URL,
              "context_size": 16384, "gpu_layers": 99,
              "batch_size": 256, "ubatch_size": 128, "dll_directories": []}
    for role in ("model", "mmproj", "server"):
        content = ("fixture-" + role).encode()
        filename = role + ".bin"
        (base / filename).write_bytes(content)
        config[role + "_file"] = filename
        config[role + "_sha256"] = hashlib.sha256(content).hexdigest()
    (base / "runtime.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, base, config


def fake_port(monkeypatch, occupied=False):
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    probe.connect_ex.return_value = 0 if occupied else 1
    monkeypatch.setattr(localruntime.socket, "socket", Mock(return_value=probe))
    return probe


def test_installed_status_never_launches_or_contacts_a_server(installed, monkeypatch):
    root, _, _ = installed
    popen = Mock(side_effect=AssertionError("Status must not launch"))
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    monkeypatch.setattr(localruntime, "_healthy", Mock(side_effect=AssertionError("No probe")))
    assert localruntime.status(root)["available"]
    assert not localruntime.status(root)["running"]
    popen.assert_not_called()


def test_unknown_process_on_port_is_neither_attached_to_nor_stopped(installed, monkeypatch):
    root, _, _ = installed
    fake_port(monkeypatch, occupied=True)
    popen = Mock(side_effect=AssertionError("Unknown port owner must not be replaced"))
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="already in use by another process"):
        localruntime.ensure_running(root)
    assert not localruntime._PROCESSES
    localruntime.stop(root)
    popen.assert_not_called()


def test_changed_model_is_rejected_before_launch(installed, monkeypatch):
    root, base, config = installed
    (base / config["model_file"]).write_bytes(b"changed model")
    popen = Mock(side_effect=AssertionError("Unverified runtime must not launch"))
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="hash mismatch"):
        localruntime.ensure_running(root)
    popen.assert_not_called()


def test_manifest_paths_cannot_escape_runtime_directory(installed):
    root, base, config = installed
    config["model_file"] = "../../outside.gguf"
    (base / "runtime.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="stay inside"):
        localruntime.ensure_running(root)
    assert not localruntime.status(root)["available"]


def test_launch_is_loopback_and_strips_provider_credentials(installed, monkeypatch):
    root, _, _ = installed
    fake_port(monkeypatch)
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    monkeypatch.setattr(localruntime, "_healthy", lambda: True)
    for name in ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLAMA_ARG_RPC", "LLAMA_ARG_HF_REPO"):
        monkeypatch.setenv(name, "credential-sentinel")
    assert localruntime.ensure_running(root) == localruntime.BASE_URL
    command = popen.call_args.args[0]
    kwargs = popen.call_args.kwargs
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8766"
    assert "--offline" in command
    assert "credential-sentinel" not in str((command, kwargs["env"]))
    assert kwargs["stdin"] is localruntime.subprocess.DEVNULL
    assert localruntime._PROCESSES[root] is process
    # A healthy project-owned process is reused instead of spawning a second.
    assert localruntime.ensure_running(root) == localruntime.BASE_URL
    assert popen.call_count == 1
    localruntime.stop(root)
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=10)
    assert not localruntime._PROCESSES


def test_stop_only_touches_process_registered_for_requested_project(installed):
    root, _, _ = installed
    owned, other = Mock(), Mock()
    owned.poll.return_value = other.poll.return_value = None
    localruntime._PROCESSES[root] = owned
    localruntime._PROCESSES[root / "other"] = other
    localruntime.stop(root)
    owned.terminate.assert_called_once()
    other.terminate.assert_not_called()
    assert localruntime._PROCESSES[root / "other"] is other


def test_dead_owned_process_does_not_hide_startup_failure(installed, monkeypatch):
    root, _, _ = installed
    fake_port(monkeypatch)
    process = Mock()
    process.poll.return_value = 1
    monkeypatch.setattr(localruntime.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(localruntime, "_healthy", Mock(side_effect=AssertionError("Dead process")))
    with pytest.raises(RuntimeError, match="could not start"):
        localruntime.ensure_running(root)
    assert not localruntime.status(root)["running"]


@pytest.mark.parametrize("invalid", [[], None, "runtime", 42])
def test_status_handles_non_object_manifests_as_unavailable(installed, invalid):
    root, base, _ = installed
    (base / "runtime.json").write_text(json.dumps(invalid))
    report = localruntime.status(root)
    assert not report["available"] and not report["running"]
    with pytest.raises(ValueError, match="Unexpected local Gemma runtime manifest"):
        localruntime.ensure_running(root)


def test_changed_manifest_restarts_owned_process_before_reuse(installed, monkeypatch):
    root, base, config = installed
    fake_port(monkeypatch)
    first, second = Mock(), Mock()
    first.poll.return_value = second.poll.return_value = None
    popen = Mock(side_effect=[first, second])
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    monkeypatch.setattr(localruntime, "_healthy", lambda: True)
    localruntime.ensure_running(root)
    first_identity = localruntime._LOADED_MANIFESTS[root]
    config["context_size"] = 8192
    (base / "runtime.json").write_text(json.dumps(config))
    localruntime.ensure_running(root)
    assert popen.call_count == 2
    first.terminate.assert_called_once()
    first.wait.assert_called_once_with(timeout=10)
    second.terminate.assert_not_called()
    command = popen.call_args.args[0]
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert localruntime._PROCESSES[root] is second
    assert localruntime._LOADED_MANIFESTS[root] != first_identity


def test_changed_file_hash_is_checked_even_when_process_is_healthy(installed, monkeypatch):
    root, base, config = installed
    fake_port(monkeypatch)
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(localruntime.subprocess, "Popen", popen)
    monkeypatch.setattr(localruntime, "_healthy", lambda: True)
    localruntime.ensure_running(root)
    (base / config["mmproj_file"]).write_bytes(b"changed projector")
    with pytest.raises(ValueError, match="hash mismatch"):
        localruntime.ensure_running(root)
    assert popen.call_count == 1
