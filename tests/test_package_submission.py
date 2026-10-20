"""Submission boundaries and integrity, using tiny synthetic project fixtures."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


_SPEC = importlib.util.spec_from_file_location("package_submission", Path(__file__).resolve().parents[1] / "scripts/package_submission.py")
package = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(package)


def put(root, name, content="placeholder"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path):
    put(tmp_path, "pyproject.toml", '[project]\nname="fixture"\n')
    put(tmp_path, "run.py", 'print("fixture")\n')
    put(tmp_path, "README.md", "# Fixture\n")
    put(tmp_path, ".env.example", "GEMINI_API_KEY=\nPROJECTMEMORY_PROVIDER=local\n")
    put(tmp_path, "src/projectmemory/main.py", 'ANSWER = "blue"\n')
    return tmp_path


def test_allowlist_excludes_credentials_datasets_caches_upload_answers_and_oversized_unlisted_files(project):
    included = ["src/projectmemory/datasets/ami.py", "tests/test_main.py", "scripts/setup.py", "docs/REPORT.md"]
    for name in included:
        put(project, name)
    result_names = ["results/answers-mmqa-gemini-bm25-5/report.json", "results/final-mmqa-local-bm25/predictions.jsonl",
                    "results/final-local-av-combined/run.json", "results/final-local-av-combined/source_manifest.json",
                    "results/final-mmqa-local-dense-text/retrieval_source.json", "results/final-analysis/analysis.json",
                    "results/demo-av-live.json", "docs/TESTED_ENVIRONMENT.json"]
    for name in result_names:
        put(project, name, '{"score": 0.8}\n')
    (project / "results/demo-av-browser.jpg").write_bytes(b"\xff\xd8public demo screenshot\xff\xd9")
    excluded = [".env", ".env.backup", "data/prepared/mmqa/corpus.jsonl", "data/uploads/private.txt",
                ".cache/local-gemma4/model.gguf", ".venv/lib/secret.py", "src/projectmemory/__pycache__/module.py",
                "docs/.env", "docs/credentials.json", "docs/private.pem", "scripts/.private.py", "results/upload-answer.json",
                "results/av-embedding-smoke.corpus.jsonl", "results/final-private/report.json", "results/final-analysis/raw.json",
                "results/private-browser.jpg", "results/demo-upload.jpg",
                "results/final-mmqa-local-bm25/corpus.jsonl", "results/final-mmqa-local-bm25/private.mp4"]
    for name in excluded:
        put(project, name, "private input must never be read")
    put(project, "data/downloads/bulk.json", "X" * (package.MAX_FILE_BYTES + 1))
    payloads, _ = package.snapshot(project)
    assert set(included + result_names).issubset(payloads)
    assert "results/demo-av-browser.jpg" in payloads
    assert not set(excluded).intersection(payloads)
    assert ".env.example" in payloads
    assert "scripts/package_submission.py" not in payloads  # fixture is not the live project


def test_vendor_credential_in_whitelisted_file_stops_without_echoing_secret(project):
    token = "nvapi-" + "Q7mN2z" * 8
    put(project, "docs/REPORT.md", "Accidental key: " + token)
    with pytest.raises(package.PackageError) as caught:
        package.snapshot(project)
    assert token not in str(caught.value)
    assert "docs/REPORT.md" in str(caught.value)
    assert not (project / "dist").exists()


@pytest.mark.parametrize("name,contents", [
    (".env.example", "GEMINI_API_KEY=filled-example-is-not-allowed\n"),
    ("results/final-analysis/analysis.json", '{"request":{"api_key":"non-vendor-secret"}}'),
])
def test_populated_credential_fields_are_rejected_even_without_vendor_prefix(project, name, contents):
    put(project, name, contents)
    with pytest.raises(package.PackageError, match="credential|Credential"):
        package.snapshot(project)


def test_json_paths_are_portable_but_original_hashes_and_scores_are_preserved(project):
    original = {"accuracy": 0.6, "corpus_sha256": "a" * 64,
                "source": str(project / "data" / "corpus.jsonl"), "external": "C:/Users/person/private/file.wav"}
    source = put(project, "results/final-analysis/analysis.json", json.dumps(original))
    payloads, manifest = package.snapshot(project)
    portable = json.loads(payloads["results/final-analysis/analysis.json"])
    assert portable["source"] == "data/corpus.jsonl"
    assert portable["external"] == "<external local path omitted>"
    assert portable["accuracy"] == original["accuracy"] and portable["corpus_sha256"] == original["corpus_sha256"]
    record = next(item for item in manifest["files"] if item["path"] == "results/final-analysis/analysis.json")
    assert record["source_sha256"] == package._hash(source.read_bytes())
    assert record["sha256"] == package._hash(payloads[record["path"]])
    assert record["source_sha256"] != record["sha256"]
    assert record["transformations"] == ["local paths rebased or omitted"]
    assert json.loads(source.read_text()) == original


def test_archive_is_deterministic_verified_and_contains_reproduction_instructions(project):
    payloads, manifest = package.snapshot(project)
    first = package.write_archive(project, payloads, "first.zip")
    second = package.write_archive(project, payloads, "second.zip")
    assert first.read_bytes() == second.read_bytes()
    check = package.verify_archive(first)
    assert check["verified"] and check["files"] == len(manifest["files"]) + 2
    with zipfile.ZipFile(first) as archive:
        guide = archive.read("SUBMISSION_README.md").decode()
        assert "scripts/setup_local_gemma.py" in guide and "run.py prepare mmqa" in guide
        assert "NEW results directory" in guide
        assert "Archived reports are measured snapshots, not resumable runs on new paths" in guide
        assert "--output-root results/reproduced --retrieval-root results/reproduced/mmqa-comparison" in guide
        assert "--runs-root results/reproduced --output results/reproduced/final-analysis" in guide
        assert "--normalize-audio-rounding --seed 42" in guide
        assert '".[embeddings,ingestion,analysis,dev]"' in guide
        assert "pip install matplotlib" not in guide
        assert "results/demo-av-live.json" in guide and "demo-av-answer.json" not in guide
        assert "SHA256SUMS.txt" in archive.namelist()


def test_default_cli_is_dry_run(project, monkeypatch, capsys):
    monkeypatch.setattr(package, "PROJECT_ROOT", project)
    assert package.main([]) == 0
    assert "Dry run only" in capsys.readouterr().out
    assert not (project / "dist").exists()


def test_tampered_archive_is_detected(project):
    payloads, _ = package.snapshot(project)
    payloads["run.py"] = b"tampered source"
    with pytest.raises(package.PackageError, match="checksum mismatch"):
        package.write_archive(project, payloads)
    assert not (project / "dist/ProjectMemory-submission.zip").exists()
    assert not list((project / "dist").iterdir())


@pytest.mark.parametrize("name", ["../outside.zip", "C:/outside.zip", "nested/archive.zip", "archive.txt"])
def test_output_name_cannot_leave_dist(project, name):
    payloads, _ = package.snapshot(project)
    with pytest.raises(package.PackageError, match="simple .zip"):
        package.write_archive(project, payloads, name)


def test_linked_source_outside_project_is_rejected_without_reading_target(project, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-external.txt")
    outside.write_text("private outside content")
    link = project / "src/projectmemory/link.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(package.PackageError, match="linked submission input"):
        package.snapshot(project)


def test_safe_file_rejects_direct_external_input(project, tmp_path):
    with pytest.raises(package.PackageError, match="leaves the project"):
        package._safe_file(project, project.parent / "outside.py")


def test_archive_member_traversal_is_rejected_without_extracting(project):
    path = project / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(package.PackageError, match="unsafe member"):
        package.verify_archive(path)
    assert not (project.parent / "escape.txt").exists()


def test_oversized_allowlisted_file_fails_instead_of_silently_omitting_it(project, monkeypatch):
    put(project, "docs/REPORT.md", "X" * 101)
    monkeypatch.setattr(package, "MAX_FILE_BYTES", 100)
    with pytest.raises(package.PackageError, match="exceeds"):
        package.snapshot(project)
