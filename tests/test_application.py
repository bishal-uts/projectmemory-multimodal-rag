from pathlib import Path

import pytest

from projectmemory.application import ask_application_question, select_local_evidence


def media(tmp_path, identity, modality, start=0, duration=20, source="meeting"):
    path = tmp_path / (identity + (".wav" if modality == "audio" else ".mp4"))
    path.write_bytes(b"fixture")
    return {"id": identity, "source_id": source, "modality": modality, "path": str(path),
            "metadata": {"start_seconds": start, "end_seconds": start + duration}}


def test_local_selection_pairs_and_limits_whole_segments(tmp_path, monkeypatch):
    rows = [media(tmp_path, kind + str(start), kind, start) for start in (0, 20) for kind in ("video", "audio")]
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: {"duration_seconds": 20})
    chosen, plan = select_local_evidence(rows, [rows[0], rows[2], rows[3]])
    assert [row["id"] for row in chosen] == ["video0", "audio0"]
    assert plan["added_ids"] == ["audio0"]
    assert {row["id"] for row in plan["omitted"]} == {"video20", "audio20"}
    assert plan["totals"]["audio_seconds"] == 20
    assert "not cropped" in plan["notice"]


def test_actual_duration_overrides_metadata_and_rejects_empty_selection(tmp_path, monkeypatch):
    row = media(tmp_path, "audio", "audio", duration=1)
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: {"duration_seconds": 31})
    with pytest.raises(ValueError, match="None of the retrieved"):
        select_local_evidence([row], [row])


def test_delayed_audio_pairs_by_ingestion_segment_without_changing_timestamps(tmp_path, monkeypatch):
    video = media(tmp_path, "v", "video")
    audio = media(tmp_path, "a", "audio", start=.7, duration=19.3)
    video["id"] = "av:" + "a" * 20 + ":video:000000"
    audio["id"] = "av:" + "a" * 20 + ":audio:000000"
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: {"duration_seconds": 20})
    chosen, plan = select_local_evidence([video, audio], [video])
    assert [row["modality"] for row in chosen] == ["video", "audio"]
    assert chosen[1]["metadata"]["start_seconds"] == .7
    assert plan["added_ids"] == [audio["id"]]


def test_pairing_does_not_cross_source_or_interval(tmp_path, monkeypatch):
    rows = [media(tmp_path, "v", "video"), media(tmp_path, "other", "audio", source="other"),
            media(tmp_path, "later", "audio", start=20)]
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: {"duration_seconds": 20})
    chosen, plan = select_local_evidence(rows, rows[:1])
    assert [row["id"] for row in chosen] == ["v"]
    assert plan["added_ids"] == []


def test_application_generator_and_display_use_same_selected_ids(tmp_path, monkeypatch):
    rows = [media(tmp_path, "v", "video"), media(tmp_path, "a", "audio")]
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: {"duration_seconds": 20})
    class Retriever:
        def search(self, query, top_k, candidate_ids):
            return [{"id": "v", "score": 1}]
    class Generator:
        provider = "local"
        def answer(self, question, evidence):
            assert [row["id"] for row in evidence] == ["v", "a"]
            return {"answer": "fixture", "citations": ["a"]}
    result = ask_application_question(rows, "Question?", Retriever(), Generator())
    assert result["retrieved_ids"] == ["v"]
    assert [row["id"] for row in result["evidence"]] == ["v", "a"]
    assert result["evidence_plan"]["added_ids"] == ["a"]


def test_search_and_hosted_generation_keep_original_rankings(tmp_path, monkeypatch):
    rows = [media(tmp_path, "v", "video"), media(tmp_path, "a", "audio")]
    monkeypatch.setattr("projectmemory.application.probe_media", lambda path: pytest.fail("No budget selection"))
    class Retriever:
        def search(self, query, top_k, candidate_ids):
            return [{"id": "v", "score": 1}]
    class Generator:
        provider = "gemini"
        def answer(self, question, evidence):
            assert [row["id"] for row in evidence] == ["v"]
            return {}
    for generator in (None, Generator()):
        result = ask_application_question(rows, "Question?", Retriever(), generator)
        assert [row["id"] for row in result["evidence"]] == ["v"]
        assert "evidence_plan" not in result
