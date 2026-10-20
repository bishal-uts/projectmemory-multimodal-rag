import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from projectmemory.generation import GeminiGenerator, GenerationError
from projectmemory.nvidia import NvidiaGenerator, DEFAULT_NVIDIA_MODEL


EVIDENCE = [{"id": "doc:1", "modality": "text", "title": "Meeting", "text": "The decision was blue.",
             "reference_answers": ["LEAK_SENTINEL"], "answer_label": "LEAK_LABEL",
             "metadata": {"gold": "LEAK_METADATA"}}]
ANSWER = {"answer": "Blue.", "answer_items": ["blue"], "citations": ["doc:1"],
          "abstained": False, "answer_label": ""}


def gemini_response(answer=None):
    return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(answer or ANSWER)}]}}],
            "modelVersion": "gemini-3.8-flash", "usageMetadata": {"totalTokenCount": 10}}


def nvidia_response(answer=None):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer or ANSWER),
                                                               "reasoning": "DO_NOT_STORE_REASONING"}}],
            "model": DEFAULT_NVIDIA_MODEL, "usage": {"total_tokens": 10}}


class FakeResponse(io.BytesIO):
    def __init__(self, payload, headers=None, status=200):
        super().__init__(json.dumps(payload).encode() if payload is not None else b"")
        self.headers = headers or {}
        self.status = status


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize("cls,response", [(GeminiGenerator, gemini_response), (NvidiaGenerator, nvidia_response)])
def test_grounded_request_filters_gold_and_keeps_key_only_in_header(tmp_path, cls, response):
    generator = cls(api_key="secret-example", cache_dir=tmp_path)
    opener = generator._opener = FakeOpener([FakeResponse(response())])
    result = generator.answer("What colour was chosen?", EVIDENCE)
    request = opener.requests[0]
    payload = request.data.decode()
    assert "LEAK_" not in payload
    assert "secret-example" not in payload + request.full_url
    assert "secret-example" in str(dict(request.header_items()))
    assert "only the provided evidence" in payload
    assert result["citations"] == ["doc:1"]
    assert result["sources"][0]["title"] == "Meeting"
    assert "DO_NOT_STORE_REASONING" not in next(tmp_path.glob("*.json")).read_text()
    second = generator.answer("What colour was chosen?", EVIDENCE)
    assert second["cache_hit"] and not second["api_called"] and len(opener.requests) == 1


@pytest.mark.parametrize("cls", [GeminiGenerator, NvidiaGenerator])
def test_missing_key_and_empty_evidence(monkeypatch, cls):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    generator = cls(cache_dir=None)
    assert generator.answer("Question?", [])["abstained"]
    with pytest.raises(GenerationError, match="API_KEY"):
        generator.answer("Question?", EVIDENCE)


@pytest.mark.parametrize("cls,response", [(GeminiGenerator, gemini_response), (NvidiaGenerator, nvidia_response)])
def test_rejects_invented_citation_and_corrupt_cache(tmp_path, cls, response):
    generator = cls(api_key="secret", cache_dir=tmp_path, **({"format_retries": 0} if cls is NvidiaGenerator else {}))
    generator._opener = FakeOpener([FakeResponse(response({**ANSWER, "citations": ["invented"]}))])
    with pytest.raises(GenerationError, match="invented|valid grounded JSON"):
        generator.answer("Question?", EVIDENCE)
    assert not list(tmp_path.glob("*.json"))
    generator._opener = FakeOpener([FakeResponse(response())])
    generator._last_call = None
    generator.answer("Question?", EVIDENCE)
    cache = next(tmp_path.glob("*.json"))
    content = json.loads(cache.read_text())
    content["citations"] = ["invented"]
    cache.write_text(json.dumps(content))
    with pytest.raises(GenerationError, match="Cached answer failed"):
        generator.answer("Question?", EVIDENCE)


@pytest.mark.parametrize("cls,response", [(GeminiGenerator, gemini_response), (NvidiaGenerator, nvidia_response)])
def test_inline_media_cache_tracks_bytes(tmp_path, cls, response):
    path = tmp_path / "image.png"
    path.write_bytes(b"first-image")
    evidence = [{"id": "doc:1", "modality": "image", "path": str(path)}]
    generator = cls(api_key="secret", cache_dir=tmp_path / "cache")
    opener = generator._opener = FakeOpener([FakeResponse(response()), FakeResponse(response())])
    first = generator.answer("Question?", evidence)
    assert "Zmlyc3QtaW1hZ2U=" in opener.requests[0].data.decode()
    assert str(path) not in opener.requests[0].data.decode()
    path.write_bytes(b"second-image")
    generator._last_call = None
    second = generator.answer("Question?", evidence)
    assert first["cache_key"] != second["cache_key"]
    assert len(opener.requests) == 2


@pytest.mark.parametrize("cls", [GeminiGenerator, NvidiaGenerator])
def test_missing_media_and_size_are_not_silently_text_fallback(tmp_path, cls):
    generator = cls(api_key="secret", cache_dir=None, max_inline_bytes=2)
    with pytest.raises(ValueError, match="absolute media path"):
        generator.answer("Question?", [{"id": "x", "modality": "image", "path": "missing.png"}])
    path = tmp_path / "image.png"
    path.write_bytes(b"three")
    if cls is NvidiaGenerator:
        with pytest.raises(ValueError, match="max_file_bytes"):
            generator.answer("Question?", [{"id": "x", "modality": "image", "path": str(path)}])


def test_gemini_files_upload_poll_and_cleanup(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    generator = GeminiGenerator(api_key="secret", cache_dir=None, max_inline_bytes=2)
    generator._pause = lambda _: None
    uri = "https://generativelanguage.googleapis.com/v1beta/files/example"
    opener = generator._opener = FakeOpener([
        FakeResponse(None, {"X-Goog-Upload-URL": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=example"}),
        FakeResponse({"file": {"name": "files/example", "uri": uri, "state": "PROCESSING"}}),
        FakeResponse({"name": "files/example", "uri": uri, "state": "ACTIVE"}),
        FakeResponse(gemini_response()), FakeResponse(None),
    ])
    generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    assert [request.method for request in opener.requests] == ["POST", "POST", "GET", "POST", "DELETE"]
    assert "fileData" in opener.requests[3].data.decode()
    assert "secret" not in str(dict(opener.requests[1].header_items()))


def test_upload_rejects_foreign_redirect_target(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    generator = GeminiGenerator(api_key="secret", cache_dir=None, max_inline_bytes=2)
    opener = generator._opener = FakeOpener([FakeResponse(None, {"X-Goog-Upload-URL": "https://example.com/upload"})])
    with pytest.raises(GenerationError, match="invalid upload endpoint"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "image", "path": str(path)}])
    assert len(opener.requests) == 1


@pytest.mark.parametrize("cls,response", [(GeminiGenerator, gemini_response), (NvidiaGenerator, nvidia_response)])
def test_redacts_errors_and_respects_retry_after(cls, response):
    generator = cls(api_key="secret-example", cache_dir=None, max_retries=1)
    waits = []
    generator._pause = waits.append
    generator._throttle = lambda: None
    generator._opener = FakeOpener([
        HTTPError("https://example/?key=secret-example", 429, "secret-example", {"Retry-After": "3"}, None),
        FakeResponse(response()),
    ])
    generator.answer("Question?", EVIDENCE)
    assert waits == [3.0]
    generator._opener = FakeOpener([URLError("secret-example"), URLError("secret-example")])
    with pytest.raises(GenerationError) as caught:
        generator.answer("A different question?", EVIDENCE)
    assert "secret-example" not in str(caught.value)


def test_choice_and_abstention_validation():
    validate = GeminiGenerator._validate_answer
    assert validate({**ANSWER, "answer_label": "B"}, {"doc:1"}, ["a", "b"])["answer_label"] == "B"
    for bad in ({**ANSWER, "answer_label": "C"}, {**ANSWER, "answer_label": "AB"}, {**ANSWER, "answer_label": ""},
                {**ANSWER, "abstained": True}, {**ANSWER, "citations": []}):
        with pytest.raises(GenerationError):
            validate(bad, {"doc:1"}, ["a", "b"])


def test_gemini_schema_allows_empty_label_without_invalid_empty_enum():
    generator = GeminiGenerator(api_key="secret", cache_dir=None)
    opener = generator._opener = FakeOpener([FakeResponse(gemini_response())])
    assert generator.answer("Question?", EVIDENCE)["answer_label"] == ""
    schema = json.loads(opener.requests[0].data)["generationConfig"]["responseSchema"]
    assert schema["properties"]["answer_label"] == {"type": "STRING"}


def test_nvidia_native_audio_video_and_reasoning_are_not_mixed_with_answer(tmp_path):
    evidence = []
    for modality, suffix in (("audio", ".wav"), ("video", ".mp4")):
        path = tmp_path / (modality + suffix)
        path.write_bytes(modality.encode())
        evidence.append({"id": "doc:1" if modality == "audio" else "doc:2", "modality": modality, "path": str(path)})
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    opener = generator._opener = FakeOpener([FakeResponse(nvidia_response())])
    result = generator.answer("Question?", evidence)
    payload = json.loads(opener.requests[0].data)
    assert {part["type"] for part in payload["messages"][1]["content"]} == {"text", "audio_url", "video_url"}
    assert "response_format" not in payload and "mm_processor_kwargs" not in payload
    assert result["answer"] == "Blue." and "reasoning" not in result


def test_nvidia_polls_pending_response():
    request_id = "ef4c1967-d543-467c-af7e-8c7899d75be8"
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    generator._pause = lambda _: None
    opener = generator._opener = FakeOpener([
        FakeResponse({"reqId": request_id}, {"NVCF-REQID": request_id}, status=202),
        FakeResponse(nvidia_response()),
    ])
    assert generator.answer("Question?", EVIDENCE)["answer"] == "Blue."
    assert opener.requests[1].full_url.endswith("/v1/status/" + request_id)


def test_nvidia_never_uses_google_key(monkeypatch):
    for key in ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "google-secret")
    with pytest.raises(GenerationError, match="NVIDIA_API_KEY"):
        NvidiaGenerator(cache_dir=None).answer("Question?", EVIDENCE)


def test_gemini_upload_is_deleted_when_generation_fails(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    generator = GeminiGenerator(api_key="secret", cache_dir=None, max_inline_bytes=2, max_retries=0)
    uri = "https://generativelanguage.googleapis.com/v1beta/files/example"
    opener = generator._opener = FakeOpener([
        FakeResponse(None, {"X-Goog-Upload-URL": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=example"}),
        FakeResponse({"file": {"name": "files/example", "uri": uri, "state": "ACTIVE"}}),
        HTTPError("https://example", 403, "secret", {}, None),
        FakeResponse(None),
    ])
    with pytest.raises(GenerationError, match="HTTP 403"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    assert opener.requests[-1].method == "DELETE"


def test_long_retry_after_stops_without_hammering_api():
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    opener = generator._opener = FakeOpener([
        HTTPError("https://example", 429, "secret", {"Retry-After": "120"}, None),
    ])
    with pytest.raises(GenerationError, match="retry the command later"):
        generator.answer("Question?", EVIDENCE)
    assert len(opener.requests) == 1


def test_nvidia_repairs_format_once_without_echoing_bad_output():
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    generator._throttle = lambda: None
    bad = nvidia_response()
    bad["choices"][0]["message"]["content"] = "INVALID_RESPONSE_SENTINEL"
    opener = generator._opener = FakeOpener([FakeResponse(bad), FakeResponse(nvidia_response())])
    result = generator.answer("Question?", EVIDENCE)
    assert result["format_retries_used"] == 1 and result["generation_requests"] == 2
    assert result["usage"]["total_tokens"] == 20
    assert "Formatting reminder" in opener.requests[1].data.decode()
    assert "INVALID_RESPONSE_SENTINEL" not in opener.requests[1].data.decode()
    assert "DO_NOT_STORE_REASONING" not in json.dumps(result)


def test_nvidia_format_retry_has_a_hard_limit():
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    generator._throttle = lambda: None
    bad = nvidia_response()
    bad["choices"][0]["message"]["content"] = "bad JSON"
    opener = generator._opener = FakeOpener([FakeResponse(bad), FakeResponse(bad)])
    with pytest.raises(GenerationError, match="valid grounded JSON"):
        generator.answer("Question?", EVIDENCE)
    assert len(opener.requests) == 2


def test_nvidia_video_duration_uses_segment_interval(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    generator = NvidiaGenerator(api_key="secret", cache_dir=None)
    with pytest.raises(ValueError, match="two minutes"):
        generator.answer("Question?", [{"id": "video", "modality": "video", "path": str(path),
                                        "metadata": {"start_seconds": 20, "end_seconds": 150}}])
