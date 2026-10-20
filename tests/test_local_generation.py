import base64
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler

from PIL import Image
import pytest

from projectmemory import local_generation as local
from projectmemory.generation import GenerationError
from projectmemory.local_generation import LocalGemmaGenerator, DEFAULT_LOCAL_MODEL, DEFAULT_LOCAL_ALIAS


EVIDENCE = [{"id": "doc:1", "modality": "text", "text": "The selected colour was blue.",
             "reference_answers": ["GOLD_SENTINEL"], "metadata": {"answer": "GOLD_SENTINEL"}}]
ANSWER = {"answer": "Blue.", "answer_items": ["blue"], "citations": ["doc:1"],
          "abstained": False, "answer_label": ""}


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())


class Opener:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def open(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)


def response(answer=ANSWER, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(answer),
              "reasoning_content": "PRIVATE_REASONING_SENTINEL"}}], "usage": {"total_tokens": 11}}


def ready(alias=DEFAULT_LOCAL_ALIAS, modalities=("text", "image", "audio")):
    return [{"status": "ok"}, {"data": [{"id": alias, "architecture": {"input_modalities": list(modalities)}}]}]


@pytest.fixture
def generator(tmp_path):
    manifest = {"model": DEFAULT_LOCAL_MODEL, "model_alias": DEFAULT_LOCAL_ALIAS,
                "model_sha256": "a" * 64, "mmproj_sha256": "b" * 64,
                "runtime_version": "test-build", "model_revision": "test-pinned-revision"}
    (tmp_path / "runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    return LocalGemmaGenerator(project_root=tmp_path, runtime_manifest="runtime.json",
                               cache_dir="cache", autostart=False)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8766", "http://localhost:8766", "http://example.com:8766",
                                  "http://127.0.0.1:8766/v1", "http://127.0.0.1:8766?redirect=evil",
                                  "http://secret@127.0.0.1:8766", "http://127.0.0.1:8766#x", "http://127.1:8766"])
def test_only_numeric_loopback_is_allowed(url):
    with pytest.raises(ValueError, match="numeric loopback"):
        LocalGemmaGenerator(base_url=url)


def test_proxy_credentials_and_gold_are_never_used(generator, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-nvidia")
    monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.invalid:8080")
    # An explicitly empty ProxyHandler overrides environment proxies. urllib
    # omits the inactive handler from its final handler list.
    rebuilt = LocalGemmaGenerator(project_root=generator.project_root, runtime_manifest="runtime.json", autostart=False)
    assert not any(isinstance(handler, ProxyHandler) and handler.proxies for handler in rebuilt._opener.handlers)
    opener = generator._opener = Opener(ready() + [response()])
    first = generator.answer("Which colour?", EVIDENCE)
    request = opener.requests[-1]
    payload = json.loads(request.data)
    all_sent = request.full_url + str(request.header_items()) + request.data.decode()
    assert "secret-" not in all_sent and "GOLD_SENTINEL" not in all_sent
    assert not any(name.lower() == "authorization" for name, _ in request.header_items())
    assert payload["model"] == DEFAULT_LOCAL_ALIAS
    # Match b11429 server-common.cpp's reader, not its outdated flat README
    # example: a misplaced schema silently permits arbitrary JSON objects.
    wrapper = payload["response_format"]
    assert wrapper["type"] == "json_schema"
    assert "schema" not in wrapper
    schema = wrapper["json_schema"]["schema"]
    assert schema["properties"]["citations"]["items"]["enum"] == ["doc:1"]
    assert schema["properties"]["citations"]["maxItems"] == 1
    assert payload["reasoning_effort"] == "none"
    assert first["provider"] == "local" and first["local_inference"]
    second = generator.answer("Which colour?", EVIDENCE)
    assert second["cache_hit"] and not second["api_called"] and not second["local_inference"]
    assert len(opener.requests) == 3
    assert "PRIVATE_REASONING_SENTINEL" not in next(generator.cache_dir.glob("*.json")).read_text()


def test_empty_evidence_needs_neither_runtime_nor_manifest(tmp_path):
    generator = LocalGemmaGenerator(project_root=tmp_path, autostart=False)
    assert generator.answer("Question?", [])["abstained"]
    with pytest.raises(GenerationError, match="manifest"):
        generator.answer("Question?", EVIDENCE)


def test_runtime_identity_changes_invalidate_cached_answer(generator):
    opener = generator._opener = Opener(ready() + [response()] + ready() + [response()])
    first = generator.answer("Question?", EVIDENCE)
    manifest = json.loads(generator.runtime_manifest.read_text())
    manifest["mmproj_sha256"] = "c" * 64
    generator.runtime_manifest.write_text(json.dumps(manifest))
    second = generator.answer("Question?", EVIDENCE)
    assert first["cache_key"] != second["cache_key"]
    assert len(opener.requests) == 6


def test_evaluation_config_freezes_runtime_media_and_generation_parameters(generator, monkeypatch):
    monkeypatch.setattr(local.av, "_tool", lambda name: name)
    monkeypatch.setattr(local.av, "_run", lambda command, timeout: "ffmpeg test-build\n")
    first = generator.evaluation_config()
    generator.max_output_tokens = 123
    second = generator.evaluation_config()
    assert first["generation_parameters"]["max_tokens"] == 4096
    assert second["generation_parameters"]["max_tokens"] == 123
    assert second["media_protocol"]["max_audio_seconds_per_request"] == 30
    assert second["media_protocol"]["ffmpeg_version"] == "ffmpeg test-build"
    assert len(second["runtime_manifest_sha256"]) == 64
    assert len(second["adapter_sha256"]) == 64


@pytest.mark.parametrize("model_info", [ready("wrong-model"), ready(modalities=())])
def test_wrong_model_or_capability_is_rejected_before_evidence_is_sent(generator, model_info):
    opener = generator._opener = Opener(model_info)
    with pytest.raises(GenerationError, match="alias|encoder"):
        generator.answer("Question?", EVIDENCE)
    assert all(request.method == "GET" for request in opener.requests)


def test_bounded_repair_rejects_unknown_citations_and_never_echoes_output(generator):
    invented = {**ANSWER, "answer": "MALFORMED_ANSWER_SENTINEL", "citations": ["invented"]}
    opener = generator._opener = Opener(ready() + [response(invented), response()])
    result = generator.answer("Question?", EVIDENCE)
    assert result["format_retries_used"] == 1 and result["usage"]["total_tokens"] == 22
    assert "MALFORMED_ANSWER_SENTINEL" not in opener.requests[-1].data.decode()
    assert "PRIVATE_REASONING_SENTINEL" not in opener.requests[-1].data.decode()
    generator._opener = Opener(ready() + [response(invented), response(invented)])
    with pytest.raises(GenerationError, match="valid grounded JSON"):
        generator.answer("Different question?", EVIDENCE)


def test_corrupt_cache_and_truncated_output_do_not_get_scored(generator):
    generator._opener = Opener(ready() + [response()])
    generator.answer("Question?", EVIDENCE)
    cache = next(generator.cache_dir.glob("*.json"))
    bad = json.loads(cache.read_text())
    bad["citations"] = ["invented"]
    cache.write_text(json.dumps(bad))
    with pytest.raises(GenerationError, match="Cached answer failed"):
        generator.answer("Question?", EVIDENCE)
    generator._opener = Opener(ready() + [response(finish="length")])
    with pytest.raises(GenerationError, match="did not finish"):
        generator.answer("Different question?", EVIDENCE)
    assert len(list(generator.cache_dir.glob("*.json"))) == 1


def test_native_image_bytes_and_changed_image_invalidate_cache(generator, tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 3), "blue").save(path)
    evidence = [{"id": "doc:1", "modality": "image", "path": str(path)}]
    opener = generator._opener = Opener(ready() + [response()] + ready() + [response()])
    first = generator.answer("Which colour?", evidence)
    parts = json.loads(opener.requests[-1].data)["messages"][-1]["content"]
    image = next(part for part in parts if part["type"] == "image_url")
    assert base64.b64decode(image["image_url"]["url"].split(",")[1]) == path.read_bytes()
    assert str(path) not in opener.requests[-1].data.decode()
    Image.new("RGB", (3, 3), "red").save(path)
    second = generator.answer("Which colour?", evidence)
    assert first["cache_key"] != second["cache_key"]


def fake_probe(modality, duration, *, audio_in_video=False):
    return {"duration_seconds": duration, "start_seconds": 0,
            "video_streams": [{"index": 0}] if modality == "video" else [],
            "audio_streams": [{"index": 0}] if modality == "audio" or audio_in_video else []}


def fake_media(monkeypatch, probe):
    monkeypatch.setattr(local.av, "probe_media", probe)
    monkeypatch.setattr(local.av, "_tool", lambda name: name)
    def run(command, timeout):
        if "-show_frames" in command:
            duration = probe(Path(command[-1]))["duration_seconds"]
            return json.dumps({"format": {"start_time": "0"}, "frames": [
                {"best_effort_timestamp_time": str(second)} for second in range(local.math.ceil(duration))]})
        return "ffmpeg test-build\n"
    monkeypatch.setattr(local.av, "_run", run)


def test_audio_uses_native_input_audio_and_actual_duration(generator, tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"native-audio-bytes")
    fake_media(monkeypatch, lambda _: fake_probe("audio", 3))
    opener = generator._opener = Opener(ready() + [response()])
    generator.answer("Question?", [{"id": "doc:1", "modality": "audio", "path": str(path),
                                     "metadata": {"duration_seconds": 999, "gold_answer": "GOLD_SENTINEL"}}])
    parts = json.loads(opener.requests[-1].data)["messages"][-1]["content"]
    audio = next(part for part in parts if part["type"] == "input_audio")
    assert base64.b64decode(audio["input_audio"]["data"]) == path.read_bytes()
    assert audio["input_audio"]["format"] == "wav"
    assert "GOLD_SENTINEL" not in opener.requests[-1].data.decode()


@pytest.mark.parametrize("modality,durations", [("audio", [30.1]), ("audio", [20, 20]), ("video", [60.1]), ("video", [31, 31])])
def test_duration_limits_cover_entire_request_and_do_not_trust_metadata(generator, tmp_path, monkeypatch, modality, durations):
    paths = []
    for index, duration in enumerate(durations):
        path = tmp_path / f"{index}.{'wav' if modality == 'audio' else 'mp4'}"
        path.write_bytes(b"media")
        paths.append(path)
    fake_media(monkeypatch, lambda path: fake_probe(modality, durations[int(path.stem)]))
    generator._opener = Opener([])
    with pytest.raises(ValueError, match="seconds of " + modality):
        generator.answer("Question?", [{"id": str(i), "modality": modality, "path": str(path),
                                         "metadata": {"duration_seconds": 1}} for i, path in enumerate(paths)])
    assert not generator._opener.requests


def test_video_audio_is_not_silently_discarded(generator, tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video-with-audio")
    fake_media(monkeypatch, lambda _: fake_probe("video", 3, audio_in_video=True))
    with pytest.raises(ValueError, match="silent video stream"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])


def test_video_sampling_covers_integer_timestamps_and_combines_with_native_audio(generator, tmp_path, monkeypatch):
    video, audio = tmp_path / "clip.mp4", tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    fake_media(monkeypatch, lambda path: fake_probe("audio" if path.suffix == ".wav" else "video", 2.2))
    commands = []
    def run(command, timeout):
        if "-version" in command:
            return "ffmpeg test-build\n"
        if "-show_frames" in command:
            return json.dumps({"format": {"start_time": "0"}, "frames": [
                {"best_effort_timestamp_time": str(second)} for second in (0, 1, 2)]})
        commands.append(command)
        for index in range(int(command[command.index("-frames:v") + 1])):
            Image.new("RGB", (2, 2), "green").save(command[-1] % index)
        return ""
    monkeypatch.setattr(local.av, "_run", run)
    opener = generator._opener = Opener(ready() + [response()])
    generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(video)},
                                     {"id": "doc:2", "modality": "audio", "path": str(audio)}])
    parts = json.loads(opener.requests[-1].data)["messages"][-1]["content"]
    assert len(commands) == 1 and "-ss" not in commands[0]
    assert "select='eq(n,0)+eq(n,1)+eq(n,2)'" in commands[0][commands[0].index("-vf") + 1]
    assert commands[0][commands[0].index("-pix_fmt") + 1] == "yuvj420p"
    assert "-xerror" in commands[0]
    assert all("file,pipe" in command and "-an" in command for command in commands)
    assert sum(part["type"] == "image_url" for part in parts) == 3
    assert sum(part["type"] == "input_audio" for part in parts) == 1
    text = " ".join(part["text"] for part in parts if part["type"] == "text")
    assert "clip-relative 2.000000 seconds" in text


def test_partial_frame_decode_aborts_before_completion(generator, tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    fake_media(monkeypatch, lambda _: fake_probe("video", 2))
    opener = generator._opener = Opener(ready())
    with pytest.raises(GenerationError, match="no partial video"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    assert all(request.method == "GET" for request in opener.requests)


def test_fractional_duration_samples_actual_last_frame_not_unreachable_seek(generator, tmp_path, monkeypatch):
    path = tmp_path / "fractional.mp4"
    path.write_bytes(b"video")
    fake_media(monkeypatch, lambda _: fake_probe("video", 30.03))
    commands = []
    timestamps = [f"{index * 1001 / 24000:.6f}" for index in range(720)]
    def run(command, timeout):
        if "-version" in command:
            return "ffmpeg test-build\n"
        if "-show_frames" in command:
            return json.dumps({"format": {"start_time": "0"}, "frames": [
                {"best_effort_timestamp_time": value} for value in timestamps]})
        commands.append(command)
        for index in range(int(command[command.index("-frames:v") + 1])):
            Image.new("RGB", (2, 2), "blue").save(command[-1] % index)
        return ""
    monkeypatch.setattr(local.av, "_run", run)
    opener = generator._opener = Opener(ready() + [response()])
    generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    parts = json.loads(opener.requests[-1].data)["messages"][-1]["content"]
    frames = [part for part in parts if part["type"] == "image_url"]
    labels = [part["text"] for part in parts if part["type"] == "text" and "video frame at" in part["text"]]
    assert len(frames) == len(labels) == 31
    assert "29.988292 seconds (active at sample second(s) 30)" in labels[-1]
    assert "eq(n,719)" in commands[0][commands[0].index("-vf") + 1]
    assert "clip-relative 30.000000" not in labels[-1]


@pytest.mark.parametrize("timestamps", [["0", "nan"], ["0", "inf"], ["0", "2", "1"], [], ["0.2", "1"]])
def test_invalid_video_timeline_fails_without_generation(generator, tmp_path, monkeypatch, timestamps):
    path = tmp_path / "bad-timeline.mp4"
    path.write_bytes(b"video")
    fake_media(monkeypatch, lambda _: fake_probe("video", 2))
    def run(command, timeout):
        if "-version" in command:
            return "ffmpeg test-build\n"
        assert "-show_frames" in command
        return json.dumps({"format": {"start_time": "0"}, "frames": [
            {"best_effort_timestamp_time": value} for value in timestamps]})
    monkeypatch.setattr(local.av, "_run", run)
    opener = generator._opener = Opener(ready())
    with pytest.raises(GenerationError, match="no partial video"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    assert all(request.method == "GET" for request in opener.requests)


def test_decoder_error_is_not_treated_as_end_of_stream(generator, tmp_path, monkeypatch):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"video")
    fake_media(monkeypatch, lambda _: fake_probe("video", 2))
    probe_run = local.av._run
    def run(command, timeout):
        if "-version" in command or "-show_frames" in command:
            return probe_run(command, timeout)
        raise RuntimeError("decoder rejected a corrupt packet")
    monkeypatch.setattr(local.av, "_run", run)
    opener = generator._opener = Opener(ready())
    with pytest.raises(GenerationError, match="Video decoding failed; no partial video"):
        generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    assert all(request.method == "GET" for request in opener.requests)


def test_low_fps_frames_are_deduplicated_and_timestamps_are_clip_relative(generator, tmp_path, monkeypatch):
    path = tmp_path / "slow.mp4"
    path.write_bytes(b"video")
    fake_media(monkeypatch, lambda _: fake_probe("video", 3))
    def run(command, timeout):
        if "-version" in command:
            return "ffmpeg test-build\n"
        if "-show_frames" in command:
            return json.dumps({"format": {"start_time": "5"}, "frames": [
                {"best_effort_timestamp_time": value} for value in ("5", "6.5")]})
        for index in range(int(command[command.index("-frames:v") + 1])):
            Image.new("RGB", (2, 2), "green").save(command[-1] % index)
        return ""
    monkeypatch.setattr(local.av, "_run", run)
    opener = generator._opener = Opener(ready() + [response()])
    generator.answer("Question?", [{"id": "doc:1", "modality": "video", "path": str(path)}])
    parts = json.loads(opener.requests[-1].data)["messages"][-1]["content"]
    labels = [part["text"] for part in parts if part["type"] == "text" and "video frame at" in part["text"]]
    assert len(labels) == 2
    assert "clip-relative 0.000000 seconds (active at sample second(s) 0, 1)" in labels[0]
    assert "clip-relative 1.500000 seconds (active at sample second(s) 2)" in labels[1]


def test_local_transport_does_not_follow_redirects_or_leak_error_text(generator):
    generator._opener = Opener([HTTPError("http://remote/?key=secret", 302, "secret", {"Location": "https://example.com"}, None)])
    with pytest.raises(GenerationError) as caught:
        generator._request_local("GET", "/health")
    assert "secret" not in str(caught.value) and "302" in str(caught.value)
    with pytest.raises(GenerationError, match="unexpected"):
        generator._request_local("POST", "https://example.com", b"private")
