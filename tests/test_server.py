import base64
import io
import json
from pathlib import Path
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from projectmemory.io import write_jsonl
from projectmemory.server import AppState, Handler, make_server


@pytest.fixture
def running_server(tmp_path):
    root = tmp_path / "workspace"
    data = root / "data"
    media = data / "media" / "example.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"image-bytes")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret-outside")
    corpus = [
        {"id": "doc:1", "title": "Meeting", "modality": "text", "text": "The remote control will be blue.", "source_id": "meeting1"},
        {"id": "doc:2", "title": "Remote image", "modality": "image", "text": "", "path": str(media), "source_id": "image1"},
        {"id": "doc:outside", "title": "Invalid source", "modality": "image", "text": "", "path": str(outside)},
    ]
    write_jsonl(data / "prepared" / "sample" / "corpus.jsonl", corpus)
    write_jsonl(data / "prepared" / "sample" / "queries.jsonl", [{"id": "q1", "question": "Which colour?", "reference_answers": ["GOLD_SECRET"]}])
    server = make_server(root=root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def request(server_info, path, payload=None, extra_headers=None, csrf=True):
    server, root, base = server_info
    headers = {"Content-Type": "application/json"}
    if csrf:
        headers["X-ProjectMemory"] = server.state.csrf_token
    headers.update(extra_headers or {})
    req = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    return urlopen(req, timeout=10)


def test_index_and_status_do_not_disclose_credentials(running_server, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "SECRET_API_TOKEN")
    with request(running_server, "/") as response:
        html = response.read().decode()
        assert "ProjectMemory" in html and "__CSRF_TOKEN__" not in html
        assert "SECRET_API_TOKEN" not in html
        assert response.headers["X-Frame-Options"] == "DENY"
    with request(running_server, "/api/status") as response:
        raw = response.read().decode()
        data = json.loads(raw)
        assert data["collections"][0]["records"] == 3
        assert data["providers"]["nvidia"]["configured"] is True
        assert "SECRET_API_TOKEN" not in raw and "GOLD_SECRET" not in raw


@pytest.mark.parametrize("headers,csrf", [({}, False), ({"Origin": "https://malicious.example"}, True)])
def test_post_rejects_missing_csrf_and_cross_site_requests(running_server, headers, csrf):
    with pytest.raises(HTTPError) as caught:
        request(running_server, "/api/search", {"collection": "sample", "question": "blue"}, headers, csrf)
    assert caught.value.code == 403


@pytest.mark.parametrize("body_size", [54, 131072])
@pytest.mark.parametrize("reason,status", [("csrf", 403), ("origin", 403), ("content_type", 415)])
def test_rejected_post_returns_error_when_body_arrives_after_headers(running_server, monkeypatch, body_size, reason, status):
    server = running_server[0]
    monkeypatch.setattr(server.state, "query", lambda *args, **kwargs: pytest.fail("Rejected request was dispatched"))
    port = server.server_address[1]
    headers = ["POST /api/search HTTP/1.1", f"Host: 127.0.0.1:{port}",
               f"Content-Length: {body_size}", "Connection: close",
               "Content-Type: text/plain" if reason == "content_type" else "Content-Type: application/json"]
    if reason != "csrf":
        headers.append(f"X-ProjectMemory: {server.state.csrf_token}")
    if reason == "origin":
        headers.append("Origin: https://malicious.example")
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = connection.recv(65536)
            assert chunk, "Server closed without an HTTP rejection"
            response += chunk
        assert response.startswith(f"HTTP/1.0 {status} ".encode())
        # Deliberately separate headers/body to reproduce the Windows close race.
        # The rejection must already be available, before waiting for body data.
        time.sleep(0.01)
        connection.sendall(b"{" + b" " * (body_size - 2) + b"}")
        while chunk := connection.recv(65536):
            response += chunk
    assert "error" in json.loads(response.split(b"\r\n\r\n", 1)[1])


def test_rejected_post_does_not_wait_indefinitely_for_missing_body(running_server):
    port = running_server[0].server_address[1]
    with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
        connection.sendall((f"POST /api/search HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                            "Content-Length: 100\r\nContent-Type: application/json\r\n\r\n").encode())
        response = b""
        while chunk := connection.recv(65536):
            response += chunk
    assert response.startswith(b"HTTP/1.0 403 ")
    assert "error" in json.loads(response.split(b"\r\n\r\n", 1)[1])


@pytest.mark.parametrize("length", ["not-an-integer", "-1", "0", "68000001"])
def test_rejected_post_does_not_read_invalid_or_oversized_bodies(length):
    handler = Handler.__new__(Handler)
    handler.headers = {"Content-Length": length}
    handler.wfile = io.BytesIO()
    handler.rfile = io.BytesIO(b"unread body")
    handler._error = lambda *args: None
    # No connection is needed: invalid lengths must never start draining.
    handler._reject_post(403, "Rejected")
    assert handler.rfile.tell() == 0
    assert handler.close_connection is True


def test_rejected_post_drain_enforces_total_deadline_for_trickling_body(monkeypatch):
    now = [100.0]
    read_sizes, timeouts = [], []

    class Connection:
        def gettimeout(self):
            return None

        def settimeout(self, timeout):
            timeouts.append(timeout)

    class Trickle:
        def read1(self, size):
            read_sizes.append(size)
            now[0] += 0.3
            return b"x"

    handler = Handler.__new__(Handler)
    handler.headers = {"Content-Length": "131072"}
    handler.wfile = io.BytesIO()
    handler.rfile = Trickle()
    handler.connection = Connection()
    handler._error = lambda *args: None
    monkeypatch.setattr("projectmemory.server.time.monotonic", lambda: now[0])
    handler._reject_post(403, "Rejected")
    assert read_sizes == [65536, 65536]
    assert timeouts[:2] == pytest.approx([0.5, 0.2])
    assert timeouts[-1] is None


def test_host_check_rejects_rebinding_without_using_network(running_server):
    # Windows may abort malicious-Host HTTP traffic before delivering the
    # response. Exercise this guard directly; CSRF/Origin are HTTP-tested above.
    handler = Handler.__new__(Handler)
    handler.server = running_server[0]
    handler.headers = {"Host": "malicious.example"}
    assert not handler._safe_host()


def test_search_returns_evidence_without_api(running_server):
    with request(running_server, "/api/search", {"collection": "sample", "question": "blue", "method": "bm25"}) as response:
        result = json.load(response)
    assert result["generated_answer"] is None
    assert result["evidence"][0]["id"] == "doc:1"
    assert "reference_answers" not in json.dumps(result)


def test_answer_uses_server_generator_only(running_server, monkeypatch):
    class Generator:
        def answer(self, question, evidence):
            return {"answer": "Blue", "citations": [evidence[0]["id"]], "abstained": False,
                    "answer_items": ["Blue"], "provider": "nvidia", "model": "test"}
    monkeypatch.setattr("projectmemory.server.key_available", lambda provider: True)
    monkeypatch.setattr("projectmemory.server.create_generator", lambda **kwargs: Generator())
    with request(running_server, "/api/answer", {"collection": "sample", "question": "blue", "provider": "nvidia"}) as response:
        result = json.load(response)
    assert result["generated_answer"]["answer"] == "Blue"
    with pytest.raises(HTTPError) as caught:
        request(running_server, "/api/answer", {"collection": "sample", "question": "blue", "api_key": "secret"})
    assert caught.value.code == 400


def test_local_answer_uses_project_runtime_without_api_key(running_server, monkeypatch):
    calls = []
    released = []
    class PreviousDenseRetriever:
        def unload(self):
            released.append(True)
    running_server[0].state.retrievers[("old-corpus", "dense-text", "cuda")] = PreviousDenseRetriever()
    class Generator:
        def answer(self, question, evidence):
            assert released == [True]
            return {"answer": "Blue", "citations": [evidence[0]["id"]], "abstained": False,
                    "answer_items": ["Blue"], "provider": "local", "model": "local-fixture"}
    def create(**kwargs):
        calls.append(kwargs)
        return Generator()
    monkeypatch.setattr("projectmemory.server.provider_status", lambda provider, root=None: {"configured": True})
    monkeypatch.setattr("projectmemory.server.key_available", lambda provider: pytest.fail("Local answers must not check an API key"))
    monkeypatch.setattr("projectmemory.server.create_generator", create)
    with request(running_server, "/api/answer", {"collection": "sample", "question": "blue", "provider": "local"}) as response:
        result = json.load(response)
    assert result["generated_answer"]["provider"] == "local"
    assert calls[0]["project_root"] == running_server[1]


def test_uninstalled_local_answer_reports_setup_instead_of_key(running_server, monkeypatch):
    monkeypatch.setattr("projectmemory.server.provider_status", lambda provider, root=None: {"configured": False})
    with pytest.raises(HTTPError) as caught:
        request(running_server, "/api/answer", {"collection": "sample", "question": "blue", "provider": "local"})
    error = json.load(caught.value)["error"]
    assert "not installed" in error
    assert "API key" not in error


def test_dense_gpu_cache_hit_stops_owned_answer_runtime_first(tmp_path, monkeypatch):
    from projectmemory.evaluation import corpus_fingerprint
    calls = []
    monkeypatch.setattr("projectmemory.localruntime.stop", lambda root=None: calls.append(root))
    state = AppState(tmp_path)
    corpus = [{"id": "doc", "modality": "text", "text": "Blue"}]
    cached = object()
    state.retrievers[(corpus_fingerprint(corpus), "dense-text", "cuda")] = cached
    assert state._retriever(corpus, "dense-text", "cuda") is cached
    assert calls == [tmp_path]


def test_media_only_serves_registered_in_project_sources_and_ranges(running_server):
    with request(running_server, "/media?collection=sample&id=doc%3A2", extra_headers={"Range": "bytes=0-4"}) as response:
        assert response.status == 206
        assert response.read() == b"image"
    for path in ("/media?collection=sample&id=doc%3Aoutside", "/media?collection=../&id=x", "/.env", "/media?path=C:/Windows/win.ini"):
        with pytest.raises(HTTPError) as caught:
            request(running_server, path)
        assert caught.value.code == 404


def test_upload_imports_text_and_rejects_filename_traversal(running_server):
    raw = base64.b64encode(b"The customer needs a blue remote control.").decode()
    with request(running_server, "/api/ingest", {"filename": "notes.txt", "data": raw}) as response:
        result = json.load(response)
    assert result["collection"] == "uploads" and result["added_records"] == 1
    assert (running_server[1] / "data/prepared/uploads/corpus.jsonl").exists()
    for filename in ("../evil.txt", "C:\\evil.txt", ".env", "CON.txt", "malicious.svg"):
        with pytest.raises(HTTPError) as caught:
            request(running_server, "/api/ingest", {"filename": filename, "data": raw})
        assert caught.value.code == 400


def test_results_are_actual_compact_metrics(running_server):
    root = running_server[1]
    (root / "results").mkdir()
    report = {"method": "bm25", "query_count": 2, "metrics": {"recall@5": 0.5},
              "queries": [{"reference_answers": ["GOLD_SECRET"]}],
              "config": {"dataset": "sample"}, "k": 5}
    (root / "results/test.json").write_text(json.dumps(report))
    with request(running_server, "/api/results") as response:
        raw = response.read().decode()
        result = json.loads(raw)
    assert result["reports"][0]["metrics"]["recall@5"] == 0.5
    assert "GOLD_SECRET" not in raw


def test_server_rejects_public_binding(tmp_path):
    with pytest.raises(ValueError, match="localhost"):
        make_server(root=tmp_path, host="0.0.0.0", port=0)
