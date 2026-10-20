"""Local research interface. Bind to loopback; keep API credentials server-side."""
from __future__ import annotations

import base64
from collections import Counter
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit

from .evaluation import corpus_fingerprint
from .io import read_jsonl
from .retrieval import create_retriever
from .settings import DEFAULT_MODELS, configured_provider, create_generator, key_available, load_project_env, provider_status


_MAX_UPLOAD = 50_000_000
_MAX_BODY = 68_000_000
_REJECT_DRAIN_SECONDS = 0.5
_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def _within(path: Path, parent: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("The requested path is outside the project data directory")
    return path


class AppState:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.data = self.root / "data"
        self.prepared = self.data / "prepared"
        self.csrf_token = secrets.token_urlsafe(32)
        self.work_lock = threading.Lock()
        self.retrievers = {}
        self.generators = {}
        self.busy = False
        load_project_env(self.root / ".env")

    def collection_path(self, collection: str) -> Path:
        if not isinstance(collection, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", collection):
            raise ValueError("Choose a valid prepared collection")
        folder = _within(self.prepared / collection, self.data)
        if not (folder / "corpus.jsonl").is_file():
            raise ValueError("The selected collection has not been prepared")
        return folder

    def corpus(self, collection: str) -> list[dict]:
        return read_jsonl(self.collection_path(collection) / "corpus.jsonl")

    def status(self) -> dict:
        load_project_env(self.root / ".env")
        collections = []
        if self.prepared.is_dir():
            for folder in sorted(self.prepared.iterdir()):
                if not folder.is_dir() or not (folder / "corpus.jsonl").is_file():
                    continue
                try:
                    corpus = self.corpus(folder.name)
                except (OSError, ValueError):
                    continue
                counts = Counter(item.get("modality", "text") for item in corpus)
                sources = sorted({str(item.get("source_id", "")) for item in corpus} - {""})
                missing = sum(1 for item in corpus if item.get("modality", "text") != "text"
                              and (not item.get("path") or not Path(item["path"]).is_file()))
                sample_questions = []
                queries_path = folder / "queries.jsonl"
                if queries_path.is_file():
                    for query in read_jsonl(queries_path)[:3]:
                        if isinstance(query.get("question"), str) and not query.get("options"):
                            sample_questions.append(query["question"])
                collections.append({"id": folder.name, "records": len(corpus), "modalities": dict(counts),
                                    "missing_media": missing, "source_count": len(sources),
                                    "sources": sources if len(sources) <= 100 else [],
                                    "sample_questions": sample_questions})
        return {"collections": collections, "providers": {
            provider: provider_status(provider, root=self.root)
            for provider in DEFAULT_MODELS}, "default_provider": configured_provider(), "busy": self.busy,
            "local_only": True, "max_upload_bytes": _MAX_UPLOAD}

    def close(self):
        from .localruntime import stop
        stop(root=self.root)

    def _retriever(self, corpus: list[dict], method: str, device: str):
        if method != "bm25" and device == "cuda":
            # Embeddings and the answer model take turns using the laptop GPU.
            from .localruntime import stop
            stop(root=self.root)
        key = (corpus_fingerprint(corpus), method, device)
        if key in self.retrievers:
            return self.retrievers[key]
        if method != "bm25":
            # Keep only one dense model in this process to avoid double GPU
            # allocations when a user changes collection or representation.
            for old in list(self.retrievers):
                if old[1] != "bm25":
                    del self.retrievers[old]
            gc.collect()
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        kwargs = {} if method == "bm25" else {
            "device": device, "cache_dir": self.root / ".cache" / "embeddings",
            "revision": _REVISION, "batch_size": 2, "dtype": "float32"}
        retriever = create_retriever(corpus, method, **kwargs)
        self.retrievers[key] = retriever
        while len(self.retrievers) > 5:
            del self.retrievers[next(iter(self.retrievers))]
        return retriever

    def query(self, payload: dict, *, generate: bool) -> dict:
        allowed = {"collection", "question", "method", "provider", "device", "top_k", "source_id"}
        if set(payload) - allowed:
            raise ValueError("Unexpected search setting")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 6000:
            raise ValueError("Enter a question of 1–6,000 characters")
        method, device = payload.get("method", "bm25"), payload.get("device", "cpu")
        provider = payload.get("provider", configured_provider())
        if method not in {"bm25", "dense-text", "multimodal"} or device not in {"cpu", "cuda"}:
            raise ValueError("Choose a supported retriever and device")
        if provider not in DEFAULT_MODELS:
            raise ValueError("Choose local Gemma, NVIDIA or Gemini")
        top_k = payload.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("Evidence count must be between 1 and 10")
        source_id = payload.get("source_id") or None
        if source_id is not None and not isinstance(source_id, str):
            raise ValueError("Source must be a source ID")
        if not self.work_lock.acquire(blocking=False):
            raise RuntimeError("Another search, answer, or import is running. Please wait for it to finish.")
        self.busy = True
        started = time.perf_counter()
        try:
            collection = payload.get("collection")
            corpus = self.corpus(collection)
            if generate and provider == "local" and not provider_status(provider, root=self.root)["configured"]:
                raise ValueError("Local Gemma 4 E2B is not installed or its runtime is unavailable. Complete the local setup before generating answers; search is still available.")
            if generate and provider != "local" and not key_available(provider):
                load_project_env(self.root / ".env")
                if not key_available(provider):
                    raise ValueError(f"Add the {provider.upper()} API key to the local .env file, then try again")
            retriever = self._retriever(corpus, method, device)
            generator = None
            if generate:
                if provider not in self.generators:
                    local_options = {"project_root": self.root} if provider == "local" else {}
                    self.generators[provider] = create_generator(provider=provider, model=DEFAULT_MODELS[provider],
                                                                 cache_dir=self.root / ".cache" / "generation", **local_options)
                generator = self.generators[provider]
            if generate and provider == "local":
                # A BM25 search may leave an older dense retriever cached. The
                # RAG step unloads the current retriever after searching; free
                # every other model here before the local generator can start.
                for cached in self.retrievers.values():
                    if cached is not retriever and hasattr(cached, "unload"):
                        cached.unload()
            from .application import ask_application_question
            result = ask_application_question(corpus, question, retriever, generator=generator,
                                              top_k=top_k, source_id=source_id, provider=provider)
            evidence = []
            for record in result.get("evidence", []):
                item = {key: record[key] for key in ("id", "title", "modality", "text", "source_id", "score") if key in record}
                item["text"] = str(item.get("text", ""))[:8000]
                metadata = record.get("metadata", {})
                item["location"] = {key: metadata[key] for key in
                                    ("page", "slide", "utterance_start", "utterance_end", "start_seconds", "end_seconds", "start_word", "end_word_exclusive")
                                    if isinstance(metadata, dict) and key in metadata}
                if record.get("path"):
                    try:
                        path = _within(Path(record["path"]), self.data)
                        if path.is_file():
                            item["media_url"] = "/media?collection=" + quote(collection) + "&id=" + quote(record["id"], safe="")
                    except ValueError:
                        pass
                if isinstance(metadata, dict) and metadata.get("source_file"):
                    try:
                        path = _within(Path(metadata["source_file"]), self.data)
                        if path.is_file():
                            item["original_url"] = "/media?collection=" + quote(collection) + "&id=" + quote(record["id"], safe="") + "&original=1"
                    except ValueError:
                        pass
                evidence.append(item)
            generated = result.get("generated_answer")
            if generated is not None:
                generated = {key: generated[key] for key in
                             ("answer", "answer_items", "citations", "abstained", "provider", "model", "model_version", "cache_hit", "usage", "file_cleanup_warning")
                             if key in generated}
            return {"question": question, "collection": collection, "method": method,
                    "evidence": evidence, "generated_answer": generated,
                    "evidence_plan": result.get("evidence_plan"),
                    "elapsed_seconds": time.perf_counter() - started}
        finally:
            self.busy = False
            self.work_lock.release()

    def ingest(self, payload: dict) -> dict:
        from .ingestion import SUPPORTED_EXTENSIONS, ingest_paths
        if set(payload) != {"filename", "data"}:
            raise ValueError("Upload must contain one filename and base64 file data")
        filename, encoded = payload["filename"], payload["data"]
        if (not isinstance(filename, str) or not filename or len(filename) > 160
                or filename in {".", ".."} or re.search(r'[\\/<>:"|?*\x00-\x1f]', filename)
                or filename[-1] in {" ", "."} or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS
                or re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", filename, re.I)):
            raise ValueError("Choose a supported file with a safe filename")
        if not isinstance(encoded, str) or len(encoded) > ((_MAX_UPLOAD + 2) // 3) * 4:
            raise ValueError("The upload exceeds the 50 MB file limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError:
            raise ValueError("Upload data must be valid base64") from None
        if not raw or len(raw) > _MAX_UPLOAD:
            raise ValueError("The file must contain between 1 byte and 50 MB")
        if not self.work_lock.acquire(blocking=False):
            raise RuntimeError("Another search, answer, or import is running. Please wait for it to finish.")
        self.busy = True
        try:
            digest = hashlib.sha256(raw).hexdigest()
            destination = _within(self.data / "uploads" / digest / filename, self.data)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            output = _within(self.prepared / "uploads", self.data)
            report = ingest_paths([destination], output)
            return {"collection": "uploads", "filename": filename, "records": report["corpus_records"],
                    "added_records": sum(item["records"] for item in report["last_ingested"])}
        finally:
            self.busy = False
            self.work_lock.release()

    def results(self) -> dict:
        reports = []
        folder = self.root / "results"
        if folder.is_dir():
            for path in sorted(folder.rglob("*.json")):
                if not path.resolve().is_relative_to(folder.resolve()) or path.stat().st_size > 5_000_000:
                    continue
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                    metrics = report.get("metrics", {})
                    if not isinstance(metrics, dict) or not metrics or not report.get("method"):
                        continue
                    metrics = {key: value for key, value in metrics.items()
                               if isinstance(value, (int, float)) and not isinstance(value, bool)}
                    config = report.get("config", {})
                    dataset = config.get("dataset_manifest", {}).get("dataset") or Path(config.get("dataset", "unknown")).name
                    reports.append({"id": path.relative_to(folder).as_posix(), "dataset": dataset,
                                    "method": report["method"], "task": report.get("task", "retrieval"),
                                    "media_mode": report.get("media_mode"), "provider": report.get("provider"),
                                    "model": report.get("model"),
                                    "split": report.get("split"), "query_count": report.get("query_count"),
                                    "k": report.get("k", 5), "metrics": metrics,
                                    "created_at": report.get("created_at"),
                                    "corpus_fingerprint": report.get("corpus_fingerprint"),
                                    "query_fingerprint": report.get("query_fingerprint")})
                except (ValueError, AttributeError, OSError, TypeError):
                    continue
        return {"reports": reports, "note": "Measured local results. Compare rows only when corpus, questions, split and k match."}


class ProjectMemoryServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, Handler)

    def server_close(self):
        try:
            self.state.close()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "ProjectMemory"
    sys_version = ""

    def log_message(self, format, *args):
        # Avoid request bodies, private queries and credentials in server output.
        pass

    @property
    def state(self):
        return self.server.state

    def _safe_host(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _respond(self, status: int, body: bytes, mime="application/json; charset=utf-8", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        self._respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _error(self, status, message):
        self._json(status, {"error": message})

    def _reject_post(self, status, message):
        self.close_connection = True
        self._error(status, message)
        self.wfile.flush()
        # Closing while a client is still sending its body can turn the HTTP
        # error into a TCP reset on Windows. Send the error first, then discard
        # a bounded declared body without parsing or dispatching it. An absent,
        # oversized or stalled body must not keep this worker alive indefinitely.
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return
        if not 0 < remaining <= _MAX_BODY:
            return
        previous_timeout = self.connection.gettimeout()
        deadline = time.monotonic() + _REJECT_DRAIN_SECONDS
        try:
            while remaining:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                # read1 performs at most one raw read, so a slow trickle cannot
                # reset the socket timeout repeatedly inside BufferedReader.read.
                chunk = self.rfile.read1(min(remaining, 64 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass
        finally:
            self.connection.settimeout(previous_timeout)

    def do_GET(self):
        if not self._safe_host():
            self._error(403, "This local app requires a localhost address")
            return
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/":
                page = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
                self._respond(200, page.replace("__CSRF_TOKEN__", self.state.csrf_token).encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/status":
                self._json(200, self.state.status())
            elif parsed.path == "/api/results":
                self._json(200, self.state.results())
            elif parsed.path == "/media":
                self._media(parse_qs(parsed.query))
            else:
                self._error(404, "Resource not found")
        except (ValueError, KeyError, OSError):
            self._error(404, "Requested collection or media is unavailable")

    def do_POST(self):
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if (not self._safe_host() or origin not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}")
                or not secrets.compare_digest(self.headers.get("X-ProjectMemory", "").encode("utf-8"), self.state.csrf_token.encode("utf-8"))):
            self._reject_post(403, "Request verification failed; reload the local app")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._reject_post(415, "Use JSON request data")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reject_post(400, "Content-Length must be a valid integer")
            return
        if not 0 < size <= _MAX_BODY:
            self._reject_post(413, "Request exceeds the upload limit")
            return
        try:
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Request data must be an object")
            if self.path in {"/api/search", "/api/answer"}:
                self._json(200, self.state.query(payload, generate=self.path.endswith("answer")))
            elif self.path == "/api/ingest":
                self._json(200, self.state.ingest(payload))
            else:
                self._error(404, "Resource not found")
        except (ValueError, KeyError) as exc:
            self._error(400, str(exc))
        except RuntimeError as exc:
            self._error(409, str(exc))
        except (OSError, ImportError):
            self._error(500, "A required file or local dependency is unavailable. Check the project setup.")
        except Exception:
            self._error(500, "The operation could not finish. Check the input and project setup.")

    def _media(self, query: dict):
        collection = query.get("collection", [None])[0]
        doc_id = query.get("id", [None])[0]
        record = next((item for item in self.state.corpus(collection) if item["id"] == doc_id), None)
        if record is None:
            raise ValueError("Unknown evidence ID")
        original = query.get("original", ["0"])[0] == "1"
        raw_path = record.get("metadata", {}).get("source_file") if original else record.get("path")
        if not raw_path:
            raise ValueError("No media for this evidence")
        path = _within(Path(raw_path), self.state.data)
        if not path.is_file():
            raise ValueError("Missing media")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not original and (not mime.startswith(("image/", "audio/", "video/")) or mime == "image/svg+xml"):
            raise ValueError("This file is not a previewable media type")
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if match is None:
                self._respond(416, b"", extra={"Content-Range": f"bytes */{size}"})
                return
            start = int(match.group(1))
            end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
            if start > end or start >= size:
                self._respond(416, b"", extra={"Content-Range": f"bytes */{size}"})
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream" if original else mime)
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if original:
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(path.name))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def make_server(root: Path = Path("."), host: str = "127.0.0.1", port: int = 8765) -> ProjectMemoryServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("ProjectMemory's development server must bind to localhost")
    return ProjectMemoryServer(("127.0.0.1", port), AppState(root))


def serve(host: str = "127.0.0.1", port: int = 8765, root: Path = Path(".")) -> None:
    server = make_server(root=root, host=host, port=port)
    print(f"ProjectMemory is ready at http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
