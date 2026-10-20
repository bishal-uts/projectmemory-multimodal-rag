"""Grounded Gemma 4 E2B answers through the project-owned llama.cpp runtime.

Only a numeric loopback address is accepted. No credentials, proxy, redirect,
cloud fallback, downloads, or implicit media truncation occur in this adapter.
Video is represented explicitly as one timestamped frame per second; audio
uses the model's native audio encoder through llama.cpp's input_audio API.
"""
from __future__ import annotations

import base64
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from . import av
from .generation import GeminiGenerator, GenerationError, _NoRedirect, _SYSTEM, _hash_file, _json_bytes


DEFAULT_LOCAL_MODEL = "google/gemma-4-E2B-it-qat-q4_0-gguf"
DEFAULT_LOCAL_ALIAS = "gemma-4-e2b"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8766"
_ROOT = Path(__file__).resolve().parents[2]
_MEDIA_PROTOCOL = {
    "version": 2, "audio": "native wav/mp3/flac, unchanged bytes",
    "max_audio_seconds_per_request": 30, "max_video_seconds_per_request": 60,
    "video": "silent source; sample the frame active at 0,1,... seconds using decoded presentation timestamps; label actual frame timestamps",
    "video_sampling": "latest frame PTS <= sample time; repeated frame indices deduplicated; retain the final active frame for fractional duration",
    "video_pixel_format": "yuvj420p", "video_decoder_errors": "fail request; never send partial video",
    "video_fps": 1, "video_frame_long_edge": 768, "jpeg_quality": 3,
    "max_images_and_frames": 64,
}
_JSON_INSTRUCTION = """
Return exactly one JSON object with answer (string), answer_items (string array),
citations (evidence ID array), abstained (boolean), and answer_label (string).
Use an empty answer_label for free text. Do not include Markdown or reasoning.
Video evidence consists of sampled, timestamped frames; do not assume unseen
motion, speech, or events between the frames. Audio evidence is native audio.
For media records, the text field can be empty: the attached image, audio, or
video frames that follow the record are the evidence. An empty text field does
not mean that no media was provided. Listen to attached audio before answering.
"""


def _schema(ids: set[str], options: list[str] | None) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "answer": {"type": "string"},
                "answer_items": {"type": "array", "items": {"type": "string"}},
                "citations": {"type": "array", "maxItems": len(ids),
                              "items": {"type": "string", "enum": sorted(ids)}},
                "abstained": {"type": "boolean"},
                "answer_label": {"type": "string", "enum": [""] + (list("ABCD"[:len(options)]) if options else [])},
            }, "required": ["answer", "answer_items", "citations", "abstained", "answer_label"]}


class LocalGemmaGenerator(GeminiGenerator):
    """The shared answer contract, with a local-only transport and media protocol.

    requests_per_minute is accepted for factory compatibility; local inference
    has no remote quota and is not throttled. Runtime launch and GPU ownership
    belong to localruntime. autostart=False is useful with an explicitly started
    server (and in tests); a verified runtime manifest is still required.
    """

    provider = "local"
    provider_label = "Gemma 4 E2B (local)"
    key_hint = "No API key is required; prepare the local Gemma runtime"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL,
                 cache_dir: Path | str | None = Path(".cache/generation"),
                 max_output_tokens: int = 4096, requests_per_minute: float = 5, *,
                 base_url: str = DEFAULT_LOCAL_BASE_URL,
                 runtime_manifest: Path | str = Path(".cache/local-gemma4/runtime.json"),
                 project_root: Path | str | None = None, timeout: float = 300,
                 format_retries: int = 1, autostart: bool = True) -> None:
        if model != DEFAULT_LOCAL_MODEL:
            raise ValueError(f"This local adapter requires {DEFAULT_LOCAL_MODEL}")
        if (isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int)
                or not 1 <= max_output_tokens <= 8192):
            raise ValueError("max_output_tokens must be an integer from 1 to 8192")
        if (isinstance(requests_per_minute, bool) or not isinstance(requests_per_minute, (int, float))
                or not math.isfinite(requests_per_minute) or requests_per_minute <= 0):
            raise ValueError("requests_per_minute must be positive and finite")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("timeout must be positive and at most 600 seconds")
        if isinstance(format_retries, bool) or not isinstance(format_retries, int) or format_retries not in {0, 1}:
            raise ValueError("format_retries must be 0 or 1")
        if not isinstance(autostart, bool):
            raise ValueError("autostart must be a boolean")
        try:
            parsed = urlsplit(base_url)
            valid = (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
                     and parsed.port is not None and not parsed.username and not parsed.password
                     and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Local inference base_url must be an HTTP numeric loopback address with a port")
        self.base_url = base_url.rstrip("/")
        self.project_root = Path(project_root).resolve() if project_root is not None else _ROOT
        self.runtime_manifest = Path(runtime_manifest)
        if not self.runtime_manifest.is_absolute():
            self.runtime_manifest = self.project_root / self.runtime_manifest
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None and not self.cache_dir.is_absolute():
            self.cache_dir = self.project_root / self.cache_dir
        self.model, self.model_alias = model, DEFAULT_LOCAL_ALIAS
        self.max_output_tokens, self.requests_per_minute = max_output_tokens, float(requests_per_minute)
        self.timeout, self.format_retries, self.autostart = timeout, format_retries, autostart
        self.max_file_bytes = self.max_media_bytes = 32_000_000
        # Do not call the cloud parent constructor: it reads provider keys and
        # enables environment-configured proxies. Neither belongs here.
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _manifest(self) -> tuple[dict, str]:
        try:
            value = json.loads(self.runtime_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise GenerationError("Local Gemma runtime manifest is missing or invalid; run the local setup first") from None
        if (not isinstance(value, dict) or value.get("model") != self.model
                or value.get("model_alias") != self.model_alias
                or not isinstance(value.get("runtime_version"), str) or not value["runtime_version"].strip()
                or any(not isinstance(value.get(key), str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value[key])
                       for key in ("model_sha256", "mmproj_sha256"))):
            raise GenerationError("Local runtime manifest does not identify the expected Gemma model, projector, and runtime")
        return value, hashlib.sha256(_json_bytes(value)).hexdigest()

    def _generation_config(self) -> dict:
        return {"max_tokens": self.max_output_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                "min_p": 0.0, "repeat_penalty": 1.0, "seed": 42, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}

    def evaluation_config(self) -> dict:
        """Nonsecret configuration for a resumable, fixed-runtime evaluation."""
        manifest, manifest_hash = self._manifest()
        protocol = dict(_MEDIA_PROTOCOL)
        try:
            protocol["ffmpeg_version"] = av._run([av._tool("ffmpeg"), "-version"], 10).splitlines()[0]
        except (OSError, RuntimeError, ValueError, IndexError):
            # Text/image evaluation does not require FFmpeg. Audio/video will
            # fail explicitly in _inspect_assets when no decoder is available.
            protocol["ffmpeg_version"] = None
        return {"runtime_manifest_sha256": manifest_hash, "model_revision": manifest.get("model_revision"),
                "runtime_version": manifest["runtime_version"], "media_protocol": protocol,
                "generation_parameters": self._generation_config(), "format_retries": self.format_retries,
                "response_format": "grounded JSON schema with exact evidence citation IDs",
                "adapter_sha256": _hash_file(Path(__file__)),
                "runtime_adapter_sha256": _hash_file(Path(__file__).with_name("localruntime.py"))}

    def _request_local(self, method: str, path: str, data: bytes | None = None) -> dict:
        if (method, path) not in {("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions")}:
            raise GenerationError("Refusing an unexpected local inference endpoint")
        request = Request(self.base_url + path, data=data, method=method,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise GenerationError("Local inference response exceeded 4 MB")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise GenerationError("Local inference returned a non-object response")
                return value
        except HTTPError as exc:
            code = exc.code
            exc.close()
            hint = " Reduce the evidence or output budget if the context is too large." if code == 400 else ""
            raise GenerationError(f"Local Gemma returned HTTP {code}.{hint}") from None
        except (URLError, TimeoutError, OSError):
            raise GenerationError("Could not reach local Gemma; start the local runtime and retry") from None
        except (UnicodeDecodeError, ValueError):
            raise GenerationError("Local inference returned invalid JSON") from None

    def _ready(self, modalities: set[str]) -> None:
        if self.autostart:
            from .localruntime import ensure_running
            if ensure_running(root=self.project_root).rstrip("/") != self.base_url:
                raise GenerationError("The managed local runtime uses a different loopback endpoint")
        if self._request_local("GET", "/health").get("status") != "ok":
            raise GenerationError("Local Gemma is not ready")
        models = self._request_local("GET", "/v1/models").get("data")
        if not isinstance(models, list):
            raise GenerationError("Local server did not identify its loaded model")
        matching = [item for item in models if isinstance(item, dict) and item.get("id") == self.model_alias]
        if len(matching) != 1:
            raise GenerationError("Local server is not serving the expected Gemma 4 E2B alias")
        architecture = matching[0].get("architecture", {})
        supported = architecture.get("input_modalities", []) if isinstance(architecture, dict) else []
        # Video is explicitly sampled into images before transport.
        needed = (modalities - {"video"}) | ({"image"} if "video" in modalities else set())
        if not needed.issubset(set(supported)):
            raise GenerationError("Local server does not advertise the required media encoder; check the multimodal projector")

    def _inspect_assets(self, records: list[dict], assets: list[dict]) -> dict:
        totals = {"audio": 0.0, "video": 0.0}
        frames_and_images = 0
        by_id = {record["id"]: record for record in records}
        for asset in assets:
            modality = by_id[asset["id"]]["modality"]
            if modality == "pdf":
                raise ValueError("Local Gemma requires PDF pages rendered as images; ingest the PDF first")
            if modality == "image":
                if asset["mime_type"] not in {"image/jpeg", "image/png"}:
                    raise ValueError("Local Gemma image evidence must be JPEG or PNG; ingest the source first")
                from PIL import Image
                try:
                    with Image.open(asset["path"]) as picture:
                        if picture.width * picture.height > 30_000_000:
                            raise ValueError("Local image evidence exceeds 30 megapixels")
                        picture.verify()
                except (OSError, SyntaxError):
                    raise ValueError("Local image evidence is not a valid JPEG/PNG") from None
                frames_and_images += 1
                continue
            if modality == "audio" and asset["mime_type"] not in {"audio/wav", "audio/mpeg", "audio/flac"}:
                raise ValueError("Local Gemma audio evidence must be WAV, MP3, or FLAC")
            info = av.probe_media(asset["path"])
            if modality == "audio" and (len(info["audio_streams"]) != 1 or info["video_streams"]):
                raise ValueError("Audio evidence must contain one audio stream without a temporal video stream")
            if modality == "video" and (len(info["video_streams"]) != 1 or info["audio_streams"]):
                raise ValueError("Video evidence must have one silent video stream; ingest it to separate native audio and video evidence")
            duration = info["duration_seconds"]
            totals[modality] += duration
            asset["duration_seconds"] = duration
            by_id[asset["id"]]["media_processing"] = {
                "duration_seconds": duration,
                "representation": "native audio" if modality == "audio" else "1 fps timestamped frames",
            }
            if modality == "video":
                frames_and_images += math.ceil(duration)
        if totals["audio"] > 30:
            raise ValueError("Local Gemma accepts at most 30 seconds of audio per request; segment the source and retrieve fewer clips")
        if totals["video"] > 60:
            raise ValueError("Local Gemma accepts at most 60 seconds of video per request; segment the source and retrieve fewer clips")
        if frames_and_images > 64:
            raise ValueError("Local Gemma request exceeds 64 images/video frames; retrieve fewer media records")
        protocol = dict(_MEDIA_PROTOCOL)
        if totals["audio"] or totals["video"]:
            protocol["ffmpeg_version"] = av._run([av._tool("ffmpeg"), "-version"], 10).splitlines()[0]
        return protocol

    @staticmethod
    def _read_asset(asset: dict) -> bytes:
        raw = asset["path"].read_bytes()
        if len(raw) != asset["size"] or hashlib.sha256(raw).hexdigest() != asset["sha256"]:
            raise ValueError("Media changed while preparing the request; retry")
        return raw

    def _video_parts(self, asset: dict) -> list[dict]:
        parts = []
        # Container duration includes the display interval of the final frame.
        # Seeking to an integer within that interval can pass the final frame's
        # PTS and produce no image. Probe the real frame timeline and select by
        # decoded index, retaining the frame actually displayed at each sample.
        try:
            probe = json.loads(av._run([
                av._tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
                "-select_streams", "v:0", "-show_frames", "-show_format",
                "-show_entries", "frame=best_effort_timestamp_time:format=start_time",
                "-of", "json", str(asset["path"]),
            ], 60))
            start = float(probe.get("format", {}).get("start_time", 0))
            timestamps = [float(frame["best_effort_timestamp_time"]) - start
                          for frame in probe["frames"]]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise GenerationError("Could not inspect video presentation timestamps; no partial video was sent") from exc
        if (not math.isfinite(start) or not timestamps
                or any(not math.isfinite(value) or value < -1e-6 for value in timestamps)
                or any(right < left for left, right in zip(timestamps, timestamps[1:]))):
            raise GenerationError("Invalid video presentation timestamps; no partial video was sent")
        samples = {}
        for second in range(math.ceil(asset["duration_seconds"])):
            index = bisect_right(timestamps, second + 1e-6) - 1
            if index < 0:
                raise GenerationError("No video frame exists at a requested sample time; no partial video was sent")
            samples.setdefault(index, []).append(second)
        if not samples:
            raise GenerationError("No video samples were selected; no partial video was sent")
        expression = "+".join(f"eq(n,{index})" for index in samples)
        with tempfile.TemporaryDirectory(prefix="projectmemory_gemma_frames_") as folder:
            pattern = Path(folder) / "frame_%03d.jpg"
            try:
                av._run([
                    av._tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-xerror",
                    "-protocol_whitelist", "file,pipe", "-i", str(asset["path"]),
                    "-map", "0:v:0", "-frames:v", str(len(samples)), "-an",
                    "-vf", f"select='{expression}',scale=w='min(768,iw)':h='min(768,ih)':force_original_aspect_ratio=decrease,format=yuvj420p",
                    "-fps_mode", "vfr", "-c:v", "mjpeg", "-pix_fmt", "yuvj420p",
                    "-q:v", "3", "-threads", "2", "-start_number", "0", str(pattern),
                ], 60)
            except RuntimeError as exc:
                raise GenerationError("Video decoding failed; no partial video was sent") from exc
            if len(list(Path(folder).glob("frame_*.jpg"))) != len(samples):
                raise GenerationError("Could not decode every requested video sample; no partial video was sent")
            for position, (index, seconds) in enumerate(samples.items()):
                target = Path(folder) / f"frame_{position:03d}.jpg"
                if not target.is_file() or not target.stat().st_size:
                    raise GenerationError("Could not decode every requested video sample; no partial video was sent")
                requested = ", ".join(str(second) for second in seconds)
                parts.extend([{"type": "text", "text": f"Evidence {asset['id']} video frame at clip-relative {timestamps[index]:.6f} seconds (active at sample second(s) {requested}):"},
                              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(target.read_bytes()).decode("ascii")}}])
        if _hash_file(asset["path"]) != asset["sha256"]:
            raise ValueError("Video changed while preparing the request; retry")
        return parts

    def answer(self, question: str, evidence: list[dict], options: list[str] | None = None) -> dict:
        records, assets, _ = self._prepare(question, evidence, options)
        if not records:
            return {"answer": "The supplied evidence is insufficient to answer this question.", "answer_items": [],
                    "citations": [], "abstained": True, "answer_label": "", "sources": [], "model": self.model,
                    "provider": self.provider, "model_version": None, "usage": {}, "cache_hit": False,
                    "cache_key": hashlib.sha256(_json_bytes({"model": self.model, "question": question, "options": options})).hexdigest(),
                    "api_called": False, "cloud_api_called": False, "local_inference": False}
        protocol = self._inspect_assets(records, assets)
        manifest, manifest_hash = self._manifest()
        ids = {item["id"] for item in records}
        system = _SYSTEM + _JSON_INSTRUCTION
        # b11429 reads json_schema.schema for this type. Its README's flat
        # schema example silently falls back to unconstrained JSON objects.
        config = {**self._generation_config(), "response_format": {
            "type": "json_schema", "json_schema": {"name": "grounded_answer", "strict": True,
                                                       "schema": _schema(ids, options)}}}
        signature = {"prompt_version": 1, "provider": self.provider, "model": self.model,
                     "runtime_manifest_sha256": manifest_hash, "adapter_sha256": _hash_file(Path(__file__)),
                     "runtime_adapter_sha256": _hash_file(Path(__file__).with_name("localruntime.py")),
                     "system": system, "config": config, "question": question, "options": options,
                     "evidence": records, "media_protocol": protocol, "format_retries": self.format_retries,
                     "assets": [{key: value for key, value in item.items() if key != "path"} for item in assets]}
        cache_key = hashlib.sha256(_json_bytes(signature)).hexdigest()
        cache_path = self.cache_dir / (cache_key + ".json") if self.cache_dir is not None else None
        if cache_path is not None and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (cached.get("cache_key") != cache_key or cached.get("provider") != self.provider
                        or cached.get("model") != self.model or cached.get("runtime_manifest_sha256") != manifest_hash):
                    raise GenerationError("Cached answer belongs to another local model/runtime request")
                clean = self._validate_answer(cached, ids, options)
                return {**cached, **clean, "sources": self._sources(evidence, clean["citations"]),
                        "cache_hit": True, "api_called": False, "local_inference": False}
            except (ValueError, AttributeError, OSError, GenerationError):
                raise GenerationError("Cached answer failed validation; remove this cache entry before retrying") from None
        self._ready({item["modality"] for item in records})
        parts = [{"type": "text", "text": "Question and choices:\n" + json.dumps({"question": question, "options": options}, ensure_ascii=False)}]
        by_id = {asset["id"]: asset for asset in assets}
        for record in records:
            parts.append({"type": "text", "text": "Evidence record:\n" + json.dumps(record, ensure_ascii=False)})
            asset = by_id.get(record["id"])
            if asset is None:
                continue
            if record["modality"] == "video":
                parts.extend(self._video_parts(asset))
            else:
                encoded = base64.b64encode(self._read_asset(asset)).decode("ascii")
                if record["modality"] == "image":
                    parts.append({"type": "image_url", "image_url": {"url": "data:" + asset["mime_type"] + ";base64," + encoded}})
                else:
                    parts.append({"type": "input_audio", "input_audio": {"data": encoded, "format": asset["path"].suffix[1:].lower()}})
        payload = {"model": self.model_alias, "messages": [{"role": "system", "content": system},
                   {"role": "user", "content": parts}], **config}
        usage = {}
        for attempt in range(self.format_retries + 1):
            response = self._request_local("POST", "/v1/chat/completions", _json_bytes(payload))
            response_usage = response.get("usage", {})
            if isinstance(response_usage, dict):
                for key, value in response_usage.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[key] = usage.get(key, 0) + value
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise GenerationError("Local Gemma returned no answer candidate")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise GenerationError("Local Gemma did not finish normally; reduce evidence or increase the output budget")
            try:
                clean = self._validate_answer(json.loads(choice.get("message", {}).get("content")), ids, options)
                break
            except (ValueError, TypeError, AttributeError, GenerationError):
                if attempt == self.format_retries:
                    raise GenerationError("Local Gemma did not return a valid grounded JSON answer; no score was recorded") from None
                parts.append({"type": "text", "text": "Return exactly the requested JSON object. Cite only provided evidence IDs, with empty answer_label for free text. If evidence is insufficient, use abstained=true and empty answer_items, citations, and answer_label."})
        result = {**clean, "provider": self.provider, "model": self.model,
                  "model_version": manifest.get("model_revision", manifest["model_sha256"]),
                  "runtime_version": manifest["runtime_version"], "runtime_manifest_sha256": manifest_hash,
                  "media_protocol": protocol, "usage": usage, "format_retries_used": attempt,
                  "generation_requests": attempt + 1, "cache_key": cache_key, "cache_hit": False,
                  "api_called": True, "cloud_api_called": False, "local_inference": True, "created_at": datetime.now(timezone.utc).isoformat(),
                  "sources": self._sources(evidence, clean["citations"])}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent, delete=False) as handle:
                    temporary = handle.name
                    json.dump(result, handle, ensure_ascii=False, indent=2)
                os.replace(temporary, cache_path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return result
