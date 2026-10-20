"""NVIDIA's hosted Nemotron Omni endpoint with grounded evidence and local cache."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit

from .generation import GeminiGenerator, GenerationError, _SYSTEM, _json_bytes


DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
_BASE = "https://integrate.api.nvidia.com"
_JSON_INSTRUCTION = '''\nReturn only one JSON object (no Markdown) with exactly these fields:
{"answer": "concise readable answer", "answer_items": ["each answer entity"],
 "citations": ["exact evidence ID"], "abstained": false, "answer_label": ""}.
Use answer_label A, B, C, or D only for a multiple-choice question. On insufficient
evidence, use abstained=true, empty answer_items/citations/answer_label, and an
explanation in answer. Do not include your reasoning trace in the JSON answer.'''


class NvidiaGenerator(GeminiGenerator):
    """Hosted NIM adapter, using only documented hosted endpoint fields.

    Local files are sent inline (8 MB raw-media default, configurable up to
    12 MB). Larger clips must be compressed or segmented upstream. No upload
    service, automatic model fallback, or external search tools are enabled.
    """

    provider = "nvidia"
    provider_label = "NVIDIA"
    key_hint = "NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY / NGC_API_KEY)"

    def __init__(self, model: str = DEFAULT_NVIDIA_MODEL,
                 cache_dir: Path | str | None = Path(".cache/generation"),
                 api_key: str | None = None, max_output_tokens: int = 4096,
                 requests_per_minute: float = 5, *, reasoning_budget: int = 1024,
                 max_inline_bytes: int = 8_000_000, max_retries: int = 2,
                 timeout: float = 60, poll_timeout: float = 180,
                 format_retries: int = 1) -> None:
        if not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+", model):
            raise ValueError("model must be an NVIDIA catalog ID such as nvidia/model-name")
        if (isinstance(reasoning_budget, bool) or not isinstance(reasoning_budget, int)
                or not 0 <= reasoning_budget <= 32768 or reasoning_budget >= max_output_tokens):
            raise ValueError("reasoning_budget must be 0–32768 and below max_output_tokens")
        if not 0 < poll_timeout <= 300:
            raise ValueError("poll_timeout must be positive and at most 300 seconds")
        if isinstance(format_retries, bool) or not isinstance(format_retries, int) or format_retries not in {0, 1}:
            raise ValueError("format_retries must be 0 or 1")
        # Shared transport, validation and evidence handling; set NVIDIA key
        # explicitly afterwards so Google credentials can never reach NVIDIA.
        super().__init__(cache_dir=cache_dir, api_key="unused", max_output_tokens=max_output_tokens,
                         requests_per_minute=requests_per_minute, max_inline_bytes=max_inline_bytes,
                         max_media_bytes=max_inline_bytes, max_file_bytes=max_inline_bytes,
                         max_retries=max_retries, timeout=timeout)
        if max_output_tokens > 65536:
            raise ValueError("max_output_tokens cannot exceed 65536")
        self.model = model
        self._api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NGC_API_KEY")
        self.reasoning_budget, self.poll_timeout = reasoning_budget, poll_timeout
        self.format_retries = format_retries

    @staticmethod
    def _provider_url(url: str) -> bool:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname == "integrate.api.nvidia.com"
                and parsed.port in (None, 443) and not parsed.username and not parsed.password)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self._api_key, "Accept": "application/json"}

    def _complete(self, payload: dict) -> dict:
        response, headers = self._request("POST", _BASE + "/v1/chat/completions", _json_bytes(payload), throttle=True)
        deadline = time.monotonic() + self.poll_timeout
        request_id = headers.get("nvcf-reqid") or response.get("reqId")
        while headers.get("_http_status") == "202":
            if not isinstance(request_id, str) or not re.fullmatch(r"[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", request_id):
                raise GenerationError("NVIDIA returned a pending response without a valid request ID")
            if time.monotonic() >= deadline:
                raise GenerationError("NVIDIA inference exceeded poll_timeout; retry later")
            self._pause(min(2, max(0, deadline - time.monotonic())))
            response, headers = self._request("GET", _BASE + "/v1/status/" + request_id)
        return response

    def answer(self, question: str, evidence: list[dict], options: list[str] | None = None) -> dict:
        records, assets, _ = self._prepare(question, evidence, options)
        allowed = {"image/jpeg", "image/png", "audio/wav", "audio/mpeg", "audio/flac", "video/mp4"}
        if any(asset["mime_type"] not in allowed for asset in assets):
            raise ValueError("NVIDIA Omni requires JPEG/PNG images, WAV/MP3/FLAC audio, or MP4 video; render PDF pages as images")
        for item in evidence:
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            duration = item.get("duration_seconds", metadata.get("duration_seconds"))
            if duration is None and "start_seconds" in metadata and "end_seconds" in metadata:
                duration = float(metadata["end_seconds"]) - float(metadata["start_seconds"])
            if duration is not None and item.get("modality") == "video" and float(duration) > 120:
                raise ValueError("NVIDIA Omni videos must be at most two minutes; segment the source first")
        system = _SYSTEM + _JSON_INSTRUCTION
        config = {"max_tokens": self.max_output_tokens, "reasoning_budget": self.reasoning_budget,
                  "temperature": 0.6, "top_p": 0.95, "stream": False, "seed": 42}
        signature = {"prompt_version": 1, "provider": self.provider, "model": self.model,
                     "system": system, "config": config, "question": question, "options": options,
                     "evidence": records,
                     "assets": [{key: value for key, value in asset.items() if key != "path"} for asset in assets]}
        cache_key = hashlib.sha256(_json_bytes(signature)).hexdigest()
        cache_path = self.cache_dir / (cache_key + ".json") if self.cache_dir is not None else None
        ids = {item["id"] for item in records}
        if cache_path is not None and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != cache_key or cached.get("model") != self.model or cached.get("provider") != self.provider:
                    raise GenerationError("Cached answer belongs to a different request")
                clean = self._validate_answer(cached, ids, options)
                return {**cached, **clean, "sources": self._sources(evidence, clean["citations"]), "cache_hit": True, "api_called": False}
            except (ValueError, AttributeError, OSError, GenerationError):
                raise GenerationError("Cached answer failed validation; remove this cache entry before retrying") from None
        if not records:
            return {"answer": "The supplied evidence is insufficient to answer this question.",
                    "answer_items": [], "citations": [], "abstained": True, "answer_label": "",
                    "sources": [], "model": self.model, "model_version": None, "usage": {},
                    "provider": self.provider, "cache_hit": False, "cache_key": cache_key, "api_called": False}
        if not self._api_key:
            raise GenerationError(f"Set {self.key_hint} before calling the API")
        parts = [{"type": "text", "text": "Question and choices:\n" + json.dumps({"question": question, "options": options}, ensure_ascii=False)}]
        by_id = {asset["id"]: asset for asset in assets}
        for record in records:
            parts.append({"type": "text", "text": "Evidence record:\n" + json.dumps(record, ensure_ascii=False)})
            asset = by_id.get(record["id"])
            if asset is not None:
                raw = asset["path"].read_bytes()
                if len(raw) != asset["size"] or hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                    raise ValueError("Media changed while preparing the request; retry")
                part_type = record["modality"] + "_url"
                parts.append({"type": part_type, part_type: {"url": "data:" + asset["mime_type"] + ";base64," + base64.b64encode(raw).decode("ascii")}})
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                   {"role": "user", "content": parts}], **config}
        usage = {}
        for format_attempt in range(self.format_retries + 1):
            response = self._complete(payload)
            for key, value in response.get("usage", {}).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise GenerationError("NVIDIA returned no answer candidate")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise GenerationError("NVIDIA did not finish the answer normally; increase the output budget if truncated")
            answer_text = choice.get("message", {}).get("content")
            try:
                if not isinstance(answer_text, str):
                    raise GenerationError("NVIDIA returned no final answer text")
                # Strip only an exact outer JSON fence; never extract an object
                # from a reasoning trace or mixed prose.
                answer_text = answer_text.strip()
                fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", answer_text, flags=re.S | re.I)
                if fence:
                    answer_text = fence.group(1)
                decoded = json.loads(answer_text)
                clean = self._validate_answer(decoded, ids, options)
                break
            except (ValueError, TypeError, GenerationError):
                if format_attempt == self.format_retries:
                    raise GenerationError("NVIDIA did not return a valid grounded JSON answer; no score was recorded") from None
                # Retry the same question/evidence once, without sending the
                # malformed answer, any private reasoning, or any gold label.
                payload["messages"][-1]["content"].append({
                    "type": "text", "text": "Formatting reminder: return exactly the JSON object described in the system instruction. Use double-quoted keys and strings, no prose or Markdown. Citations must exactly match provided evidence IDs. Use empty answer_label for free text, or one A-D letter for choices. If evidence is insufficient, return the specified abstention object."})
        result = {**clean, "provider": self.provider, "model": self.model,
                  "model_version": response.get("model"), "system_fingerprint": response.get("system_fingerprint"),
                  "usage": usage, "usage_last_request": response.get("usage", {}),
                  "format_retries_used": format_attempt, "generation_requests": format_attempt + 1,
                  "cache_key": cache_key, "cache_hit": False,
                  "api_called": True, "created_at": datetime.now(timezone.utc).isoformat(),
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
