"""Optional EmbeddingGemma 2 adapter for text and original-media retrieval.

Uses the official Sentence Transformers input protocol. Install the optional
embedding dependencies only when running dense retrieval. Long sources must be
segmented upstream: the model shares an 8,192-token context across modalities.
Video encoding samples visual frames; index an extracted audio clip separately
when the soundtrack matters. Metadata timestamps do not crop a media file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .retrieval import candidate_indices, record_text, validate_corpus, validate_top_k


DEFAULT_MODEL = "google/embeddinggemma-2"
_CACHE_VERSION = 2


def _normalize(vector: Any) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Encoder produced an empty or non-finite embedding; check model precision.")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Encoder produced a zero or invalid embedding.")
    return [value / norm for value in values]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Publish one complete cache shard, leaving no half-written JSON on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, allow_nan=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class EmbeddingRetriever:
    """Lazy exact cosine search over normalized EmbeddingGemma 2 vectors.

    ``mode='text'`` searches title/text fields (including upstream OCR and
    transcripts). ``mode='multimodal'`` encodes original local media files and
    does not substitute their captions. Each record must be a prepared retrieval
    unit. Text-only records with neither text nor title have no representation
    and are omitted. Every media record in native mode must have a valid path.

    Cache keys include corpus text, full media content hashes, model, revision,
    precision, and processing settings. Pin ``revision`` to a commit for strict
    reproducibility; an unpinned remote model may change without a new key.
    """

    def __init__(self, corpus: list[dict[str, Any]], mode: str = "text",
                 model_name: str = DEFAULT_MODEL, device: str = "cpu",
                 cache_dir: Path | None = None, *, dtype: str = "float32",
                 batch_size: int = 8, revision: str | None = None,
                 processing_kwargs: dict[str, Any] | None = None) -> None:
        if mode not in {"text", "multimodal"}:
            raise ValueError("mode must be 'text' or 'multimodal'.")
        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("EmbeddingGemma 2 requires float32 or bfloat16; float16 is unsafe.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.corpus = validate_corpus(corpus)
        self.mode, self.model_name, self.device = mode, model_name, device
        self.dtype, self.batch_size, self.revision = dtype, batch_size, revision
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Uniform sampling avoids silently considering just a clip's beginning.
        self.processing_kwargs = {
            "video": {"fps": 1, "max_frames": 32, "overflow_strategy": "uniform",
                      "add_timestamps": True},
            **(processing_kwargs or {}),
        }
        self._model: Any = None
        self._vectors: list[list[float]] | None = None
        self._records: list[dict[str, Any]] = []
        self._inputs: list[str | dict[str, str]] = []
        self._prepare_inputs()

    def _prepare_inputs(self) -> None:
        for item in self.corpus:
            modality = item.get("modality", "text")
            if modality not in {"text", "image", "audio", "video"}:
                raise ValueError(f"Record {item['id']!r} has unsupported modality {modality!r}.")
            if self.mode == "text" or modality == "text":
                if not record_text(item):
                    continue
                # This is the official Document prefix, with a real title when
                # available. Preformatting permits documents with different
                # titles to share one encode batch without double prompting.
                title = (item.get("title") or "").strip() or "none"
                prepared: str | dict[str, str] = f"title: {title} | text: {item.get('text') or ''}"
            else:
                raw_path = item.get("path")
                if not raw_path:
                    raise FileNotFoundError(
                        f"Native {modality} record {item['id']!r} needs a local media path; "
                        "captions cannot replace original media. Download/prepare the asset first."
                    )
                path = Path(raw_path)
                if not path.is_absolute():
                    raise ValueError(f"Record {item['id']!r} needs an absolute local media path.")
                if not path.is_file():
                    raise FileNotFoundError(f"Media for {item['id']!r} does not exist: {path}")
                prepared = {modality: str(path.resolve())}
            self._records.append(item)
            self._inputs.append(prepared)

    def _load_model(self) -> Any:
        from .av import configure_media_runtime
        configure_media_runtime()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval needs the optional embedding packages. Install current "
                "sentence-transformers, transformers, torch, numpy, pillow, soundfile, and "
                "torchcodec compatible with EmbeddingGemma 2. BM25 works without them."
            ) from exc
        kwargs: dict[str, Any] = {
            "device": self.device,
            "model_kwargs": {"torch_dtype": getattr(torch, self.dtype)},
        }
        if self.cache_dir is not None:
            kwargs["cache_folder"] = str((self.cache_dir / "model").resolve())
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.mode == "text":
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        try:
            return SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.model_name!r}. Verify network/model access and a "
                "Sentence Transformers/Transformers release supporting EmbeddingGemma 2. "
                "No fallback model was substituted."
            ) from exc

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def unload(self) -> None:
        """Release the encoder, retaining document vectors for the next search."""
        self._model = None
        import gc
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def build_index(self) -> None:
        """Prepare index and query model before measuring search latency.

        A cache hit avoids document encoding but still initializes the model
        needed to embed subsequent queries. Empty corpora need no model.
        """
        self._build_index()
        if self._records:
            self._get_model()

    def _fingerprint(self) -> str:
        records = []
        for item, prepared in zip(self._records, self._inputs):
            record = {"id": item["id"], "title": item.get("title"), "input": prepared}
            if isinstance(prepared, dict):
                record["media_sha256"] = _sha256_file(Path(next(iter(prepared.values()))))
            records.append(record)
        signature = {
            "version": _CACHE_VERSION, "model": self.model_name, "revision": self.revision,
            "mode": self.mode, "dtype": self.dtype, "processing": self.processing_kwargs,
            "records": records,
        }
        return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _encode(self, values: list[Any], **kwargs: Any) -> list[list[float]]:
        try:
            output = self._get_model().encode(
                values, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, **kwargs,
            )
        except (TypeError, ValueError, OSError) as exc:
            raise RuntimeError(
                "Embedding failed. Check native-media decoder dependencies, prepared clip "
                "lengths, and library support for EmbeddingGemma 2 multimodal inputs."
            ) from exc
        vectors = [_normalize(row) for row in output]
        if len(vectors) != len(values) or len({len(row) for row in vectors}) > 1:
            raise ValueError("Encoder returned the wrong number or shape of embeddings.")
        return vectors

    def _build_index(self) -> None:
        if self._vectors is not None:
            return
        if not self._records:
            self._vectors = []
            return
        cache_path = None
        fingerprint = None
        if self.cache_dir is not None:
            fingerprint = self._fingerprint()
            cache_path = self.cache_dir / f"embeddings-{fingerprint}.json"
            if cache_path.is_file():
                try:
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    if payload["fingerprint"] != fingerprint:
                        raise ValueError("Wrong cache fingerprint")
                    vectors = [_normalize(row) for row in payload["vectors"]]
                    if len(vectors) != len(self._records) or len({len(row) for row in vectors}) != 1:
                        raise ValueError("Wrong cached embedding shape")
                    self._vectors = vectors
                    return
                except (OSError, ValueError, TypeError, KeyError, OverflowError):
                    pass  # A damaged cache can be rebuilt from the source files.
        # Group by modality, retaining original positions. Text already carries
        # its title-aware Document prefix; native media have no text prefix.
        groups: dict[str, list[int]] = {}
        for index, prepared in enumerate(self._inputs):
            key = next(iter(prepared)) if isinstance(prepared, dict) else "text"
            groups.setdefault(key, []).append(index)
        indexed: dict[int, list[float]] = {}
        partial_dir = self.cache_dir / f"partial-{fingerprint}" if self.cache_dir is not None else None
        if partial_dir is not None and partial_dir.is_dir():
            # Independent atomic shards allow interrupted long CPU runs to
            # resume, including when the requested batch size changes.
            for path in sorted(partial_dir.glob("batch-*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload["fingerprint"] != fingerprint:
                        raise ValueError("Wrong partial-cache fingerprint")
                    positions = payload["indices"]
                    vectors = [_normalize(row) for row in payload["vectors"]]
                    if (len(positions) != len(vectors) or not positions
                            or any(type(index) is not int or not 0 <= index < len(self._records)
                                   for index in positions)
                            or len(set(positions)) != len(positions)
                            or len({len(row) for row in vectors}) != 1
                            or (indexed and len(vectors[0]) != len(next(iter(indexed.values()))))):
                        raise ValueError("Wrong partial-cache shape")
                    indexed.update(zip(positions, vectors))
                except (OSError, ValueError, TypeError, KeyError, OverflowError):
                    pass  # Re-encode only the missing or damaged batches.
        started = last_progress = time.monotonic()
        report_progress = len(self._records) > self.batch_size
        if report_progress:
            print(f"Embedding index: {len(indexed)}/{len(self._records)} cached; "
                  f"batch size {self.batch_size} ({self.mode}).", file=sys.stderr, flush=True)
        for kind, indices in groups.items():
            kwargs: dict[str, Any] = {"prompt": ""}
            if kind != "text":
                kwargs["processing_kwargs"] = self.processing_kwargs
            pending = [index for index in indices if index not in indexed]
            if kind == "text" and len(pending) > self.batch_size:
                # Like Sentence Transformers' own batching, group similar
                # lengths to avoid padding short passages to the longest one.
                pending.sort(key=lambda index: len(self._inputs[index]), reverse=True)
            for start in range(0, len(pending), self.batch_size):
                batch_indices = pending[start:start + self.batch_size]
                vectors = self._encode([self._inputs[index] for index in batch_indices], **kwargs)
                if indexed and len(vectors[0]) != len(next(iter(indexed.values()))):
                    raise ValueError("Different batches returned incompatible embedding dimensions.")
                indexed.update(zip(batch_indices, vectors))
                if partial_dir is not None:
                    key = hashlib.sha256(json.dumps(batch_indices).encode()).hexdigest()[:20]
                    _write_cache(partial_dir / f"batch-{key}.json", {
                        "fingerprint": fingerprint, "indices": batch_indices, "vectors": vectors,
                    })
                now = time.monotonic()
                if report_progress and (now - last_progress >= 30 or len(indexed) == len(self._records)):
                    print(f"Embedding index: {len(indexed)}/{len(self._records)} sources; "
                          f"{now - started:.0f}s elapsed ({kind}).", file=sys.stderr, flush=True)
                    last_progress = now
        ordered_vectors = [indexed[index] for index in range(len(self._records))]
        if len({len(row) for row in ordered_vectors}) != 1:
            raise ValueError("Different modalities returned incompatible embedding dimensions.")
        self._vectors = ordered_vectors
        if cache_path is not None:
            _write_cache(cache_path, {"fingerprint": fingerprint, "vectors": self._vectors})

    def search(self, query: str, top_k: int = 5,
               candidate_ids: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not top_k or not query.strip() or not self._records:
            return []
        candidates = candidate_indices(self._records, candidate_ids)
        if not candidates:
            return []
        self._build_index()
        query_vector = self._encode([query], prompt_name="SearchQuery")[0]
        assert self._vectors is not None
        if len(query_vector) != len(self._vectors[0]):
            raise ValueError("Query and index embedding dimensions differ; rebuild the index.")
        scored = [{"id": item["id"], "score": max(-1.0, min(1.0, sum(
            left * right for left, right in zip(vector, query_vector)
        )))} for index, (item, vector) in enumerate(zip(self._records, self._vectors))
                  if index in candidates]
        return sorted(scored, key=lambda item: (-item["score"], item["id"]))[:top_k]
