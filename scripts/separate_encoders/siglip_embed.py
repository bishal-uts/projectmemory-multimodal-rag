"""Embed the MultiModalQA images and questions with SigLIP 2 (runs on the cluster, offline).

Input bundle (made by run_separate_encoders.py export): images/, images.jsonl with
{"id", "file", "sha256"}, and questions.jsonl with {"id", "question"} only. No answers or
relevance labels leave the laptop. Writes L2-normalised float32 vectors to an .npz file
and a JSON record of the model revision, library versions, timings and text truncation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-so400m-patch14-384"
REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
MAX_TEXT_TOKENS = 64  # SigLIP 2 was trained with text padded to 64 tokens.


def features(output) -> torch.Tensor:
    """get_*_features returns a tensor in older transformers and an output object in newer ones."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(threads)
    images = read_jsonl(args.bundle / "images.jsonl")
    questions = read_jsonl(args.bundle / "questions.jsonl")
    if any(set(row) != {"id", "question"} for row in questions):
        raise SystemExit("questions.jsonl must hold only id and question")
    for row in images:
        digest = hashlib.sha256((args.bundle / "images" / row["file"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"Image {row['id']} changed in transfer")
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).eval()
    load_seconds = time.monotonic() - started

    texts = [row["question"] for row in questions]
    lengths = [len(ids) for ids in processor.tokenizer(texts)["input_ids"]]
    image_vectors, image_started = [], time.monotonic()
    with torch.inference_mode():
        text_inputs = processor(text=texts, padding="max_length", max_length=MAX_TEXT_TOKENS,
                                truncation=True, return_tensors="pt")
        text_vectors = torch.nn.functional.normalize(features(model.get_text_features(**text_inputs)), dim=-1)
        for start in range(0, len(images), args.batch_size):
            batch = images[start:start + args.batch_size]
            pixels = [Image.open(args.bundle / "images" / row["file"]).convert("RGB") for row in batch]
            inputs = processor(images=pixels, return_tensors="pt")
            image_vectors.append(torch.nn.functional.normalize(features(model.get_image_features(**inputs)), dim=-1))
            if start == 0:
                # The vectors must be the ones the model itself compares: check against its forward pass.
                check = model(**inputs, **{name: value[:2] for name, value in text_inputs.items()})
                for name, ours, theirs in (("image", image_vectors[0], check.image_embeds),
                                           ("text", text_vectors[:2], check.text_embeds)):
                    if not torch.allclose(ours, torch.nn.functional.normalize(theirs, dim=-1), atol=1e-4):
                        raise SystemExit(f"{name} features differ from the model's forward pass")
            done = min(start + args.batch_size, len(images))
            if done % 80 == 0 or done == len(images):
                print(f"{done}/{len(images)} images, {time.monotonic() - image_started:.0f} s", flush=True)
    image_matrix = torch.cat(image_vectors).numpy().astype(np.float32)
    text_matrix = text_vectors.numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, image_ids=np.array([row["id"] for row in images]), image_vectors=image_matrix,
             question_ids=np.array([row["id"] for row in questions]), questions=np.array(texts),
             question_vectors=text_matrix)
    record = {
        "model": MODEL, "revision": REVISION, "transformers": transformers.__version__, "torch": torch.__version__,
        "python": platform.python_version(), "host": platform.node(), "device": "cpu", "threads": threads,
        "dtype": "float32", "images": len(images), "questions": len(questions), "dimension": int(image_matrix.shape[1]),
        "text_max_tokens": MAX_TEXT_TOKENS, "questions_truncated": sum(length > MAX_TEXT_TOKENS for length in lengths),
        "longest_question_tokens": max(lengths), "load_seconds": round(load_seconds, 1),
        "image_seconds": round(time.monotonic() - image_started, 1),
        "image_processing": "SigLIP 2 processor defaults (resize to 384x384); RGB conversion",
        "text_processing": "Question text as written, padded to 64 tokens, no prompt template",
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
