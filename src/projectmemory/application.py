"""Application evidence selection; controlled evaluations use rag.py directly."""
from __future__ import annotations

import math
from pathlib import Path
import re

from .av import probe_media
from .rag import ask_question


def _pair_key(record):
    if record.get("modality") not in {"audio", "video"} or not record.get("source_id"):
        return None
    meta = record.get("metadata", {})
    segment = re.fullmatch(r"av:([0-9a-f]{20}):(audio|video):(\d{6})", record.get("id", ""))
    if segment and segment.group(2) == record["modality"]:
        # Ingestion retains stream offsets: corresponding audio/video may have
        # different real timestamps but share this configuration/segment index.
        return (record["source_id"], "segment", segment.group(1), segment.group(3))
    start, end = meta.get("start_seconds"), meta.get("end_seconds")
    if start is not None and end is not None:
        return (record["source_id"], float(start), float(end))
    if meta.get("evaluation_protocol") == "known_video_qa" or meta.get("dataset") == "daily_omni":
        return (record["source_id"], "full-clip")
    return None


def select_local_evidence(corpus, ranked):
    """Greedily retain whole ranked units and synchronized AV counterparts.

    Durations come from actual files, not metadata. No source is cropped. A
    paired segment is atomic: if it cannot fit, skip both and explain why.
    This policy is deliberately excluded from the retrieval/answer experiments.
    """
    groups = {}
    for record in corpus:
        key = _pair_key(record)
        if key is not None:
            groups.setdefault(key, []).append(record)
    selected, selected_ids, processed, omitted = [], set(), set(), []
    totals = {"audio_seconds": 0.0, "video_seconds": 0.0, "images_and_frames": 0, "media_bytes": 0}
    limits = {"audio_seconds": 30, "video_seconds": 60, "images_and_frames": 64, "media_bytes": 32_000_000}
    costs = {}

    def cost(record):
        identity = record["id"]
        if identity in costs:
            return costs[identity]
        value = dict.fromkeys(totals, 0)
        modality = record.get("modality", "text")
        if modality != "text":
            path = Path(record.get("path", ""))
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"Evidence {identity} has no readable local media file")
            value["media_bytes"] = path.stat().st_size
            if modality == "image":
                value["images_and_frames"] = 1
            elif modality in {"audio", "video"}:
                duration = probe_media(path)["duration_seconds"]
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Evidence {identity} has an invalid duration")
                value[modality + "_seconds"] = duration
                if modality == "video":
                    value["images_and_frames"] = math.ceil(duration)
        costs[identity] = value
        return value

    for record in ranked:
        if record["id"] in processed:
            continue
        unit = [record]
        key = _pair_key(record)
        counterparts = [item for item in groups.get(key, [])
                        if item["modality"] != record["modality"]]
        # Ambiguous duplicates must not be guessed into a synchronized pair.
        if len(counterparts) == 1:
            unit.append(dict(counterparts[0], paired_with=record["id"]))
        unit = [item for item in unit if item["id"] not in selected_ids]
        proposed = {name: totals[name] + sum(cost(item)[name] for item in unit) for name in totals}
        exceeded = [name for name in totals if proposed[name] > limits[name]]
        processed.update(item["id"] for item in unit)
        if exceeded:
            omitted.extend({"id": item["id"], "reason": "Local request limit: " + ", ".join(exceeded)} for item in unit)
            continue
        selected.extend(unit)
        selected_ids.update(item["id"] for item in unit)
        totals = proposed
    if ranked and not selected:
        raise ValueError("None of the retrieved media segments fit the local model limits. Import the source as shorter clips or choose another source.")
    ranked_ids = {item["id"] for item in ranked}
    added = [item["id"] for item in selected if item["id"] not in ranked_ids]
    notice = ""
    if added or omitted:
        notice = f"Answer uses {len(selected)} evidence records: {len(added)} matching audio/video records added; {len(omitted)} records omitted to fit local limits. Sources were not cropped."
    return selected, {"policy": "whole-segment AV pairing and ranked budget selection v1",
                      "selected_ids": [item["id"] for item in selected], "added_ids": added,
                      "omitted": omitted, "totals": totals, "limits": limits, "notice": notice}


def ask_application_question(corpus, question, retriever, generator=None, *, top_k=5, source_id=None, provider=None):
    if generator is None or (provider or getattr(generator, "provider", None)) != "local":
        return ask_question(corpus, question, retriever, generator, top_k, source_id)

    class PlannedGenerator:
        provider = "local"
        evidence = None
        plan = None

        def answer(self, prompt, evidence):
            self.evidence, self.plan = select_local_evidence(corpus, evidence)
            return generator.answer(prompt, self.evidence)

    planned = PlannedGenerator()
    result = ask_question(corpus, question, retriever, planned, top_k, source_id)
    result["retrieved_ids"] = [item["id"] for item in result["evidence"]]
    result["evidence"] = planned.evidence
    result["evidence_plan"] = planned.plan
    return result
