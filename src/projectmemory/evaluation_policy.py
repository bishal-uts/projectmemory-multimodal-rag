"""Explicit benchmark policy for invalid local-model outputs, never outages.

The policy is opt-in. It changes scoring treatment after the existing bounded
formatting retries; it does not change prompts, sampling, or inference retries.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .generation import GenerationError
from .io import read_jsonl, write_json


INVALID_LOCAL_RESPONSE = "Local Gemma did not return a valid grounded JSON answer; no score was recorded"


class FormatFailureRecorder:
    """Count a precisely identified invalid model output as unanswered/incorrect.

    Network, startup, media, I/O, output-limit, and other failures still propagate.
    Successful answers are returned unchanged. Failures are checkpointed by the
    ordinary evaluator rather than inserted into the generation response cache.
    """

    def __init__(self, generator):
        if getattr(generator, "provider", None) != "local":
            raise ValueError("Recording format failures is supported only for the local Gemma provider")
        self.generator = generator
        self._last_config = {}

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def evaluation_config(self) -> dict:
        configuration = dict(self.generator.evaluation_config()) if hasattr(self.generator, "evaluation_config") else {}
        configuration["invalid_response_policy"] = {
            "version": 1,
            "action": "record_as_unanswered_and_incorrect_after_existing_format_retries",
            "eligible_exception": "GenerationError",
            "eligible_message": INVALID_LOCAL_RESPONSE,
            "all_other_failures": "abort_with_completed_predictions_preserved",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self._last_config = configuration
        return configuration

    def answer(self, question, evidence, **kwargs):
        try:
            return self.generator.answer(question, evidence, **kwargs)
        except GenerationError as exc:
            if str(exc) != INVALID_LOCAL_RESPONSE:
                raise
            return {
                "answer": "", "answer_items": [], "citations": [], "answer_label": "", "abstained": True,
                "evaluation_failure": True, "failure_type": "invalid_model_response",
                "failure_explanation": (
                    "Local generation exhausted its existing formatting retries without a valid grounded JSON answer. "
                    "Recorded as an unanswered, incorrect response under the fixed evaluation policy."
                ),
                "provider": self.provider, "model": self.model,
                "model_version": self._last_config.get("model_revision"),
                "usage": {}, "usage_unavailable": True, "cache_hit": False,
                "api_called": True, "cloud_api_called": False, "local_inference": True,
                "created_at": datetime.now(timezone.utc).isoformat(), "sources": [],
            }


def summarize_evaluation_failures(report: dict, output: Path) -> dict:
    """Attach counts from persisted predictions without changing their scores."""
    predictions = read_jsonl(Path(output) / "predictions.jsonl")
    if len(predictions) != report["completed"]:
        raise ValueError("Cannot summarize failures: prediction count differs from the completed report")
    counts = Counter(row["result"]["failure_type"] for row in predictions
                     if row["result"].get("evaluation_failure") is True)
    updated = dict(report)
    updated["evaluation_failures"] = {"count": sum(counts.values()), "by_type": dict(sorted(counts.items()))}
    updated["limitations"] = list(report.get("limitations", [])) + [
        "The opt-in policy records exhausted local formatting failures as unanswered and incorrect, not as service outages.",
        "Usage is unavailable for invalid responses; successful-response token totals exclude those failed attempts.",
    ]
    write_json(Path(output) / "report.json", updated)
    return updated
