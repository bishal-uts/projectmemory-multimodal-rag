"""Audit completed local answer experiments and render paired comparisons.

No model is loaded and no network request is made. All six conditions must be
complete before this script writes statistics or figures. It recomputes every
saved metric from the original prepared questions and checks run identities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from projectmemory.evaluation import corpus_fingerprint, validate_dataset
from projectmemory.generation import GeminiGenerator, GenerationError
from projectmemory.io import fingerprint, read_jsonl
from projectmemory.rag import answer_metrics


MMQA_METHODS = ("bm25", "dense-text", "multimodal")
AV_MODES = ("audio", "video", "combined")
MODEL = "google/gemma-4-E2B-it-qat-q4_0-gguf"
LABELS = {"bm25": "BM25", "dense-text": "Dense text", "multimodal": "Native multimodal",
          "audio": "Audio", "video": "Silent video", "combined": "Audio + video"}


class IntegrityError(ValueError):
    """Saved experiments cannot be compared without silently changing protocol."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise IntegrityError(f"Missing or invalid JSON: {path}") from None
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected a JSON object: {path}")
    return value


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def run_directories(root: Path, runs_root: Path | None = None) -> dict[str, dict[str, Path]]:
    directory = Path(runs_root) if runs_root is not None else Path("results")
    if not directory.is_absolute():
        directory = root / directory
    return {"mmqa": {method: directory / f"final-mmqa-local-{method}" for method in MMQA_METHODS},
            "av": {mode: directory / f"final-local-av-{mode}" for mode in AV_MODES}}


def readiness(root: Path, expected_mmqa_count: int = 100, expected_av_count: int = 20, *,
              runs_root: Path | None = None) -> dict:
    conditions = []
    for task, runs in run_directories(root, runs_root).items():
        expected = expected_mmqa_count if task == "mmqa" else expected_av_count
        for condition, directory in runs.items():
            missing = [name for name in ("run.json", "predictions.jsonl", "report.json") if not (directory / name).is_file()]
            prediction_file = directory / "predictions.jsonl"
            # An actively written final partial line is not a saved prediction.
            count = sum(bool(line.strip()) and line.endswith(b"\n") for line in prediction_file.read_bytes().splitlines(keepends=True)) if prediction_file.is_file() else 0
            conditions.append({"task": task, "condition": condition, "expected_questions": expected,
                               "saved_prediction_lines": count, "missing_files": missing,
                               "status": "ready_for_validation" if not missing and count == expected else "incomplete"})
    return {"status": "ready_for_validation" if all(item["status"] == "ready_for_validation" for item in conditions) else "incomplete",
            "conditions": conditions, "note": "No comparison statistics are produced until every condition is complete and passes integrity checks."}


def _same_metrics(saved: dict, recomputed: dict, context: str) -> None:
    require(isinstance(saved, dict) and set(saved) == set(recomputed), f"Metric names differ from recomputation: {context}")
    for name, expected in recomputed.items():
        value = saved[name]
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12),
                f"Stale or invalid metric {name}: {context}")


def aggregate_metrics(rows: list[dict]) -> dict:
    keys = {name for row in rows for name in row["metrics"]}
    return {name: mean(row["metrics"][name] for row in rows if name in row["metrics"]) for name in sorted(keys)}


def _validate_result(result: dict, evidence_ids: set[str], options, model: str) -> bool:
    require(isinstance(result, dict), "Prediction result is not an object")
    require(result.get("provider") == "local" and result.get("model") == model, "Prediction model/provider differs from run manifest")
    if result.get("evaluation_failure") is True:
        require(result.get("failure_type") == "invalid_model_response" and result.get("abstained") is True
                and result.get("answer") == "" and result.get("answer_items") == [] and result.get("citations") == []
                and result.get("answer_label") == "", "Invalid-response record must contain the explicit empty failure result")
        return True
    require(result.get("evaluation_failure") in (None, False) and not result.get("failure_type"), "Unrecognized evaluation failure record")
    try:
        GeminiGenerator._validate_answer(result, evidence_ids, options)
    except GenerationError as exc:
        raise IntegrityError(f"Saved answer failed grounded-answer validation: {exc}") from None
    return False


def load_run(directory: Path, corpus: list[dict], queries: list[dict], *, task: str, condition: str,
             expected_count: int, expected_corpus_fingerprint: str) -> dict:
    manifest, report = read_object(directory / "run.json"), read_object(directory / "report.json")
    try:
        predictions = read_jsonl(directory / "predictions.jsonl")
    except (OSError, ValueError):
        raise IntegrityError(f"Invalid predictions file: {directory}") from None
    require(len(queries) == expected_count, f"Prepared {task} question denominator changed")
    require(manifest.get("schema_version") == 2 and manifest.get("questions") == expected_count
            and report.get("completed") == expected_count and report.get("query_count") == expected_count
            and len(predictions) == expected_count, f"Incomplete or inconsistent question denominator: {directory.name}")
    require(manifest.get("provider") == "local" and manifest.get("model") == MODEL, "Final comparison requires the frozen local Gemma model")
    require(manifest.get("prediction_only") is False and manifest.get("split") is None, "Final comparison requires all selected questions with measured answers")
    require(manifest.get("corpus_fingerprint") == expected_corpus_fingerprint, f"Corpus/media fingerprint mismatch: {directory.name}")
    require(manifest.get("query_fingerprint") == fingerprint(queries), f"Question fingerprint mismatch: {directory.name}")
    require(report.get("task") == "answer_evaluation", "Expected an answer evaluation report")
    for key, value in manifest.items():
        require(key in report and report[key] == value, f"Report and run manifest differ at {key}: {directory.name}")
    require(isinstance(manifest.get("generator_config"), dict) and manifest["generator_config"].get("runtime_manifest_sha256"),
            "Final local run lacks frozen runtime/generator configuration")
    if task == "mmqa":
        require(manifest.get("method") == condition and manifest.get("k") == 5, "MMQA method or evidence cutoff differs from the planned comparison")
    else:
        require(manifest.get("media_mode") == condition, "AV condition does not match its media_mode")
    lookup = {record["id"]: record for record in corpus}
    expected_ids = [query["id"] for query in queries]
    actual_ids = [row.get("id") for row in predictions]
    require(len(actual_ids) == len(set(actual_ids)) and actual_ids == expected_ids,
            f"Prediction IDs/order do not exactly cover the selected questions: {directory.name}")
    invalid_count = 0
    for row, query in zip(predictions, queries):
        require(row.get("question") == query["question"], "Saved question text differs from reference data")
        evidence = row.get("evidence_ids")
        require(isinstance(evidence, list) and all(isinstance(item, str) for item in evidence)
                and len(evidence) == len(set(evidence)) and set(evidence).issubset(lookup), "Prediction evidence contains missing or duplicate corpus IDs")
        if task == "mmqa":
            require(0 < len(evidence) <= 5 and bool(query.get("relevant_ids")), "MMQA evidence or gold support list is invalid")
        else:
            require(len(query.get("options", [])) == 4 and isinstance(query.get("answer_label"), str)
                    and query["answer_label"] in {"A", "B", "C", "D"}, "AV baseline requires exactly four choices and a valid gold letter")
            kinds = {"audio", "video"} if condition == "combined" else {condition}
            expected_evidence = [identity for identity in query["metadata"]["known_media_ids"] if lookup[identity]["modality"] in kinds]
            require(evidence == expected_evidence, "AV conditions must use the identical known clip with only the specified modalities")
        result = row.get("result")
        invalid_count += _validate_result(result, set(evidence), query.get("options"), manifest["model"])
        require(row.get("answer_label") == result.get("answer_label"), "Prediction and result choice labels differ")
        measured = answer_metrics(query, result)
        _same_metrics(row.get("metrics"), measured, query["id"])
    _same_metrics(report.get("metrics"), aggregate_metrics(predictions), directory.name)
    failures = {"count": invalid_count, "by_type": {"invalid_model_response": invalid_count} if invalid_count else {}}
    require(report.get("evaluation_failures") == failures, "Report invalid-model-response count differs from saved predictions")
    return {"manifest": manifest, "report": report, "predictions": predictions, "invalid_model_response_count": invalid_count,
            "files": {name: file_hash(directory / name) for name in ("run.json", "report.json", "predictions.jsonl")}}


def compatible_runs(runs: dict[str, dict], *, task: str) -> None:
    first = next(iter(runs.values()))["manifest"]
    common = ("schema_version", "corpus_fingerprint", "query_fingerprint", "questions", "k", "split", "provider", "model",
              "max_output_tokens", "reasoning_budget", "prediction_only", "generator_config", "implementation")
    for condition, run in runs.items():
        manifest = run["manifest"]
        for key in common:
            require(manifest.get(key) == first.get(key), f"Unpaired {task} experiment: {key} differs for {condition}")
        if task == "av":
            require(manifest.get("method") == first.get("method") and manifest.get("retriever_config") == first.get("retriever_config"),
                    "AV conditions changed configuration beyond media_mode")
        else:
            require(manifest.get("media_mode") == first.get("media_mode"), "MMQA generation media_mode differs across retrieval methods")
    # Retrieval methods legitimately differ. For replay runs, only the saved
    # ranking's revision should vary, not the replay backend configuration.
    replay = [run["manifest"].get("retriever_config", {}) for run in runs.values()
              if run["manifest"].get("retriever_config", {}).get("mode") == "verified-ranking-replay"]
    if len(replay) > 1:
        stripped = [{key: value for key, value in config.items() if key != "revision"} for config in replay]
        require(all(config == stripped[0] for config in stripped), "Saved-ranking replay settings differ beyond the report revision")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(baseline: list[float], comparison: list[float], *, clusters: list[str] | None = None,
                     repetitions: int = 10_000, seed: int = 42) -> dict:
    require(len(baseline) == len(comparison) and bool(baseline), "Paired bootstrap needs matching nonempty vectors")
    require(all(value in (0, 1) for value in baseline + comparison), "Accuracy bootstrap inputs must be binary")
    require(isinstance(repetitions, int) and not isinstance(repetitions, bool) and repetitions >= 100, "Use at least 100 bootstrap resamples")
    differences = [float(right) - float(left) for left, right in zip(baseline, comparison)]
    cluster_labels = clusters if clusters is not None else [str(index) for index in range(len(differences))]
    require(len(cluster_labels) == len(differences) and all(isinstance(label, str) and label for label in cluster_labels), "Invalid paired bootstrap cluster labels")
    grouped = defaultdict(list)
    for label, difference in zip(cluster_labels, differences):
        grouped[label].append(difference)
    units = [(sum(group), len(group)) for group in grouped.values()]
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [units[rng.randrange(len(units))] for _ in units]
        estimates.append(sum(value for value, _ in sampled) / sum(size for _, size in sampled))
    return {"difference": mean(differences), "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
            "questions": len(differences), "resampling_units": len(units),
            "resampling_unit": "video_id cluster" if clusters is not None else "question",
            "repetitions": repetitions, "seed": seed, "interval": "paired percentile bootstrap; exploratory, not multiplicity-adjusted"}


def summarize_rows(rows: list[dict], primary_metric: str) -> dict:
    return {"questions": len(rows), "correct": sum(row["metrics"][primary_metric] == 1 for row in rows),
            "answered": sum(not row["result"]["abstained"] for row in rows),
            "abstained": sum(row["result"]["abstained"] for row in rows),
            "invalid_model_response_count": sum(row["result"].get("evaluation_failure") is True for row in rows),
            "metrics": aggregate_metrics(rows),
            "metric_denominators": {name: sum(name in row["metrics"] for row in rows) for name in aggregate_metrics(rows)}}


def question_group(query: dict, lookup: dict, task: str) -> str:
    if task == "av":
        return query["metadata"].get("question_type", "unknown")
    modalities = {lookup[identity].get("metadata", {}).get("original_modality", lookup[identity]["modality"])
                  for identity in query["relevant_ids"]}
    return "+".join(sorted(modalities))


def summarize_task(runs: dict[str, dict], corpus: list[dict], queries: list[dict], *, task: str, repetitions: int = 10_000) -> dict:
    lookup = {record["id"]: record for record in corpus}
    primary = "answer_set_exact_match" if task == "mmqa" else "choice_accuracy"
    result = {"primary_metric": primary, "question_count": len(queries), "conditions": {}, "paired_comparisons": [],
              "query_ids": [query["id"] for query in queries],
              "corpus_fingerprint": next(iter(runs.values()))["manifest"]["corpus_fingerprint"],
              "query_fingerprint": next(iter(runs.values()))["manifest"]["query_fingerprint"]}
    for condition, run in runs.items():
        rows = run["predictions"]
        summary = summarize_rows(rows, primary)
        groups = defaultdict(list)
        for query, row in zip(queries, rows):
            groups[question_group(query, lookup, task)].append(row)
        summary["by_gold_modalities" if task == "mmqa" else "by_question_category"] = {
            group: summarize_rows(group_rows, primary) for group, group_rows in sorted(groups.items())}
        summary["input_file_sha256"] = run["files"]
        if task == "mmqa":
            cells = defaultdict(list)
            for query, row in zip(queries, rows):
                all_gold = set(query["relevant_ids"]).issubset(row["evidence_ids"])
                cell = ("all_gold_retrieved" if all_gold else "missing_some_gold") + ("_correct" if row["metrics"][primary] == 1 else "_incorrect")
                cells[cell].append(row)
            summary["retrieval_answer_association"] = {
                cell: summarize_rows(cells[cell], primary) for cell in (
                    "all_gold_retrieved_correct", "all_gold_retrieved_incorrect", "missing_some_gold_correct", "missing_some_gold_incorrect")}
            summary["retrieval_answer_association_note"] = "Descriptive association only. Missing annotated sources does not prove retrieval caused an error; all gold retrieved does not establish sufficient understanding. Incorrect includes abstentions and recorded invalid outputs."
        result["conditions"][condition] = summary
    clusters = None
    if task == "av":
        clusters = [query["metadata"]["video_id"] for query in queries]
        require(all(isinstance(clip, str) and clip for clip in clusters), "AV questions lack stable video_id clusters")
        result["unique_clips"] = len(set(clusters))
        result["clip_question_counts"] = dict(sorted(Counter(clusters).items()))
        result["baselines"] = {"always_A": {"correct": sum(query["answer_label"] == "A" for query in queries),
                                               "questions": len(queries), "choice_accuracy": mean(query["answer_label"] == "A" for query in queries)},
                               "uniform_random_expected": {"choices": 4, "choice_accuracy": 0.25,
                                                           "note": "Analytical expectation, not a sampled model or post-hoc majority baseline."}}
        result["gold_label_counts"] = dict(sorted(Counter(query["answer_label"] for query in queries).items()))
    for baseline, comparison in itertools.combinations(runs, 2):
        interval = paired_bootstrap([row["metrics"][primary] for row in runs[baseline]["predictions"]],
                                    [row["metrics"][primary] for row in runs[comparison]["predictions"]],
                                    clusters=clusters, repetitions=repetitions)
        result["paired_comparisons"].append({"baseline": baseline, "comparison": comparison,
                                            "direction": f"{comparison} minus {baseline}", **interval})
    return result


def analyze(root: Path, *, runs_root: Path | None = None, mmqa_dataset: Path | None = None, av_dataset: Path | None = None,
            expected_mmqa_count: int = 100, expected_av_count: int = 20, repetitions: int = 10_000) -> dict:
    root = Path(root).resolve()
    status = readiness(root, expected_mmqa_count, expected_av_count, runs_root=runs_root)
    if status["status"] != "ready_for_validation":
        return status
    datasets = {"mmqa": mmqa_dataset or root / "data/prepared/mmqa",
                "av": av_dataset or root / "data/prepared/daily-omni-local-normalized"}
    output = {"schema_version": 1, "status": "complete", "model": MODEL, "provider": "local", "experiments": {},
              "analysis_script_sha256": file_hash(Path(__file__)),
              "metric_implementation_sha256": file_hash(ROOT / "src/projectmemory/rag.py"),
              "limitations": [
                  "MMQA uses the same 100-question development pool used during development, not an untouched test set or official leaderboard protocol.",
                  "The AV experiment uses a duration-filtered, boundary-normalized subset; it is not an untouched full-clip benchmark.",
                  "Invalid model responses and abstentions remain in the accuracy denominator. Transport/runtime failures abort evaluation and are not fabricated as answers.",
                  "Citation gold overlap is not a measure of claim entailment or factual faithfulness.",
                  "Intervals are exploratory paired percentile bootstrap intervals without multiplicity adjustment. MMQA questions may share supporting sources; AV resampling preserves video clusters.",
                  "Retrieval-error associations are descriptive, not causal. Lexical exact match can penalize semantically equivalent wording.",
              ]}
    loaded = {}
    for task, directories in run_directories(root, runs_root).items():
        dataset = Path(datasets[task])
        corpus, queries = read_jsonl(dataset / "corpus.jsonl"), read_jsonl(dataset / "queries.jsonl")
        validate_dataset(corpus, queries)
        current_fingerprint = corpus_fingerprint(corpus)
        expected = expected_mmqa_count if task == "mmqa" else expected_av_count
        runs = {condition: load_run(directory, corpus, queries, task=task, condition=condition,
                                    expected_count=expected, expected_corpus_fingerprint=current_fingerprint)
                for condition, directory in directories.items()}
        compatible_runs(runs, task=task)
        loaded[task] = runs
        output["experiments"][task] = summarize_task(runs, corpus, queries, task=task, repetitions=repetitions)
        task_manifest = next(iter(runs.values()))["manifest"]
        output["experiments"][task]["generator_config"] = task_manifest["generator_config"]
        output["experiments"][task]["implementation"] = task_manifest.get("implementation", {})
        output["experiments"][task]["dataset_files_sha256"] = {name: file_hash(dataset / name) for name in ("corpus.jsonl", "queries.jsonl")}
        if task == "av":
            manifest = read_object(dataset / "manifest.json")
            require(manifest.get("dataset") == "daily_omni_local_boundary_normalized_subset"
                    and manifest.get("query_fingerprint") == fingerprint(queries)
                    and manifest.get("corpus_records_fingerprint") == fingerprint(corpus)
                    and manifest.get("selected_ids") == [query["id"] for query in queries],
                    "AV subset manifest does not identify the exact boundary-normalized questions/corpus")
            output["experiments"][task]["subset_protocol"] = {key: manifest[key] for key in ("dataset", "selection", "eligibility", "normalized_audio_files") if key in manifest}
            output["experiments"][task]["dataset_files_sha256"]["manifest.json"] = file_hash(dataset / "manifest.json")
    frozen = loaded["mmqa"]["bm25"]["manifest"]
    av_frozen = loaded["av"]["audio"]["manifest"]
    differing_fields = [key for key in ("max_output_tokens", "reasoning_budget", "generator_config", "implementation")
                        if frozen.get(key) != av_frozen.get(key)]
    output["cross_experiment_configuration"] = {
        "identical": not differing_fields, "differing_fields": differing_fields,
        "note": "Code and generation configuration must match within each paired experiment. MMQA and AV are separate tasks; no accuracy difference between them is estimated. Historical code hashes are retained without requiring the current generator source to match.",
    }
    if differing_fields:
        output["limitations"].append("MMQA and AV use separately frozen configurations, differing at: " + ", ".join(differing_fields) + ". Each task retains its own generator configuration and implementation hashes; comparisons are only within each task.")
    return output


def markdown(summary: dict) -> str:
    lines = ["# Final local answer comparison", "", f"Generator: `{summary['model']}`. All values below were recomputed and checked against saved predictions and reports.", ""]
    for task, title in (("mmqa", "MultiModalQA development pool"), ("av", "Daily-Omni boundary-normalized subset")):
        experiment = summary["experiments"][task]
        lines += [f"## {title}", "", f"Questions: **{experiment['question_count']}**." + (f" Unique clips: **{experiment['unique_clips']}**." if task == "av" else " Same pooled corpus and k=5 across retrieval methods."), "",
                  "| Condition | Correct / N | Accuracy | Coverage | Invalid outputs | Gold citation precision | Gold citation recall |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for condition, result in experiment["conditions"].items():
            metrics = result["metrics"]
            citation = [f"{metrics[key]:.1%}" if key in metrics else "Not annotated" for key in ("citation_gold_precision", "citation_gold_recall")]
            lines.append(f"| {LABELS[condition]} | {result['correct']} / {result['questions']} | {metrics[experiment['primary_metric']]:.1%} | {metrics['answer_coverage']:.1%} | {result['invalid_model_response_count']} | {' | '.join(citation)} |")
        if task == "av":
            baseline = experiment["baselines"]["always_A"]
            lines += ["", f"Prespecified always-A baseline: **{baseline['choice_accuracy']:.1%}** ({baseline['correct']}/{baseline['questions']}). Uniform random choice has **25% expected accuracy**; this is an analytical expectation, not a measured random trial."]
        lines += ["", "### Paired accuracy differences", "", "Positive values favor the first named condition. Intervals are exploratory 95% percentile bootstrap intervals (10,000 resamples by default, seed 42).", "",
                  "| Comparison minus baseline | Difference (percentage points) | 95% interval (pp) | Resampling unit |",
                  "|---|---:|---:|---|"]
        for interval in experiment["paired_comparisons"]:
            lines.append(f"| {LABELS[interval['comparison']]} − {LABELS[interval['baseline']]} | {interval['difference'] * 100:+.2f} | [{interval['ci95'][0] * 100:+.2f}, {interval['ci95'][1] * 100:+.2f}] | {interval['resampling_unit']} ({interval['resampling_units']}) |")
        group_key = "by_gold_modalities" if task == "mmqa" else "by_question_category"
        lines += ["", "### " + ("Gold-source modality groups" if task == "mmqa" else "Question categories"), "",
                  "| Condition | Group | N | Accuracy | Coverage | Invalid outputs |", "|---|---|---:|---:|---:|---:|"]
        for condition, result in experiment["conditions"].items():
            for group, row in result[group_key].items():
                lines.append(f"| {LABELS[condition]} | {group} | {row['questions']} | {row['metrics'][experiment['primary_metric']]:.1%} | {row['metrics']['answer_coverage']:.1%} | {row['invalid_model_response_count']} |")
        if task == "mmqa":
            lines += ["", "### Retrieval and answer error association", "", "| Condition | All gold / correct | All gold / incorrect | Missing gold / correct | Missing gold / incorrect |", "|---|---:|---:|---:|---:|"]
            for condition, result in experiment["conditions"].items():
                cells = result["retrieval_answer_association"]
                lines.append(f"| {LABELS[condition]} | " + " | ".join(str(cells[cell]["questions"]) for cell in ("all_gold_retrieved_correct", "all_gold_retrieved_incorrect", "missing_some_gold_correct", "missing_some_gold_incorrect")) + " |")
            lines += ["", "These are associations, not causal diagnoses. Incorrect includes abstentions and invalid outputs; retrieving all annotated sources does not prove that the answer model understood them."]
        lines += [""]
    lines += ["## Interpretation limits", "", *[f"- {note}" for note in summary["limitations"]], "", "![Answer accuracy and coverage](answer-comparison.png)", "", "![Paired accuracy differences](paired-differences.png)", ""]
    return "\n".join(lines)


def plot(summary: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.80, wspace=0.18)
    figure.suptitle("Local Gemma 4 E2B: answer accuracy and coverage", fontsize=16, y=0.96)
    for axis, task in zip(axes, ("mmqa", "av")):
        experiment = summary["experiments"][task]
        conditions = experiment["conditions"]
        primary = experiment["primary_metric"]
        positions = list(range(len(conditions)))
        for offset, metric, color, label in ((-0.18, primary, "#2563eb", "Accuracy"), (0.18, "answer_coverage", "#0f766e", "Coverage")):
            values = [result["metrics"][metric] * 100 for result in conditions.values()]
            bars = axis.bar([position + offset for position in positions], values, 0.32, color=color, label=label, zorder=3)
            axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=9)
        axis.set_xticks(positions, [LABELS[condition].replace("Native ", "Native\n").replace("Audio + ", "Audio +\n") for condition in conditions])
        axis.set_ylim(0, 110)
        axis.set_yticks(range(0, 101, 20))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(axis="y", color="#e2e8f0", zorder=0)
        axis.set_title(("MMQA exact match" if task == "mmqa" else "AV choice accuracy") + f" | n={experiment['question_count']}")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if task == "av":
            axis.axhline(experiment["baselines"]["always_A"]["choice_accuracy"] * 100, color="#c2410c", linestyle=":", linewidth=1.5, label="Always A")
            axis.axhline(25, color="#475569", linestyle="--", linewidth=1.2, label="Random expectation")
    axes[0].set_ylabel("Questions (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.1, -0.17), ncol=4, frameon=False, fontsize=9)
    figure.text(0.08, 0.035, "Fixed development subsets. Abstentions and invalid outputs remain in accuracy denominators. Coverage = a non-abstaining answer.", fontsize=9, color="#475569")
    for extension in ("png", "svg"):
        figure.savefig(destination / f"answer-comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    figure.subplots_adjust(left=0.19, right=0.98, top=0.79, bottom=0.22, wspace=0.72)
    figure.suptitle("Paired accuracy differences with exploratory 95% intervals", fontsize=15, y=0.96)
    largest = max(abs(bound) * 100 for experiment in summary["experiments"].values() for row in experiment["paired_comparisons"] for bound in row["ci95"])
    extent = min(100, max(15, math.ceil((largest + 5) / 5) * 5))
    for axis, task in zip(axes, ("mmqa", "av")):
        experiment = summary["experiments"][task]
        for index, row in enumerate(experiment["paired_comparisons"]):
            value, (low, high) = row["difference"] * 100, [bound * 100 for bound in row["ci95"]]
            axis.plot([low, high], [index, index], color="#2563eb", linewidth=2)
            axis.scatter([value], [index], color="#0f172a", s=30, zorder=4)
        axis.set_yticks(range(3), [f"{LABELS[row['comparison']]}\n− {LABELS[row['baseline']]}" for row in experiment["paired_comparisons"]])
        axis.invert_yaxis()
        axis.set_ylim(2.7, -0.7)
        axis.set_xlim(-extent, extent)
        axis.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
        axis.grid(axis="x", color="#e2e8f0")
        axis.set_xlabel("Accuracy difference (percentage points)")
        axis.set_title("MMQA: question resampling" if task == "mmqa" else f"AV: {experiment['unique_clips']} clip clusters")
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    figure.text(0.08, 0.035, "Positive favors the first named condition. Percentile bootstrap, seed 42; exploratory intervals without multiplicity adjustment.", fontsize=9, color="#475569")
    for extension in ("png", "svg"):
        figure.savefig(destination / f"paired-differences.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_analysis(summary: dict, output: Path) -> None:
    require(summary.get("status") == "complete", "Cannot publish statistics for incomplete runs")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analysis_", dir=output) as temporary:
        stage = Path(temporary)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "comparison.md").write_text(markdown(summary), encoding="utf-8")
        plot(summary, stage)
        for source in stage.iterdir():
            os.replace(source, output / source.name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path,
                        help="Directory holding all six run folders; defaults to <root>/results. Relative paths are resolved from --root.")
    parser.add_argument("--mmqa-dataset", type=Path)
    parser.add_argument("--av-dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.root, runs_root=args.runs_root, mmqa_dataset=args.mmqa_dataset,
                          av_dataset=args.av_dataset, repetitions=args.resamples)
        if summary["status"] != "complete":
            print(json.dumps(summary, indent=2))
            return 2
        output = args.output or args.root / "results/final-analysis"
        write_analysis(summary, output)
        print(json.dumps({"status": "complete", "output": str(output.resolve()), "files": ["summary.json", "comparison.md", "answer-comparison.png", "answer-comparison.svg", "paired-differences.png", "paired-differences.svg"]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Final analysis stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
