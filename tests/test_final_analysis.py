import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from projectmemory.evaluation import corpus_fingerprint
from projectmemory.io import fingerprint, write_json, write_jsonl
from projectmemory.rag import answer_metrics


_SPEC = importlib.util.spec_from_file_location("analyze_final_results", Path(__file__).resolve().parents[1] / "scripts/analyze_final_results.py")
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def answer(value, citation, label=""):
    return {"answer": value, "answer_items": [value], "citations": [citation], "answer_label": label,
            "abstained": False, "provider": "local", "model": analysis.MODEL}


def invalid():
    return {"answer": "", "answer_items": [], "citations": [], "answer_label": "", "abstained": True,
            "evaluation_failure": True, "failure_type": "invalid_model_response", "provider": "local", "model": analysis.MODEL}


def write_run(path, corpus, queries, condition, task, answers, evidence):
    manifest = {"schema_version": 2, "corpus_fingerprint": corpus_fingerprint(corpus), "query_fingerprint": fingerprint(queries),
                "questions": len(queries), "method": condition if task == "mmqa" else "bm25", "k": 5, "split": None,
                "provider": "local", "model": analysis.MODEL, "max_output_tokens": 4096, "reasoning_budget": None,
                "retriever_config": ({"model_name": "saved-retrieval-rankings", "mode": "verified-ranking-replay", "revision": condition} if task == "mmqa" else {"k1": 1.5, "b": .75}),
                "implementation": {"rag.py": "fixed-code-fingerprint"}, "media_mode": "combined" if task == "mmqa" else condition,
                "prediction_only": False, "generator_config": {"runtime_manifest_sha256": "a" * 64, "model_revision": "fixed-revision"}}
    rows = [{"id": query["id"], "question": query["question"], "answer_label": result["answer_label"],
             "evidence_ids": ids, "result": result, "metrics": answer_metrics(query, result)}
            for query, result, ids in zip(queries, answers, evidence)]
    count = sum(row["result"].get("evaluation_failure") is True for row in rows)
    report = {**manifest, "task": "answer_evaluation", "completed": len(rows), "query_count": len(rows),
              "metrics": analysis.aggregate_metrics(rows), "evaluation_failures": {"count": count, "by_type": {"invalid_model_response": count} if count else {}}}
    write_json(path / "run.json", manifest)
    write_json(path / "report.json", report)
    write_jsonl(path / "predictions.jsonl", rows)


@pytest.fixture
def project(tmp_path):
    corpus = [{"id": "doc:1", "modality": "text", "text": "Blue.", "metadata": {"original_modality": "text"}},
              {"id": "doc:2", "modality": "text", "text": "Two.", "metadata": {"original_modality": "table"}}]
    queries = [{"id": "q1", "question": "Colour?", "reference_answers": ["blue"], "relevant_ids": ["doc:1"], "metadata": {"dataset": "multimodalqa"}},
               {"id": "q2", "question": "Number?", "reference_answers": ["two"], "relevant_ids": ["doc:2"], "metadata": {"dataset": "multimodalqa"}}]
    av_corpus = [{"id": "audio:1", "modality": "audio", "text": ""}, {"id": "video:1", "modality": "video", "text": ""}]
    av_queries = [{"id": "a1", "question": "First?", "reference_answers": ["second"], "answer_label": "B", "options": ["first", "second", "third", "fourth"],
                   "metadata": {"dataset": "daily_omni", "video_id": "clip-1", "question_type": "sequence", "known_media_ids": ["audio:1", "video:1"]}},
                  {"id": "a2", "question": "Next?", "reference_answers": ["first"], "answer_label": "A", "options": ["first", "second", "third", "fourth"],
                   "metadata": {"dataset": "daily_omni", "video_id": "clip-1", "question_type": "alignment", "known_media_ids": ["audio:1", "video:1"]}}]
    for name, records, questions in (("mmqa", corpus, queries), ("daily-omni-local-normalized", av_corpus, av_queries)):
        folder = tmp_path / "data/prepared" / name
        write_jsonl(folder / "corpus.jsonl", records)
        write_jsonl(folder / "queries.jsonl", questions)
    write_json(tmp_path / "data/prepared/daily-omni-local-normalized/manifest.json", {
        "dataset": "daily_omni_local_boundary_normalized_subset", "query_fingerprint": fingerprint(av_queries),
        "corpus_records_fingerprint": fingerprint(av_corpus), "selected_ids": [q["id"] for q in av_queries],
        "normalized_audio_files": 1, "selection": {"seed": 42}, "eligibility": {"max_terminal_audio_trim_seconds": .001}})
    configs = {
        "bm25": ([answer("blue", "doc:1"), answer("red", "doc:1")], [["doc:1"], ["doc:1"]]),
        "dense-text": ([answer("blue", "doc:1"), answer("two", "doc:2")], [["doc:1"], ["doc:2"]]),
        "multimodal": ([invalid(), answer("two", "doc:2")], [["doc:1"], ["doc:2"]]),
    }
    for method, (answers, evidence) in configs.items():
        write_run(tmp_path / f"results/final-mmqa-local-{method}", corpus, queries, method, "mmqa", answers, evidence)
    for mode in analysis.AV_MODES:
        ids = ["audio:1", "video:1"] if mode == "combined" else [mode + ":1"]
        answers = ([answer("first", ids[0], "A"), answer("first", ids[0], "A")] if mode == "audio"
                   else [answer("second", ids[0], "B"), answer("first", ids[0], "A")] if mode == "video"
                   else [answer("second", ids[0], "B"), invalid()])
        write_run(tmp_path / f"results/final-local-av-{mode}", av_corpus, av_queries, mode, "av", answers, [ids, ids])
    return tmp_path


def run_analysis(project):
    return analysis.analyze(project, expected_mmqa_count=2, expected_av_count=2, repetitions=200)


def test_recomputation_counts_failures_subgroups_retrieval_association_and_baselines(project):
    result = run_analysis(project)
    mmqa, av = result["experiments"]["mmqa"], result["experiments"]["av"]
    assert result["status"] == "complete"
    assert mmqa["conditions"]["bm25"]["metrics"]["answer_set_exact_match"] == .5
    assert mmqa["conditions"]["dense-text"]["metrics"]["answer_set_exact_match"] == 1
    native = mmqa["conditions"]["multimodal"]
    assert native["invalid_model_response_count"] == 1 and native["answered"] == 1 and native["questions"] == 2
    assert native["metrics"]["answer_coverage"] == .5
    assert native["by_gold_modalities"]["table"]["correct"] == 1
    assert native["retrieval_answer_association"]["all_gold_retrieved_incorrect"]["questions"] == 1
    assert mmqa["conditions"]["bm25"]["retrieval_answer_association"]["missing_some_gold_incorrect"]["questions"] == 1
    assert av["baselines"]["always_A"]["choice_accuracy"] == .5
    assert av["baselines"]["uniform_random_expected"]["choice_accuracy"] == .25
    assert av["unique_clips"] == 1 and av["paired_comparisons"][0]["resampling_units"] == 1
    assert av["conditions"]["combined"]["invalid_model_response_count"] == 1
    assert "citation_gold_precision" not in av["conditions"]["audio"]["metrics"]
    assert mmqa["paired_comparisons"][0]["difference"] == .5


def test_paired_bootstrap_preserves_clip_clusters_and_query_weighted_estimate():
    clustered = analysis.paired_bootstrap([0, 0, 1], [1, 1, 0], clusters=["same", "same", "other"], repetitions=1000)
    assert clustered["difference"] == pytest.approx(1 / 3)
    assert clustered["resampling_units"] == 2 and clustered["ci95"] == [-1, 1]
    assert clustered == analysis.paired_bootstrap([0, 0, 1], [1, 1, 0], clusters=["same", "same", "other"], repetitions=1000)
    uniform = analysis.paired_bootstrap([0, 0], [1, 1], repetitions=100)
    assert uniform["difference"] == 1 and uniform["ci95"] == [1, 1]


def test_incomplete_runs_produce_status_only_without_invented_metrics(project):
    (project / "results/final-local-av-combined/report.json").unlink()
    result = run_analysis(project)
    assert result["status"] == "incomplete" and "experiments" not in result
    pending = next(row for row in result["conditions"] if row["task"] == "av" and row["condition"] == "combined")
    assert pending["saved_prediction_lines"] == 2 and pending["missing_files"] == ["report.json"]
    with pytest.raises(analysis.IntegrityError, match="incomplete"):
        analysis.write_analysis(result, project / "results/final-analysis")
    assert not (project / "results/final-analysis").exists()


def mutate_json(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    write_json(path, payload)


def test_stale_aggregate_metrics_are_rejected(project):
    mutate_json(project / "results/final-mmqa-local-bm25/report.json", lambda report: report["metrics"].update(answer_set_exact_match=1.0))
    with pytest.raises(analysis.IntegrityError, match="Stale or invalid metric"):
        run_analysis(project)


def test_tampering_both_row_and_report_metrics_still_fails_recomputation(project):
    path = project / "results/final-mmqa-local-bm25/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["metrics"]["answer_set_exact_match"] = 1.0
    write_jsonl(path, rows)
    mutate_json(path.parent / "report.json", lambda report: report["metrics"].update(answer_set_exact_match=1.0))
    with pytest.raises(analysis.IntegrityError, match="Stale or invalid metric"):
        run_analysis(project)


@pytest.mark.parametrize("field", ["corpus_fingerprint", "query_fingerprint"])
def test_stale_fingerprints_are_rejected_even_when_reports_match(project, field):
    for name in ("run.json", "report.json"):
        mutate_json(project / "results/final-mmqa-local-bm25" / name, lambda value: value.update({field: "wrong"}))
    with pytest.raises(analysis.IntegrityError, match="fingerprint mismatch"):
        run_analysis(project)


def test_question_ids_must_uniquely_cover_the_planned_order(project):
    path = project / "results/final-mmqa-local-bm25/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_jsonl(path, [rows[0], rows[0]])
    with pytest.raises(analysis.IntegrityError, match="IDs/order"):
        run_analysis(project)


def test_generator_changes_are_not_compared_as_retrieval_effects(project):
    for name in ("run.json", "report.json"):
        mutate_json(project / "results/final-mmqa-local-dense-text" / name,
                    lambda value: value["generator_config"].update(runtime_manifest_sha256="b" * 64))
    with pytest.raises(analysis.IntegrityError, match="generator_config differs"):
        run_analysis(project)


def test_separate_tasks_retain_their_own_historical_code_and_config(project):
    for mode in analysis.AV_MODES:
        for name in ("run.json", "report.json"):
            def update(value):
                value["generator_config"].update(adapter_sha256="historical-video-decoder-fix")
                value["implementation"].update({"local_generation.py": "historical-av-source"})
            mutate_json(project / f"results/final-local-av-{mode}" / name, update)
    result = run_analysis(project)
    assert result["status"] == "complete"
    assert result["cross_experiment_configuration"]["differing_fields"] == ["generator_config", "implementation"]
    assert result["experiments"]["av"]["generator_config"]["adapter_sha256"] == "historical-video-decoder-fix"
    assert "adapter_sha256" not in result["experiments"]["mmqa"]["generator_config"]
    assert "separately frozen" in analysis.markdown(result)
    # A video-only change within AV would confound the modality comparison.
    for name in ("run.json", "report.json"):
        mutate_json(project / "results/final-local-av-video" / name,
                    lambda value: value["generator_config"].update(adapter_sha256="another-decoder"))
    with pytest.raises(analysis.IntegrityError, match="Unpaired av experiment: generator_config differs"):
        run_analysis(project)


@pytest.mark.parametrize("absolute", [False, True])
def test_reproduction_uses_selected_runs_directory_and_preserves_snapshots(project, absolute):
    destination = project / "results/reproduced"
    for runs in analysis.run_directories(project).values():
        for source in runs.values():
            shutil.copytree(source, destination / source.name)
    original = project / "results/final-mmqa-local-bm25/report.json"
    mutate_json(original, lambda report: report["metrics"].update(answer_set_exact_match=0.123))
    original_bytes = original.read_bytes()
    selected = destination if absolute else Path("results/reproduced")
    result = analysis.analyze(project, runs_root=selected, expected_mmqa_count=2,
                              expected_av_count=2, repetitions=100)
    assert result["status"] == "complete"
    assert result["experiments"]["mmqa"]["conditions"]["bm25"]["metrics"]["answer_set_exact_match"] == .5
    assert original.read_bytes() == original_bytes
    with pytest.raises(analysis.IntegrityError, match="Stale or invalid metric"):
        run_analysis(project)


def test_cli_accepts_separate_runs_and_output_roots(project, monkeypatch):
    calls = {}
    def fake_analyze(root, **kwargs):
        calls.update(root=root, **kwargs)
        return {"status": "complete"}
    monkeypatch.setattr(analysis, "analyze", fake_analyze)
    monkeypatch.setattr(analysis, "write_analysis", lambda summary, output: calls.update(output=output))
    assert analysis.main(["--root", str(project), "--runs-root", "results/reproduced",
                          "--output", str(project / "results/reproduced/final-analysis")]) == 0
    assert calls["runs_root"] == Path("results/reproduced")
    assert calls["output"] == project / "results/reproduced/final-analysis"


def test_av_evidence_must_match_the_exact_supplied_clip_and_ablation(project):
    path = project / "results/final-local-av-video/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["evidence_ids"] = ["audio:1", "video:1"]
    write_jsonl(path, rows)
    with pytest.raises(analysis.IntegrityError, match="identical known clip"):
        run_analysis(project)


def test_invalid_output_denominator_and_failure_counts_cannot_be_hidden(project):
    mutate_json(project / "results/final-mmqa-local-multimodal/report.json", lambda report: report.update(evaluation_failures={"count": 0, "by_type": {}}))
    with pytest.raises(analysis.IntegrityError, match="invalid-model-response count"):
        run_analysis(project)


def test_only_explicit_invalid_model_response_is_scored_as_failure(project):
    path = project / "results/final-mmqa-local-multimodal/predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["result"]["failure_type"] = "network_error"
    write_jsonl(path, rows)
    with pytest.raises(analysis.IntegrityError, match="explicit empty failure"):
        run_analysis(project)


def test_publication_outputs_contain_checked_denominators_and_limits(project, monkeypatch):
    summary = run_analysis(project)
    output = project / "results/final-analysis"
    # This test isolates publication integrity from optional plotting dependencies.
    monkeypatch.setattr(analysis, "plot", lambda summary, folder: (folder / "answer-comparison.svg").write_text("<svg/>", encoding="utf-8"))
    analysis.write_analysis(summary, output)
    saved = json.loads((output / "summary.json").read_text())
    text = (output / "comparison.md").read_text(encoding="utf-8")
    assert saved == summary
    assert "1 / 2" in text and "not causal" in text and "25% expected accuracy" in text
    assert (output / "answer-comparison.svg").is_file()
