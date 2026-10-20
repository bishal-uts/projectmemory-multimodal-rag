"""Exact-ranking answer replay, using tiny records and fake generators only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from projectmemory.evaluation import corpus_fingerprint
from projectmemory.io import fingerprint, write_jsonl
from projectmemory.evaluation_policy import FormatFailureRecorder, INVALID_LOCAL_RESPONSE
from projectmemory.generation import GenerationError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_saved_rankings.py"
SPEC = importlib.util.spec_from_file_location("evaluate_saved_rankings", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def make_report(corpus, queries, *, split="dev", method="multimodal", k=2):
    selected = [q for q in queries if split is None or q.get("split") == split]
    rows = []
    for query in selected:
        scope = query.get("metadata", {}).get("candidate_source_ids")
        candidates = [row["id"] for row in corpus if scope is None or row.get("source_id") in scope]
        rows.append({"id": query["id"], "question": query["question"],
                     "relevant_ids": query["relevant_ids"], "candidate_source_ids": scope,
                     "candidate_count": len(candidates),
                     "hits": [{"id": identity, "score": 1 - index / 10}
                              for index, identity in enumerate(reversed(candidates[-k:]))]})
    return {"schema_version": 1, "task": "retrieval", "method": method, "k": k, "split": split,
            "corpus_fingerprint": corpus_fingerprint(corpus), "query_fingerprint": fingerprint(selected),
            "corpus_size": len(corpus), "query_count": len(selected), "skipped_count": 0,
            "skipped": [], "queries": rows}


@pytest.fixture
def fixture(tmp_path):
    dataset = tmp_path / "dataset"
    corpus = [{"id": "s1", "source_id": "a", "modality": "text", "text": "blue remote"},
              {"id": "s2", "source_id": "a", "modality": "text", "text": "red alternative"},
              {"id": "s3", "source_id": "b", "modality": "text", "text": "green other"}]
    queries = [{"id": "q1", "question": "Which colour?", "relevant_ids": ["s1"], "split": "dev",
                "reference_answers": ["BLUE_GOLD_SENTINEL"], "metadata": {"candidate_source_ids": ["a"]}},
               {"id": "q2", "question": "Which alternative?", "relevant_ids": ["s2"], "split": "dev",
                "reference_answers": ["RED_GOLD_SENTINEL"], "metadata": {"candidate_source_ids": ["a"]}},
               {"id": "q3", "question": "What other colour?", "relevant_ids": ["s3"], "split": "test",
                "reference_answers": ["GREEN_GOLD_SENTINEL"]}]
    write_jsonl(dataset / "corpus.jsonl", corpus)
    write_jsonl(dataset / "queries.jsonl", queries)
    path = tmp_path / "retrieval.json"
    report = make_report(corpus, queries)
    path.write_text(json.dumps(report), encoding="utf-8")
    return corpus, queries, path, report, dataset


class FakeGenerator:
    provider, model, max_output_tokens = "local", "fake-local-generator", 4096

    def __init__(self, fail_question=None):
        self.calls = []
        self.fail_question = fail_question

    def answer(self, question, evidence):
        self.calls.append((question, copy.deepcopy(evidence)))
        assert "GOLD_SENTINEL" not in json.dumps([question, evidence])
        if question == self.fail_question:
            raise RuntimeError("fixture interruption")
        return {"answer": "red", "answer_items": ["red"], "citations": [evidence[0]["id"]],
                "abstained": False, "answer_label": "", "provider": "local", "model": self.model}


def test_replay_is_exact_and_checks_caller_scope_and_k(fixture):
    corpus, queries, path, report, _ = fixture
    retriever = replay.SavedRankingsRetriever(corpus, queries, path)
    question = queries[0]["question"]
    hits = retriever.search(question, 2, {"s1", "s2"})
    assert hits == report["queries"][0]["hits"]
    hits[0]["score"] = -999
    assert retriever.search(question, 2, {"s1", "s2"}) == report["queries"][0]["hits"]
    with pytest.raises(ValueError, match="exactly"):
        retriever.search(question, 1, {"s1", "s2"})
    with pytest.raises(ValueError, match="source scope"):
        retriever.search(question, 2, {"s1"})
    with pytest.raises(ValueError, match="source scope"):
        retriever.search(question, 2)
    with pytest.raises(ValueError, match="Question text"):
        retriever.search("Unrecorded question?", 2)


def test_run_honors_split_limit_provenance_and_never_loads_an_encoder(fixture, tmp_path, monkeypatch):
    from projectmemory.embeddings import EmbeddingRetriever
    monkeypatch.setattr(EmbeddingRetriever, "_load_model", lambda self: pytest.fail("Replay must not load embeddings"))
    corpus, queries, path, report, dataset = fixture
    output = tmp_path / "answers"
    generator = FakeGenerator()
    result = replay.run(dataset, path, output, limit=1, generator=generator)
    assert result["completed"] == 1 and result["method"] == "multimodal" and result["split"] == "dev"
    assert len(generator.calls) == 1 and generator.calls[0][0] == queries[0]["question"]
    assert [r["id"] for r in generator.calls[0][1]] == [h["id"] for h in report["queries"][0]["hits"]]
    source = json.loads((output / "retrieval_source.json").read_text())
    assert source["complete_report_query_ids"] == ["q1", "q2"]
    assert source["evaluated_query_ids"] == ["q1"] and not source["embedding_model_loaded"]
    report_hash, script_hash = hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert source["retrieval_report_sha256"] == report_hash and source["replay_script_sha256"] == script_hash
    assert report_hash in result["retriever_config"]["revision"]
    assert script_hash in result["retriever_config"]["revision"]
    replay.run(dataset, path, output, limit=1, generator=generator)
    assert len(generator.calls) == 1  # Completed predictions resume through the existing evaluator.


def test_partial_run_resumes_with_exact_remaining_rankings(fixture, tmp_path):
    _, queries, path, report, dataset = fixture
    output = tmp_path / "answers"
    first = FakeGenerator(fail_question=queries[1]["question"])
    with pytest.raises(RuntimeError, match="fixture interruption"):
        replay.run(dataset, path, output, generator=first)
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 1
    resumed = FakeGenerator()
    result = replay.run(dataset, path, output, generator=resumed)
    assert result["completed"] == 2 and len(resumed.calls) == 1
    assert resumed.calls[0][0] == queries[1]["question"]
    assert [r["id"] for r in resumed.calls[0][1]] == [h["id"] for h in report["queries"][1]["hits"]]


@pytest.mark.parametrize("field,value,message", [
    ("task", "answer_evaluation", "retrieval report"),
    ("schema_version", 99, "retrieval report"),
    ("method", "unknown", "method"),
    ("k", True, "positive integer"),
    ("k", 0, "positive integer"),
    ("split", [], "split"),
    ("corpus_fingerprint", "wrong", "corpus/media"),
    ("query_fingerprint", "wrong", "query fingerprint"),
    ("corpus_size", 999, "corpus size"),
    ("query_count", 1, "complete rankings"),
    ("skipped_count", 1, "complete rankings"),
    ("skipped", [{"id": "q2"}], "complete rankings"),
])
def test_invalid_report_metadata_is_rejected_before_generator_creation(fixture, tmp_path, monkeypatch, field, value, message):
    _, _, path, report, dataset = fixture
    report[field] = value
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, "create_generator", lambda **kwargs: pytest.fail("Invalid rankings must not initialize generation"))
    with pytest.raises(ValueError, match=message):
        replay.run(dataset, path, tmp_path / "answers")
    assert not (tmp_path / "answers").exists()


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r["queries"].pop(), "uniquely cover"),
    (lambda r: r["queries"].append(copy.deepcopy(r["queries"][0])), "uniquely cover"),
    (lambda r: r["queries"][0].update(id="unknown"), "uniquely cover"),
    (lambda r: r["queries"][0].update(question="Changed question"), "question text"),
    (lambda r: r["queries"][0].update(relevant_ids=["s2"]), "relevance annotations"),
    (lambda r: r["queries"][0].update(candidate_source_ids=["b"]), "source scope"),
    (lambda r: r["queries"][0].update(candidate_count=99), "candidate count"),
    (lambda r: r["queries"][0]["hits"][0].update(id="missing"), "unknown evidence"),
    (lambda r: r["queries"][0]["hits"][0].update(id="s3"), "outside"),
    (lambda r: r["queries"][0]["hits"][0].update(id=r["queries"][0]["hits"][1]["id"]), "duplicate"),
    (lambda r: r["queries"][0]["hits"].append({"id": "s3", "score": .2}), "retrieval k"),
    (lambda r: r["queries"][0]["hits"][0].update(score=float("nan")), "finite"),
    (lambda r: r["queries"][0]["hits"][0].update(score=True), "finite"),
])
def test_incomplete_or_changed_rankings_are_rejected(fixture, mutation, message):
    corpus, queries, path, report, _ = fixture
    mutation(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        replay.SavedRankingsRetriever(corpus, queries, path)


def test_duplicate_question_text_is_rejected_even_with_unique_ids(fixture):
    corpus, queries, path, _, _ = fixture
    queries[1]["question"] = queries[0]["question"]
    path.write_text(json.dumps(make_report(corpus, queries)))
    with pytest.raises(ValueError, match="Ambiguous duplicate question text"):
        replay.SavedRankingsRetriever(corpus, queries, path)


def test_unscoped_ranking_rejects_an_added_scope(fixture):
    corpus, queries, path, _, _ = fixture
    path.write_text(json.dumps(make_report(corpus, queries, split="test")))
    retriever = replay.SavedRankingsRetriever(corpus, queries, path)
    assert retriever.search(queries[2]["question"], 2)
    with pytest.raises(ValueError, match="source scope"):
        retriever.search(queries[2]["question"], 2, {"s1", "s2", "s3"})


def test_full_report_is_validated_even_when_answer_limit_is_one(fixture, tmp_path):
    _, _, path, report, dataset = fixture
    report["queries"][1]["hits"][0]["id"] = "missing"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unknown evidence"):
        replay.run(dataset, path, tmp_path / "answers", limit=1, generator=FakeGenerator())


def test_changed_ranking_content_cannot_resume_prior_output(fixture, tmp_path):
    _, _, path, report, dataset = fixture
    output = tmp_path / "answers"
    generator = FakeGenerator()
    replay.run(dataset, path, output, generator=generator)
    prior = (output / "retrieval_source.json").read_bytes()
    report["queries"][0]["hits"][0]["score"] = 0.99
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different replay provenance"):
        replay.run(dataset, path, output, generator=generator)
    assert (output / "retrieval_source.json").read_bytes() == prior and len(generator.calls) == 2


def test_existing_nonreplay_run_cannot_be_relabelled(fixture, tmp_path):
    _, _, path, _, dataset = fixture
    output = tmp_path / "answers"
    output.mkdir()
    (output / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="no replay provenance"):
        replay.run(dataset, path, output, generator=FakeGenerator())
    assert not (output / "retrieval_source.json").exists()


def test_cli_defaults_are_local_100_questions_and_4096_tokens(fixture, tmp_path, monkeypatch):
    _, _, path, _, dataset = fixture
    calls = []
    def fake_run(dataset_arg, report_arg, output_arg, **kwargs):
        calls.append((dataset_arg, report_arg, output_arg, kwargs))
        return {"provider": "local", "model": "fake", "method": "multimodal", "completed": 2, "metrics": {}}
    monkeypatch.setattr(replay, "run", fake_run)
    assert replay.main(["--dataset", str(dataset), "--retrieval-report", str(path), "--output", str(tmp_path / "answers")]) == 0
    assert calls[0][3] == {"provider": "local", "limit": 100, "max_output_tokens": 4096,
                           "record_format_failures": False}


def test_opt_in_records_invalid_response_with_zero_coverage_and_failure_count(fixture, tmp_path):
    _, queries, path, _, dataset = fixture
    class InvalidFirstGenerator(FakeGenerator):
        def answer(self, question, evidence):
            if question == queries[0]["question"]:
                self.calls.append((question, evidence))
                raise GenerationError(INVALID_LOCAL_RESPONSE)
            return super().answer(question, evidence)
    generator = InvalidFirstGenerator()
    output = tmp_path / "answers"
    report = replay.run(dataset, path, output, generator=generator, record_format_failures=True)
    assert report["completed"] == 2 and report["metrics"]["answer_coverage"] == .5
    assert report["evaluation_failures"] == {"count": 1, "by_type": {"invalid_model_response": 1}}
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    failed = predictions[0]["result"]
    assert failed["evaluation_failure"] is True and failed["abstained"] is True
    assert failed["failure_type"] == "invalid_model_response" and failed["failure_explanation"]
    assert failed["answer"] == failed["answer_label"] == ""
    assert failed["answer_items"] == failed["citations"] == failed["sources"] == []
    assert failed["usage"] == {} and failed["usage_unavailable"] and not failed["cache_hit"]
    assert predictions[0]["metrics"]["answer_set_exact_match"] == 0
    assert predictions[0]["metrics"]["answer_coverage"] == 0
    policy = report["generator_config"]["invalid_response_policy"]
    assert policy["eligible_message"] == INVALID_LOCAL_RESPONSE
    assert len(policy["implementation_sha256"]) == 64
    assert json.loads((output / "report.json").read_text())["evaluation_failures"]["count"] == 1
    resumed = FakeGenerator()
    report = replay.run(dataset, path, output, generator=resumed, record_format_failures=True)
    assert not resumed.calls and report["evaluation_failures"]["count"] == 1


def test_format_failure_without_opt_in_still_aborts(fixture, tmp_path):
    _, _, path, _, dataset = fixture
    class InvalidGenerator(FakeGenerator):
        def answer(self, question, evidence):
            raise GenerationError(INVALID_LOCAL_RESPONSE)
    output = tmp_path / "answers"
    with pytest.raises(GenerationError, match="valid grounded JSON"):
        replay.run(dataset, path, output, generator=InvalidGenerator())
    assert not (output / "predictions.jsonl").exists() and not (output / "report.json").exists()


@pytest.mark.parametrize("failure", [
    GenerationError("Could not reach local Gemma; start the local runtime and retry"),
    GenerationError("Local Gemma returned HTTP 503."),
    GenerationError("Local Gemma did not finish normally; reduce evidence or increase the output budget"),
    RuntimeError("The local CUDA runtime could not start"),
    RuntimeError(INVALID_LOCAL_RESPONSE),  # Message alone is insufficient: exception type must match.
    OSError("fixture media file unavailable"),
    ValueError("Local Gemma accepts at most 30 seconds of audio per request"),
])
def test_wrapper_propagates_infrastructure_media_and_other_failures(fixture, tmp_path, failure):
    _, _, path, _, dataset = fixture
    class BrokenGenerator(FakeGenerator):
        def answer(self, question, evidence):
            raise failure
    output = tmp_path / "answers"
    with pytest.raises(type(failure)) as caught:
        replay.run(dataset, path, output, generator=BrokenGenerator(), record_format_failures=True)
    assert caught.value is failure
    assert not (output / "predictions.jsonl").exists() and not (output / "report.json").exists()


def test_reusable_wrapper_preserves_success_and_choice_arguments():
    expected = {"answer": "B", "citations": ["s"], "abstained": False, "answer_label": "B"}
    calls = []
    class ChoiceGenerator(FakeGenerator):
        def evaluation_config(self):
            return {"model_revision": "pinned-model", "generation_parameters": {"temperature": 1}}
        def answer(self, question, evidence, **kwargs):
            calls.append((question, evidence, kwargs))
            return expected
    wrapper = FormatFailureRecorder(ChoiceGenerator())
    config = wrapper.evaluation_config()
    assert config["model_revision"] == "pinned-model" and config["generation_parameters"] == {"temperature": 1}
    evidence, choices = [{"id": "s"}], ["A", "B", "C", "D"]
    assert wrapper.answer("Question?", evidence, options=choices) is expected
    assert calls == [("Question?", evidence, {"options": choices})]


def test_reusable_wrapper_failure_scores_as_incorrect_supplied_media_qa():
    from projectmemory.rag import answer_metrics
    class InvalidChoiceGenerator(FakeGenerator):
        def answer(self, question, evidence, **kwargs):
            raise GenerationError(INVALID_LOCAL_RESPONSE)
    wrapper = FormatFailureRecorder(InvalidChoiceGenerator())
    failure = wrapper.answer("Which event?", [{"id": "audio"}], options=["A", "B", "C", "D"])
    scores = answer_metrics({"reference_answers": ["B"], "options": ["A", "B", "C", "D"], "answer_label": "B"}, failure)
    assert scores == {"choice_accuracy": 0, "answer_coverage": 0}


def test_policy_is_local_only_and_cannot_silently_change_existing_run(fixture, tmp_path):
    _, _, path, _, dataset = fixture
    with pytest.raises(ValueError, match="only for the local"):
        replay.run(dataset, path, tmp_path / "api-answers", provider="gemini", record_format_failures=True)
    assert not (tmp_path / "api-answers").exists()
    output = tmp_path / "answers"
    replay.run(dataset, path, output, generator=FakeGenerator())
    with pytest.raises(ValueError, match="different replay provenance"):
        replay.run(dataset, path, output, generator=FakeGenerator(), record_format_failures=True)
