from projectmemory.rag import answer_metrics, ask_question, evaluate_answers
from projectmemory.retrieval import BM25Retriever
import pytest


def test_required_answer_lists_are_not_alternative_aliases():
    query = {"reference_answers": ["red", "blue"], "relevant_ids": ["a", "b"]}
    result = answer_metrics(query, {"answer_items": ["red"], "citations": ["a"], "abstained": False})
    assert result["answer_set_exact_match"] == 0
    assert result["answer_set_recall"] == .5
    assert result["citation_gold_recall"] == .5


def test_rag_never_passes_gold_and_resumes_only_matching_run(tmp_path):
    corpus = [{"id": "s", "source_id": "meeting", "modality": "text", "text": "blue remote"}]
    queries = [{"id": "q", "question": "remote?", "relevant_ids": ["s"], "reference_answers": ["secret_gold"],
                "metadata": {"candidate_source_ids": ["meeting"]}}]
    class Generator:
        provider, model, max_output_tokens = "test", "fixture", 100
        calls = 0
        def answer(self, question, evidence):
            self.calls += 1
            assert "secret_gold" not in str((question, evidence))
            return {"answer": "blue", "answer_items": ["blue"], "citations": ["s"], "abstained": False}
    generator = Generator()
    retriever = BM25Retriever(corpus)
    first = evaluate_answers(corpus, queries, retriever, generator, tmp_path)
    second = evaluate_answers(corpus, queries, retriever, generator, tmp_path)
    assert generator.calls == 1 and first["metrics"] == second["metrics"]
    with pytest.raises(ValueError, match="different"):
        evaluate_answers(corpus, queries, retriever, generator, tmp_path, k=3)


def test_bad_citation_stops_before_persisting_prediction(tmp_path):
    corpus = [{"id": "s", "modality": "text", "text": "blue"}]
    queries = [{"id": "q", "question": "blue?", "relevant_ids": ["s"], "reference_answers": ["blue"]}]
    class Generator:
        provider, model, max_output_tokens = "test", "fixture", 100
        def answer(self, question, evidence):
            return {"answer": "blue", "answer_items": ["blue"], "citations": ["invented"], "abstained": False}
    with pytest.raises(ValueError, match="outside"):
        evaluate_answers(corpus, queries, BM25Retriever(corpus), Generator(), tmp_path)
    assert not (tmp_path / "predictions.jsonl").exists()


def test_local_evaluation_rejects_changed_runtime_identity_on_resume(tmp_path):
    corpus = [{"id": "s", "modality": "text", "text": "blue"}]
    queries = [{"id": "q", "question": "blue?", "relevant_ids": ["s"], "reference_answers": ["blue"]}]
    class Generator:
        provider, model, max_output_tokens = "local", "fixture", 100
        runtime_hash = "a" * 64
        calls = 0
        def evaluation_config(self):
            return {"runtime_manifest_sha256": self.runtime_hash, "format_retries": 1}
        def answer(self, question, evidence):
            self.calls += 1
            return {"answer": "blue", "answer_items": ["blue"], "citations": ["s"], "abstained": False}
    generator = Generator()
    first = evaluate_answers(corpus, queries, BM25Retriever(corpus), generator, tmp_path)
    assert first["generator_config"]["runtime_manifest_sha256"] == "a" * 64
    generator.runtime_hash = "b" * 64
    with pytest.raises(ValueError, match="different evaluation configuration"):
        evaluate_answers(corpus, queries, BM25Retriever(corpus), generator, tmp_path)
    assert generator.calls == 1


def test_gpu_retrieval_and_local_generation_take_turns(monkeypatch):
    from projectmemory import localruntime
    events = []
    monkeypatch.setattr(localruntime, "stop", lambda *args, **kwargs: events.append("stop generator"))
    corpus = [{"id": "s", "modality": "text", "text": "blue"}]
    class Retriever:
        device = "cuda:0"
        def search(self, question, top_k, candidate_ids):
            events.append("search")
            return [{"id": "s", "score": 1}]
        def unload(self):
            events.append("unload retriever")
    class Generator:
        provider = "local"
        def answer(self, question, evidence):
            events.append("answer")
            return {"answer": "blue"}
    ask_question(corpus, "blue?", Retriever(), Generator())
    assert events == ["stop generator", "search", "unload retriever", "answer"]


def test_api_generation_does_not_unload_cpu_retriever(monkeypatch):
    from projectmemory import localruntime
    monkeypatch.setattr(localruntime, "stop", lambda *args, **kwargs: pytest.fail("CPU retrieval must not stop runtime"))
    corpus = [{"id": "s", "modality": "text", "text": "blue"}]
    class Retriever:
        device = "cpu"
        def search(self, question, top_k, candidate_ids):
            return [{"id": "s", "score": 1}]
        def unload(self):
            pytest.fail("API answer does not require freeing encoder memory")
    class Generator:
        provider = "gemini"
        def answer(self, question, evidence):
            return {"answer": "blue"}
    assert ask_question(corpus, "blue?", Retriever(), Generator())["generated_answer"]["answer"] == "blue"
