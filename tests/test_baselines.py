import json

from signpost.baselines.vanilla_llm import run_vanilla_llm
from signpost.baselines.vanilla_rag import run_vanilla_rag
from signpost.baselines.hybrid_rag import run_hybrid_rag
from signpost.baselines.agrag import run_agrag
from signpost.baselines.cluerag import _openai_embedding_base, convert_cluerag_outputs, prepare_cluerag_inputs, run_cluerag_shared
from signpost.baselines.graphrag_r1 import run_graphrag_r1
from signpost.baselines.hiprag import run_hiprag
from signpost.parsing.io import write_jsonl


class FakeLLM:
    def chat(self, messages, *, model=None, thinking=False):
        text = "\n".join(str(message.get("content", "")) for message in messages)
        if "named_entities" in text or "extract named entities" in text.lower():
            return json.dumps({"named_entities": ["Signpost"]})
        return "fake answer"


class SequenceLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def chat(self, messages, *, model=None, thinking=False):
        if self.outputs:
            return self.outputs.pop(0)
        return "<answer>Evidence.</answer>"


def _write_questions(path):
    write_jsonl(
        path,
        [
            {
                "question_id": "q1",
                "question": "What does Signpost retrieve?",
                "answer": "Evidence.",
                "metadata": {"gold_chunk_ids": ["c1"]},
            }
        ],
    )


def test_vanilla_llm_writes_unified_prediction_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("signpost.baselines.vanilla_llm.OpenAICompatibleClient", lambda: FakeLLM())
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")

    count = run_vanilla_llm(dataset="mini")

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "vanilla_llm.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["method"] == "vanilla_llm"
    assert row["retrieved_chunks"] == []
    assert row["llm_calls"] == 1.0
    assert "<answer>" in row["prediction"]


def test_vanilla_rag_local_bm25_writes_retrieved_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("signpost.baselines.vanilla_rag.OpenAICompatibleClient", lambda: FakeLLM())
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )

    count = run_vanilla_rag(dataset="mini", use_es=False, mode="bm25", top_k=1)

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "vanilla_rag.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["method"] == "vanilla_rag"
    assert row["metadata"]["use_es"] is False
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert row["citations"][0]["locate"] == "doc.txt:L1-L2"
    assert row["tool_calls"] == 1.0


def test_hybrid_rag_alias_writes_hybrid_method_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("signpost.baselines.vanilla_rag.OpenAICompatibleClient", lambda: FakeLLM())
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )

    count = run_hybrid_rag(dataset="mini", use_es=False, mode="bm25", top_k=1)

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "hybrid_rag.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["method"] == "hybrid_rag"
    assert row["metadata"]["mode"] == "bm25"
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"


def test_agrag_adapter_builds_graph_and_writes_unified_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("signpost.baselines.agrag.OpenAICompatibleClient", lambda: FakeLLM())
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )
    write_jsonl(
        base / "semantic_llm.extractions.jsonl",
        [
            {
                "chunk_id": "c1",
                "extraction": {
                    "entities": [{"name": "Signpost"}, {"name": "Evidence"}],
                    "relations": [
                        {
                            "source": "Signpost",
                            "target": "Evidence",
                            "description": "Signpost retrieves evidence from chunks.",
                            "keywords": ["retrieves"],
                            "weight": 1.0,
                        }
                    ],
                },
            }
        ],
    )

    count = run_agrag(dataset="mini", use_es=False, mode="bm25", top_k=1, graph_top_k=1, link_top_k=1, embedding_provider="hash")

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "agrag.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["method"] == "agrag"
    assert row["metadata"]["retrieval"] == "agrag_ppr_mcmi_hybrid"
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert row["citations"][0]["locate"] == "doc.txt:L1-L2"
    assert row["graph_ppr_calls"] == 1.0
    graph = json.loads((tmp_path / "outputs" / "mini" / "baselines" / "agrag" / "graph.json").read_text(encoding="utf-8"))
    assert graph["triples"] == 1


def test_cluerag_prepare_and_convert_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "documents.jsonl",
        [
            {
                "doc_id": "d1",
                "file_name": "doc.txt",
                "text": "Signpost retrieves evidence from chunks.",
            }
        ],
    )
    repo = tmp_path / "baselines" / "ClueRAG"

    manifest = prepare_cluerag_inputs(dataset="mini", repo_path=repo)

    assert manifest["cluerag_dataset"] == "signpost_mini"
    assert (repo / "data" / "signpost_mini.json").exists()
    assert (repo / "data" / "signpost_mini_corpus.json").exists()

    official = tmp_path / "outputs" / "mini" / "baselines" / "cluerag" / "official_outputs" / "COSINE_1.00"
    official.mkdir(parents=True)
    (official / "retrieval_results.json").write_text(
        json.dumps(
            {
                "retrieval_results": [],
                "metadata": {"prompt_tokens": 4, "completion_tokens": 1, "num_requests": 1},
            }
        ),
        encoding="utf-8",
    )
    (official / "generation_results.json").write_text(
        json.dumps(
            {
                "generation_results": [
                    {
                        "qid": "q1",
                        "question": "What does Signpost retrieve?",
                        "answer": "Evidence.",
                        "chunks": ["chunk-a"],
                        "generation": json.dumps({"thought": "grounded", "answer": "Evidence."}),
                    }
                ],
                "metadata": {"prompt_tokens": 10, "completion_tokens": 2, "num_requests": 1},
            }
        ),
        encoding="utf-8",
    )

    count = convert_cluerag_outputs(dataset="mini", official_output_dir=official)

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "cluerag.jsonl").read_text(encoding="utf-8"))
    assert row["metadata"]["method"] == "cluerag"
    assert row["retrieved_chunks"][0]["chunk_id"] == "chunk-a"
    assert "Evidence." in row["prediction"]
    assert row["input_tokens"] == 14.0
    assert row["output_tokens"] == 3.0
    assert row["llm_calls"] == 2.0


def test_cluerag_shared_local_builds_own_graph_and_records_model_costs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("signpost.llm.client.OpenAICompatibleClient", lambda timeout=600: FakeLLM())
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "metadata": {"token_count": 5},
            }
        ],
    )
    write_jsonl(
        base / "semantic_llm.extractions.jsonl",
        [
            {
                "chunk_id": "c1",
                "extraction": {
                    "entities": [
                        {"name": "Signpost", "entity_type": "CONCEPT", "description": "A retrieval method."},
                        {"name": "Evidence", "entity_type": "CONCEPT", "description": "Information retrieved from chunks."},
                    ],
                    "relations": [
                        {
                            "source": "Signpost",
                            "target": "Evidence",
                            "description": "Signpost retrieves evidence from chunks.",
                        }
                    ],
                },
            }
        ],
    )

    status = run_cluerag_shared(
        dataset="mini",
        use_es=False,
        embedding_provider="hash",
        direct_top_k=1,
        ku_top_k=2,
        graph_top_k=2,
        top_n=1,
        depth=1,
        rerank_url="",
    )

    assert status["converted_predictions"] == 1
    manifest = json.loads((tmp_path / "outputs" / "mini" / "baselines" / "cluerag" / "shared_graph" / "manifest.json").read_text())
    assert manifest["graph_organization"] == "cluerag_multilayer_chunk_knowledge_unit_entity"
    assert manifest["knowledge_units"] == 3
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "cluerag.jsonl").read_text(encoding="utf-8"))
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert row["llm_calls"] == 2.0
    assert row["query_ner_calls"] == 1.0
    assert row["rerank_calls"] == 0.0
    assert row["embedding_calls"] == 0.0
    metrics = json.loads((tmp_path / "outputs" / "mini" / "baselines" / "cluerag" / "run_metrics.json").read_text())
    assert metrics["offline_llm_calls"] == 0.0
    assert metrics["offline_disk_bytes"] > 0.0
    assert metrics["offline_metadata"]["shared_knowledge_units"] == 3.0


def test_cluerag_embedding_base_strips_embeddings_suffix() -> None:
    assert _openai_embedding_base("http://localhost:8001/v1/embeddings") == "http://localhost:8001/v1"
    assert _openai_embedding_base("http://localhost:8001/v1") == "http://localhost:8001/v1"


def test_hiprag_forces_initial_search_and_writes_evidence_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "signpost.baselines.hiprag.OpenAICompatibleClient",
        lambda: SequenceLLM(["<think><answer>Ungrounded.</answer>", "</think><answer>Evidence.</answer>"]),
    )
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )

    count = run_hiprag(dataset="mini", use_es=False, mode="bm25", search_top_k=1, max_steps=1, embedding_provider="hash")

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "hiprag.jsonl").read_text(encoding="utf-8"))
    assert row["evidence_chunks"][0]["chunk_id"] == "c1"
    assert row["evidence_chunks"][0]["source"] == "retrieval_context"
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert "Evidence." in row["prediction"]
    assert any(event.get("stage") == "hiprag_forced_initial_search" for event in row["trace"])


def test_graphrag_r1_forces_initial_query_and_writes_evidence_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "signpost.baselines.graphrag_r1.OpenAICompatibleClient",
        lambda: SequenceLLM(["<think><answer>Ungrounded.</answer>", "</think><answer>Evidence.</answer>"]),
    )
    base = tmp_path / "datasets" / "processed" / "mini"
    base.mkdir(parents=True)
    _write_questions(base / "questions.jsonl")
    write_jsonl(
        base / "chunks.jsonl",
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "doc.txt",
                "content": "Signpost retrieves evidence from chunks.",
                "start_line": 1,
                "end_line": 2,
            }
        ],
    )
    write_jsonl(
        base / "semantic_llm.extractions.jsonl",
        [
            {
                "chunk_id": "c1",
                "extraction": {
                    "entities": [{"name": "Signpost"}, {"name": "Evidence"}],
                    "relations": [{"source": "Signpost", "target": "Evidence", "description": "Signpost retrieves evidence from chunks."}],
                },
            }
        ],
    )

    count = run_graphrag_r1(
        dataset="mini",
        use_es=False,
        mode="bm25",
        graph_top_k=1,
        chunk_top_k=1,
        link_top_k=1,
        max_steps=1,
        embedding_provider="hash",
    )

    assert count == 1
    row = json.loads((tmp_path / "outputs" / "mini" / "predictions" / "graphrag_r1.jsonl").read_text(encoding="utf-8"))
    assert row["evidence_chunks"][0]["chunk_id"] == "c1"
    assert row["evidence_chunks"][0]["source"] == "retrieval_context"
    assert row["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert "Evidence." in row["prediction"]
    assert any(event.get("stage") == "graphrag_r1_forced_initial_query" for event in row["trace"])
