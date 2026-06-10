import json

from signpost.agent.batch import run_batch
from signpost.agent.supervisor import AgentConfig, Supervisor, collect_locates, count_online_signpost_recommendations, deterministic_decompose
from signpost.agent.tools import KnowledgeSearchConfig, KnowledgeSearchTool, ReadFileConfig, ReadFileTool, default_search_config
from signpost.parsing.io import write_jsonl


def _write_artifacts(tmp_path):
    graph_path = tmp_path / "graph.unified.json"
    chunks_path = tmp_path / "chunks.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    graph = {
        "nodes": [
            {"node_id": "chunk:c1", "node_type": "chunk", "chunk_id": "c1", "file_name": "mini.txt", "start_line": 1, "end_line": 2, "section_path": ["概述"]},
            {"node_id": "summary:s1", "node_type": "summary", "title": "方法", "summary": "PPR 推荐方法", "source_chunk_ids": ["c1"], "source_locates": ["mini.txt:L1-L2"]},
            {"node_id": "entity:e1", "node_type": "entity", "name": "PPR", "entity_type": "METHOD", "source_chunk_ids": ["c1"], "source_locates": ["mini.txt:L1-L2"]},
        ],
        "edges": [
            {"source": "summary:s1", "target": "chunk:c1", "edge_type": "structure"},
            {"source": "entity:e1", "target": "chunk:c1", "edge_type": "source"},
        ],
    }
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "file_name": "mini.txt",
                "content": "Signpost 使用 PPR 推荐。",
                "start_line": 1,
                "end_line": 2,
                "section_path": ["概述"],
            }
        ],
    )
    write_jsonl(
        documents_path,
        [
            {
                "doc_id": "d1",
                "file_name": "mini.txt",
                "lines": [
                    {"line_no": 1, "text": "Signpost 使用 PPR 推荐。"},
                    {"line_no": 2, "text": "它为智能体提供在线路标。"},
                ],
            }
        ],
    )
    return graph_path, chunks_path, documents_path


def test_deterministic_decompose_splits_question() -> None:
    assert deterministic_decompose("A？B。C", 2) == ["A", "B"]


def test_collect_locates_from_retrieval() -> None:
    retrieval = {"text_group": {"items": [{"offline_signpost": {"provenance": {"locate": "a.txt:L1-L2"}}}]}, "graph_group": {"items": []}}
    assert collect_locates(retrieval) == ["a.txt:L1-L2"]


def test_supervisor_runs_search_readfile_and_synthesizes_answer(tmp_path) -> None:
    graph_path, chunks_path, documents_path = _write_artifacts(tmp_path)
    search_tool = KnowledgeSearchTool(KnowledgeSearchConfig(namespace="mini", graph_path=graph_path, chunks_path=chunks_path))
    read_tool = ReadFileTool(ReadFileConfig(dataset="mini", documents_path=documents_path, before=0, after=0))
    result = Supervisor(AgentConfig(namespace="mini", read_top_k=2), search_tool, read_tool).run("PPR 推荐是什么？")

    assert "PPR 推荐" in result["answer"]
    assert result["citations"][0]["locate"] == "mini.txt:L1-L2"
    assert any(event["tool"] == "knowledge_search" for event in result["trace"] if event["event_type"] == "tool_call")
    assert any(event["tool"] == "read_file" for event in result["trace"] if event["event_type"] == "tool_call")


def test_batch_writes_prediction_jsonl(tmp_path, monkeypatch) -> None:
    output = tmp_path / "predictions.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "PPR 推荐是什么？", "answer": "gold"}])

    def fake_run_agent(**kwargs):
        return {"answer": "answer", "citations": [], "trace_id": "t1", "trace": []}

    monkeypatch.setattr("signpost.agent.batch.run_agent", fake_run_agent)
    count = run_batch(namespace="mini", questions_path=str(questions), output_path=str(output))

    assert count == 1
    row = list(open(output, encoding="utf-8"))[0]
    payload = json.loads(row)
    assert payload["metadata"]["method"] == "signpost"
    assert payload["answer"] == "gold"
    assert "<answer>" in payload["prediction"]


def test_batch_extracts_json_synthesis_answer(tmp_path, monkeypatch) -> None:
    output = tmp_path / "predictions.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "What is PPR?", "answer": "gold"}])

    def fake_run_agent(**kwargs):
        return {
            "answer": (
                "```json\n"
                '{"rationale":"The question asks for the definition.","answer":"PPR is personalized PageRank."}'
                "\n```"
            ),
            "citations": [],
            "trace_id": "t1",
            "trace": [],
        }

    monkeypatch.setattr("signpost.agent.batch.run_agent", fake_run_agent)
    count = run_batch(namespace="mini", questions_path=str(questions), output_path=str(output))

    assert count == 1
    payload = json.loads(list(open(output, encoding="utf-8"))[0])
    assert "<answer>\nPPR is personalized PageRank.\n</answer>" in payload["prediction"]
    assert '"answer"' not in payload["prediction"]
    assert "The question asks for the definition." in payload["prediction"]


def test_batch_passes_embedding_provider_to_agent(tmp_path, monkeypatch) -> None:
    output = tmp_path / "predictions.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "PPR 推荐是什么？", "answer": "gold"}])
    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {"answer": "answer", "citations": [], "trace_id": "t1", "trace": []}

    monkeypatch.setattr("signpost.agent.batch.run_agent", fake_run_agent)
    count = run_batch(
        namespace="legal_test-ecnu",
        dataset="legal_test",
        questions_path=str(questions),
        output_path=str(output),
        embedding_provider="ecnu",
        use_es=True,
    )

    assert count == 1
    assert captured["namespace"] == "legal_test-ecnu"
    assert captured["dataset"] == "legal_test"
    assert captured["embedding_provider"] == "ecnu"
    assert captured["use_es"] is True


def test_batch_passes_signpost_variant_to_agent(tmp_path, monkeypatch) -> None:
    output = tmp_path / "predictions.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "PPR 推荐是什么？", "answer": "gold"}])
    captured = {}

    def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {"answer": "answer", "citations": [], "trace_id": "t1", "trace": []}

    monkeypatch.setattr("signpost.agent.batch.run_agent", fake_run_agent)
    count = run_batch(
        namespace="legal_test",
        dataset="legal_test",
        questions_path=str(questions),
        output_path=str(output),
        signpost_variant="no_online",
    )

    assert count == 1
    assert captured["signpost_variant"] == "no_online"
    payload = json.loads(list(open(output, encoding="utf-8"))[0])
    assert payload["metadata"]["signpost_variant"] == "no_online"


def test_count_online_signpost_recommendations_from_grouped_result() -> None:
    retrieval = {
        "text_group": {"online_signpost": {"recommended_entities": [{"name": "A"}, {"name": "B"}]}},
        "graph_group": {"online_signpost": {"recommended_entities": [{"name": "C"}]}},
    }

    assert count_online_signpost_recommendations(retrieval) == 3


def test_default_search_config_prefers_processed_dataset_artifacts(tmp_path, monkeypatch) -> None:
    base = tmp_path / "datasets" / "processed" / "legal_test"
    base.mkdir(parents=True)
    (base / "graph.unified.json").write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
    (base / "chunks.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

    config = default_search_config("legal_test")

    assert config.graph_path.as_posix().endswith("datasets/processed/legal_test/graph.unified.json")
    assert config.chunks_path is not None
    assert config.chunks_path.as_posix().endswith("datasets/processed/legal_test/chunks.jsonl")


def test_default_search_config_uses_dataset_for_processed_artifacts_with_es_namespace(tmp_path, monkeypatch) -> None:
    base = tmp_path / "datasets" / "processed" / "legal_test"
    base.mkdir(parents=True)
    (base / "graph.unified.json").write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
    (base / "chunks.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr("signpost.config.context.PROJECT_ROOT", tmp_path)

    config = default_search_config("legal_test-ecnu", dataset="legal_test", use_es=True, embedding_provider_name="ecnu")

    assert config.namespace == "legal_test-ecnu"
    assert config.use_es is True
    assert config.embedding_provider_name == "ecnu"
    assert config.graph_path.as_posix().endswith("datasets/processed/legal_test/graph.unified.json")
    assert config.chunks_path is not None
    assert config.chunks_path.as_posix().endswith("datasets/processed/legal_test/chunks.jsonl")
