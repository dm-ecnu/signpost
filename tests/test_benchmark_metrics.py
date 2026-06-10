import json

from signpost.benchmark.cost_quality import break_even_queries, pareto_frontier, summarize_methods
from signpost.benchmark.index_metrics import summarize_graph, summarize_semantic_extractions, summarize_stage_log
from signpost.benchmark.method_summary import build_method_summary, upsert_summary
from signpost.benchmark.query_metrics import extract_query_cost, summarize_prediction_file
from signpost.benchmark.time_stage import run_timed_stage
from signpost.evaluation.schema import build_prediction_text
from signpost.parsing.io import write_jsonl


def test_query_metrics_infer_trace_cost_and_evidence_recall(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "question_id": "q1",
                "question": "Q",
                "answer": "alpha beta",
                "prediction": build_prediction_text(answer="alpha beta"),
                "metadata": {"method": "signpost", "dataset": "mini", "gold_chunk_ids": ["c1"]},
                "retrieved_chunks": [{"chunk_id": "c0"}, {"chunk_id": "c1"}],
                "trace": [
                    {"event_type": "supervisor_start", "timestamp": 1.0},
                    {"event_type": "tool_call", "tool": "knowledge_search", "timestamp": 2.0, "output_summary": {"text_items": 2}},
                    {"event_type": "tool_call", "tool": "read_file", "timestamp": 3.0},
                    {"event_type": "final_answer", "timestamp": 4.0},
                ],
            }
        ],
    )

    result = summarize_prediction_file(predictions)

    assert result["quality"]["exact_match"] == 1.0
    assert result["cost"]["totals"]["tool_calls"] == 2.0
    assert result["cost"]["totals"]["read_file_calls"] == 1.0
    assert result["cost"]["totals"]["latency_seconds"] == 3.0
    assert result["retrieval"]["recall_at_k"]["recall@3"] == 1.0


def test_extract_query_cost_prefers_explicit_values() -> None:
    cost = extract_query_cost({"latency_seconds": 7, "input_tokens": 10, "output_tokens": 5, "trace": [{"timestamp": 1}, {"timestamp": 2}]})

    assert cost["latency_seconds"] == 7.0
    assert cost["total_tokens"] == 15.0


def test_index_metrics_summarize_logs_cache_and_graph(tmp_path) -> None:
    stage_log = tmp_path / "stage.jsonl"
    write_jsonl(
        stage_log,
        [
            {
                "dataset": "mini",
                "stage": "F5_chunk_index",
                "wall_time_seconds": 2,
                "status": "ok",
                "llm_calls": 1,
                "extra_metrics": {"indexed_chunks": 2},
            },
            {
                "dataset": "mini",
                "stage": "F5_chunk_index",
                "wall_time_seconds": 4,
                "status": "ok",
                "llm_calls": 2,
                "extra_metrics": {"indexed_chunks": 3},
            },
        ],
    )
    cache = tmp_path / "semantic.extractions.jsonl"
    write_jsonl(
        cache,
        [
            {"chunk_id": "c1", "extraction": {"entities": [{"name": "A"}], "relations": [{"source": "A", "target": "B"}]}},
            {"chunk_id": "c2", "extraction": {"entities": [{"name": "B"}, {"name": "C"}], "relations": []}},
        ],
    )
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"node_id": "n1", "node_type": "entity"}, {"node_id": "n2", "node_type": "chunk"}],
                "edges": [{"source": "n1", "target": "n2", "edge_type": "source"}],
            }
        ),
        encoding="utf-8",
    )

    stage_summary = summarize_stage_log(stage_log)
    cache_summary = summarize_semantic_extractions(cache, gleaning_rounds=1)
    graph_summary = summarize_graph(graph)

    assert stage_summary["stages"]["F5_chunk_index"]["wall_time_seconds"]["sum"] == 6.0
    assert stage_summary["stages"]["F5_chunk_index"]["extra_metrics"]["indexed_chunks"]["sum"] == 5.0
    assert cache_summary["estimated_llm_calls"] == 4
    assert graph_summary["connected_components"]["count"] == 1
    assert graph_summary["edge_type_ratio"]["source"] == 1.0


def test_cost_quality_amortization_pareto_and_break_even() -> None:
    methods = [
        {
            "method": "baseline",
            "num_queries": 10,
            "quality_score": 0.5,
            "online_tokens_mean": 100,
            "online_latency_seconds_mean": 10,
            "offline_tokens": 0,
            "offline_wall_time_seconds": 0,
        },
        {
            "method": "signpost",
            "num_queries": 10,
            "quality_score": 0.7,
            "online_tokens_mean": 50,
            "online_latency_seconds_mean": 6,
            "offline_tokens": 1000,
            "offline_wall_time_seconds": 60,
        },
    ]

    result = summarize_methods(methods, workload_sizes=[10, 100])

    assert break_even_queries(1000, 0, 50, 100) == 20
    assert {"method": "signpost", "quality_score": 0.7, "online_tokens_mean": 50.0} in pareto_frontier(result["methods"])
    assert result["amortized"]["signpost"][1]["amortized_tokens"] == 60.0


def test_time_stage_appends_jsonl(tmp_path) -> None:
    log = tmp_path / "stage_timing.jsonl"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")
    stdout_log = tmp_path / "stdout.log"
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"chunks": 3, "notes": "stage local metrics"}), encoding="utf-8")

    row = run_timed_stage(
        dataset="mini",
        stage="F0_test",
        log_path=log,
        command=["python", "-c", "print('ok')"],
        output_path=str(artifact),
        llm_calls=2,
        input_tokens=10,
        output_tokens=5,
        metrics_path=metrics,
        stdout_log=stdout_log,
    )

    assert row["status"] == "ok"
    assert row["disk_bytes"] == 5
    assert row["llm_calls"] == 2
    assert row["extra_metrics"]["chunks"] == 3
    assert stdout_log.read_text(encoding="utf-8").strip() == "ok"
    saved = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["stage"] == "F0_test"


def test_method_summary_builds_and_upserts_rows(tmp_path) -> None:
    query_metrics = tmp_path / "query_metrics.json"
    query_metrics.write_text(
        json.dumps(
            {
                "num_queries": 2,
                "quality": {"f1": 0.5},
                "cost": {"means": {"total_tokens": 10}},
                "retrieval": {"mrr": 0.25},
            }
        ),
        encoding="utf-8",
    )
    stage_log = tmp_path / "stage.jsonl"
    write_jsonl(
        stage_log,
        [
            {"stage": "F5_chunk_index", "wall_time_seconds": 3, "llm_calls": 1, "input_tokens": 2, "output_tokens": 4, "disk_bytes": 8},
            {"stage": "F16_evaluation", "wall_time_seconds": 99},
        ],
    )

    summary = build_method_summary(
        method="hybrid",
        dataset="mini",
        query_metrics_path=query_metrics,
        stage_log_path=stage_log,
        offline_stages=["F5_chunk_index"],
    )
    output = tmp_path / "method_summaries.json"
    rows = upsert_summary(output, summary)

    assert summary["offline"]["wall_time_seconds"] == 3
    assert summary["offline"]["input_tokens"] == 2
    assert rows[0]["method"] == "hybrid"
