from __future__ import annotations

"""Small artifact summaries used by time_stage after a stage finishes."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from signpost.benchmark.query_metrics import extract_query_cost
from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def collect_stage_metrics(
    *,
    stage: str,
    input_path: str = "",
    output_path: str = "",
) -> dict[str, Any]:
    input_file = resolve_project_path(input_path) if input_path else None
    output_file = resolve_project_path(output_path) if output_path else None
    if stage == "F3_data_prepare":
        base_dir = output_file.parent if output_file and output_file.suffix == ".jsonl" else output_file
        return summarize_prepared_dataset(base_dir) if base_dir else {}
    if stage == "F3_5_parse_normalize":
        return summarize_documents(output_file) if output_file else {}
    if stage == "F4_chunk_tree":
        return summarize_chunks_and_trees(output_file) if output_file else {}
    if stage == "F5_chunk_index":
        metrics = summarize_chunks(input_file) if input_file else {}
        return {f"input_{key}": value for key, value in metrics.items()}
    if stage == "F6_semantic_graph":
        return summarize_semantic_stage(output_file) if output_file else {}
    if stage in {"F7_structure_graph", "F8_sequence_graph", "F9_unified_graph"}:
        return summarize_graph(output_file) if output_file else {}
    if stage == "F10_graph_es_sync":
        metrics = summarize_graph(input_file) if input_file else {}
        return {f"input_{key}": value for key, value in metrics.items()}
    if stage in {"F11_offline_signpost", "F12_online_ppr", "F13_retrieval", "F14_read_file"}:
        return summarize_json_result(output_file) if output_file else {}
    if stage == "F15_agent_batch":
        return summarize_predictions(output_file) if output_file else {}
    if stage == "F16_evaluation":
        return summarize_eval(output_file) if output_file else {}
    return summarize_existing_path(output_file)


def summarize_existing_path(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix == ".jsonl":
        return {"jsonl_rows": float(sum(1 for _ in read_jsonl(path)))}
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {"json_keys": float(len(data))}
    return {}


def summarize_prepared_dataset(base_dir: Path) -> dict[str, Any]:
    raw_corpus = base_dir / "raw_corpus.jsonl"
    questions = base_dir / "questions.jsonl"
    result: dict[str, Any] = {}
    if raw_corpus.exists():
        docs = list(read_jsonl(raw_corpus))
        result["docs"] = float(len(docs))
        result["raw_text_chars"] = float(sum(len(str(row.get("text", ""))) for row in docs))
    if questions.exists():
        q_rows = list(read_jsonl(questions))
        result["questions"] = float(len(q_rows))
    return result


def summarize_documents(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    return {
        "documents": float(len(rows)),
        "document_lines": float(sum(len(row.get("lines") or []) for row in rows)),
        "document_chars": float(sum(len(str(row.get("text", ""))) for row in rows)),
        "placeholders": float(sum(len(row.get("placeholders") or []) for row in rows)),
    }


def summarize_chunks(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    rows = list(read_jsonl(path))
    token_counts = [float((row.get("metadata") or {}).get("token_count") or 0) for row in rows]
    docs = {str(row.get("doc_id")) for row in rows if row.get("doc_id")}
    merge_counts = Counter(str((row.get("metadata") or {}).get("merge") or "unknown") for row in rows)
    result: dict[str, Any] = {
        "chunks": float(len(rows)),
        "chunk_docs": float(len(docs)),
        "chunk_tokens_sum": float(sum(token_counts)),
        "chunk_tokens_max": float(max(token_counts, default=0.0)),
        "chunk_tokens_mean": float(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
    }
    result.update({f"chunks_merge_{key}": float(value) for key, value in merge_counts.items()})
    return result


def summarize_chunks_and_trees(chunks_path: Path) -> dict[str, Any]:
    result = summarize_chunks(chunks_path)
    tree_path = chunks_path.parent / "document_trees.jsonl"
    if tree_path.exists():
        trees = list(read_jsonl(tree_path))
        result["trees"] = float(len(trees))
        result["headers"] = float(sum(len(row.get("headers") or []) for row in trees))
    return result


def summarize_graph(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    node_counts = Counter(str(row.get("node_type") or "unknown") for row in nodes if isinstance(row, dict))
    edge_counts = Counter(str(row.get("edge_type") or "unknown") for row in edges if isinstance(row, dict))
    result: dict[str, Any] = {
        "graph_nodes": float(len(nodes)),
        "graph_edges": float(len(edges)),
    }
    result.update({f"graph_nodes_{key}": float(value) for key, value in node_counts.items()})
    result.update({f"graph_edges_{key}": float(value) for key, value in edge_counts.items()})
    return result


def summarize_semantic_stage(graph_path: Path) -> dict[str, Any]:
    result = summarize_graph(graph_path)
    cache_path = graph_path.parent / "semantic_llm.extractions.jsonl"
    if cache_path.exists():
        rows = list(read_jsonl(cache_path))
        result["semantic_extraction_rows"] = float(len(rows))
        result["semantic_completed_chunks"] = float(len({row.get("chunk_id") for row in rows if row.get("chunk_id")}))
        result["entities_before_merge"] = float(sum(len((row.get("extraction") or {}).get("entities") or []) for row in rows))
        result["relations_before_merge"] = float(sum(len((row.get("extraction") or {}).get("relations") or []) for row in rows))
    return result


def summarize_json_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    if isinstance(data, dict):
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        for key in ("text_items", "graph_items", "online_signposts", "sequence_context_items"):
            if isinstance(metadata.get(key), (int, float)):
                result[key] = float(metadata[key])
        for key in ("results", "signposts", "seeds", "lines"):
            if isinstance(data.get(key), list):
                result[key] = float(len(data[key]))
        if isinstance(data.get("text_group"), dict):
            result["text_items"] = float(len(data["text_group"].get("items") or []))
        if isinstance(data.get("graph_group"), dict):
            result["graph_items"] = float(len(data["graph_group"].get("items") or []))
        if isinstance(data.get("online_signposts"), list):
            result["online_signposts"] = float(len(data["online_signposts"]))
    return result


def summarize_predictions(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    costs = [extract_query_cost(row) for row in rows]
    return {
        "queries": float(len(rows)),
        "tool_calls": float(sum(cost.get("tool_calls", 0) for cost in costs)),
        "read_file_calls": float(sum(cost.get("read_file_calls", 0) for cost in costs)),
        "knowledge_search_calls": float(sum(cost.get("knowledge_search_calls", 0) for cost in costs)),
        "llm_calls": float(sum(cost.get("llm_calls", 0) for cost in costs)),
        "input_tokens": float(sum(cost.get("input_tokens", 0) for cost in costs)),
        "output_tokens": float(sum(cost.get("output_tokens", 0) for cost in costs)),
        "total_tokens": float(sum(cost.get("total_tokens", 0) for cost in costs)),
        "latency_seconds": float(sum(cost.get("latency_seconds", 0) for cost in costs)),
    }


def summarize_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {}
    for key in ("num_samples", "num_scored", "num_skipped"):
        if isinstance(data.get(key), (int, float)):
            result[key] = float(data[key])
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            result[f"eval_{key}"] = float(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a stage artifact into a flat metrics JSON object.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--input-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = collect_stage_metrics(stage=args.stage, input_path=args.input_path, output_path=args.output_path)
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output} metrics={len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
