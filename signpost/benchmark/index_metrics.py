from __future__ import annotations

"""Offline index and graph-structure metric aggregation."""

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from signpost.benchmark.stats import summarize_values
from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def summarize_stage_log(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("stage") or "unknown")
        by_stage[key].append(row)
    stages = {}
    for stage, items in by_stage.items():
        status_counts = Counter(str(row.get("status", "unknown")) for row in items)
        extra_metric_values: dict[str, list[float]] = defaultdict(list)
        for row in items:
            extra = row.get("extra_metrics")
            if not isinstance(extra, dict):
                continue
            for key, value in extra.items():
                if isinstance(value, (int, float)):
                    extra_metric_values[str(key)].append(float(value))
        stages[stage] = {
            "runs": len(items),
            "status_counts": dict(status_counts),
            "wall_time_seconds": summarize_values(row.get("wall_time_seconds", 0) for row in items),
            "llm_calls": summarize_values(row.get("llm_calls", 0) for row in items),
            "input_tokens": summarize_values(row.get("input_tokens", 0) for row in items),
            "output_tokens": summarize_values(row.get("output_tokens", 0) for row in items),
            "disk_bytes": summarize_values(row.get("disk_bytes", 0) for row in items),
            "extra_metrics": {key: summarize_values(values) for key, values in extra_metric_values.items()},
        }
    return {"input": str(path), "rows": len(rows), "stages": stages}


def summarize_semantic_extractions(path: Path, *, gleaning_rounds: int | None = None) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    entity_counts = []
    relation_counts = []
    chunks = set()
    for row in rows:
        chunks.add(row.get("chunk_id"))
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}
        entity_counts.append(len(extraction.get("entities", []) if isinstance(extraction.get("entities"), list) else []))
        relation_counts.append(len(extraction.get("relations", []) if isinstance(extraction.get("relations"), list) else []))
    llm_calls_per_chunk = 1 + gleaning_rounds if gleaning_rounds is not None else None
    estimated_calls = len(chunks) * llm_calls_per_chunk if llm_calls_per_chunk is not None else None
    return {
        "input": str(path),
        "rows": len(rows),
        "unique_chunks": len(chunks),
        "gleaning_rounds": gleaning_rounds,
        "estimated_llm_calls": estimated_calls,
        "entities_per_chunk": summarize_values(entity_counts),
        "relations_per_chunk": summarize_values(relation_counts),
        "total_entities_before_merge": sum(entity_counts),
        "total_relations_before_merge": sum(relation_counts),
    }


def summarize_graph(path: Path) -> dict[str, Any]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges", []) if isinstance(graph.get("edges"), list) else []
    node_counts = Counter(str(node.get("node_type", "unknown")) for node in nodes if isinstance(node, dict))
    edge_counts = Counter(str(edge.get("edge_type", "unknown")) for edge in edges if isinstance(edge, dict))
    degrees = _degree_counts(nodes, edges)
    components = _components(nodes, edges)
    return {
        "input": str(path),
        "metadata": graph.get("metadata", {}),
        "nodes": len(nodes),
        "edges": len(edges),
        "node_counts": dict(node_counts),
        "edge_counts": dict(edge_counts),
        "edge_type_ratio": {key: value / len(edges) if edges else 0.0 for key, value in edge_counts.items()},
        "degree": summarize_values(degrees.values()),
        "connected_components": {
            "count": len(components),
            "largest": max((len(component) for component in components), default=0),
            "sizes": summarize_values(len(component) for component in components),
        },
    }


def _degree_counts(nodes: list[Any], edges: list[Any]) -> dict[str, int]:
    node_ids = {str(node.get("node_id")) for node in nodes if isinstance(node, dict) and node.get("node_id")}
    degrees = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in degrees:
            degrees[source] += 1
        if target in degrees and target != source:
            degrees[target] += 1
    return degrees


def _components(nodes: list[Any], edges: list[Any]) -> list[set[str]]:
    node_ids = {str(node.get("node_id")) for node in nodes if isinstance(node, dict) and node.get("node_id")}
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    seen = set()
    components = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        component = set()
        queue = deque([node_id])
        seen.add(node_id)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize offline index logs and graph JSON artifacts.")
    parser.add_argument("--stage-log", action="append", default=[])
    parser.add_argument("--semantic-cache", action="append", default=[])
    parser.add_argument("--graph", action="append", default=[])
    parser.add_argument("--gleaning-rounds", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = {
        "stage_logs": [summarize_stage_log(resolve_project_path(path)) for path in args.stage_log],
        "semantic_extractions": [summarize_semantic_extractions(resolve_project_path(path), gleaning_rounds=args.gleaning_rounds) for path in args.semantic_cache],
        "graphs": [summarize_graph(resolve_project_path(path)) for path in args.graph],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} graphs={len(result['graphs'])}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
