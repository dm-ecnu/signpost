from __future__ import annotations

"""F9 multi-view topological graph merge.

The unified graph follows the thesis model:

V = Vchunk + Vsummary + Ventity
E = Estruct + Esem + Eseq + Esource

Input graphs from F6/F7/F8 keep their own local type names.  This module
normalizes them while preserving the original type fields for downstream
inspection and indexing.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


NODE_TYPE_MAP = {
    "chunk": "chunk",
    "entity": "entity",
    "raptor": "summary",
    "summary": "summary",
}

EDGE_TYPE_MAP = {
    "structure": "structure",
    "semantic_relation": "semantic",
    "semantic": "semantic",
    "sequence": "sequence",
    "source": "source",
}


def merge_graphs(
    *,
    semantic_graph: dict[str, Any] | None = None,
    structure_graph: dict[str, Any] | None = None,
    sequence_graph: dict[str, Any] | None = None,
    namespace: str,
) -> dict[str, Any]:
    node_map: dict[str, dict[str, Any]] = {}
    edge_map: dict[tuple[Any, ...], dict[str, Any]] = {}

    for view_name, graph in (("semantic", semantic_graph), ("structure", structure_graph), ("sequence", sequence_graph)):
        if not graph:
            continue
        _merge_nodes(node_map, graph.get("nodes", []), view_name)
        _merge_edges(edge_map, graph.get("edges", []), view_name)

    nodes = sorted(node_map.values(), key=lambda node: (_node_type_rank(node.get("node_type")), node.get("node_id", "")))
    edges = sorted(edge_map.values(), key=lambda edge: (_edge_type_rank(edge.get("edge_type")), edge.get("source", ""), edge.get("target", ""), edge.get("direction", "")))
    metadata = _metadata(namespace, nodes, edges, semantic_graph, structure_graph, sequence_graph)
    return {"metadata": metadata, "nodes": nodes, "edges": edges}


def save_graph_atomic(graph: dict[str, Any], path: Path) -> None:
    """Persist a graph JSON with a temporary file followed by atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{int(time.time() * 1000)}")
    temp_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_unified_graph(graph: dict[str, Any]) -> dict[str, int]:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("unified graph must contain nodes[] and edges[]")
    node_ids = {node.get("node_id") for node in nodes if isinstance(node, dict)}
    missing_ids = [idx for idx, node in enumerate(nodes) if not isinstance(node, dict) or not node.get("node_id") or node.get("node_type") not in {"chunk", "summary", "entity"}]
    if missing_ids:
        raise ValueError(f"invalid unified node indexes: {missing_ids[:5]}")
    if len(node_ids) != len(nodes):
        raise ValueError("unified graph has duplicate node_id")
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict) or edge.get("edge_type") not in {"structure", "semantic", "sequence", "source"}:
            raise ValueError(f"invalid unified edge at index {idx}")
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            raise ValueError(f"unified edge {idx} references missing node")
    node_counts = Counter(node["node_type"] for node in nodes)
    edge_counts = Counter(edge["edge_type"] for edge in edges)
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "chunk_nodes": node_counts.get("chunk", 0),
        "summary_nodes": node_counts.get("summary", 0),
        "entity_nodes": node_counts.get("entity", 0),
        "structure_edges": edge_counts.get("structure", 0),
        "semantic_edges": edge_counts.get("semantic", 0),
        "sequence_edges": edge_counts.get("sequence", 0),
        "source_edges": edge_counts.get("source", 0),
    }


def _merge_nodes(node_map: dict[str, dict[str, Any]], nodes: list[dict[str, Any]], view_name: str) -> None:
    for node in nodes:
        if not isinstance(node, dict) or not node.get("node_id"):
            continue
        normalized = _normalize_node(node, view_name)
        node_id = normalized["node_id"]
        if node_id not in node_map:
            node_map[node_id] = normalized
            continue
        node_map[node_id] = _deep_merge(node_map[node_id], normalized)
        node_map[node_id]["views"] = sorted(set(node_map[node_id].get("views", [])) | {view_name})


def _merge_edges(edge_map: dict[tuple[Any, ...], dict[str, Any]], edges: list[dict[str, Any]], view_name: str) -> None:
    for edge in edges:
        if not isinstance(edge, dict) or not edge.get("source") or not edge.get("target"):
            continue
        normalized = _normalize_edge(edge, view_name)
        edge_key = _edge_key(normalized)
        if edge_key not in edge_map:
            edge_map[edge_key] = normalized
            continue
        edge_map[edge_key] = _deep_merge(edge_map[edge_key], normalized)
        edge_map[edge_key]["views"] = sorted(set(edge_map[edge_key].get("views", [])) | {view_name})


def _normalize_node(node: dict[str, Any], view_name: str) -> dict[str, Any]:
    original_type = str(node.get("node_type", ""))
    normalized_type = NODE_TYPE_MAP.get(original_type, original_type)
    row = dict(node)
    row["node_type"] = normalized_type
    row["views"] = sorted(set(row.get("views", [])) | {view_name})
    if original_type != normalized_type:
        row["original_node_type"] = original_type
    if original_type == "raptor":
        row["summary_type"] = "raptor"
    return row


def _normalize_edge(edge: dict[str, Any], view_name: str) -> dict[str, Any]:
    original_type = str(edge.get("edge_type", ""))
    normalized_type = EDGE_TYPE_MAP.get(original_type, original_type)
    row = dict(edge)
    row["edge_type"] = normalized_type
    row["views"] = sorted(set(row.get("views", [])) | {view_name})
    if original_type != normalized_type:
        row["original_edge_type"] = original_type
    if normalized_type == "semantic" and "relation_types" in row:
        row["relation_types"] = sorted(set(row.get("relation_types") or []))
    return row


def _edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
    edge_type = edge.get("edge_type")
    source = edge.get("source")
    target = edge.get("target")
    if edge_type == "semantic":
        source, target = sorted([source, target])
        return (edge_type, source, target)
    if edge_type == "sequence":
        return (edge_type, source, target, edge.get("direction"))
    if edge_type == "source":
        return (edge_type, source, target)
    if edge_type == "structure":
        return (edge_type, source, target)
    return (edge_type, source, target, edge.get("direction"))


def _metadata(
    namespace: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    semantic_graph: dict[str, Any] | None,
    structure_graph: dict[str, Any] | None,
    sequence_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    node_counts = Counter(node.get("node_type") for node in nodes)
    edge_counts = Counter(edge.get("edge_type") for edge in edges)
    return {
        "namespace": namespace,
        "graph_type": "unified",
        "views": [name for name, graph in (("semantic", semantic_graph), ("structure", structure_graph), ("sequence", sequence_graph)) if graph],
        "nodes": len(nodes),
        "edges": len(edges),
        "chunk_nodes": node_counts.get("chunk", 0),
        "summary_nodes": node_counts.get("summary", 0),
        "entity_nodes": node_counts.get("entity", 0),
        "structure_edges": edge_counts.get("structure", 0),
        "semantic_edges": edge_counts.get("semantic", 0),
        "sequence_edges": edge_counts.get("sequence", 0),
        "source_edges": edge_counts.get("source", 0),
        "input_graphs": {
            "semantic": (semantic_graph or {}).get("metadata", {}),
            "structure": (structure_graph or {}).get("metadata", {}),
            "sequence": (sequence_graph or {}).get("metadata", {}),
        },
    }


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if value is None:
            continue
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
        elif isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = _merge_lists(merged[key], value)
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        elif key in {"description", "content"} and isinstance(value, str) and value and value not in str(merged[key]):
            merged[key] = str(merged[key]) or value
    return merged


def _merge_lists(left: list[Any], right: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in left + right:
        marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    try:
        return sorted(result)
    except TypeError:
        return result


def _node_type_rank(node_type: str | None) -> int:
    return {"chunk": 0, "summary": 1, "entity": 2}.get(str(node_type), 99)


def _edge_type_rank(edge_type: str | None) -> int:
    return {"structure": 0, "sequence": 1, "semantic": 2, "source": 3}.get(str(edge_type), 99)
