from __future__ import annotations

"""F11 offline signposts for retrieved graph results.

Offline signposts are precomputed navigation cues attached to retrieval results:
vertical hierarchy cues, horizontal adjacent-position cues, and provenance cues.
They are deterministic and depend only on the unified graph.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.retrieval.chunk_search import search_chunks
from signpost.retrieval.graph_search import search_graph


def build_offline_signpost(graph: dict[str, Any], result: dict[str, Any] | str) -> dict[str, Any]:
    index = GraphIndex(graph)
    node_id = index.resolve_node_id(result)
    edge = index.resolve_edge(result)
    if node_id:
        node = index.node_by_id[node_id]
        if node.get("node_type") == "chunk":
            return _chunk_signpost(index, node)
        if node.get("node_type") == "summary":
            return _summary_signpost(index, node)
        if node.get("node_type") == "entity":
            return _entity_signpost(index, node)
    if edge:
        return _relation_signpost(index, edge)
    return {"result_type": "unknown", "vertical": {}, "horizontal": {}, "provenance": {}}


def attach_offline_signposts(graph: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**result, "offline_signpost": build_offline_signpost(graph, result)} for result in results]


class GraphIndex:
    def __init__(self, graph: dict[str, Any]) -> None:
        self.graph = graph
        self.node_by_id = {node["node_id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("node_id")}
        self.chunk_node_by_chunk_id = {node.get("chunk_id"): node for node in self.node_by_id.values() if node.get("node_type") == "chunk" and node.get("chunk_id")}
        self.structure_children: dict[str, list[str]] = defaultdict(list)
        self.structure_parents: dict[str, list[str]] = defaultdict(list)
        self.sequence_next: dict[str, str] = {}
        self.sequence_prev: dict[str, str] = {}
        self.semantic_neighbors: dict[str, set[str]] = defaultdict(set)
        self.source_chunks_by_entity: dict[str, set[str]] = defaultdict(set)
        self.edge_by_signature: dict[tuple[str, str], dict[str, Any]] = {}

        for edge in graph.get("edges", []):
            if edge.get("edge_type") == "structure":
                self.structure_children[edge["source"]].append(edge["target"])
                self.structure_parents[edge["target"]].append(edge["source"])
            elif edge.get("edge_type") == "sequence":
                if edge.get("direction") == "next":
                    self.sequence_next[edge["source"]] = edge["target"]
                elif edge.get("direction") == "prev":
                    self.sequence_prev[edge["source"]] = edge["target"]
            elif edge.get("edge_type") == "semantic":
                self.semantic_neighbors[edge["source"]].add(edge["target"])
                self.semantic_neighbors[edge["target"]].add(edge["source"])
                self.edge_by_signature[(edge["source"], edge["target"])] = edge
                self.edge_by_signature[(edge["target"], edge["source"])] = edge
            elif edge.get("edge_type") == "source":
                chunk_id = _chunk_id_from_node_id(edge.get("target", ""))
                if chunk_id:
                    self.source_chunks_by_entity[edge["source"]].add(chunk_id)

    def resolve_node_id(self, result: dict[str, Any] | str) -> str | None:
        if isinstance(result, str):
            if result in self.node_by_id:
                return result
            chunk_node = self.chunk_node_by_chunk_id.get(result)
            return chunk_node.get("node_id") if chunk_node else None
        for key in ("node_id", "id"):
            value = result.get(key)
            if value in self.node_by_id:
                return value
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id in self.chunk_node_by_chunk_id:
            return self.chunk_node_by_chunk_id[chunk_id]["node_id"]
        return None

    def resolve_edge(self, result: dict[str, Any] | str) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        source = result.get("source")
        target = result.get("target")
        if source and target:
            return self.edge_by_signature.get((source, target))
        edge_id = result.get("edge_id") or result.get("id")
        if not edge_id:
            return None
        for edge in self.graph.get("edges", []):
            if edge.get("edge_type") == "semantic" and _edge_id(edge) == edge_id:
                return edge
        return None


def _chunk_signpost(index: GraphIndex, node: dict[str, Any]) -> dict[str, Any]:
    node_id = node["node_id"]
    parent_ids = index.structure_parents.get(node_id, [])
    parent_nodes = [_summary_ref(index, parent_id) for parent_id in parent_ids]
    prev_node = index.node_by_id.get(index.sequence_prev.get(node_id, ""))
    next_node = index.node_by_id.get(index.sequence_next.get(node_id, ""))
    return {
        "result_type": "chunk",
        "vertical": {
            "section_path": node.get("section_path") or [],
            "parent_summaries": parent_nodes,
            "nearest_parent_summary": parent_nodes[0] if parent_nodes else None,
        },
        "horizontal": {
            "previous_chunk": _chunk_ref(prev_node) if prev_node else None,
            "next_chunk": _chunk_ref(next_node) if next_node else None,
        },
        "provenance": {
            "file_name": node.get("file_name"),
            "start_line": node.get("start_line"),
            "end_line": node.get("end_line"),
            "locate": _locate(node),
        },
    }


def _summary_signpost(index: GraphIndex, node: dict[str, Any]) -> dict[str, Any]:
    node_id = node["node_id"]
    parent_ids = index.structure_parents.get(node_id, [])
    child_ids = index.structure_children.get(node_id, [])
    child_summaries = [_summary_ref(index, child_id) for child_id in child_ids if index.node_by_id.get(child_id, {}).get("node_type") == "summary"]
    child_chunks = [_chunk_ref(index.node_by_id[child_id]) for child_id in child_ids if index.node_by_id.get(child_id, {}).get("node_type") == "chunk"]
    return {
        "result_type": "summary",
        "vertical": {
            "level": node.get("level"),
            "section_path": node.get("section_path") or [],
            "parent_summary": _summary_ref(index, parent_ids[0]) if parent_ids else None,
            "child_summaries": child_summaries,
            "child_chunks": child_chunks,
        },
        "horizontal": {},
        "provenance": {
            "source_chunk_ids": node.get("source_chunk_ids") or [],
            "source_locates": _merge_locates(node.get("source_locates") or []),
        },
    }


def _entity_signpost(index: GraphIndex, node: dict[str, Any]) -> dict[str, Any]:
    neighbor_ids = sorted(index.semantic_neighbors.get(node["node_id"], set()))
    source_chunk_ids = sorted(set(node.get("source_chunk_ids") or []) | index.source_chunks_by_entity.get(node["node_id"], set()))
    return {
        "result_type": "entity",
        "vertical": {},
        "horizontal": {},
        "provenance": {
            "source_chunk_ids": source_chunk_ids,
            "source_locates": _merge_locates(node.get("source_locates") or []),
            "source_mapping": node.get("source_mapping") or {},
        },
        "semantic": {
            "neighboring_entities": [_entity_ref(index, neighbor_id) for neighbor_id in neighbor_ids],
        },
    }


def _relation_signpost(index: GraphIndex, edge: dict[str, Any]) -> dict[str, Any]:
    source = index.node_by_id.get(edge.get("source"), {})
    target = index.node_by_id.get(edge.get("target"), {})
    neighbor_ids = sorted((index.semantic_neighbors.get(edge.get("source", ""), set()) | index.semantic_neighbors.get(edge.get("target", ""), set())) - {edge.get("source"), edge.get("target")})
    return {
        "result_type": "relation",
        "vertical": {},
        "horizontal": {},
        "provenance": {
            "source_chunk_ids": edge.get("source_chunk_ids") or [],
            "source_locates": _merge_locates(edge.get("source_locates") or []),
            "source_mapping": edge.get("source_mapping") or {},
        },
        "semantic": {
            "source_entity": _entity_ref(index, source.get("node_id", "")) if source else {"node_id": edge.get("source")},
            "target_entity": _entity_ref(index, target.get("node_id", "")) if target else {"node_id": edge.get("target")},
            "neighboring_entities": [_entity_ref(index, neighbor_id) for neighbor_id in neighbor_ids],
        },
    }


def _summary_ref(index: GraphIndex, node_id: str) -> dict[str, Any]:
    node = index.node_by_id.get(node_id, {})
    return {
        "node_id": node_id,
        "title": node.get("title"),
        "level": node.get("level"),
        "section_path": node.get("section_path") or [],
    }


def _chunk_ref(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    return {
        "node_id": node.get("node_id"),
        "chunk_id": node.get("chunk_id"),
        "file_name": node.get("file_name"),
        "start_line": node.get("start_line"),
        "end_line": node.get("end_line"),
        "locate": _locate(node),
        "section_path": node.get("section_path") or [],
    }


def _entity_ref(index: GraphIndex, node_id: str) -> dict[str, Any]:
    node = index.node_by_id.get(node_id, {})
    return {
        "node_id": node_id,
        "name": node.get("name"),
        "entity_type": node.get("entity_type"),
        "source_chunk_ids": node.get("source_chunk_ids") or [],
    }


def _locate(node: dict[str, Any]) -> str | None:
    if not node.get("file_name") or node.get("start_line") is None or node.get("end_line") is None:
        return None
    return f"{node.get('file_name')}:L{node.get('start_line')}-L{node.get('end_line')}"


def _merge_locates(locates: list[str]) -> list[str]:
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    passthrough: list[str] = []
    for locate in locates:
        parsed = _parse_locate(locate)
        if not parsed:
            passthrough.append(locate)
            continue
        file_name, start, end = parsed
        by_file[file_name].append((start, end))
    result: list[str] = []
    for file_name, ranges in sorted(by_file.items()):
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        result.extend(f"{file_name}:L{start}-L{end}" for start, end in merged)
    return result + passthrough


def _parse_locate(locate: str) -> tuple[str, int, int] | None:
    match = re.match(r"^(?P<file>.+):L(?P<start>\d+)-L(?P<end>\d+)$", locate)
    if not match:
        return None
    return match.group("file"), int(match.group("start")), int(match.group("end"))


def _chunk_id_from_node_id(node_id: str) -> str | None:
    return node_id.removeprefix("chunk:") if node_id.startswith("chunk:") else None


def _edge_id(edge: dict[str, Any]) -> str:
    import hashlib

    seed = json.dumps(
        {"source": edge.get("source"), "target": edge.get("target"), "edge_type": edge.get("edge_type"), "relation_types": edge.get("relation_types") or []},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "edge:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _load_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_project_path(str(path)).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="F11 attach offline signposts to retrieval results")
    parser.add_argument("--namespace")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--query")
    parser.add_argument("--node-id", action="append")
    parser.add_argument("--chunk-id", action="append")
    parser.add_argument("--result-json", help="JSON list of retrieval results to enrich")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    args = parser.parse_args()

    graph_path = args.graph
    if not graph_path:
        if not args.namespace:
            parser.error("--graph or --namespace is required")
        graph_path = f"outputs/{args.namespace}/graph.unified.json"
    graph = _load_graph(graph_path)

    results: list[dict[str, Any]] = []
    if args.result_json:
        loaded = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("--result-json must contain a JSON list")
        results.extend(loaded)
    for node_id in args.node_id or []:
        results.append({"node_id": node_id})
    for chunk_id in args.chunk_id or []:
        results.append({"chunk_id": chunk_id})
    if args.query:
        if not args.namespace:
            parser.error("--namespace is required when --query is used")
        chunk_items = search_chunks(namespace=args.namespace, query=args.query, mode=args.mode, top_k=args.top_k, embedding_provider_name=args.embedding_provider, hash_dimensions=args.hash_dimensions).get("items", [])
        graph_items = search_graph(namespace=args.namespace, query=args.query, mode=args.mode, top_k=args.top_k, embedding_provider_name=args.embedding_provider, hash_dimensions=args.hash_dimensions).get("items", [])
        results.extend(chunk_items)
        results.extend(graph_items)
    if not results:
        parser.error("provide --node-id, --chunk-id, --result-json, or --query")

    print(json.dumps({"items": attach_offline_signposts(graph, results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
