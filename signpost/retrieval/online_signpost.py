from __future__ import annotations

"""Auxiliary group-level jump recommender via Personalized PageRank.

This is NOT the core serving loop (that is agent/sketch_chaining.py +
agent/supervisor.py, Alg. 3). It is an optional online enrichment that augments
the materialized jump (C_s) family with query-adaptive entity suggestions: it
uses the current result group as PPR seeds, prunes G_D by scene, and returns top
entity nodes as explainable jump targets. It is the component ablated by the
NO_ONLINE / no_online variant (see retrieval/signpost_variants.py).
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path


TEXT_SCENE = "text"
GRAPH_SCENE = "graph"


def compute_online_signpost(
    graph: dict[str, Any],
    seeds: list[str | dict[str, Any]],
    *,
    scene: str = "auto",
    top_k: int = 5,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    index = OnlineGraphIndex(graph)
    seed_ids = [seed_id for seed in seeds if (seed_id := index.resolve_seed(seed))]
    seed_ids = list(dict.fromkeys(seed_ids))
    resolved_scene = _resolve_scene(index, seed_ids, scene)
    subgraph_nodes = _subgraph_nodes(index, seed_ids, resolved_scene)
    scores = personalized_pagerank(index, subgraph_nodes, seed_ids, damping=damping, max_iter=max_iter, tolerance=tolerance)
    recommendations = _rank_entities(index, scores, seed_ids, top_k)
    return {
        "scene": resolved_scene,
        "seeds": seed_ids,
        "subgraph": {
            "nodes": len(subgraph_nodes),
            "edges": index.count_edges(subgraph_nodes),
        },
        "recommended_entities": recommendations,
    }


def personalized_pagerank(
    index: "OnlineGraphIndex",
    nodes: set[str],
    seeds: list[str],
    *,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    valid_seeds = [seed for seed in seeds if seed in nodes]
    if not nodes or not valid_seeds:
        return {}
    n = len(nodes)
    seed_weight = 1.0 / len(valid_seeds)
    personalization = {node: (seed_weight if node in valid_seeds else 0.0) for node in nodes}
    scores = {node: 1.0 / n for node in nodes}

    for _ in range(max_iter):
        next_scores = {node: (1.0 - damping) * personalization[node] for node in nodes}
        dangling = 0.0
        for node in nodes:
            neighbors = [(nbr, weight) for nbr, weight in index.neighbors(node).items() if nbr in nodes]
            total_weight = sum(weight for _, weight in neighbors)
            if not neighbors or total_weight <= 0:
                dangling += scores[node]
                continue
            share = damping * scores[node]
            for neighbor, weight in neighbors:
                next_scores[neighbor] += share * (weight / total_weight)
        if dangling:
            for node in nodes:
                next_scores[node] += damping * dangling * personalization[node]
        delta = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if delta < tolerance:
            break
    return scores


class OnlineGraphIndex:
    def __init__(self, graph: dict[str, Any]) -> None:
        self.graph = graph
        self.node_by_id = {node["node_id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("node_id")}
        self.chunk_node_by_chunk_id = {node.get("chunk_id"): node["node_id"] for node in self.node_by_id.values() if node.get("node_type") == "chunk" and node.get("chunk_id")}
        self.adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        self.structure_children: dict[str, set[str]] = defaultdict(set)
        self.structure_parents: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in self.node_by_id or target not in self.node_by_id:
                continue
            weight = _edge_weight(edge)
            self.adjacency[source][target] = max(self.adjacency[source].get(target, 0.0), weight)
            self.adjacency[target][source] = max(self.adjacency[target].get(source, 0.0), weight)
            if edge.get("edge_type") == "structure":
                self.structure_children[source].add(target)
                self.structure_parents[target].add(source)

    def neighbors(self, node_id: str) -> dict[str, float]:
        return self.adjacency.get(node_id, {})

    def count_edges(self, nodes: set[str]) -> int:
        count = 0
        for node in nodes:
            count += sum(1 for neighbor in self.adjacency.get(node, {}) if neighbor in nodes)
        return count // 2

    def resolve_seed(self, seed: str | dict[str, Any]) -> str | None:
        if isinstance(seed, str):
            if seed in self.node_by_id:
                return seed
            if seed in self.chunk_node_by_chunk_id:
                return self.chunk_node_by_chunk_id[seed]
            return None
        for key in ("node_id", "id"):
            value = seed.get(key)
            if value in self.node_by_id:
                return value
        chunk_id = seed.get("chunk_id")
        if chunk_id in self.chunk_node_by_chunk_id:
            return self.chunk_node_by_chunk_id[chunk_id]
        source = seed.get("source")
        if source in self.node_by_id:
            return source
        target = seed.get("target")
        if target in self.node_by_id:
            return target
        return None


def _subgraph_nodes(index: OnlineGraphIndex, seeds: list[str], scene: str) -> set[str]:
    if scene == GRAPH_SCENE:
        return {node_id for node_id, node in index.node_by_id.items() if node.get("node_type") in {"entity", "chunk"}}
    nodes = {node_id for node_id, node in index.node_by_id.items() if node.get("node_type") in {"entity", "chunk"}}
    summary_nodes: set[str] = set()
    for seed in seeds:
        node = index.node_by_id.get(seed, {})
        if node.get("node_type") == "summary":
            summary_nodes.add(seed)
            summary_nodes.update(_walk_structure(index.structure_children, seed))
            summary_nodes.update(_walk_structure(index.structure_parents, seed))
        elif node.get("node_type") == "chunk":
            summary_nodes.update(_walk_structure(index.structure_parents, seed))
    nodes.update(summary_nodes)
    return nodes


def _walk_structure(edges: dict[str, set[str]], start: str) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in edges.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _resolve_scene(index: OnlineGraphIndex, seeds: list[str], scene: str) -> str:
    if scene in {TEXT_SCENE, GRAPH_SCENE}:
        return scene
    seed_types = {index.node_by_id.get(seed, {}).get("node_type") for seed in seeds}
    if seed_types and seed_types <= {"entity"}:
        return GRAPH_SCENE
    return TEXT_SCENE


def _rank_entities(index: OnlineGraphIndex, scores: dict[str, float], seeds: list[str], top_k: int) -> list[dict[str, Any]]:
    seed_set = set(seeds)
    rows = []
    for node_id, score in scores.items():
        node = index.node_by_id.get(node_id, {})
        if node.get("node_type") != "entity" or node_id in seed_set:
            continue
        rows.append(
            {
                "node_id": node_id,
                "name": node.get("name"),
                "entity_type": node.get("entity_type"),
                "score": score,
                "source_chunk_ids": node.get("source_chunk_ids") or [],
                "source_locates": node.get("source_locates") or [],
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:top_k]


def _edge_weight(edge: dict[str, Any]) -> float:
    try:
        weight = float(edge.get("weight") or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    if edge.get("edge_type") == "semantic":
        return max(weight, 1.0)
    return 1.0


def _load_seeds(result_json: str | None, seed_values: list[str] | None) -> list[str | dict[str, Any]]:
    seeds: list[str | dict[str, Any]] = []
    for seed in seed_values or []:
        seeds.append(seed)
    if result_json:
        loaded = json.loads(Path(result_json).read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "items" in loaded:
            loaded = loaded["items"]
        if not isinstance(loaded, list):
            raise ValueError("--result-json must contain a JSON list or an object with items[]")
        seeds.extend(loaded)
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="F12 compute online signposts with PPR")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--seed", action="append")
    parser.add_argument("--result-json")
    parser.add_argument("--scene", choices=["auto", TEXT_SCENE, GRAPH_SCENE], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    seeds = _load_seeds(args.result_json, args.seed)
    if not seeds:
        parser.error("provide --seed or --result-json")
    graph = json.loads(resolve_project_path(args.graph).read_text(encoding="utf-8"))
    result = compute_online_signpost(graph, seeds, scene=args.scene, top_k=args.top_k, damping=args.damping, max_iter=args.max_iter)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
