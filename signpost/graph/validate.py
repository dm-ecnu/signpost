from __future__ import annotations

"""Graph JSON validator used by F6 and later graph stages."""

import argparse
import json
from pathlib import Path

from signpost.config.context import resolve_project_path


def validate_graph(path: Path) -> dict[str, int]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph must contain nodes[] and edges[]")
    node_ids = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or not node.get("node_id") or not node.get("node_type"):
            raise ValueError(f"invalid node at index {idx}")
        if node["node_id"] in node_ids:
            raise ValueError(f"duplicate node_id={node['node_id']}")
        node_ids.add(node["node_id"])
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict) or not edge.get("source") or not edge.get("target") or not edge.get("edge_type"):
            raise ValueError(f"invalid edge at index {idx}")
        if edge["source"] not in node_ids:
            raise ValueError(f"edge {idx} source missing: {edge['source']}")
        if edge["target"] not in node_ids:
            raise ValueError(f"edge {idx} target missing: {edge['target']}")
    return {"nodes": len(nodes), "edges": len(edges), "entity_nodes": sum(1 for node in nodes if node.get("node_type") == "entity")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate graph JSON")
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    result = validate_graph(resolve_project_path(args.graph))
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

