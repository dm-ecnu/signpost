from __future__ import annotations

"""Compact graph inspection CLI."""

import argparse
import json
from collections import Counter

from signpost.config.context import resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect graph JSON")
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    graph = json.loads(resolve_project_path(args.graph).read_text(encoding="utf-8"))
    node_counts = Counter(node.get("node_type") for node in graph.get("nodes", []))
    edge_counts = Counter(edge.get("edge_type") for edge in graph.get("edges", []))
    print(json.dumps({"metadata": graph.get("metadata", {}), "node_counts": dict(node_counts), "edge_counts": dict(edge_counts)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

