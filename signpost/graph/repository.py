from __future__ import annotations

"""Small local graph repository used by the research pipeline.

The original product stored graphs in MinIO with tenant/kb keys and an LRU
cache.  The refactored research version keeps the useful persistence semantics
without user, tenant, or knowledge-base business logic.
"""

from pathlib import Path
from typing import Any

from signpost.graph.unified import load_graph, save_graph_atomic


class LocalGraphRepository:
    """Load and save graph JSON documents under one local root directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def graph_path(self, namespace: str, name: str = "graph.unified.json") -> Path:
        return self.root / namespace / name

    def exists(self, namespace: str, name: str = "graph.unified.json") -> bool:
        return self.graph_path(namespace, name).exists()

    def load(self, namespace: str, name: str = "graph.unified.json") -> dict[str, Any]:
        return load_graph(self.graph_path(namespace, name))

    def save(self, namespace: str, graph: dict[str, Any], name: str = "graph.unified.json") -> Path:
        path = self.graph_path(namespace, name)
        save_graph_atomic(graph, path)
        return path
