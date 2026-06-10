from __future__ import annotations

"""F9 CLI: merge semantic, structure, and sequence graphs into one graph."""

import argparse

from signpost.config.context import resolve_project_path
from signpost.graph.unified import load_graph, merge_graphs, save_graph_atomic, validate_unified_graph


def merge_graph_files(
    *,
    semantic_path: str | None,
    structure_path: str | None,
    sequence_path: str | None,
    output_path: str,
    namespace: str,
) -> dict[str, int | str]:
    semantic_graph = load_graph(resolve_project_path(semantic_path)) if semantic_path else None
    structure_graph = load_graph(resolve_project_path(structure_path)) if structure_path else None
    sequence_graph = load_graph(resolve_project_path(sequence_path)) if sequence_path else None
    graph = merge_graphs(
        semantic_graph=semantic_graph,
        structure_graph=structure_graph,
        sequence_graph=sequence_graph,
        namespace=namespace,
    )
    summary = validate_unified_graph(graph)
    output = resolve_project_path(output_path)
    save_graph_atomic(graph, output)
    return {"output": str(output), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build F9 unified multi-view graph")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--semantic")
    parser.add_argument("--structure")
    parser.add_argument("--sequence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not any([args.semantic, args.structure, args.sequence]):
        parser.error("at least one input graph is required")

    result = merge_graph_files(
        semantic_path=args.semantic,
        structure_path=args.structure,
        sequence_path=args.sequence,
        output_path=args.output,
        namespace=args.namespace,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
