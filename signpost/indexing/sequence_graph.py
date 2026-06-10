from __future__ import annotations

"""F8 CLI: chunks.jsonl -> graph.sequence.json."""

import argparse
import json
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.graph.sequence import build_sequence_graph
from signpost.parsing.io import read_jsonl


def build_sequence_graph_file(
    chunks_path: Path,
    output_path: Path,
    *,
    namespace: str,
    max_chunks: int | None = None,
) -> dict[str, int | str]:
    chunks = list(read_jsonl(chunks_path))
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    graph = build_sequence_graph(chunks, namespace=namespace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "chunks": graph["metadata"]["chunks"],
        "documents": graph["metadata"]["documents"],
        "sequence_edges": graph["metadata"]["sequence_edges"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build F8 sequence graph")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", default="outputs/mini/graph.sequence.json")
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()

    result = build_sequence_graph_file(
        resolve_project_path(args.chunks),
        resolve_project_path(args.output),
        namespace=args.namespace,
        max_chunks=args.max_chunks,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
