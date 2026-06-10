from __future__ import annotations

"""F7 CLI: chunks + document trees -> graph.structure.json."""

import argparse
import json
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.graph.structure import build_structure_graph
from signpost.indexing.summarizer import create_summarizer
from signpost.parsing.io import read_jsonl


def build_structure_graph_file(
    chunks_path: Path,
    trees_path: Path,
    output_path: Path,
    *,
    namespace: str,
    summarizer_name: str = "deterministic",
    max_chunks: int | None = None,
    max_summary_tokens: int = 512,
    cluster_token_budget: int = 4096,
) -> dict[str, int | str]:
    chunks = list(read_jsonl(chunks_path))
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    trees = list(read_jsonl(trees_path))
    summarizer = create_summarizer(summarizer_name)
    graph = build_structure_graph(
        chunks,
        trees,
        summarizer,
        namespace=namespace,
        max_summary_tokens=max_summary_tokens,
        cluster_token_budget=cluster_token_budget,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "chunks": graph["metadata"]["chunks"],
        "raptor_nodes": graph["metadata"]["raptor_nodes"],
        "structure_edges": graph["metadata"]["structure_edges"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build F7 structure/RAPTOR graph")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--document-trees", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summarizer", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-summary-tokens", type=int, default=512)
    parser.add_argument("--cluster-token-budget", type=int, default=4096)
    args = parser.parse_args()

    result = build_structure_graph_file(
        resolve_project_path(args.chunks),
        resolve_project_path(args.document_trees),
        resolve_project_path(args.output),
        namespace=args.namespace,
        summarizer_name=args.summarizer,
        max_chunks=args.max_chunks,
        max_summary_tokens=args.max_summary_tokens,
        cluster_token_budget=args.cluster_token_budget,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

