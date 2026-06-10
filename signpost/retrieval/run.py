from __future__ import annotations

"""F13 graph retrieval engine.

One query returns the paper's two result groups:

- text_group: original chunks + RAPTOR summaries + PPR(text scene)
- graph_group: entities + relations + PPR(graph scene)

Each item is enriched with F11 offline signposts, and each group receives F12
online signposts.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.retrieval.chunk_search import search_chunks
from signpost.retrieval.graph_search import search_graph
from signpost.retrieval.offline_signpost import attach_offline_signposts
from signpost.retrieval.online_signpost import compute_online_signpost
from signpost.retrieval.signpost_variants import FULL, VALID_VARIANTS, apply_signpost_variant


def run_retrieval(
    *,
    namespace: str,
    query: str,
    graph: dict[str, Any],
    mode: str = "hybrid",
    chunk_top_k: int = 5,
    summary_top_k: int = 5,
    graph_top_k: int = 5,
    ppr_top_k: int = 5,
    embedding_provider_name: str = "ecnu",
    hash_dimensions: int = 128,
    chunk_index_name: str | None = None,
    graph_index_name: str | None = None,
    signpost_variant: str = FULL,
) -> dict[str, Any]:
    chunk_items = search_chunks(
        namespace=namespace,
        query=query,
        mode=mode,
        top_k=chunk_top_k,
        index_name=chunk_index_name,
        embedding_provider_name=embedding_provider_name,
        hash_dimensions=hash_dimensions,
    ).get("items", [])
    summary_items = search_graph(
        namespace=namespace,
        query=query,
        mode=mode,
        object_types=["summary"],
        top_k=summary_top_k,
        index_name=graph_index_name,
        embedding_provider_name=embedding_provider_name,
        hash_dimensions=hash_dimensions,
    ).get("items", [])
    graph_items = search_graph(
        namespace=namespace,
        query=query,
        mode=mode,
        object_types=["entity", "relation"],
        top_k=graph_top_k,
        index_name=graph_index_name,
        embedding_provider_name=embedding_provider_name,
        hash_dimensions=hash_dimensions,
    ).get("items", [])
    return build_grouped_retrieval_result(
        query=query,
        graph=graph,
        chunk_items=chunk_items,
        summary_items=summary_items,
        graph_items=graph_items,
        ppr_top_k=ppr_top_k,
        signpost_variant=signpost_variant,
    )


def build_grouped_retrieval_result(
    *,
    query: str,
    graph: dict[str, Any],
    chunk_items: list[dict[str, Any]],
    summary_items: list[dict[str, Any]],
    graph_items: list[dict[str, Any]],
    ppr_top_k: int = 5,
    signpost_variant: str = FULL,
) -> dict[str, Any]:
    text_items = _tag_items(chunk_items, "chunk") + _tag_items(summary_items, "summary")
    graph_group_items = _tag_graph_items(graph_items)
    text_items = attach_offline_signposts(graph, text_items)
    graph_group_items = attach_offline_signposts(graph, graph_group_items)
    text_online = compute_online_signpost(graph, text_items, scene="text", top_k=ppr_top_k)
    graph_online = compute_online_signpost(graph, graph_group_items, scene="graph", top_k=ppr_top_k)
    result = {
        "query": query,
        "text_group": {
            "items": text_items,
            "online_signpost": text_online,
        },
        "graph_group": {
            "items": graph_group_items,
            "online_signpost": graph_online,
        },
        "metadata": {
            "text_items": len(text_items),
            "graph_items": len(graph_group_items),
            "ppr_top_k": ppr_top_k,
        },
    }
    return apply_signpost_variant(result, signpost_variant)


def _tag_items(items: list[dict[str, Any]], retrieval_type: str) -> list[dict[str, Any]]:
    return [{**item, "retrieval_type": retrieval_type} for item in items]


def _tag_graph_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tagged = []
    for item in items:
        object_type = item.get("object_type") or "graph"
        retrieval_type = "entity" if object_type == "entity" else "relation" if object_type == "relation" else object_type
        tagged.append({**item, "retrieval_type": retrieval_type})
    return tagged


def _load_graph(path: str) -> dict[str, Any]:
    return json.loads(resolve_project_path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="F13 Signpost graph retrieval engine")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--graph")
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--summary-top-k", type=int, default=5)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument("--ppr-top-k", type=int, default=5)
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    parser.add_argument("--chunk-index-name")
    parser.add_argument("--graph-index-name")
    parser.add_argument("--signpost-variant", choices=sorted(VALID_VARIANTS), default=FULL)
    args = parser.parse_args()

    graph_path = args.graph or f"outputs/{args.namespace}/graph.unified.json"
    result = run_retrieval(
        namespace=args.namespace,
        query=args.query,
        graph=_load_graph(graph_path),
        mode=args.mode,
        chunk_top_k=args.chunk_top_k,
        summary_top_k=args.summary_top_k,
        graph_top_k=args.graph_top_k,
        ppr_top_k=args.ppr_top_k,
        embedding_provider_name=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
        chunk_index_name=args.chunk_index_name,
        graph_index_name=args.graph_index_name,
        signpost_variant=args.signpost_variant,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} text_items={result['metadata']['text_items']} graph_items={result['metadata']['graph_items']}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
