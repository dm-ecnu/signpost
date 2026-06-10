from __future__ import annotations

"""F10 search over Elasticsearch graph-object index."""

import argparse
import json
from typing import Any

from signpost.indexing.embedding import create_embedding_provider
from signpost.indexing.graph_schema import graph_index_name
from signpost.storage.elasticsearch import ElasticsearchClient


def search_graph(
    *,
    namespace: str,
    query: str,
    mode: str = "hybrid",
    object_types: list[str] | None = None,
    top_k: int = 5,
    index_name: str | None = None,
    embedding_provider_name: str = "ecnu",
    hash_dimensions: int = 128,
    es: ElasticsearchClient | None = None,
) -> dict[str, Any]:
    client = es or ElasticsearchClient()
    target_index = index_name or graph_index_name(namespace)
    types = object_types or ["entity", "relation", "summary"]
    if mode == "bm25":
        return {"items": _dedupe_items(_bm25(client, target_index, namespace, query, types, top_k * 4), top_k)}
    provider = create_embedding_provider(embedding_provider_name, dimensions=hash_dimensions)
    vector = provider.embed([query])[0]
    if mode == "dense":
        return {"items": _dedupe_items(_dense(client, target_index, namespace, vector, types, top_k * 4), top_k)}
    bm25_items = _bm25(client, target_index, namespace, query, types, top_k * 2)
    dense_items = _dense(client, target_index, namespace, vector, types, top_k * 2)
    return {"items": _rrf_fuse(bm25_items, dense_items, top_k)}


def _filters(namespace: str, object_types: list[str]) -> list[dict[str, Any]]:
    return [{"term": {"namespace": namespace}}, {"term": {"type": "graph"}}, {"terms": {"object_type": object_types}}]


def _searchable_filter(field: str) -> dict[str, Any]:
    return {
        "bool": {
            "should": [
                {"term": {field: True}},
                {"bool": {"must_not": [{"exists": {"field": field}}]}},
            ],
            "minimum_should_match": 1,
        }
    }


def _bm25(client: ElasticsearchClient, index_name: str, namespace: str, query: str, object_types: list[str], top_k: int) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "filter": _filters(namespace, object_types) + [_searchable_filter("text_searchable")],
                "must": [{"multi_match": {"query": query, "fields": ["title^2", "name^2", "content"]}}],
            }
        },
    }
    return _hydrate_parent_hits(client, index_name, _hits(client.request("POST", f"{index_name}/_search", body), "bm25"))


def _dense(client: ElasticsearchClient, index_name: str, namespace: str, vector: list[float], object_types: list[str], top_k: int) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"bool": {"filter": _filters(namespace, object_types) + [_searchable_filter("vector_searchable")]}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": {"query_vector": vector},
                },
            }
        },
    }
    return _hydrate_parent_hits(client, index_name, _hits(client.request("POST", f"{index_name}/_search", body), "dense"))


def _hits(response: dict[str, Any], source: str) -> list[dict[str, Any]]:
    items = []
    for hit in response.get("hits", {}).get("hits", []):
        doc = hit.get("_source", {})
        graph_parent_id = doc.get("graph_parent_id") or doc.get("id")
        items.append(
            {
                "id": graph_parent_id,
                "vector_doc_id": doc.get("id"),
                "graph_parent_id": graph_parent_id,
                "vector_part": doc.get("vector_part", 0),
                "vector_part_count": doc.get("vector_part_count", 1),
                "is_vector_part": bool(doc.get("is_vector_part", False)),
                "is_vector_parent": bool(doc.get("is_vector_parent", False)),
                "object_type": doc.get("object_type"),
                "node_id": doc.get("node_id"),
                "edge_id": doc.get("edge_id"),
                "title": doc.get("title"),
                "name": doc.get("name"),
                "content": doc.get("content"),
                "source": doc.get("source"),
                "target": doc.get("target"),
                "source_chunk_ids": doc.get("source_chunk_ids") or [],
                "source_locates": doc.get("source_locates") or [],
                "score": hit.get("_score", 0.0),
                "score_source": source,
            }
        )
    return items


def _hydrate_parent_hits(client: ElasticsearchClient, index_name: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_ids = sorted(
        {
            str(item.get("graph_parent_id") or item.get("id") or "")
            for item in items
            if item.get("is_vector_part") and str(item.get("graph_parent_id") or item.get("id") or "")
        }
    )
    if not parent_ids:
        return items
    try:
        response = client.request("POST", f"{index_name}/_mget", {"ids": parent_ids})
    except Exception:
        return items
    parents = {
        str(doc.get("_id")): doc.get("_source", {})
        for doc in response.get("docs", [])
        if doc.get("found") and isinstance(doc.get("_source"), dict)
    }
    if not parents:
        return items

    hydrated: list[dict[str, Any]] = []
    parent_fields = (
        "object_type",
        "node_id",
        "edge_id",
        "title",
        "name",
        "content",
        "source",
        "target",
        "source_chunk_ids",
        "source_locates",
    )
    for item in items:
        parent = parents.get(str(item.get("graph_parent_id") or item.get("id") or ""))
        if not item.get("is_vector_part") or not parent:
            hydrated.append(item)
            continue
        merged = dict(item)
        for field in parent_fields:
            if parent.get(field) is not None:
                merged[field] = parent.get(field)
        merged["matched_vector_doc_id"] = item.get("vector_doc_id")
        merged["matched_vector_part"] = item.get("vector_part", 0)
        merged["matched_vector_part_count"] = item.get("vector_part_count", 1)
        merged["is_vector_parent"] = bool(parent.get("is_vector_parent", False))
        hydrated.append(merged)
    return hydrated


def _rrf_fuse(bm25_items: list[dict[str, Any]], dense_items: list[dict[str, Any]], top_k: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(bm25_items, start=1):
        item_id = item["id"]
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        docs[item_id] = {**item, "score_source": "hybrid"}
    for rank, item in enumerate(dense_items, start=1):
        item_id = item["id"]
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        docs[item_id] = {**item, "score_source": "hybrid"}
    return [{**docs[item_id], "score": score} for item_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]]


def _dedupe_items(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("id") or item.get("vector_doc_id") or "")
        if not item_id:
            continue
        if item_id not in best or float(item.get("score") or 0.0) > float(best[item_id].get("score") or 0.0):
            best[item_id] = item
    return sorted(best.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="F10 search graph-object index")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--object-type", action="append", choices=["entity", "relation", "summary"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-name")
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    args = parser.parse_args()

    result = search_graph(
        namespace=args.namespace,
        query=args.query,
        mode=args.mode,
        object_types=args.object_type,
        top_k=args.top_k,
        index_name=args.index_name,
        embedding_provider_name=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
