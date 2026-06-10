from __future__ import annotations

"""F5 chunk search over the Elasticsearch chunk index."""

import argparse
import json
from typing import Any

from signpost.indexing.chunk_schema import chunk_index_name
from signpost.indexing.embedding import create_embedding_provider
from signpost.storage.elasticsearch import ElasticsearchClient


def search_chunks(
    *,
    namespace: str,
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    index_name: str | None = None,
    embedding_provider_name: str = "ecnu",
    hash_dimensions: int = 128,
    es: ElasticsearchClient | None = None,
) -> dict[str, Any]:
    client = es or ElasticsearchClient()
    target_index = index_name or chunk_index_name(namespace)
    if mode == "bm25":
        return {"items": _bm25(client, target_index, namespace, query, top_k)}
    provider = create_embedding_provider(embedding_provider_name, dimensions=hash_dimensions)
    vector = provider.embed([query])[0]
    if mode == "dense":
        return {"items": _dense(client, target_index, namespace, vector, top_k)}
    bm25_items = _bm25(client, target_index, namespace, query, top_k * 2)
    dense_items = _dense(client, target_index, namespace, vector, top_k * 2)
    return {"items": _rrf_fuse(bm25_items, dense_items, top_k)}


def _bm25(client: ElasticsearchClient, index_name: str, namespace: str, query: str, top_k: int) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "filter": [{"term": {"namespace": namespace}}, {"term": {"type": "chunk"}}],
                "must": [{"match": {"content": {"query": query}}}],
            }
        },
    }
    return _hits(client.request("POST", f"{index_name}/_search", body), "bm25")


def _dense(client: ElasticsearchClient, index_name: str, namespace: str, vector: list[float], top_k: int) -> list[dict[str, Any]]:
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"bool": {"filter": [{"term": {"namespace": namespace}}, {"term": {"type": "chunk"}}]}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": {"query_vector": vector},
                },
            }
        },
    }
    return _hits(client.request("POST", f"{index_name}/_search", body), "dense")


def _hits(response: dict[str, Any], source: str) -> list[dict[str, Any]]:
    items = []
    for hit in response.get("hits", {}).get("hits", []):
        doc = hit.get("_source", {})
        items.append(
            {
                "chunk_id": doc.get("id"),
                "doc_id": doc.get("doc_id"),
                "file_name": doc.get("file_name"),
                "content": doc.get("content"),
                "start_line": doc.get("start_line"),
                "end_line": doc.get("end_line"),
                "section_path": doc.get("section_path") or [],
                "prev_chunk_id": doc.get("prev_chunk_id"),
                "next_chunk_id": doc.get("next_chunk_id"),
                "score": hit.get("_score", 0.0),
                "score_source": source,
            }
        )
    return items


def _rrf_fuse(bm25_items: list[dict[str, Any]], dense_items: list[dict[str, Any]], top_k: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(bm25_items, start=1):
        chunk_id = item["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "hybrid"}
    for rank, item in enumerate(dense_items, start=1):
        chunk_id = item["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "hybrid"}
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    result = []
    for chunk_id, score in ranked:
        item = docs[chunk_id]
        item["score"] = score
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="F5 search chunk index")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-name")
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    args = parser.parse_args()

    result = search_chunks(
        namespace=args.namespace,
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        index_name=args.index_name,
        embedding_provider_name=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

