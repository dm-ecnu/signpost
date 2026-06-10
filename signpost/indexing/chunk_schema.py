from __future__ import annotations

"""Elasticsearch schema for the chunk index I_C over V_c (paper Section 3.3,
"Indexes and physical schema").

Each document carries lexical fields, a dense-vector field, object type, payload
text, source-locator fields, and a serialized sketch pointer/payload, so
query-time routing need not join through the graph.
"""

import re
from typing import Any


def chunk_index_name(namespace: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "-", namespace.lower()).strip("-")
    if not safe:
        raise ValueError("namespace must contain at least one index-safe character")
    return f"signpost-{safe}-chunks"


def chunk_index_mapping(vector_dimensions: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "signpost_text": {
                        "type": "standard",
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "namespace": {"type": "keyword"},
                "dataset_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "signpost_text"},
                "content_vector": {"type": "dense_vector", "dims": vector_dimensions, "index": True, "similarity": "cosine"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "section_path": {"type": "keyword"},
                "prev_chunk_id": {"type": "keyword"},
                "next_chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "token_count": {"type": "integer"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }


def chunk_to_es_doc(chunk: dict[str, Any], *, namespace: str, dataset_id: str, vector: list[float]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    return {
        "id": chunk["chunk_id"],
        "type": "chunk",
        "namespace": namespace,
        "dataset_id": dataset_id,
        "doc_id": chunk["doc_id"],
        "file_name": chunk["file_name"],
        "content": chunk["content"],
        "content_vector": vector,
        "start_line": int(chunk["start_line"]),
        "end_line": int(chunk["end_line"]),
        "section_path": chunk.get("section_path") or [],
        "prev_chunk_id": chunk.get("prev_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "token_count": int(metadata.get("token_count", 0)),
        "metadata": metadata,
    }

