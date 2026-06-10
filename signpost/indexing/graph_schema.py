from __future__ import annotations

"""Elasticsearch schema and adapters for F10 graph-object indexing."""

import hashlib
import json
import re
from typing import Any


def graph_index_name(namespace: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "-", namespace.lower()).strip("-")
    if not safe:
        raise ValueError("namespace must contain at least one index-safe character")
    return f"signpost-{safe}-graph"


def graph_index_mapping(vector_dimensions: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"signpost_text": {"type": "standard"}}},
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "namespace": {"type": "keyword"},
                "object_type": {"type": "keyword"},
                "graph_parent_id": {"type": "keyword"},
                "vector_part": {"type": "integer"},
                "vector_part_count": {"type": "integer"},
                "is_vector_part": {"type": "boolean"},
                "is_vector_parent": {"type": "boolean"},
                "vector_searchable": {"type": "boolean"},
                "text_searchable": {"type": "boolean"},
                "node_id": {"type": "keyword"},
                "edge_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "target": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "signpost_text", "fields": {"keyword": {"type": "keyword"}}},
                "name": {"type": "text", "analyzer": "signpost_text", "fields": {"keyword": {"type": "keyword"}}},
                "entity_type": {"type": "keyword"},
                "relation_types": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "signpost_text"},
                "content_vector": {"type": "dense_vector", "dims": vector_dimensions, "index": True, "similarity": "cosine"},
                "level": {"type": "integer"},
                "parent_node_id": {"type": "keyword"},
                "child_node_ids": {"type": "keyword"},
                "source_chunk_ids": {"type": "keyword"},
                "source_locates": {"type": "keyword"},
                "section_path": {"type": "keyword"},
                "weight": {"type": "float"},
                "pagerank": {"type": "float"},
                "metadata": {"type": "object", "enabled": False},
            }
        },
    }


def graph_to_index_documents(graph: dict[str, Any], *, namespace: str) -> list[dict[str, Any]]:
    node_by_id = {node["node_id"]: node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("node_id")}
    docs: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if node.get("node_type") == "entity":
            docs.append(entity_to_es_doc(node, namespace=namespace))
        elif node.get("node_type") == "summary":
            docs.append(summary_to_es_doc(node, namespace=namespace))
    for edge in graph.get("edges", []):
        if edge.get("edge_type") == "semantic":
            docs.append(relation_to_es_doc(edge, node_by_id=node_by_id, namespace=namespace))
    return docs


def entity_to_es_doc(node: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    content = _entity_text(node)
    return {
        "id": node["node_id"],
        "type": "graph",
        "namespace": namespace,
        "object_type": "entity",
        "node_id": node["node_id"],
        "name": node.get("name", ""),
        "entity_type": node.get("entity_type", ""),
        "title": node.get("name", ""),
        "content": content,
        "source_chunk_ids": node.get("source_chunk_ids") or [],
        "source_locates": node.get("source_locates") or [],
        "pagerank": float(node.get("pagerank") or 0.0),
        "metadata": {
            "source_mapping": node.get("source_mapping") or {},
            "type_counts": node.get("type_counts") or {},
            "auto_created": bool(node.get("auto_created", False)),
            "views": node.get("views") or [],
        },
    }


def relation_to_es_doc(edge: dict[str, Any], *, node_by_id: dict[str, dict[str, Any]], namespace: str) -> dict[str, Any]:
    relation_types = edge.get("relation_types") or edge.get("relation_type") or ["related_to"]
    if isinstance(relation_types, str):
        relation_types = [relation_types]
    source_label = _node_label(node_by_id.get(edge.get("source"), {}), edge.get("source", ""))
    target_label = _node_label(node_by_id.get(edge.get("target"), {}), edge.get("target", ""))
    content = _relation_text(edge, source_label, target_label, relation_types)
    edge_id = _edge_id(edge)
    return {
        "id": edge_id,
        "type": "graph",
        "namespace": namespace,
        "object_type": "relation",
        "edge_id": edge_id,
        "source": edge.get("source"),
        "target": edge.get("target"),
        "title": f"{source_label} -> {target_label}",
        "content": content,
        "relation_types": sorted(set(relation_types)),
        "source_chunk_ids": edge.get("source_chunk_ids") or [],
        "source_locates": edge.get("source_locates") or [],
        "weight": float(edge.get("weight") or 1.0),
        "metadata": {
            "source_name": source_label,
            "target_name": target_label,
            "source_mapping": edge.get("source_mapping") or {},
            "views": edge.get("views") or [],
        },
    }


def summary_to_es_doc(node: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    content = _summary_text(node)
    return {
        "id": node["node_id"],
        "type": "graph",
        "namespace": namespace,
        "object_type": "summary",
        "node_id": node["node_id"],
        "title": node.get("title", ""),
        "content": content,
        "level": int(node.get("level") or 0),
        "parent_node_id": node.get("parent_node_id"),
        "child_node_ids": node.get("child_node_ids") or [],
        "source_chunk_ids": node.get("source_chunk_ids") or [],
        "source_locates": node.get("source_locates") or [],
        "section_path": node.get("section_path") or [],
        "metadata": {
            "summary_type": node.get("summary_type") or node.get("original_node_type") or "summary",
            "views": node.get("views") or [],
            "raw_metadata": node.get("metadata") or {},
        },
    }


def attach_vectors(docs: list[dict[str, Any]], vectors: list[list[float]]) -> list[dict[str, Any]]:
    if len(docs) != len(vectors):
        raise ValueError("docs and vectors length mismatch")
    return [{**doc, "content_vector": vector} for doc, vector in zip(docs, vectors, strict=True)]


def chunk_parent_updates(graph: dict[str, Any]) -> dict[str, list[str]]:
    """Return chunk_id -> direct parent summary node ids from structure edges."""

    updates: dict[str, list[str]] = {}
    summary_ids = {node["node_id"] for node in graph.get("nodes", []) if node.get("node_type") == "summary"}
    chunk_ids = {node["node_id"]: node.get("chunk_id") for node in graph.get("nodes", []) if node.get("node_type") == "chunk"}
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "structure":
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in summary_ids or target not in chunk_ids:
            continue
        chunk_id = chunk_ids[target]
        if not chunk_id:
            continue
        updates.setdefault(chunk_id, [])
        if source not in updates[chunk_id]:
            updates[chunk_id].append(source)
    return updates


def _entity_text(node: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            str(node.get("name") or ""),
            str(node.get("entity_type") or ""),
            str(node.get("description") or ""),
            " ".join(node.get("source_locates") or []),
        ]
        if part
    )


def _relation_text(edge: dict[str, Any], source_label: str, target_label: str, relation_types: list[str]) -> str:
    return "\n".join(
        part
        for part in [
            f"{source_label} -> {target_label}",
            " ".join(relation_types),
            str(edge.get("description") or ""),
            " ".join(edge.get("source_locates") or []),
        ]
        if part
    )


def _summary_text(node: dict[str, Any]) -> str:
    title = str(node.get("title") or "")
    content = str(node.get("content") or "")
    if title and title not in content:
        return f"{title}\n{content}".strip()
    return content or title


def _node_label(node: dict[str, Any], fallback: str) -> str:
    return str(node.get("name") or node.get("title") or fallback)


def _edge_id(edge: dict[str, Any]) -> str:
    seed = json.dumps(
        {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "edge_type": edge.get("edge_type"),
            "relation_types": edge.get("relation_types") or [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "edge:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
