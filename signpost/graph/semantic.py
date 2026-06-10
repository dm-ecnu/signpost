from __future__ import annotations

"""F6 semantic graph construction.

The graph JSON deliberately keeps source mappings instead of overwriting entity
descriptions.  This matches the paper's multi-source evidence strategy and lets
later F11/F14 features trace an entity back to exact chunks and line ranges.
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from signpost.indexing.semantic_extractor import ExtractionResult, SemanticExtractor
from signpost.llm.client import OpenAICompatibleClient


def build_semantic_graph(
    chunks: list[dict[str, Any]],
    extractor: SemanticExtractor,
    *,
    namespace: str,
    synthesize_descriptions: bool = False,
    progress_every: int = 0,
    progress_file: Path | None = None,
    extraction_results: dict[str, ExtractionResult] | None = None,
) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str], dict[str, Any]] = {}
    chunk_nodes: dict[str, dict[str, Any]] = {}
    source_edges: dict[tuple[str, str], dict[str, Any]] = {}

    total = len(chunks)
    if progress_every:
        print(f"semantic_graph namespace={namespace} chunks={total} extractor={extractor.__class__.__name__}", file=sys.stderr, flush=True)
    _write_progress(progress_file, {"event": "start", "namespace": namespace, "chunks": total, "extractor": extractor.__class__.__name__})
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk["chunk_id"]
        chunk_node_id = f"chunk:{chunk_id}"
        chunk_nodes[chunk_node_id] = _chunk_node(chunk)
        from_cache = extraction_results is not None and chunk_id in extraction_results
        if progress_every:
            print(
                f"semantic_graph {'merging_cache' if from_cache else 'extracting'}={idx}/{total} chunk_id={chunk_id} tokens={chunk.get('metadata', {}).get('token_count')}",
                file=sys.stderr,
                flush=True,
            )
        _write_progress(
            progress_file,
            {
                "event": "merging_cache" if from_cache else "extracting",
                "index": idx,
                "total": total,
                "chunk_id": chunk_id,
                "tokens": chunk.get("metadata", {}).get("token_count"),
                "file_name": chunk.get("file_name"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
            },
        )
        result = extraction_results[chunk_id] if from_cache else extractor.extract(chunk)
        _merge_extraction(result, chunk, entities, relations, source_edges)
        if progress_every and idx % progress_every == 0:
            print(f"semantic_graph processed={idx}/{total} entities={len(entities)} relations={len(relations)}", file=sys.stderr, flush=True)
        _write_progress(
            progress_file,
            {
                "event": "processed",
                "index": idx,
                "total": total,
                "chunk_id": chunk_id,
                "entities": len(entities),
                "relations": len(relations),
                "source_edges": len(source_edges),
            },
        )

    nodes = list(chunk_nodes.values()) + list(entities.values())
    semantic_edges = [_relation_to_edge(edge) for edge in relations.values()]
    source_edge_rows = list(source_edges.values())
    if synthesize_descriptions:
        _synthesize_entity_and_relation_descriptions(list(entities.values()), semantic_edges)
    result = {
        "metadata": {
            "namespace": namespace,
            "graph_type": "semantic",
            "chunks": len(chunks),
            "entities": len(entities),
            "relations": len(relations),
            "source_edges": len(source_edge_rows),
        },
        "nodes": nodes,
        "edges": semantic_edges + source_edge_rows,
    }
    _write_progress(progress_file, {"event": "finish", "namespace": namespace, "chunks": len(chunks), "entities": len(entities), "relations": len(relations), "source_edges": len(source_edge_rows)})
    return result


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.time(), **payload}, ensure_ascii=False, separators=(",", ":")) + "\n")


def _merge_extraction(
    result: ExtractionResult,
    chunk: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    relations: dict[tuple[str, str], dict[str, Any]],
    source_edges: dict[tuple[str, str], dict[str, Any]],
) -> None:
    chunk_id = chunk["chunk_id"]
    source_key = f"{chunk['doc_id']}:{chunk_id}"
    extracted_entity_ids: set[str] = set()

    for record in result.entities:
        entity_id = entity_node_id(record.name)
        extracted_entity_ids.add(entity_id)
        node = entities.setdefault(
            entity_id,
            {
                "node_id": entity_id,
                "node_type": "entity",
                "name": record.name,
                "entity_type": record.entity_type,
                "description": "",
                "source_chunk_ids": [],
                "source_locates": [],
                "source_mapping": {},
                "type_counts": {},
                "auto_created": False,
            },
        )
        node["source_chunk_ids"] = sorted(set(node["source_chunk_ids"]) | {chunk_id})
        node["source_locates"] = sorted(set(node["source_locates"]) | {_source_locate(chunk)})
        node["source_mapping"][source_key] = {
            "description": record.description,
            "entity_type": record.entity_type,
            "file_name": chunk.get("file_name"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
        }
        counts = Counter(node.get("type_counts", {}))
        counts[record.entity_type] += 1
        node["type_counts"] = dict(counts)
        node["entity_type"] = counts.most_common(1)[0][0]

        source_edges[(entity_id, chunk_id)] = _source_edge(entity_id, chunk)

    for record in result.relations:
        src_id = entity_node_id(record.source)
        tgt_id = entity_node_id(record.target)
        for endpoint_id, endpoint_name in ((src_id, record.source), (tgt_id, record.target)):
            if endpoint_id not in entities:
                entities[endpoint_id] = _placeholder_entity(endpoint_id, endpoint_name, chunk)
            source_edges[(endpoint_id, chunk_id)] = _source_edge(endpoint_id, chunk)

        edge_key = _edge_key(src_id, tgt_id)
        edge = relations.setdefault(
            edge_key,
            {
                "source": edge_key[0],
                "target": edge_key[1],
                "edge_type": "semantic_relation",
                "description": "",
                "relation_types": [],
                "weight": 0.0,
                "source_chunk_ids": [],
                "source_locates": [],
                "source_mapping": {},
            },
        )
        edge["weight"] += record.weight
        edge["relation_types"] = sorted(set(edge["relation_types"]) | set(record.keywords or ["related_to"]))
        edge["source_chunk_ids"] = sorted(set(edge["source_chunk_ids"]) | {chunk_id})
        edge["source_locates"] = sorted(set(edge["source_locates"]) | {_source_locate(chunk)})
        edge["source_mapping"][source_key] = {
            "description": record.description,
            "relation_types": record.keywords,
            "weight": record.weight,
            "file_name": chunk.get("file_name"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
        }

    for entity_id in extracted_entity_ids:
        entities[entity_id]["description"] = _compose_description(entities[entity_id]["source_mapping"])


def _relation_to_edge(edge: dict[str, Any]) -> dict[str, Any]:
    edge = dict(edge)
    edge["description"] = _compose_description(edge["source_mapping"])
    return edge


def _chunk_node(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": f"chunk:{chunk['chunk_id']}",
        "node_type": "chunk",
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "file_name": chunk.get("file_name"),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "section_path": chunk.get("section_path") or [],
    }


def _source_edge(entity_id: str, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": entity_id,
        "target": f"chunk:{chunk['chunk_id']}",
        "edge_type": "source",
        "source_chunk_ids": [chunk["chunk_id"]],
        "source_locates": [_source_locate(chunk)],
    }


def _placeholder_entity(entity_id: str, name: str, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": entity_id,
        "node_type": "entity",
        "name": name,
        "entity_type": "UNKNOWN",
        "description": "",
        "source_chunk_ids": [chunk["chunk_id"]],
        "source_locates": [_source_locate(chunk)],
        "source_mapping": {},
        "type_counts": {},
        "auto_created": True,
    }


def entity_node_id(name: str) -> str:
    normalized = normalize_entity_name(name)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"entity:{digest}"


def normalize_entity_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _edge_key(src_id: str, tgt_id: str) -> tuple[str, str]:
    return (src_id, tgt_id) if src_id <= tgt_id else (tgt_id, src_id)


def _source_locate(chunk: dict[str, Any]) -> str:
    return f"{chunk.get('file_name')}:L{chunk.get('start_line')}-L{chunk.get('end_line')}"


def _compose_description(source_mapping: dict[str, dict[str, Any]], max_items: int = 5) -> str:
    parts = []
    for source_key, payload in list(source_mapping.items())[:max_items]:
        desc = payload.get("description") or ""
        locate = f"{payload.get('file_name')}:L{payload.get('start_line')}-L{payload.get('end_line')}"
        if desc:
            parts.append(f"{desc} ({locate}; {source_key})")
    return " ".join(parts)


def _synthesize_entity_and_relation_descriptions(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Generate unified descriptions from source evidence with one LLM pass per object.

    This is the paper's stage-three evidence synthesis.  It is optional because
    running it on full datasets can be expensive; source mappings are still
    preserved either way.
    """

    client = OpenAICompatibleClient()
    for node in nodes:
        if node.get("node_type") == "entity" and len(node.get("source_mapping", {})) > 1:
            node["description"] = _synthesize_description(client, "entity", node.get("name", ""), node.get("source_mapping", {}))
    for edge in edges:
        if edge.get("edge_type") == "semantic_relation" and len(edge.get("source_mapping", {})) > 1:
            label = f"{edge.get('source')} - {edge.get('target')}"
            edge["description"] = _synthesize_description(client, "relation", label, edge.get("source_mapping", {}))


def _synthesize_description(client: OpenAICompatibleClient, object_type: str, label: str, source_mapping: dict[str, dict[str, Any]]) -> str:
    evidence = []
    for source_key, payload in source_mapping.items():
        evidence.append(
            {
                "source": source_key,
                "description": payload.get("description", ""),
                "file_name": payload.get("file_name"),
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
            }
        )
    prompt = (
        f"Write one concise unified description for this {object_type}: {label}.\n"
        "Use only the evidence. Include compact source citations like file:Lx-Ly.\n"
        f"Evidence JSON:\n{evidence}"
    )
    return client.chat(
        [
            {"role": "system", "content": "You synthesize knowledge graph descriptions from cited evidence."},
            {"role": "user", "content": prompt},
        ]
    ).strip()
