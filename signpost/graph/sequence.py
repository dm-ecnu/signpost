from __future__ import annotations

"""F8 sequential view graph.

The sequential view keeps the original narrative order of chunks inside each
document.  It creates bidirectional edges only between adjacent chunks from the
same document, matching the thesis definition of Eseq.
"""

from collections import defaultdict, deque
from typing import Any


def build_sequence_graph(chunks: list[dict[str, Any]], *, namespace: str) -> dict[str, Any]:
    chunks_by_doc = _chunks_by_doc(chunks)
    position_by_chunk: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for doc_id, doc_chunks in chunks_by_doc.items():
        for position, chunk in enumerate(doc_chunks):
            position_by_chunk[chunk["chunk_id"]] = position
            nodes.append(_chunk_node(chunk, position, len(doc_chunks)))
        for left, right in zip(doc_chunks, doc_chunks[1:]):
            edges.append(_sequence_edge(left, right, direction="next"))
            edges.append(_sequence_edge(right, left, direction="prev"))

    return {
        "metadata": {
            "namespace": namespace,
            "graph_type": "sequence",
            "chunks": len(chunks),
            "documents": len(chunks_by_doc),
            "sequence_edges": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }


def expand_sequence_context(
    graph: dict[str, Any],
    seed_chunk_ids: list[str],
    *,
    before: int = 1,
    after: int = 1,
) -> list[dict[str, Any]]:
    """Return chunk nodes around seed chunks using sequence edges.

    The returned rows are unique chunk nodes sorted by document position.  Each
    row includes `hop_from_seed`: negative for preceding context, zero for a seed
    chunk, and positive for following context.
    """

    node_by_chunk = {node["chunk_id"]: node for node in graph.get("nodes", []) if node.get("node_type") == "chunk"}
    next_map, prev_map = _sequence_neighbors(graph)
    best_hop: dict[str, int] = {}

    for seed in seed_chunk_ids:
        if seed not in node_by_chunk:
            continue
        best_hop[seed] = 0
        _walk_direction(seed, prev_map, -1, before, best_hop)
        _walk_direction(seed, next_map, 1, after, best_hop)

    rows: list[dict[str, Any]] = []
    for chunk_id, hop in best_hop.items():
        node = dict(node_by_chunk[chunk_id])
        node["hop_from_seed"] = hop
        rows.append(node)
    rows.sort(key=lambda item: (item.get("doc_id", ""), int(item.get("doc_position", 0)), item.get("chunk_id", "")))
    return rows


def _chunk_node(chunk: dict[str, Any], position: int, doc_chunk_count: int) -> dict[str, Any]:
    return {
        "node_id": f"chunk:{chunk['chunk_id']}",
        "node_type": "chunk",
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "file_name": chunk.get("file_name"),
        "content": chunk.get("content", ""),
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "section_path": chunk.get("section_path") or [],
        "doc_position": position,
        "doc_chunk_count": doc_chunk_count,
        "prev_chunk_id": chunk.get("prev_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
    }


def _sequence_edge(source: dict[str, Any], target: dict[str, Any], *, direction: str) -> dict[str, Any]:
    return {
        "source": f"chunk:{source['chunk_id']}",
        "target": f"chunk:{target['chunk_id']}",
        "edge_type": "sequence",
        "direction": direction,
        "doc_id": source["doc_id"],
        "source_chunk_id": source["chunk_id"],
        "target_chunk_id": target["chunk_id"],
        "source_locate": _locate(source),
        "target_locate": _locate(target),
        "distance": 1,
    }


def _chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_id"]].append(chunk)
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row.get("start_line") or 0), int(row.get("end_line") or 0), row.get("chunk_id", "")))
    return grouped


def _sequence_neighbors(graph: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    next_map: dict[str, str] = {}
    prev_map: dict[str, str] = {}
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "sequence":
            continue
        source = str(edge.get("source_chunk_id") or str(edge.get("source", "")).removeprefix("chunk:"))
        target = str(edge.get("target_chunk_id") or str(edge.get("target", "")).removeprefix("chunk:"))
        if edge.get("direction") == "next":
            next_map[source] = target
        elif edge.get("direction") == "prev":
            prev_map[source] = target
    return next_map, prev_map


def _walk_direction(seed: str, neighbor_map: dict[str, str], sign: int, limit: int, best_hop: dict[str, int]) -> None:
    queue: deque[tuple[str, int]] = deque([(seed, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= limit:
            continue
        neighbor = neighbor_map.get(current)
        if not neighbor:
            continue
        hop = sign * (depth + 1)
        if neighbor not in best_hop or abs(hop) < abs(best_hop[neighbor]):
            best_hop[neighbor] = hop
        queue.append((neighbor, depth + 1))


def _locate(chunk: dict[str, Any]) -> str:
    return f"{chunk.get('file_name')}:L{chunk.get('start_line')}-L{chunk.get('end_line')}"
