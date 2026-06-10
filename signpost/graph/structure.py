from __future__ import annotations

"""F7 structure view and hierarchy-guided RAPTOR graph.

The preferred path follows the paper: reconstruct the document tree from F4 and
summarize it bottom-up.  If no usable hierarchy is present, the builder falls
back to standard RAPTOR-style recursive grouping over chunk order.
"""

import hashlib
from collections import defaultdict
from typing import Any

from signpost.chunking.tokenizer import count_tokens
from signpost.indexing.summarizer import Summarizer


def build_structure_graph(
    chunks: list[dict[str, Any]],
    document_trees: list[dict[str, Any]],
    summarizer: Summarizer,
    *,
    namespace: str,
    max_summary_tokens: int = 512,
    cluster_token_budget: int = 4096,
) -> dict[str, Any]:
    chunks_by_doc = _chunks_by_doc(chunks)
    tree_by_doc = {tree["doc_id"]: tree for tree in document_trees}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for chunk in chunks:
        nodes.append(_chunk_node(chunk))

    for doc_id, doc_chunks in chunks_by_doc.items():
        tree = tree_by_doc.get(doc_id)
        if tree and _has_real_hierarchy(tree.get("tree", {})):
            root_summary_ids = _build_tree_summaries(tree["tree"], doc_chunks, summarizer, nodes, edges, max_summary_tokens=max_summary_tokens)
            if not root_summary_ids:
                _build_standard_raptor(doc_id, doc_chunks, summarizer, nodes, edges, max_summary_tokens=max_summary_tokens, cluster_token_budget=cluster_token_budget)
            else:
                _add_document_root_summary(tree, doc_chunks, root_summary_ids, summarizer, nodes, edges, max_summary_tokens=max_summary_tokens)
        else:
            _build_standard_raptor(doc_id, doc_chunks, summarizer, nodes, edges, max_summary_tokens=max_summary_tokens, cluster_token_budget=cluster_token_budget)

    return {
        "metadata": {
            "namespace": namespace,
            "graph_type": "structure",
            "chunks": len(chunks),
            "raptor_nodes": sum(1 for node in nodes if node.get("node_type") == "raptor"),
            "structure_edges": sum(1 for edge in edges if edge.get("edge_type") == "structure"),
        },
        "nodes": nodes,
        "edges": edges,
    }


def _build_tree_summaries(
    tree_node: dict[str, Any],
    doc_chunks: list[dict[str, Any]],
    summarizer: Summarizer,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_summary_tokens: int,
    parent_summary_id: str | None = None,
) -> list[str]:
    created: list[str] = []
    for child in tree_node.get("children", []):
        child_summary_ids = _build_tree_summaries(child, doc_chunks, summarizer, nodes, edges, max_summary_tokens=max_summary_tokens)
        own_chunks = _chunks_in_range(doc_chunks, int(child.get("start_line", 0)), int(child.get("end_line", 0)))
        child_texts = [_node_content_by_id(nodes, summary_id) for summary_id in child_summary_ids]
        chunk_texts = [chunk["content"] for chunk in own_chunks]
        if not child_texts and not chunk_texts:
            continue
        summary_id = _raptor_id(child.get("title", ""), child.get("section_path", []), own_chunks)
        title, content = summarizer.summarize(str(child.get("title") or ""), child_texts + chunk_texts, max_tokens=max_summary_tokens)
        source_chunk_ids = sorted({chunk["chunk_id"] for chunk in own_chunks} | set().union(*[_source_chunks_by_id(nodes, sid) for sid in child_summary_ids] or [set()]))
        source_locates = sorted({_locate(chunk) for chunk in own_chunks} | set().union(*[_source_locates_by_id(nodes, sid) for sid in child_summary_ids] or [set()]))
        nodes.append(
            {
                "node_id": summary_id,
                "node_type": "raptor",
                "title": title,
                "content": content,
                "level": int(child.get("level", 1)),
                "parent_node_id": parent_summary_id,
                "child_node_ids": child_summary_ids + [f"chunk:{chunk['chunk_id']}" for chunk in own_chunks],
                "source_chunk_ids": source_chunk_ids,
                "source_locates": source_locates,
                "section_path": child.get("section_path") or [],
                "metadata": {"mode": "document_tree"},
            }
        )
        for child_id in child_summary_ids:
            _set_parent_node_id(nodes, child_id, summary_id)
        for child_id in child_summary_ids:
            edges.append({"source": summary_id, "target": child_id, "edge_type": "structure"})
        for chunk in own_chunks:
            edges.append({"source": summary_id, "target": f"chunk:{chunk['chunk_id']}", "edge_type": "structure"})
        created.append(summary_id)
    return created


def _add_document_root_summary(
    tree: dict[str, Any],
    doc_chunks: list[dict[str, Any]],
    child_summary_ids: list[str],
    summarizer: Summarizer,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_summary_tokens: int,
) -> None:
    if len(child_summary_ids) == 1:
        return
    title = str(tree.get("file_name") or tree.get("doc_id") or "Document")
    root_id = _raptor_id(title, [title, "root"], doc_chunks)
    child_texts = [_node_content_by_id(nodes, child_id) for child_id in child_summary_ids]
    source_chunk_ids = sorted(set().union(*[_source_chunks_by_id(nodes, sid) for sid in child_summary_ids] or [set()]))
    source_locates = sorted(set().union(*[_source_locates_by_id(nodes, sid) for sid in child_summary_ids] or [set()]))
    summary_title, content = summarizer.summarize(title, child_texts, max_tokens=max_summary_tokens)
    nodes.append(
        {
            "node_id": root_id,
            "node_type": "raptor",
            "title": summary_title,
            "content": content,
            "level": 0,
            "parent_node_id": None,
            "child_node_ids": child_summary_ids,
            "source_chunk_ids": source_chunk_ids,
            "source_locates": source_locates,
            "section_path": [title],
            "metadata": {"mode": "document_tree_root"},
        }
    )
    for child_id in child_summary_ids:
        _set_parent_node_id(nodes, child_id, root_id)
        edges.append({"source": root_id, "target": child_id, "edge_type": "structure"})


def _build_standard_raptor(
    doc_id: str,
    doc_chunks: list[dict[str, Any]],
    summarizer: Summarizer,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    max_summary_tokens: int,
    cluster_token_budget: int,
) -> None:
    current_payloads: list[dict[str, Any]] = [_chunk_payload_for_group(chunk) for chunk in doc_chunks]
    level = 1
    while len(current_payloads) > 1:
        groups = _group_by_token_budget(current_payloads, cluster_token_budget)
        next_payloads: list[dict[str, Any]] = []
        for idx, group in enumerate(groups):
            title = f"{doc_id} summary L{level}.{idx + 1}"
            summary_id = _raptor_id(title, [title], [{"chunk_id": item["id"] for item in group}])
            source_chunk_ids = sorted(set().union(*(set(item["source_chunk_ids"]) for item in group)))
            source_locates = sorted(set().union(*(set(item["source_locates"]) for item in group)))
            summary_title, content = summarizer.summarize(title, [item["content"] for item in group], max_tokens=max_summary_tokens)
            child_ids = [item["node_id"] for item in group]
            nodes.append(
                {
                    "node_id": summary_id,
                    "node_type": "raptor",
                    "title": summary_title,
                    "content": content,
                    "level": level,
                    "parent_node_id": None,
                    "child_node_ids": child_ids,
                    "source_chunk_ids": source_chunk_ids,
                    "source_locates": source_locates,
                    "section_path": [title],
                    "metadata": {"mode": "standard_raptor"},
                }
            )
            for child_id in child_ids:
                _set_parent_node_id(nodes, child_id, summary_id)
                edges.append({"source": summary_id, "target": child_id, "edge_type": "structure"})
            next_payloads.append({"id": summary_id, "node_id": summary_id, "content": content, "source_chunk_ids": source_chunk_ids, "source_locates": source_locates, "tokens": count_tokens(content)})
        current_payloads = next_payloads
        level += 1


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


def _chunks_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_id"]].append(chunk)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row.get("start_line", 0), row.get("chunk_id", "")))
    return grouped


def _has_real_hierarchy(root: dict[str, Any]) -> bool:
    children = root.get("children") or []
    return bool(children and not (len(children) == 1 and children[0].get("title", "").endswith(".txt")))


def _chunks_in_range(chunks: list[dict[str, Any]], start_line: int, end_line: int) -> list[dict[str, Any]]:
    return [chunk for chunk in chunks if int(chunk.get("start_line", 0)) >= start_line and int(chunk.get("end_line", 0)) <= end_line]


def _raptor_id(title: str, section_path: list[str], chunks: list[dict[str, Any]]) -> str:
    seed = title + "|" + ">".join(section_path) + "|" + ",".join(chunk.get("chunk_id", "") for chunk in chunks)
    return "raptor:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _locate(chunk: dict[str, Any]) -> str:
    return f"{chunk.get('file_name')}:L{chunk.get('start_line')}-L{chunk.get('end_line')}"


def _node_content_by_id(nodes: list[dict[str, Any]], node_id: str) -> str:
    for node in reversed(nodes):
        if node.get("node_id") == node_id:
            return str(node.get("content", ""))
    return ""


def _set_parent_node_id(nodes: list[dict[str, Any]], node_id: str, parent_id: str) -> None:
    for node in reversed(nodes):
        if node.get("node_id") == node_id:
            node["parent_node_id"] = parent_id
            return


def _source_chunks_by_id(nodes: list[dict[str, Any]], node_id: str) -> set[str]:
    for node in reversed(nodes):
        if node.get("node_id") == node_id:
            return set(node.get("source_chunk_ids") or [])
    return set()


def _source_locates_by_id(nodes: list[dict[str, Any]], node_id: str) -> set[str]:
    for node in reversed(nodes):
        if node.get("node_id") == node_id:
            return set(node.get("source_locates") or [])
    return set()


def _chunk_payload_for_group(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk["chunk_id"],
        "node_id": f"chunk:{chunk['chunk_id']}",
        "content": chunk["content"],
        "source_chunk_ids": [chunk["chunk_id"]],
        "source_locates": [_locate(chunk)],
        "tokens": count_tokens(chunk["content"]),
    }


def _group_by_token_budget(payloads: list[dict[str, Any]], budget: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    total = 0
    for payload in payloads:
        tokens = int(payload.get("tokens", 1))
        if current and total + tokens > budget:
            groups.append(current)
            current = []
            total = 0
        current.append(payload)
        total += tokens
    if current:
        groups.append(current)
    if len(groups) == len(payloads) and len(payloads) > 1:
        return [payloads[index : index + 2] for index in range(0, len(payloads), 2)]
    return groups
