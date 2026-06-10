from __future__ import annotations

"""F10 synchronize unified graph objects into Elasticsearch."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.indexing.chunk_schema import chunk_index_name
from signpost.indexing.embedding import EmbeddingProvider, create_embedding_provider
from signpost.indexing.graph_schema import attach_vectors, chunk_parent_updates, graph_index_mapping, graph_index_name, graph_to_index_documents
from signpost.storage.elasticsearch import ElasticsearchClient


def sync_graph_to_es(
    graph_path: Path,
    *,
    namespace: str,
    embedding_provider: EmbeddingProvider | None = None,
    index_name: str | None = None,
    batch_size: int = 32,
    recreate: bool = False,
    update_chunk_parents: bool = False,
    chunk_index: str | None = None,
    resume: bool = False,
    progress_log: Path | None = None,
    state_file: Path | None = None,
    multi_vector_parts_file: Path | None = None,
    es: ElasticsearchClient | None = None,
) -> dict[str, int | str]:
    provider = embedding_provider or create_embedding_provider("ecnu")
    client = es or ElasticsearchClient()
    target_index = index_name or graph_index_name(namespace)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    docs = graph_to_index_documents(graph, namespace=namespace)
    if not docs:
        raise ValueError(f"No graph index documents produced from {graph_path}")
    parts_by_parent = _read_parts_file(multi_vector_parts_file)
    vector_docs = _expand_vector_docs(docs, parts_by_parent)
    parent_docs = _canonical_parent_docs(docs, parts_by_parent)
    reset_parent_ids = set(parts_by_parent)

    if recreate and not resume:
        _reset_recovery_files(progress_log, state_file)
    completed_ids = _read_completed_vector_ids(progress_log, reset_parent_ids=reset_parent_ids) if resume else set()
    pending_docs = [doc for doc in vector_docs if str(doc["id"]) not in completed_ids]
    if not pending_docs:
        if parent_docs and client.exists_index(target_index):
            _ensure_vector_recovery_mapping(client, target_index)
            _bulk_index(client, target_index, parent_docs)
        parent_updates = update_chunk_parent_fields(client, graph, chunk_index or chunk_index_name(namespace)) if update_chunk_parents else 0
        client.refresh(target_index)
        return {
            "index": target_index,
            "indexed": len(completed_ids),
            "skipped_completed": len(completed_ids),
            "graph_objects": len(docs),
            "vector_documents": len(vector_docs),
            "multi_vector_objects": len(parts_by_parent),
            "dimensions": 0,
            "chunk_parent_updates": parent_updates,
        }

    for parent_id in sorted(reset_parent_ids):
        _delete_graph_parent(client, target_index, parent_id)

    first_doc = pending_docs[0]
    _append_progress(progress_log, _progress_row("started", first_doc, target_index=target_index, doc_index=0))
    try:
        first_vector = provider.embed([first_doc["content"]])[0]
    except Exception as exc:
        _append_progress(progress_log, _progress_row("failed", first_doc, target_index=target_index, doc_index=0, error=str(exc)))
        _write_state(state_file, target_index=target_index, failed_doc=first_doc, error=str(exc), completed_ids=completed_ids)
        raise
    client.create_index(target_index, graph_index_mapping(len(first_vector)), recreate=recreate and not resume)
    _ensure_vector_recovery_mapping(client, target_index)
    indexed = 0
    _bulk_index(client, target_index, attach_vectors([first_doc], [first_vector]))
    if parent_docs:
        _bulk_index(client, target_index, parent_docs)
    completed_ids.add(str(first_doc["id"]))
    _append_progress(progress_log, _progress_row("ok", first_doc, target_index=target_index, doc_index=0))
    indexed += 1
    pending = pending_docs[1:]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        for offset, doc in enumerate(batch):
            _append_progress(progress_log, _progress_row("started", doc, target_index=target_index, doc_index=start + offset + 1))
        try:
            vectors = provider.embed([doc["content"] for doc in batch])
        except Exception as exc:
            for offset, doc in enumerate(batch):
                _append_progress(progress_log, _progress_row("failed", doc, target_index=target_index, doc_index=start + offset + 1, error=str(exc)))
            failed_doc = batch[0] if len(batch) == 1 else batch[-1]
            _write_state(state_file, target_index=target_index, failed_doc=failed_doc, error=str(exc), completed_ids=completed_ids)
            raise
        _bulk_index(client, target_index, attach_vectors(batch, vectors))
        for offset, doc in enumerate(batch):
            completed_ids.add(str(doc["id"]))
            _append_progress(progress_log, _progress_row("ok", doc, target_index=target_index, doc_index=start + offset + 1))
        indexed += len(batch)
        _write_state(state_file, target_index=target_index, failed_doc=None, error="", completed_ids=completed_ids)
    client.refresh(target_index)

    parent_updates = 0
    if update_chunk_parents:
        parent_updates = update_chunk_parent_fields(client, graph, chunk_index or chunk_index_name(namespace))
    return {
        "index": target_index,
        "indexed": indexed,
        "skipped_completed": len(completed_ids) - indexed,
        "graph_objects": len(docs),
        "vector_documents": len(vector_docs),
        "multi_vector_objects": len(parts_by_parent),
        "dimensions": len(first_vector),
        "chunk_parent_updates": parent_updates,
    }


def _read_parts_file(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"multi-vector parts file must be a JSON object: {path}")
    return {str(key): max(1, int(value)) for key, value in data.items()}


def _expand_vector_docs(docs: list[dict[str, Any]], parts_by_parent: dict[str, int]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for doc in docs:
        parent_id = str(doc["id"])
        parts = parts_by_parent.get(parent_id, 1)
        if parts <= 1:
            expanded.append(
                {
                    **doc,
                    "graph_parent_id": parent_id,
                    "vector_part": 0,
                    "vector_part_count": 1,
                    "is_vector_part": False,
                    "is_vector_parent": False,
                    "vector_searchable": True,
                    "text_searchable": True,
                }
            )
            continue
        content = str(doc.get("content", ""))
        for part, window in enumerate(_split_evenly(content, parts)):
            expanded.append(
                {
                    **doc,
                    "id": f"{parent_id}::mv{parts}p{part:04d}",
                    "graph_parent_id": parent_id,
                    "vector_part": part,
                    "vector_part_count": parts,
                    "is_vector_part": True,
                    "is_vector_parent": False,
                    "vector_searchable": True,
                    "text_searchable": False,
                    "content": window,
                    "metadata": {
                        **(doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}),
                        "graph_parent_id": parent_id,
                        "vector_part": part,
                        "vector_part_count": parts,
                    },
                }
            )
    return expanded


def _canonical_parent_docs(docs: list[dict[str, Any]], parts_by_parent: dict[str, int]) -> list[dict[str, Any]]:
    parents: list[dict[str, Any]] = []
    for doc in docs:
        parent_id = str(doc.get("id") or "")
        parts = parts_by_parent.get(parent_id, 1)
        if parts <= 1:
            continue
        parents.append(
            {
                **doc,
                "id": parent_id,
                "graph_parent_id": parent_id,
                "vector_part": 0,
                "vector_part_count": parts,
                "is_vector_part": False,
                "is_vector_parent": True,
                "vector_searchable": False,
                "text_searchable": True,
                "metadata": {
                    **(doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}),
                    "graph_parent_id": parent_id,
                    "vector_part_count": parts,
                    "canonical_parent": True,
                },
            }
        )
    return parents


def _split_evenly(content: str, parts: int) -> list[str]:
    if parts <= 1 or not content:
        return [content]
    window_size = max(1, math.ceil(len(content) / parts))
    return [content[start : start + window_size] for start in range(0, len(content), window_size)] or [content]


def _read_completed_vector_ids(path: Path | None, *, reset_parent_ids: set[str]) -> set[str]:
    if path is None or not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        parent_id = str(row.get("graph_parent_id") or row.get("id") or "")
        if parent_id in reset_parent_ids:
            continue
        doc_id = str(row.get("id") or "")
        if doc_id:
            completed.add(doc_id)
    return completed


def _progress_row(status: str, doc: dict[str, Any], *, target_index: str, doc_index: int, error: str = "") -> dict[str, Any]:
    row = {
        "time": time.time(),
        "status": status,
        "index": target_index,
        "doc_index": doc_index,
        "id": doc.get("id"),
        "graph_parent_id": doc.get("graph_parent_id") or doc.get("id"),
        "object_type": doc.get("object_type"),
        "vector_part": doc.get("vector_part", 0),
        "vector_part_count": doc.get("vector_part_count", 1),
        "content_chars": len(str(doc.get("content", ""))),
    }
    if error:
        row["error"] = error
    return row


def _append_progress(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_state(path: Path | None, *, target_index: str, failed_doc: dict[str, Any] | None, error: str, completed_ids: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.time(),
        "index": target_index,
        "completed_vector_documents": len(completed_ids),
        "failed_doc": _progress_row("failed", failed_doc, target_index=target_index, doc_index=-1, error=error) if failed_doc else None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_recovery_files(progress_log: Path | None, state_file: Path | None) -> None:
    for path in (progress_log, state_file):
        if path and path.exists():
            path.unlink()


def _delete_graph_parent(client: ElasticsearchClient, index_name: str, parent_id: str) -> None:
    if not client.exists_index(index_name):
        return
    body = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"id": parent_id}},
                    {"term": {"graph_parent_id": parent_id}},
                    {"prefix": {"id": f"{parent_id}::mv"}},
                ],
                "minimum_should_match": 1,
            }
        }
    }
    client.request("POST", f"{index_name}/_delete_by_query", body)


def update_chunk_parent_fields(client: ElasticsearchClient, graph: dict[str, Any], chunk_index: str) -> int:
    updated = 0
    for chunk_id, parent_ids in chunk_parent_updates(graph).items():
        if not parent_ids:
            continue
        try:
            client.update_doc(chunk_index, chunk_id, {"parent_summary_ids": parent_ids, "parent_summary_id": parent_ids[0]})
            updated += 1
        except Exception:
            continue
    if updated:
        client.refresh(chunk_index)
    return updated


def _bulk_index(client: ElasticsearchClient, index_name: str, docs: list[dict[str, Any]]) -> None:
    operations = []
    for doc in docs:
        operations.append({"index": {"_index": index_name, "_id": doc["id"]}})
        operations.append(doc)
    client.bulk(operations)


def _ensure_vector_recovery_mapping(client: ElasticsearchClient, index_name: str) -> None:
    client.request(
        "PUT",
        f"{index_name}/_mapping",
        {
            "properties": {
                "is_vector_parent": {"type": "boolean"},
                "vector_searchable": {"type": "boolean"},
                "text_searchable": {"type": "boolean"},
            }
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="F10 sync unified graph objects into Elasticsearch")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--index-name")
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--update-chunk-parents", action="store_true")
    parser.add_argument("--chunk-index")
    parser.add_argument("--resume", action="store_true", help="Skip vector documents already marked ok in --progress-log.")
    parser.add_argument("--progress-log", help="JSONL progress log for F10 resume and recovery.")
    parser.add_argument("--state-file", help="JSON state file with the last failed vector document.")
    parser.add_argument("--multi-vector-parts-file", help="JSON object mapping original graph object id to split count.")
    args = parser.parse_args()

    provider = create_embedding_provider(args.embedding_provider, dimensions=args.hash_dimensions)
    result = sync_graph_to_es(
        resolve_project_path(args.graph),
        namespace=args.namespace,
        embedding_provider=provider,
        index_name=args.index_name,
        batch_size=args.batch_size,
        recreate=args.recreate,
        update_chunk_parents=args.update_chunk_parents,
        chunk_index=args.chunk_index,
        resume=args.resume,
        progress_log=resolve_project_path(args.progress_log) if args.progress_log else None,
        state_file=resolve_project_path(args.state_file) if args.state_file else None,
        multi_vector_parts_file=resolve_project_path(args.multi_vector_parts_file) if args.multi_vector_parts_file else None,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
