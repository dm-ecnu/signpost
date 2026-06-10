from __future__ import annotations

"""F5 CLI and library for indexing chunks into Elasticsearch."""

import argparse
import sys
import time
from pathlib import Path

from signpost.config.context import resolve_project_path
from signpost.indexing.chunk_schema import chunk_index_mapping, chunk_index_name, chunk_to_es_doc
from signpost.indexing.embedding import EmbeddingProvider, create_embedding_provider
from signpost.parsing.io import read_jsonl
from signpost.storage.elasticsearch import ElasticsearchClient


def index_chunks(
    chunks_path: Path,
    *,
    namespace: str,
    dataset_id: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    index_name: str | None = None,
    batch_size: int = 32,
    recreate: bool = False,
    progress_every: int = 0,
    embedding_retries: int = 3,
    retry_sleep: float = 2.0,
    es: ElasticsearchClient | None = None,
) -> dict[str, int | str]:
    provider = embedding_provider or create_embedding_provider("ecnu")
    client = es or ElasticsearchClient()
    target_index = index_name or chunk_index_name(namespace)
    dataset = dataset_id or namespace

    rows = list(read_jsonl(chunks_path))
    if not rows:
        raise ValueError(f"No chunks found: {chunks_path}")
    total = len(rows)
    if progress_every:
        print(f"index={target_index} chunks={total} batch_size={batch_size} provider={provider.__class__.__name__}", file=sys.stderr, flush=True)

    first_vector = _embed_batch(provider, [rows[0]], start_index=0, retries=embedding_retries, retry_sleep=retry_sleep, progress=bool(progress_every))[0]
    client.create_index(target_index, chunk_index_mapping(len(first_vector)), recreate=recreate)

    indexed = 0
    pending_rows = rows[1:]
    _bulk_index(client, target_index, [rows[0]], [first_vector], namespace=namespace, dataset_id=dataset)
    indexed += 1
    for start in range(0, len(pending_rows), batch_size):
        batch = pending_rows[start : start + batch_size]
        vectors = _embed_batch(provider, batch, start_index=start + 1, retries=embedding_retries, retry_sleep=retry_sleep, progress=bool(progress_every))
        _bulk_index(client, target_index, batch, vectors, namespace=namespace, dataset_id=dataset)
        indexed += len(batch)
        batch_no = start // batch_size + 1
        if progress_every and batch_no % progress_every == 0:
            print(f"indexed={indexed}/{total} batches={batch_no}", file=sys.stderr, flush=True)
    client.refresh(target_index)
    return {"index": target_index, "indexed": indexed, "dimensions": len(first_vector)}


def _embed_batch(
    provider: EmbeddingProvider,
    rows: list[dict],
    *,
    start_index: int,
    retries: int,
    retry_sleep: float,
    progress: bool = False,
) -> list[list[float]]:
    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return provider.embed([row["content"] for row in rows])
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                if progress:
                    print(
                        f"embedding retry row={start_index} size={len(rows)} attempt={attempt}/{attempts - 1} error={type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(retry_sleep)

    if len(rows) > 1:
        midpoint = len(rows) // 2
        if progress:
            print(f"embedding split row={start_index} size={len(rows)} -> {midpoint}+{len(rows) - midpoint}", file=sys.stderr, flush=True)
        left = _embed_batch(provider, rows[:midpoint], start_index=start_index, retries=retries, retry_sleep=retry_sleep, progress=progress)
        right = _embed_batch(provider, rows[midpoint:], start_index=start_index + midpoint, retries=retries, retry_sleep=retry_sleep, progress=progress)
        return left + right

    row = rows[0]
    details = f"row={start_index} chunk_id={row.get('chunk_id')} chars={len(str(row.get('content', '')))} tokens={row.get('metadata', {}).get('token_count')}"
    raise RuntimeError(f"Embedding failed after {attempts} attempts. {details}") from last_exc


def _bulk_index(client: ElasticsearchClient, index_name: str, chunks: list[dict], vectors: list[list[float]], *, namespace: str, dataset_id: str) -> None:
    operations = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        operations.append({"index": {"_index": index_name, "_id": chunk["chunk_id"]}})
        operations.append(chunk_to_es_doc(chunk, namespace=namespace, dataset_id=dataset_id, vector=vector))
    client.bulk(operations)


def main() -> int:
    parser = argparse.ArgumentParser(description="F5 index chunks into Elasticsearch")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index-name")
    parser.add_argument("--embedding-provider", choices=["ecnu", "hash"], default="ecnu")
    parser.add_argument("--hash-dimensions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N batches; set 0 to disable.")
    parser.add_argument("--embedding-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    provider = create_embedding_provider(args.embedding_provider, dimensions=args.hash_dimensions)
    result = index_chunks(
        resolve_project_path(args.chunks),
        namespace=args.namespace,
        dataset_id=args.dataset_id,
        embedding_provider=provider,
        index_name=args.index_name,
        batch_size=args.batch_size,
        recreate=args.recreate,
        progress_every=args.progress_every,
        embedding_retries=args.embedding_retries,
        retry_sleep=args.retry_sleep,
    )
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
