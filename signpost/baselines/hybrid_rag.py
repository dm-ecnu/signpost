from __future__ import annotations

"""Hybrid BM25+dense chunk RAG control baseline.

This module is an experiment-facing alias of ``vanilla_rag``.  The old
``vanilla_rag`` entry point is kept for backward compatibility; formal ICDE
experiments should use ``hybrid_rag`` with ES enabled and ``mode=hybrid``.
"""

import argparse

from signpost.baselines.vanilla_rag import run_vanilla_rag
from signpost.config.context import resolve_project_path


METHOD = "hybrid_rag"


def run_hybrid_rag(
    *,
    dataset: str,
    namespace: str | None = None,
    questions_path: str | None = None,
    chunks_path: str | None = None,
    output_path: str | None = None,
    query_log_path: str | None = None,
    limit: int | None = None,
    use_es: bool = False,
    mode: str = "hybrid",
    top_k: int = 5,
    max_context_tokens: int = 3500,
    embedding_provider: str = "ecnu",
    chunk_index_name: str | None = None,
    workers: int | None = None,
) -> int:
    return run_vanilla_rag(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        chunks_path=chunks_path,
        output_path=output_path,
        query_log_path=query_log_path,
        limit=limit,
        use_es=use_es,
        mode=mode,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        embedding_provider=embedding_provider,
        chunk_index_name=chunk_index_name,
        method=METHOD,
        workers=workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hybrid RAG baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--questions")
    parser.add_argument("--chunks")
    parser.add_argument("--output")
    parser.add_argument("--query-log")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=3500)
    parser.add_argument("--embedding-provider", choices=["hash", "ecnu"], default="ecnu")
    parser.add_argument("--chunk-index-name")
    args = parser.parse_args()

    count = run_hybrid_rag(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        output_path=args.output,
        query_log_path=args.query_log,
        limit=args.limit,
        use_es=args.use_es,
        mode=args.mode,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        embedding_provider=args.embedding_provider,
        chunk_index_name=args.chunk_index_name,
        workers=args.workers,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
