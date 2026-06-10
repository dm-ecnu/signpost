from __future__ import annotations

"""Flat chunk RAG baseline."""

import argparse
import time
from pathlib import Path
from typing import Any

from signpost.baselines.common import (
    BaselineResult,
    build_paths,
    chat_once,
    join_context,
    load_jsonl_list,
    locate_from_chunk,
    question_text,
    run_baseline_batch,
)
from signpost.config.context import resolve_project_path
from signpost.indexing.embedding import create_embedding_provider
from signpost.llm.client import OpenAICompatibleClient
from signpost.retrieval.chunk_search import search_chunks


METHOD = "vanilla_rag"


class VanillaRAG:
    def __init__(
        self,
        *,
        dataset: str,
        namespace: str,
        chunks_path: Path,
        use_es: bool,
        mode: str,
        top_k: int,
        max_context_tokens: int,
        embedding_provider: str,
        chunk_index_name: str | None = None,
    ):
        self.dataset = dataset
        self.namespace = namespace
        self.chunks_path = chunks_path
        self.use_es = use_es
        self.mode = mode
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        self.embedding_provider = embedding_provider
        self.chunk_index_name = chunk_index_name
        self.llm = OpenAICompatibleClient()
        self.local_chunks = [] if use_es else load_jsonl_list(chunks_path)
        self.local_embedding_provider = None if use_es or mode == "bm25" else create_embedding_provider(embedding_provider)
        self.local_vectors = None
        if self.local_embedding_provider is not None:
            self.local_vectors = self.local_embedding_provider.embed([str(item.get("content") or "") for item in self.local_chunks])

    def answer(self, row: dict[str, Any]) -> BaselineResult:
        question = question_text(row)
        retrieval_started = time.time()
        retrieved = self.retrieve(question)
        retrieval_latency = time.time() - retrieval_started
        context, used_chunks = join_context(retrieved, max_context_tokens=self.max_context_tokens)
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Retrieved context:\n"
            f"{context}\n\n"
            "Answer using only the retrieved context. If the context is insufficient, say so briefly."
        )
        messages = [
            {"role": "system", "content": "You are a retrieval-augmented QA baseline. Ground the answer in the provided chunks."},
            {"role": "user", "content": prompt},
        ]
        answer, input_tokens, output_tokens, llm_latency = chat_once(self.llm, messages, input_text=prompt)
        citations = [
            {"file_name": item.get("file_name"), "start_line": item.get("start_line"), "end_line": item.get("end_line"), "locate": locate_from_chunk(item)}
            for item in used_chunks
            if locate_from_chunk(item)
        ]
        retrieved_chunks = [
            {
                "chunk_id": str(item.get("chunk_id") or ""),
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "score_source": item.get("score_source"),
            }
            for item in retrieved
            if item.get("chunk_id")
        ]
        return BaselineResult(
            answer=answer,
            rationale="Retrieved chunks: " + "; ".join(item["chunk_id"] for item in retrieved_chunks[: self.top_k]),
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=1.0,
            tool_calls=1.0,
            retrieval_latency_seconds=retrieval_latency,
            trace=[
                {
                    "event_type": "tool_call",
                    "tool": "vanilla_rag_retrieve",
                    "latency_seconds": retrieval_latency,
                    "output_summary": {"retrieved_chunks": len(retrieved_chunks), "mode": self.mode, "use_es": self.use_es},
                },
                {
                    "event_type": "llm_call",
                    "stage": "vanilla_rag_answer",
                    "latency_seconds": llm_latency,
                    "input_tokens_estimate": input_tokens,
                    "output_tokens_estimate": output_tokens,
                },
            ],
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.use_es:
            return search_chunks(
                namespace=self.namespace,
                query=question,
                mode=self.mode,
                top_k=self.top_k,
                index_name=self.chunk_index_name,
                embedding_provider_name=self.embedding_provider,
            ).get("items", [])
        return self._local_retrieve(question)

    def _local_retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.mode == "bm25" or self.local_vectors is None:
            return _local_keyword_search(self.local_chunks, question, self.top_k)
        query_vector = self.local_embedding_provider.embed([question])[0]  # type: ignore[union-attr]
        dense_items = _local_dense_search(self.local_chunks, self.local_vectors, query_vector, self.top_k * 2)
        if self.mode == "dense":
            return dense_items[: self.top_k]
        keyword_items = _local_keyword_search(self.local_chunks, question, self.top_k * 2)
        return _rrf_fuse(keyword_items, dense_items, self.top_k)


def run_vanilla_rag(
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
    method: str = METHOD,
    workers: int | None = None,
) -> int:
    paths = build_paths(
        dataset=dataset,
        namespace=namespace,
        questions_path=questions_path,
        output_path=output_path,
        query_log_path=query_log_path,
        method=method,
    )
    runner = VanillaRAG(
        dataset=dataset,
        namespace=paths.namespace,
        chunks_path=resolve_project_path(chunks_path or f"datasets/processed/{dataset}/chunks.jsonl"),
        use_es=use_es,
        mode=mode,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        embedding_provider=embedding_provider,
        chunk_index_name=chunk_index_name,
    )
    return run_baseline_batch(
        method=method,
        paths=paths,
        answer_fn=runner.answer,
        limit=limit,
        workers=workers,
        metadata={
            "retrieval": "flat_chunk_rag",
            "use_es": use_es,
            "mode": mode,
            "top_k": top_k,
            "embedding_provider": embedding_provider,
            "chunk_index_name": chunk_index_name,
            "max_context_tokens": max_context_tokens,
        },
    )


def _local_keyword_search(chunks: list[dict[str, Any]], question: str, top_k: int) -> list[dict[str, Any]]:
    query_terms = _terms(question)
    scored = []
    for item in chunks:
        content = str(item.get("content") or "")
        score = sum(content.lower().count(term) for term in query_terms)
        if score > 0:
            scored.append({**item, "score": float(score), "score_source": "local_keyword"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def _local_dense_search(chunks: list[dict[str, Any]], vectors: list[list[float]], query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    scored = []
    for item, vector in zip(chunks, vectors, strict=True):
        scored.append({**item, "score": _dot(query_vector, vector), "score_source": "local_dense"})
    return sorted(scored, key=lambda item: (-float(item["score"]), str(item.get("chunk_id", ""))))[:top_k]


def _rrf_fuse(left: list[dict[str, Any]], right: list[dict[str, Any]], top_k: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(left, start=1):
        chunk_id = str(item.get("chunk_id"))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "local_hybrid"}
    for rank, item in enumerate(right, start=1):
        chunk_id = str(item.get("chunk_id"))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        docs[chunk_id] = {**item, "score_source": "local_hybrid"}
    return [{**docs[chunk_id], "score": score} for chunk_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]]


def _terms(text: str) -> list[str]:
    return [term for term in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(term) > 1]


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vanilla RAG baseline.")
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

    count = run_vanilla_rag(
        dataset=args.dataset,
        namespace=args.namespace,
        questions_path=args.questions,
        chunks_path=args.chunks,
        output_path=args.output,
        query_log_path=args.query_log,
        limit=args.limit,
        workers=args.workers,
        use_es=args.use_es,
        mode=args.mode,
        top_k=args.top_k,
        max_context_tokens=args.max_context_tokens,
        embedding_provider=args.embedding_provider,
        chunk_index_name=args.chunk_index_name,
    )
    output = resolve_project_path(args.output or f"outputs/{args.dataset}/predictions/{METHOD}.jsonl")
    print(f"output={output} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
