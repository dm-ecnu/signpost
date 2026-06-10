from __future__ import annotations

"""Shared helpers for baseline runners.

The baseline harness writes the same prediction JSONL shape as Signpost's F15
batch runner so F16 evaluation and benchmark summaries can be reused unchanged.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from signpost.chunking.tokenizer import count_tokens
from signpost.config.context import resolve_project_path
from signpost.evaluation.schema import build_prediction_text
from signpost.llm.client import OpenAICompatibleClient
from signpost.parsing.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class BaselinePaths:
    dataset: str
    namespace: str
    questions_path: Path
    output_path: Path
    query_log_path: Path | None = None


@dataclass(frozen=True)
class BaselineResult:
    answer: str
    rationale: str = ""
    citations: list[dict[str, Any]] | None = None
    retrieved_chunks: list[dict[str, Any]] | None = None
    evidence_chunks: list[dict[str, Any]] | None = None
    trace: list[dict[str, Any]] | None = None
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    llm_calls: float = 0.0
    tool_calls: float = 0.0
    embedding_calls: float = 0.0
    rerank_calls: float = 0.0
    graph_ppr_calls: float = 0.0
    ppr_latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0


def run_baseline_batch(
    *,
    method: str,
    paths: BaselinePaths,
    answer_fn: Callable[[dict[str, Any]], BaselineResult],
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    workers: int | None = None,
) -> int:
    rows = []
    log_rows = []
    query_log = paths.query_log_path
    if query_log:
        query_log.parent.mkdir(parents=True, exist_ok=True)
        query_log.write_text("", encoding="utf-8")

    question_rows = []
    for index, question_row in enumerate(read_jsonl(paths.questions_path), start=1):
        if limit is not None and index > limit:
            break
        question_rows.append((index, question_row))

    requested_workers = workers if workers is not None else int(os.environ.get("BASELINE_QUERY_WORKERS", "1") or 1)
    requested_workers = max(1, requested_workers)

    def run_one(index: int, question_row: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        question = question_text(question_row)
        question_id = question_id_from_row(question_row, index)
        started = time.time()
        result = answer_fn(question_row)
        finished = time.time()
        cost = baseline_cost(result, started_at=started, finished_at=finished)
        prediction = {
            "question_id": question_id,
            "question": question,
            "answer": question_row.get("answer", ""),
            "rationale": question_row.get("rationale", ""),
            "prediction": build_prediction_text(answer=result.answer, rationale=result.rationale),
            "citations": result.citations or [],
            "trace": result.trace or [],
            "retrieved_chunks": result.retrieved_chunks or [],
            "evidence_chunks": result.evidence_chunks or [],
            **cost,
            "metadata": {
                **question_row.get("metadata", {}),
                **(metadata or {}),
                "method": method,
                "dataset": paths.dataset,
                "namespace": paths.namespace,
            },
        }
        log_row = None
        if query_log:
            log_row = {
                "dataset": paths.dataset,
                "namespace": paths.namespace,
                "method": method,
                "question_id": question_id,
                "question": question,
                "started_at": started,
                "finished_at": finished,
                "citations": prediction["citations"],
                "retrieved_chunks": prediction["retrieved_chunks"],
                "evidence_chunks": prediction["evidence_chunks"],
                **cost,
            }
        return index, prediction, log_row

    completed = []
    total_questions = len(question_rows)

    def record_completed(item: tuple[int, dict[str, Any], dict[str, Any] | None]) -> None:
        index, _prediction, log_row = item
        completed.append(item)
        if query_log and log_row is not None:
            append_jsonl(query_log, log_row)
        question_id = log_row.get("question_id") if log_row else str(index)
        print(
            f"[baseline-batch] method={method} completed={len(completed)}/{total_questions} "
            f"question_id={question_id}",
            flush=True,
        )

    if requested_workers == 1 or total_questions <= 1:
        for index, question_row in question_rows:
            record_completed(run_one(index, question_row))
    else:
        with ThreadPoolExecutor(max_workers=requested_workers) as pool:
            futures = [pool.submit(run_one, index, question_row) for index, question_row in question_rows]
            for future in as_completed(futures):
                record_completed(future.result())

    for _index, prediction, log_row in sorted(completed, key=lambda item: item[0]):
        rows.append(prediction)
        if log_row is not None:
            log_rows.append(log_row)
    return write_jsonl(paths.output_path, rows)


def baseline_cost(result: BaselineResult, *, started_at: float, finished_at: float) -> dict[str, float]:
    input_tokens = float(result.input_tokens or 0.0)
    output_tokens = float(result.output_tokens or count_tokens(result.answer))
    return {
        "latency_seconds": finished_at - started_at,
        "retrieval_latency_seconds": float(result.retrieval_latency_seconds or 0.0),
        "read_file_latency_seconds": 0.0,
        "agent_reasoning_latency_seconds": max(0.0, finished_at - started_at - float(result.retrieval_latency_seconds or 0.0)),
        "llm_calls": float(result.llm_calls or 0.0),
        "online_llm_calls": float(result.llm_calls or 0.0),
        "tool_calls": float(result.tool_calls or 0.0),
        "knowledge_search_calls": float(result.tool_calls or 0.0),
        "embedding_calls": float(result.embedding_calls or 0.0),
        "rerank_calls": float(result.rerank_calls or 0.0),
        "graph_ppr_calls": float(result.graph_ppr_calls or 0.0),
        "ppr_latency_seconds": float(result.ppr_latency_seconds or 0.0),
        "read_file_calls": 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"question row is missing question/query/input: {row}")


def question_id_from_row(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("qid") or row.get("question_id") or index)


def chat_once(llm: OpenAICompatibleClient, messages: list[dict[str, str]], *, input_text: str) -> tuple[str, float, float, float]:
    started = time.time()
    retries = max(1, int(os.environ.get("BASELINE_LLM_RETRIES", os.environ.get("LLM_RETRIES", "3")) or 3))
    retry_sleep = max(0.0, float(os.environ.get("BASELINE_LLM_RETRY_SLEEP", os.environ.get("RETRY_SLEEP", "5")) or 5))
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            answer = llm.chat(messages)
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            print(f"[baseline-chat] retry={attempt}/{retries} error={exc}", flush=True)
            time.sleep(retry_sleep)
    else:
        raise RuntimeError("unreachable chat retry state") from last_exc
    latency = time.time() - started
    return answer, float(count_tokens(input_text)), float(count_tokens(answer)), latency


def build_paths(
    *,
    dataset: str,
    namespace: str | None,
    questions_path: str | None,
    output_path: str | None,
    query_log_path: str | None,
    method: str,
) -> BaselinePaths:
    resolved_namespace = namespace or dataset
    return BaselinePaths(
        dataset=dataset,
        namespace=resolved_namespace,
        questions_path=resolve_project_path(questions_path or f"datasets/processed/{dataset}/questions.jsonl"),
        output_path=resolve_project_path(output_path or f"outputs/{dataset}/predictions/{method}.jsonl"),
        query_log_path=resolve_project_path(query_log_path or f"outputs/{dataset}/logs/{method}.query.jsonl"),
    )


def load_jsonl_list(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def join_context(chunks: Iterable[dict[str, Any]], *, max_context_tokens: int) -> tuple[str, list[dict[str, Any]]]:
    used = []
    parts = []
    total = 0
    for item in chunks:
        content = str(item.get("content") or "")
        token_count = count_tokens(content)
        if parts and total + token_count > max_context_tokens:
            break
        total += token_count
        used.append(item)
        locate = locate_from_chunk(item)
        header = f"[{locate}]" if locate else f"[{item.get('chunk_id', '')}]"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts), used


def locate_from_chunk(item: dict[str, Any]) -> str:
    file_name = item.get("file_name")
    start = item.get("start_line")
    end = item.get("end_line")
    if file_name and start is not None and end is not None:
        return f"{file_name}:L{start}-L{end}"
    return ""
