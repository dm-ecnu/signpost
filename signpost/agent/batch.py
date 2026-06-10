from __future__ import annotations

"""Batch runner for F15 predictions JSONL."""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from signpost.agent.run import run_agent
from signpost.chunking.tokenizer import count_tokens
from signpost.config.context import resolve_project_path
from signpost.evaluation.schema import build_prediction_text
from signpost.parsing.io import read_jsonl, write_jsonl
from signpost.retrieval.signpost_variants import FULL, VALID_VARIANTS


def run_batch(
    *,
    namespace: str,
    dataset: str | None = None,
    questions_path: str,
    output_path: str,
    embedding_provider: str = "ecnu",
    use_llm: bool = True,
    use_es: bool = False,
    limit: int | None = None,
    query_log_path: str | None = None,
    signpost_variant: str = FULL,
    read_top_k: int = 3,
    workers: int | None = None,
) -> int:
    rows = []
    log_rows = []
    query_log = resolve_project_path(query_log_path) if query_log_path else None
    if query_log:
        query_log.parent.mkdir(parents=True, exist_ok=True)
        query_log.write_text("", encoding="utf-8")

    question_rows = []
    for index, row in enumerate(read_jsonl(resolve_project_path(questions_path)), start=1):
        if limit is not None and index > limit:
            break
        question_rows.append((index, row))

    requested_workers = workers if workers is not None else int(os.environ.get("SIGNPOST_QUERY_WORKERS", "1") or 1)
    requested_workers = max(1, requested_workers)

    def run_one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
        question = _question_text(row)
        started = time.time()
        result = run_agent(
            namespace=namespace,
            question=question,
            dataset=dataset,
            embedding_provider=embedding_provider,
            use_llm=use_llm,
            use_es=use_es,
            signpost_variant=signpost_variant,
            read_top_k=read_top_k,
        )
        finished = time.time()
        generated_answer, generated_rationale = _split_generated_answer(result["answer"])
        question_id = row.get("id") or row.get("qid") or row.get("question_id") or str(index)
        cost = _query_cost(result, started_at=started, finished_at=finished)
        prediction = {
            "question_id": question_id,
            "question": question,
            "answer": row.get("answer", ""),
            "rationale": row.get("rationale", ""),
            "prediction": build_prediction_text(
                answer=generated_answer,
                rationale=_generated_rationale(result, generated_rationale=generated_rationale),
            ),
            "citations": result["citations"],
            "trace_id": result["trace_id"],
            "trace": result["trace"],
            "retrieved_chunks": _retrieved_chunks(result),
            "evidence_chunks": _evidence_chunks(result),
            "latency_seconds": cost["latency_seconds"],
            "retrieval_latency_seconds": cost["retrieval_latency_seconds"],
            "read_file_latency_seconds": cost["read_file_latency_seconds"],
            "agent_reasoning_latency_seconds": cost["agent_reasoning_latency_seconds"],
            "llm_calls": cost["llm_calls"],
            "tool_calls": cost["tool_calls"],
            "knowledge_search_calls": cost["knowledge_search_calls"],
            "read_file_calls": cost["read_file_calls"],
            "input_tokens": cost["input_tokens"],
            "output_tokens": cost["output_tokens"],
            "total_tokens": cost["total_tokens"],
            "metadata": {
                **row.get("metadata", {}),
                "method": "signpost",
                "signpost_variant": signpost_variant,
                "dataset": dataset or namespace,
                "namespace": namespace,
                "embedding_provider": embedding_provider,
                "read_top_k": read_top_k,
            },
        }
        log_row = None
        if query_log:
            log_row = {
                "dataset": namespace,
                "artifact_dataset": dataset or namespace,
                "method": "signpost",
                "signpost_variant": signpost_variant,
                "embedding_provider": embedding_provider,
                "read_top_k": read_top_k,
                "question_id": question_id,
                "question": question,
                "started_at": started,
                "finished_at": finished,
                **cost,
                "retrieved_chunks": prediction["retrieved_chunks"],
                "evidence_chunks": prediction["evidence_chunks"],
                "citations": result["citations"],
                "trace_id": result["trace_id"],
            }
        return index, prediction, log_row

    completed = []
    total_questions = len(question_rows)

    def record_completed(item: tuple[int, dict[str, Any], dict[str, Any] | None]) -> None:
        index, _prediction, log_row = item
        completed.append(item)
        if query_log and log_row is not None:
            _append_jsonl(query_log, log_row)
        question_id = log_row.get("question_id") if log_row else str(index)
        print(
            f"[signpost-batch] completed={len(completed)}/{total_questions} "
            f"question_id={question_id} variant={signpost_variant}",
            flush=True,
        )

    if requested_workers == 1 or total_questions <= 1:
        for index, row in question_rows:
            record_completed(run_one(index, row))
    else:
        with ThreadPoolExecutor(max_workers=requested_workers) as pool:
            futures = [pool.submit(run_one, index, row) for index, row in question_rows]
            for future in as_completed(futures):
                record_completed(future.result())

    for _index, prediction, log_row in sorted(completed, key=lambda item: item[0]):
        rows.append(prediction)
        if log_row is not None:
            log_rows.append(log_row)
    return write_jsonl(resolve_project_path(output_path), rows)


def _question_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"question row is missing question/query/input: {row}")


def _split_generated_answer(raw_answer: Any) -> tuple[str, str]:
    raw_text = str(raw_answer or "").strip()
    parsed = _parse_json_object(raw_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        answer = parsed["answer"].strip()
        rationale = str(parsed.get("rationale") or "").strip()
        if answer:
            return answer, rationale
    return raw_text, ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0).strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _generated_rationale(result: dict[str, Any], *, generated_rationale: str = "") -> str:
    subquestions = result.get("subquestions") or []
    citations = result.get("citations") or []
    parts = []
    if generated_rationale:
        parts.append(generated_rationale)
    if subquestions:
        parts.append("Subquestions: " + "; ".join(str(item) for item in subquestions))
    if citations:
        parts.append("Evidence: " + "; ".join(str(item.get("locate")) for item in citations if item.get("locate")))
    return "\n".join(parts)


def _query_cost(result: dict[str, Any], *, started_at: float, finished_at: float) -> dict[str, float]:
    trace = result.get("trace") if isinstance(result.get("trace"), list) else []
    tool_events = [event for event in trace if isinstance(event, dict) and event.get("event_type") == "tool_call"]
    llm_events = [event for event in trace if isinstance(event, dict) and str(event.get("event_type", "")).startswith("llm_")]
    retrieval_latency = sum(float(event.get("latency_seconds") or 0) for event in tool_events if event.get("tool") == "knowledge_search")
    read_latency = sum(float(event.get("latency_seconds") or 0) for event in tool_events if event.get("tool") == "read_file")
    llm_latency = sum(float(event.get("latency_seconds") or 0) for event in llm_events)
    input_tokens = sum(float(event.get("input_tokens_estimate") or 0) for event in llm_events)
    output_tokens = sum(float(event.get("output_tokens_estimate") or 0) for event in llm_events)
    if not input_tokens:
        input_tokens = float(count_tokens(str(result.get("question", ""))))
    if not output_tokens:
        output_tokens = float(count_tokens(str(result.get("answer", ""))))
    return {
        "latency_seconds": finished_at - started_at,
        "retrieval_latency_seconds": retrieval_latency,
        "read_file_latency_seconds": read_latency,
        "agent_reasoning_latency_seconds": max(0.0, finished_at - started_at - retrieval_latency - read_latency),
        "tool_calls": float(len(tool_events)),
        "knowledge_search_calls": float(sum(1 for event in tool_events if event.get("tool") == "knowledge_search")),
        "read_file_calls": float(sum(1 for event in tool_events if event.get("tool") == "read_file")),
        "llm_calls": float(len(llm_events)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _retrieved_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for research in result.get("research") or []:
        retrieval = research.get("retrieval") if isinstance(research, dict) else {}
        if not isinstance(retrieval, dict):
            continue
        for group_name in ("text_group", "graph_group"):
            group = retrieval.get(group_name) if isinstance(retrieval.get(group_name), dict) else {}
            for item in group.get("items") or []:
                chunk_id = item.get("chunk_id") or item.get("id") or item.get("node_id")
                if chunk_id:
                    chunks[str(chunk_id)] = {
                        "chunk_id": str(chunk_id),
                        "doc_id": item.get("doc_id"),
                        "score": item.get("score"),
                    }
                for source_chunk_id in item.get("source_chunk_ids") or []:
                    chunks[str(source_chunk_id)] = {"chunk_id": str(source_chunk_id), "doc_id": item.get("doc_id")}
    return list(chunks.values())


def _evidence_chunks(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_index, research in enumerate(result.get("research") or [], start=1):
        if not isinstance(research, dict):
            continue
        subquestion = str(research.get("subquestion") or "")
        for rank, snippet in enumerate(research.get("evidence") or [], start=1):
            if not isinstance(snippet, dict):
                continue
            file_name = str(snippet.get("file_name") or "")
            start_line = None
            end_line = None
            lines = snippet.get("lines") if isinstance(snippet.get("lines"), list) else []
            if lines:
                first = lines[0] if isinstance(lines[0], dict) else {}
                last = lines[-1] if isinstance(lines[-1], dict) else {}
                start_line = first.get("line_no")
                end_line = last.get("line_no")
            resolved = snippet.get("resolved") if isinstance(snippet.get("resolved"), dict) else {}
            start_line = resolved.get("start_line", start_line)
            end_line = resolved.get("end_line", end_line)
            rows.append(
                {
                    "rank": rank,
                    "round": round_index,
                    "source": "read_file",
                    "query": subquestion,
                    "chunk_id": str(snippet.get("chunk_id") or ""),
                    "doc_id": snippet.get("doc_id"),
                    "file_name": file_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "score": snippet.get("rerank_score"),
                    "score_source": "read_file_rerank" if snippet.get("rerank_score") is not None else "read_file",
                    "locate": snippet.get("locate") or _locate(file_name, start_line, end_line),
                }
            )
    return rows


def _locate(file_name: str, start: Any, end: Any) -> str:
    if file_name and start is not None and end is not None:
        return f"{file_name}:L{start}-L{end}"
    return ""


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="F15 batch Supervisor-Researcher predictions")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dataset", help="Processed dataset id for graph/chunks/documents. Defaults to --namespace.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "ecnu"],
        default="ecnu",
        help="Query embedding provider. Must match the ES index embedding provider when --use-es is set.",
    )
    parser.add_argument("--use-llm", dest="use_llm", action="store_true")
    parser.add_argument("--no-use-llm", dest="use_llm", action="store_false", help="Use deterministic decomposition and template synthesis for debugging.")
    parser.set_defaults(use_llm=True)
    parser.add_argument("--use-es", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-log", help="Optional per-query JSONL cost log written during the batch run.")
    parser.add_argument("--signpost-variant", choices=sorted(VALID_VARIANTS), default=FULL)
    parser.add_argument("--read-top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    count = run_batch(
        namespace=args.namespace,
        dataset=args.dataset,
        questions_path=args.questions,
        output_path=args.output,
        embedding_provider=args.embedding_provider,
        use_llm=args.use_llm,
        use_es=args.use_es,
        limit=args.limit,
        query_log_path=args.query_log,
        signpost_variant=args.signpost_variant,
        read_top_k=args.read_top_k,
        workers=args.workers,
    )
    print(f"output={resolve_project_path(args.output)} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
