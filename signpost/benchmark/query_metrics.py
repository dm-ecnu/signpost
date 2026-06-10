from __future__ import annotations

"""Query-level quality, cost, and weak evidence metrics for ICDE experiments."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from signpost.benchmark.stats import summarize_values
from signpost.config.context import resolve_project_path
from signpost.evaluation.metrics import evaluate_rows
from signpost.evaluation.schema import normalize_prediction_record
from signpost.parsing.io import read_jsonl


COST_FIELDS = (
    "online_llm_calls",
    "llm_calls",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_seconds",
    "retrieval_latency_seconds",
    "ppr_latency_seconds",
    "read_file_latency_seconds",
    "agent_reasoning_latency_seconds",
    "retrieved_chunks",
    "read_file_calls",
    "graph_ppr_calls",
    "embedding_calls",
    "rerank_calls",
    "max_context_tokens",
)


def summarize_prediction_file(path: Path, *, normalize: bool = False, top_ks: list[int] | None = None) -> dict[str, Any]:
    """Summarize a prediction or query-log JSONL file.

    The function accepts both compact F16 prediction rows and richer query-log
    rows. Missing cost fields are inferred from trace/citation fields when
    possible and otherwise treated as zero.
    """

    rows = [normalize_prediction_record(row) if normalize else row for row in read_jsonl(path)]
    quality = evaluate_rows(rows)
    cost_rows = [extract_query_cost(row) for row in rows]
    retrieval = summarize_retrieval_quality(rows, top_ks=top_ks or [1, 3, 5, 10])
    return {
        "input": str(path),
        "num_queries": len(rows),
        "quality": quality["metrics"],
        "quality_counts": {
            "num_samples": quality["num_samples"],
            "num_scored": quality["num_scored"],
            "num_skipped": quality["num_skipped"],
        },
        "cost": summarize_query_costs(cost_rows),
        "retrieval": retrieval,
        "per_query": [{**{"question_id": row.get("question_id")}, **cost, **_per_query_quality(quality, row.get("question_id"))} for row, cost in zip(rows, cost_rows, strict=True)],
    }


def summarize_query_costs(cost_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {field: summarize_values(row.get(field, 0) for row in cost_rows) for field in COST_FIELDS}
    return {
        "fields": summaries,
        "totals": {field: summaries[field]["sum"] for field in COST_FIELDS},
        "means": {field: summaries[field]["mean"] for field in COST_FIELDS},
        "p95": {field: summaries[field]["p95"] for field in COST_FIELDS},
    }


def extract_query_cost(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    explicit = {field: row.get(field, metadata.get(field, 0)) for field in COST_FIELDS}

    tool_events = [event for event in trace if isinstance(event, dict) and event.get("event_type") == "tool_call"]
    read_events = [event for event in tool_events if str(event.get("tool", "")).lower() == "read_file"]
    search_events = [event for event in tool_events if "search" in str(event.get("tool", "")).lower()]
    llm_events = [event for event in trace if isinstance(event, dict) and str(event.get("event_type", "")).startswith("llm_")]
    citations = row.get("citations") if isinstance(row.get("citations"), list) else []

    inferred = {
        "online_llm_calls": len(llm_events),
        "llm_calls": len(llm_events),
        "tool_calls": len(tool_events),
        "retrieved_chunks": _retrieved_chunks(row, search_events),
        "read_file_calls": len(read_events),
        "graph_ppr_calls": _graph_ppr_calls(row),
        "latency_seconds": _trace_latency(trace),
    }
    if citations and not inferred["read_file_calls"]:
        inferred["read_file_calls"] = len(citations)

    result = {}
    for field in COST_FIELDS:
        value = explicit.get(field)
        if value in (None, "", [], {}):
            value = 0
        if not value and field in inferred:
            value = inferred[field]
        result[field] = _numeric_or_count(value)
    if not result["total_tokens"]:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def summarize_retrieval_quality(rows: list[dict[str, Any]], *, top_ks: list[int]) -> dict[str, Any]:
    """Compute weak evidence recall/MRR when gold evidence fields are present."""

    scored_rows = []
    for row in rows:
        gold = _gold_evidence_ids(row)
        if not gold:
            continue
        retrieved = _retrieved_evidence_ids(row)
        scored_rows.append({"gold": gold, "retrieved": retrieved})
    if not scored_rows:
        return {"num_evidence_scored": 0, "recall_at_k": {}, "mrr": 0.0}

    recall_at_k = {}
    for k in top_ks:
        hits = 0
        for item in scored_rows:
            if item["gold"] & set(item["retrieved"][:k]):
                hits += 1
        recall_at_k[f"recall@{k}"] = hits / len(scored_rows)

    reciprocal_ranks = []
    for item in scored_rows:
        rank = 0
        for idx, evidence_id in enumerate(item["retrieved"], start=1):
            if evidence_id in item["gold"]:
                rank = idx
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return {"num_evidence_scored": len(scored_rows), "recall_at_k": recall_at_k, "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)}


def _per_query_quality(quality: dict[str, Any], question_id: Any) -> dict[str, float]:
    for item in quality.get("per_example", []):
        if item.get("question_id") == question_id:
            return {key: float(item.get(key, 0.0)) for key in ("exact_match", "precision", "recall", "f1")}
    return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def _retrieved_chunks(row: dict[str, Any], search_events: list[dict[str, Any]]) -> int:
    value = row.get("retrieved_chunks")
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int | float):
        return int(value)
    total = 0
    for event in search_events:
        summary = event.get("output_summary") if isinstance(event.get("output_summary"), dict) else {}
        total += int(summary.get("text_items", 0) or 0)
    return total


def _graph_ppr_calls(row: dict[str, Any]) -> int:
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    return sum(1 for event in trace if isinstance(event, dict) and "ppr" in json.dumps(event, ensure_ascii=False).lower())


def _trace_latency(trace: list[Any]) -> float:
    timestamps = [float(event["timestamp"]) for event in trace if isinstance(event, dict) and isinstance(event.get("timestamp"), int | float)]
    if len(timestamps) < 2:
        return 0.0
    return max(timestamps) - min(timestamps)


def _numeric_or_count(value: Any) -> float:
    if isinstance(value, list | tuple | set | dict):
        return float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gold_evidence_ids(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    candidates = []
    for key in ("gold_evidence", "gold_evidence_ids", "gold_chunk_ids", "gold_doc_ids", "supporting_facts"):
        candidates.append(row.get(key))
        candidates.append(metadata.get(key))
    return {_evidence_id(item) for value in candidates for item in _as_list(value) if _evidence_id(item)}


def _retrieved_evidence_ids(row: dict[str, Any]) -> list[str]:
    candidates = []
    for key in ("retrieved_chunks", "retrieved_chunk_ids", "citations", "evidence", "contexts"):
        candidates.extend(_as_list(row.get(key)))
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    for event in trace:
        if not isinstance(event, dict):
            continue
        summary = event.get("output_summary")
        if isinstance(summary, dict):
            candidates.append(summary)
    seen = set()
    ids = []
    for item in candidates:
        evidence_id = _evidence_id(item)
        if evidence_id and evidence_id not in seen:
            ids.append(evidence_id)
            seen.add(evidence_id)
    return ids


def _evidence_id(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("chunk_id", "id", "doc_id", "locate", "file_name"):
        item = value.get(key)
        if item:
            return str(item).strip()
    return ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize F16 predictions/query logs with quality, cost, and evidence metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--top-k", type=int, nargs="*", default=[1, 3, 5, 10])
    args = parser.parse_args()

    result = summarize_prediction_file(resolve_project_path(args.input), normalize=args.normalize, top_ks=args.top_k)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} queries={result['num_queries']} f1={result['quality']['f1']:.4f}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
