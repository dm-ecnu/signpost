from __future__ import annotations

"""Final ICDE-style result aggregation for Signpost experiments.

This module is intentionally additive: the legacy EM/Precision/F1 evaluator
remains in ``signpost.evaluation.metrics`` and is reused only for traceability.
The paper-facing metrics here follow the revised v10 metric protocol:
Answer Recall, Contain Accuracy, optional LLM Judge Accuracy, evidence
navigation, trace/process cost, online interaction cost, and index efficiency.
"""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from signpost.benchmark.stats import summarize_values
from signpost.evaluation.metrics import evaluate_rows, extract_answer_from_prediction, normalize_answer


METHOD_ORDER = [
    "vanilla_llm",
    "hybrid_rag",
    "cluerag",
    "cluerag_prompt_normalized",
    "signpost.full",
    "signpost.no_offline",
    "signpost.no_online",
    "signpost.no_semantic_cues",
    "signpost.no_provenance_cues",
    "signpost.no_vertical_cues",
    "signpost.no_horizontal_cues",
]

ABLATION_LABELS = {
    "signpost.full": ("full", "none"),
    "signpost.no_offline": ("no_offline", "object sketches"),
    "signpost.no_online": ("no_online", "scene recommendation"),
    "signpost.no_semantic_cues": ("no_semantic", "semantic cues"),
    "signpost.no_provenance_cues": ("no_provenance", "provenance cues"),
    "signpost.no_vertical_cues": ("no_vertical", "structural cues"),
    "signpost.no_horizontal_cues": ("no_horizontal", "sequential cues"),
}


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    chunk_id: str = ""
    file_name: str = ""
    start_line: int | None = None
    end_line: int | None = None
    step: int | None = None
    timestamp: float | None = None

    def key(self) -> str:
        if self.kind == "chunk" and self.chunk_id:
            return f"chunk:{self.chunk_id}"
        return f"span:{self.file_name}:{self.start_line}:{self.end_line}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def extract_final_answer(row: dict[str, Any]) -> str:
    raw = str(row.get("prediction", ""))
    answer = extract_answer_from_prediction(raw)
    parsed = _extract_json_answer(answer)
    return parsed if parsed is not None else answer


def answer_quality(row: dict[str, Any]) -> dict[str, Any]:
    predicted = extract_final_answer(row)
    gold_answers = _gold_answers(row.get("answer", ""))
    recalls = [_answer_recall(gold, predicted) for gold in gold_answers]
    contains = [_contain_accuracy(gold, predicted) for gold in gold_answers]
    return {
        "question_id": row.get("question_id"),
        "answer_recall": max(recalls) if recalls else 0.0,
        "contain_accuracy": max(contains) if contains else 0.0,
        "predicted_answer": predicted,
    }


def _answer_recall(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    if not gold_tokens:
        return 1.0 if not predicted_tokens else 0.0
    if not predicted_tokens:
        return 0.0
    overlap = Counter(gold_tokens) & Counter(predicted_tokens)
    return sum(overlap.values()) / len(gold_tokens)


def _contain_accuracy(gold: str, predicted: str) -> float:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(predicted)
    return 1.0 if gold_norm and gold_norm in pred_norm else 0.0


def _gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _extract_json_answer(text: str) -> str | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    for payload in (candidate, _first_json_object(candidate)):
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    return None


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def load_targets(targets_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "silver": {str(row.get("question_id")): row for row in read_jsonl(targets_dir / "silver_evidence_chunks.jsonl")},
        "entities": {str(row.get("question_id")): row for row in read_jsonl(targets_dir / "target_entities.jsonl")},
        "units": {str(row.get("question_id")): row for row in read_jsonl(targets_dir / "target_units.jsonl")},
        "claims": {str(row.get("question_id")): row for row in read_jsonl(targets_dir / "claim_units.jsonl")},
    }


def load_chunk_index(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"by_id": {}, "by_file": {}}
    by_id: dict[str, dict[str, Any]] = {}
    by_file: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        chunk_id = str(row.get("chunk_id", ""))
        if not chunk_id:
            continue
        item = {
            "chunk_id": chunk_id,
            "file_name": str(row.get("file_name", "")),
            "start_line": _to_int(row.get("start_line")),
            "end_line": _to_int(row.get("end_line")),
            "content_norm": normalize_answer(str(row.get("content", ""))),
        }
        by_id[chunk_id] = item
        by_file.setdefault(item["file_name"], []).append(item)
    for items in by_file.values():
        items.sort(key=lambda item: (item.get("start_line") or 0, item.get("end_line") or 0))
    return {"by_id": by_id, "by_file": by_file}


def evidence_sequence(row: dict[str, Any]) -> list[EvidenceItem]:
    reads = read_file_sequence(row)
    if reads:
        return reads
    citations = citation_sequence(row)
    if citations:
        return citations
    return retrieved_chunk_sequence(row)


def retrieved_chunk_sequence(row: dict[str, Any]) -> list[EvidenceItem]:
    result = []
    for item in _as_list(row.get("retrieved_chunks")):
        if isinstance(item, dict):
            chunk_id = str(item.get("chunk_id", "")).strip()
        else:
            chunk_id = str(item).strip()
        if chunk_id:
            result.append(EvidenceItem(kind="chunk", chunk_id=chunk_id))
    return result


def read_file_sequence(row: dict[str, Any]) -> list[EvidenceItem]:
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    result: list[EvidenceItem] = []
    step = 0
    for event in trace:
        if not isinstance(event, dict) or event.get("event_type") != "tool_call":
            continue
        tool = str(event.get("tool", "")).lower()
        if "search" in tool or tool == "read_file":
            step += 1
        if tool != "read_file":
            continue
        file_name, start_line, end_line = _span_from_tool_event(event)
        if not file_name:
            continue
        result.append(
            EvidenceItem(
                kind="span",
                file_name=file_name,
                start_line=start_line,
                end_line=end_line,
                step=step,
                timestamp=_to_float_or_none(event.get("timestamp")),
            )
        )
    return result


def citation_sequence(row: dict[str, Any]) -> list[EvidenceItem]:
    result = []
    for citation in _as_list(row.get("citations")):
        if not isinstance(citation, dict):
            continue
        file_name = str(citation.get("file_name", "")).strip()
        if not file_name:
            continue
        result.append(
            EvidenceItem(
                kind="span",
                file_name=file_name,
                start_line=_to_int(citation.get("start_line")),
                end_line=_to_int(citation.get("end_line")),
            )
        )
    return result


def trace_process_metrics(row: dict[str, Any], silver_row: dict[str, Any] | None) -> dict[str, Any]:
    reads = read_file_sequence(row)
    if not silver_row or not silver_row.get("silver_chunks"):
        return {
            "read_silver_hit": None,
            "first_hit_step": None,
            "tokens_to_first_hit": None,
            "duplicate_read_ratio": duplicate_read_ratio(reads),
            "miss": None,
        }
    first_hit = None
    for item in reads:
        if silver_hits_for_item(item, silver_row):
            first_hit = item
            break
    tokens_to_hit = None
    if first_hit is not None:
        tokens_to_hit = tokens_before(row, first_hit.timestamp)
    return {
        "read_silver_hit": 1.0 if first_hit is not None else 0.0,
        "first_hit_step": float(first_hit.step) if first_hit and first_hit.step is not None else None,
        "tokens_to_first_hit": tokens_to_hit if tokens_to_hit is not None else (float(row.get("total_tokens", 0) or 0) if first_hit is None else None),
        "duplicate_read_ratio": duplicate_read_ratio(reads),
        "miss": 0.0 if first_hit is not None else 1.0,
    }


def duplicate_read_ratio(reads: list[EvidenceItem]) -> float:
    if not reads:
        return 0.0
    unique = {item.key() for item in reads}
    return 1.0 - (len(unique) / len(reads))


def tokens_before(row: dict[str, Any], timestamp: float | None) -> float | None:
    if timestamp is None:
        return None
    total = 0.0
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    for event in trace:
        if not isinstance(event, dict) or event.get("event_type") != "llm_call":
            continue
        event_ts = _to_float_or_none(event.get("timestamp"))
        if event_ts is not None and event_ts <= timestamp:
            total += float(event.get("input_tokens_estimate", 0) or 0)
            total += float(event.get("output_tokens_estimate", 0) or 0)
    return total


def evidence_navigation_metrics(
    row: dict[str, Any],
    targets: dict[str, dict[str, dict[str, Any]]],
    chunk_index: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    qid = str(row.get("question_id"))
    silver_row = targets["silver"].get(qid)
    entity_row = targets["entities"].get(qid)
    unit_row = targets["units"].get(qid)
    claim_row = targets["claims"].get(qid)
    sequence = evidence_sequence(row)
    top_k = sequence[:k]
    silver_chunks = _silver_chunks(silver_row)
    silver_hit_ids = silver_hits_for_items(top_k, silver_row) if silver_row else set()
    all_hit_ranks = silver_hit_ranks(sequence, silver_row) if silver_row else []

    silver_count = len(silver_chunks)
    silver_hit = 1.0 if silver_count and silver_hit_ids else (None if not silver_count else 0.0)
    silver_recall = (len(silver_hit_ids) / silver_count) if silver_count else None
    mrr = (1.0 / all_hit_ranks[0]) if all_hit_ranks else (None if not silver_count else 0.0)

    return {
        "silver_scored": 1 if silver_count else 0,
        "silver_hit_at_5": silver_hit,
        "silver_recall_at_5": silver_recall,
        "evidence_mrr": mrr,
        "target_unit_scored": 1 if _target_units(unit_row) else 0,
        "target_unit_recall_at_5": target_unit_recall(top_k, unit_row, chunk_index),
        "target_entity_scored": 1 if _target_entities(entity_row) else 0,
        "target_entity_recall_at_5": target_entity_recall(top_k, entity_row, chunk_index),
        "claim_scored": 1 if _claims(claim_row) else 0,
        "claim_coverage_at_5": claim_coverage(silver_hit_ids, silver_row, claim_row),
    }


def silver_hit_ranks(sequence: list[EvidenceItem], silver_row: dict[str, Any] | None) -> list[int]:
    if not silver_row:
        return []
    ranks = []
    for idx, item in enumerate(sequence, start=1):
        if silver_hits_for_item(item, silver_row):
            ranks.append(idx)
    return ranks


def silver_hits_for_items(sequence: list[EvidenceItem], silver_row: dict[str, Any] | None) -> set[str]:
    hits: set[str] = set()
    if not silver_row:
        return hits
    for item in sequence:
        hits.update(silver_hits_for_item(item, silver_row))
    return hits


def silver_hits_for_item(item: EvidenceItem, silver_row: dict[str, Any] | None) -> set[str]:
    hits: set[str] = set()
    if not silver_row:
        return hits
    for silver in _silver_chunks(silver_row):
        chunk_id = str(silver.get("chunk_id", ""))
        if item.kind == "chunk" and chunk_id and item.chunk_id == chunk_id:
            hits.add(chunk_id)
            continue
        if item.kind == "span" and _same_file_overlap(item, silver):
            hits.add(chunk_id)
    return hits


def target_entity_recall(sequence: list[EvidenceItem], entity_row: dict[str, Any] | None, chunk_index: dict[str, Any]) -> float | None:
    entities = _target_entities(entity_row)
    if not entities:
        return None
    reached = chunk_ids_for_sequence(sequence, chunk_index)
    if not reached:
        return 0.0
    covered = 0
    for entity in entities:
        matched = {str(item) for item in _as_list(entity.get("matched_chunk_ids"))}
        if matched & reached:
            covered += 1
    return covered / len(entities)


def target_unit_recall(sequence: list[EvidenceItem], unit_row: dict[str, Any] | None, chunk_index: dict[str, Any]) -> float | None:
    units = _target_units(unit_row)
    if not units:
        return None
    texts = object_texts_for_sequence(sequence, chunk_index)
    if not texts:
        return 0.0
    covered = 0
    for unit in units:
        unit_norm = str(unit.get("normalized") or normalize_answer(str(unit.get("text", "")))).strip()
        if not unit_norm:
            continue
        if any(_unit_in_text(unit_norm, text) for text in texts):
            covered += 1
    return covered / len(units)


def claim_coverage(silver_hit_ids: set[str], silver_row: dict[str, Any] | None, claim_row: dict[str, Any] | None) -> float | None:
    claims = _claims(claim_row)
    if not claims:
        return None
    matched_units: set[str] = set()
    for silver in _silver_chunks(silver_row):
        if str(silver.get("chunk_id", "")) not in silver_hit_ids:
            continue
        for key in ("matched_units", "matched_numbers"):
            for unit in _as_list(silver.get(key)):
                norm = normalize_answer(str(unit))
                if norm:
                    matched_units.add(norm)
    if not matched_units:
        return 0.0
    covered = 0
    for claim in claims:
        claim_units = {normalize_answer(str(unit)) for unit in _as_list(claim.get("target_units")) if str(unit).strip()}
        if claim_units & matched_units:
            covered += 1
            continue
        if any(_unit_in_text(unit, text) for unit in claim_units for text in matched_units):
            covered += 1
    return covered / len(claims)


def object_texts_for_sequence(sequence: list[EvidenceItem], chunk_index: dict[str, Any]) -> list[str]:
    by_id = chunk_index.get("by_id", {})
    texts = []
    seen = set()
    for chunk_id in chunk_ids_for_sequence(sequence, chunk_index):
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = by_id.get(chunk_id)
        if chunk and chunk.get("content_norm"):
            texts.append(str(chunk["content_norm"]))
    return texts


def chunk_ids_for_sequence(sequence: list[EvidenceItem], chunk_index: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in sequence:
        if item.kind == "chunk" and item.chunk_id:
            result.add(item.chunk_id)
        elif item.kind == "span":
            result.update(chunk_ids_for_span(item, chunk_index))
    return result


def chunk_ids_for_span(item: EvidenceItem, chunk_index: dict[str, Any]) -> set[str]:
    by_file = chunk_index.get("by_file", {})
    if not item.file_name or item.start_line is None or item.end_line is None:
        return set()
    result = set()
    for chunk in by_file.get(item.file_name, []):
        if _ranges_overlap(item.start_line, item.end_line, chunk.get("start_line"), chunk.get("end_line")):
            result.add(str(chunk.get("chunk_id", "")))
    return {chunk_id for chunk_id in result if chunk_id}


def online_cost(row: dict[str, Any]) -> dict[str, float]:
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    tool_events = [event for event in trace if isinstance(event, dict) and event.get("event_type") == "tool_call"]
    search_events = [event for event in tool_events if "search" in str(event.get("tool", "")).lower()]
    read_events = [event for event in tool_events if str(event.get("tool", "")).lower() == "read_file"]
    llm_events = [event for event in trace if isinstance(event, dict) and event.get("event_type") == "llm_call"]
    llm_calls = _number(row.get("llm_calls"), len(llm_events))
    tool_calls = _number(row.get("tool_calls"), len(tool_events))
    search_calls = _number(row.get("knowledge_search_calls"), len(search_events))
    read_calls = _number(row.get("read_file_calls"), len(read_events))
    input_tokens = _number(row.get("input_tokens"), 0.0)
    output_tokens = _number(row.get("output_tokens"), 0.0)
    total_tokens = _number(row.get("total_tokens"), input_tokens + output_tokens)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    return {
        "llm_calls": llm_calls,
        "search_calls": search_calls,
        "read_calls": read_calls,
        "tool_calls": tool_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": _number(row.get("latency_seconds"), 0.0),
        "retrieval_latency_seconds": _number(row.get("retrieval_latency_seconds"), 0.0),
        "read_file_latency_seconds": _number(row.get("read_file_latency_seconds"), 0.0),
        "agent_reasoning_latency_seconds": _number(row.get("agent_reasoning_latency_seconds"), 0.0),
    }


def summarize_method(
    method: str,
    rows: list[dict[str, Any]],
    targets: dict[str, dict[str, dict[str, Any]]],
    chunk_index: dict[str, Any],
    judge_labels: dict[str, float] | None = None,
) -> dict[str, Any]:
    legacy = evaluate_rows(rows)
    per_query = []
    for row in rows:
        qid = str(row.get("question_id"))
        quality = answer_quality(row)
        evidence = evidence_navigation_metrics(row, targets, chunk_index, k=5)
        process = trace_process_metrics(row, targets["silver"].get(qid))
        cost = online_cost(row)
        judge = judge_labels.get(qid) if judge_labels else None
        per_query.append(
            {
                "method": method,
                "question_id": qid,
                **quality,
                "llm_judge_correct": judge,
                **evidence,
                **process,
                **cost,
            }
        )
    summary = {
        "method": method,
        "num_queries": len(rows),
        "answer_recall": _mean(row["answer_recall"] for row in per_query),
        "contain_accuracy": _mean(row["contain_accuracy"] for row in per_query),
        "llm_judge_accuracy": _mean_defined(row.get("llm_judge_correct") for row in per_query),
        "judge_scored": sum(1 for row in per_query if row.get("llm_judge_correct") is not None),
        "legacy_exact_match": legacy["metrics"].get("exact_match", 0.0),
        "legacy_precision": legacy["metrics"].get("precision", 0.0),
        "legacy_recall": legacy["metrics"].get("recall", 0.0),
        "legacy_f1": legacy["metrics"].get("f1", 0.0),
        "silver_scored_queries": sum(int(row.get("silver_scored") or 0) for row in per_query),
        "silver_hit_at_5": _mean_defined(row.get("silver_hit_at_5") for row in per_query),
        "silver_recall_at_5": _mean_defined(row.get("silver_recall_at_5") for row in per_query),
        "evidence_mrr": _mean_defined(row.get("evidence_mrr") for row in per_query),
        "target_unit_scored_queries": sum(int(row.get("target_unit_scored") or 0) for row in per_query),
        "target_unit_recall_at_5": _mean_defined(row.get("target_unit_recall_at_5") for row in per_query),
        "target_entity_scored_queries": sum(int(row.get("target_entity_scored") or 0) for row in per_query),
        "target_entity_recall_at_5": _mean_defined(row.get("target_entity_recall_at_5") for row in per_query),
        "claim_scored_queries": sum(int(row.get("claim_scored") or 0) for row in per_query),
        "claim_coverage_at_5": _mean_defined(row.get("claim_coverage_at_5") for row in per_query),
        "read_silver_hit_rate": _mean_defined(row.get("read_silver_hit") for row in per_query),
        "first_hit_step": _mean_defined(row.get("first_hit_step") for row in per_query),
        "tokens_to_first_hit": _mean_defined(row.get("tokens_to_first_hit") for row in per_query),
        "duplicate_read_ratio": _mean_defined(row.get("duplicate_read_ratio") for row in per_query),
        "miss_rate": _mean_defined(row.get("miss") for row in per_query),
    }
    for field in (
        "llm_calls",
        "search_calls",
        "read_calls",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
        "retrieval_latency_seconds",
        "read_file_latency_seconds",
        "agent_reasoning_latency_seconds",
    ):
        stats = summarize_values(row.get(field) for row in per_query)
        summary[f"{field}_mean"] = stats["mean"]
        summary[f"{field}_p95"] = stats["p95"]
        summary[f"{field}_sum"] = stats["sum"]
    correct_count = sum(1 for row in per_query if row.get("llm_judge_correct") == 1.0)
    summary["tokens_per_correct"] = summary["total_tokens_sum"] / correct_count if correct_count else None
    summary["llm_calls_per_correct"] = summary["llm_calls_sum"] / correct_count if correct_count else None
    return {"summary": summary, "per_query": per_query}


def load_judge_labels(path: Path | None) -> dict[str, dict[str, float]]:
    """Load optional judge outputs.

    Supported JSONL fields:
    - method: optional method id. If omitted, labels are shared by question id.
    - question_id: required.
    - label/correct: "correct"/"incorrect", bool, or 1/0.
    """

    if path is None or not path.exists():
        return {}
    by_method: dict[str, dict[str, float]] = {}
    shared: dict[str, float] = {}
    for row in read_jsonl(path):
        qid = str(row.get("question_id", ""))
        if not qid:
            continue
        label = _judge_value(row.get("correct", row.get("label")))
        if label is None:
            continue
        method = str(row.get("method", "")).strip()
        if method:
            by_method.setdefault(method, {})[qid] = label
        else:
            shared[qid] = label
    if shared:
        by_method["*"] = shared
    return by_method


def _judge_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return 1.0 if float(value) >= 0.5 else 0.0
    text = str(value).strip().lower()
    if text in {"correct", "true", "yes", "1"}:
        return 1.0
    if text in {"incorrect", "false", "no", "0"}:
        return 0.0
    return None


def summarize_timing(offline_stage_timing: Path | None, online_stage_timing: Path | None) -> dict[str, Any]:
    offline_rows = read_jsonl(offline_stage_timing) if offline_stage_timing else []
    online_rows = read_jsonl(online_stage_timing) if online_stage_timing else []
    offline_by_stage = {str(row.get("stage")): row for row in offline_rows if row.get("status") == "ok"}
    latest_online: dict[str, dict[str, Any]] = {}
    for row in online_rows:
        if row.get("status") != "ok" or row.get("method_scope") != "online_query":
            continue
        if float(row.get("extra_metrics", {}).get("jsonl_rows", 0) or 0) < 100:
            continue
        latest_online[str(row.get("method"))] = row
    return {"offline_by_stage": offline_by_stage, "latest_online": latest_online}


def index_efficiency_rows(timing: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_stage = timing.get("offline_by_stage", {})
    f5 = by_stage.get("F5_chunk_index", {})
    f6 = by_stage.get("F6_semantic_graph_llm", {})
    f7 = by_stage.get("F7_structure_graph", {})
    f8 = by_stage.get("F8_sequence_graph", {})
    f9 = by_stage.get("F9_unified_graph", {})
    f10 = by_stage.get("F10_graph_es_sync", {})

    method_build = sum(float(stage.get("wall_time_seconds", 0) or 0) for stage in (f7, f8, f9, f10))
    method_storage = sum(float(stage.get("disk_bytes", 0) or 0) for stage in (f7, f8, f9, f10))
    nodes = float(f9.get("extra_metrics", {}).get("graph_nodes", 0) or 0)
    edges = float(f9.get("extra_metrics", {}).get("graph_edges", 0) or 0)
    objects = float(f9.get("extra_metrics", {}).get("graph_nodes_chunk", 0) or 0)
    throughput = objects / method_build if method_build else None
    index_rows = [
        {
            "method": "Signpost",
            "build_seconds": method_build,
            "storage_bytes": method_storage,
            "storage_mib": method_storage / (1024 * 1024),
            "nodes": nodes,
            "edges": edges,
            "retrievable_objects": objects,
            "throughput_objects_per_second": throughput,
            "scope": "layered graph and sketches",
        }
    ]
    layer_rows = [
        _layer_row("Structural", f7, "graph_nodes_raptor", "graph_edges_structure", "zoom in/out"),
        _layer_row("Sequential", f8, "graph_nodes_chunk", "graph_edges_sequence", "local reading"),
        {
            "component": "Semantic",
            "nodes": _metric(f9, "graph_nodes_entity"),
            "edges": _metric(f9, "graph_edges_semantic"),
            "build_seconds": float(f6.get("wall_time_seconds", 0) or 0),
            "storage_bytes": float(f6.get("disk_bytes", 0) or 0),
            "role": "semantic jump; shared annotation stage",
        },
        {
            "component": "Provenance",
            "nodes": 0.0,
            "edges": _metric(f9, "graph_edges_source"),
            "build_seconds": 0.0,
            "storage_bytes": 0.0,
            "role": "source verification",
        },
        {
            "component": "Unified index",
            "nodes": nodes,
            "edges": edges,
            "build_seconds": float(f9.get("wall_time_seconds", 0) or 0) + float(f10.get("wall_time_seconds", 0) or 0),
            "storage_bytes": float(f9.get("disk_bytes", 0) or 0) + float(f10.get("disk_bytes", 0) or 0),
            "role": "retrieval interface",
        },
        {
            "component": "Sketches",
            "nodes": 0.0,
            "edges": 0.0,
            "build_seconds": 0.0,
            "storage_bytes": 0.0,
            "role": "included in graph object metadata; not separately timed",
        },
    ]
    shared_rows = []
    for stage in ("F3_data_prepare", "F3_5_parse_documents", "F4_chunk_tree", "F5_chunk_index", "F6_semantic_graph_llm"):
        row = by_stage.get(stage)
        if not row:
            continue
        shared_rows.append(
            {
                "stage": stage,
                "scope": row.get("method_scope", ""),
                "wall_time_seconds": float(row.get("wall_time_seconds", 0) or 0),
                "disk_bytes": float(row.get("disk_bytes", 0) or 0),
                "note": _shared_stage_note(stage),
            }
        )
    return index_rows, layer_rows, shared_rows


def _layer_row(name: str, row: dict[str, Any], node_key: str, edge_key: str, role: str) -> dict[str, Any]:
    return {
        "component": name,
        "nodes": _metric(row, node_key),
        "edges": _metric(row, edge_key),
        "build_seconds": float(row.get("wall_time_seconds", 0) or 0),
        "storage_bytes": float(row.get("disk_bytes", 0) or 0),
        "role": role,
    }


def _metric(row: dict[str, Any], key: str) -> float:
    return float(row.get("extra_metrics", {}).get(key, 0) or 0)


def _shared_stage_note(stage: str) -> str:
    notes = {
        "F3_data_prepare": "shared dataset conversion",
        "F3_5_parse_documents": "shared parsing",
        "F4_chunk_tree": "shared chunk/tree construction",
        "F5_chunk_index": "shared chunk vector index, also used by flat retrieval",
        "F6_semantic_graph_llm": "shared semantic annotation; recorded but excluded from method-specific build time",
    }
    return notes.get(stage, "")


def write_outputs(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    per_query_rows: list[dict[str, Any]],
    timing: dict[str, Any],
) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    quality_fields = [
        "method",
        "num_queries",
        "answer_recall",
        "contain_accuracy",
        "llm_judge_accuracy",
        "judge_scored",
        "legacy_exact_match",
        "legacy_precision",
        "legacy_recall",
        "legacy_f1",
    ]
    evidence_fields = [
        "method",
        "silver_scored_queries",
        "target_unit_scored_queries",
        "target_unit_recall_at_5",
        "target_entity_scored_queries",
        "target_entity_recall_at_5",
        "silver_hit_at_5",
        "silver_recall_at_5",
        "evidence_mrr",
        "claim_scored_queries",
        "claim_coverage_at_5",
    ]
    process_fields = [
        "method",
        "read_silver_hit_rate",
        "first_hit_step",
        "tokens_to_first_hit",
        "duplicate_read_ratio",
        "miss_rate",
    ]
    online_fields = [
        "method",
        "latency_seconds_mean",
        "latency_seconds_p95",
        "llm_calls_mean",
        "search_calls_mean",
        "read_calls_mean",
        "tool_calls_mean",
        "input_tokens_mean",
        "output_tokens_mean",
        "total_tokens_mean",
        "tokens_per_correct",
        "llm_calls_per_correct",
    ]
    ablation_fields = [
        "method",
        "variant",
        "hidden_cues",
        "answer_recall",
        "llm_judge_accuracy",
        "silver_hit_at_5",
        "first_hit_step",
        "duplicate_read_ratio",
        "llm_calls_mean",
        "total_tokens_mean",
    ]

    ablation_rows = []
    for row in summaries:
        variant, hidden = ABLATION_LABELS.get(row["method"], (row["method"], ""))
        ablation_rows.append({**row, "variant": variant, "hidden_cues": hidden})

    write_csv(metrics_dir / "answer_quality_summary.csv", summaries, quality_fields)
    write_csv(metrics_dir / "evidence_navigation_summary.csv", summaries, evidence_fields)
    write_csv(metrics_dir / "agent_process_summary.csv", summaries, process_fields)
    write_csv(metrics_dir / "online_efficiency_summary.csv", summaries, online_fields)
    write_csv(metrics_dir / "ablation_summary.csv", ablation_rows, ablation_fields)
    write_csv(metrics_dir / "per_query_final_metrics.csv", per_query_rows, _per_query_fields())

    index_rows, layer_rows, shared_rows = index_efficiency_rows(timing)
    write_csv(
        metrics_dir / "index_efficiency_summary.csv",
        index_rows,
        ["method", "build_seconds", "storage_bytes", "storage_mib", "nodes", "edges", "retrievable_objects", "throughput_objects_per_second", "scope"],
    )
    write_csv(metrics_dir / "layer_graph_summary.csv", layer_rows, ["component", "nodes", "edges", "build_seconds", "storage_bytes", "role"])
    write_csv(metrics_dir / "shared_or_auxiliary_stage_summary.csv", shared_rows, ["stage", "scope", "wall_time_seconds", "disk_bytes", "note"])
    write_json(
        metrics_dir / "final_metrics.json",
        {
            "method_summaries": summaries,
            "index_efficiency": index_rows,
            "layer_breakdown": layer_rows,
            "shared_or_auxiliary_stages": shared_rows,
            "notes": {
                "llm_judge_accuracy": "todo when no judge output file is provided",
                "legacy_metrics": "legacy EM/Precision/Recall/F1 are preserved for traceability but not used as paper-facing metrics",
                "offline_cost": "F6 semantic extraction is recorded as shared/auxiliary and excluded from Signpost method-specific build time",
            },
        },
    )

    write_table(tables_dir / "table_answer_quality.md", answer_quality_table(summaries))
    write_table(tables_dir / "table_evidence_navigation.md", evidence_navigation_table(summaries))
    write_table(tables_dir / "table_agent_process.md", agent_process_table(summaries))
    write_table(tables_dir / "table_online_efficiency.md", online_efficiency_table(summaries))
    write_table(tables_dir / "table_index_efficiency.md", index_efficiency_table(index_rows))
    write_table(tables_dir / "table_layer_breakdown.md", layer_breakdown_table(layer_rows))
    write_table(tables_dir / "table_shared_or_auxiliary_stages.md", shared_stages_table(shared_rows))
    write_table(tables_dir / "table_ablation.md", ablation_table(ablation_rows))


def _per_query_fields() -> list[str]:
    return [
        "method",
        "question_id",
        "answer_recall",
        "contain_accuracy",
        "llm_judge_correct",
        "target_unit_recall_at_5",
        "target_entity_recall_at_5",
        "silver_hit_at_5",
        "silver_recall_at_5",
        "evidence_mrr",
        "claim_coverage_at_5",
        "read_silver_hit",
        "first_hit_step",
        "tokens_to_first_hit",
        "duplicate_read_ratio",
        "miss",
        "llm_calls",
        "search_calls",
        "read_calls",
        "tool_calls",
        "total_tokens",
        "latency_seconds",
    ]


def write_table(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def answer_quality_table(summaries: list[dict[str, Any]]) -> str:
    full = _summary_by_method(summaries).get("signpost.full", {})
    rows = [
        ["No retrieval", "Vanilla LLM", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Flat retrieval", "Hybrid RAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Graph RAG", "Clue-RAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Graph RAG", "AGRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Hierarchical RAG", "LinearRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Agentic RAG", "HiPRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Agentic GraphRAG", "GraphRAG-R1", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        [
            "Navigation index",
            "Signpost",
            _fmt(full.get("answer_recall")),
            _fmt(full.get("contain_accuracy")),
            _fmt(full.get("llm_judge_accuracy")),
            "todo",
            "todo",
            "todo",
            _fmt(full.get("llm_judge_accuracy")),
        ],
    ]
    return _markdown_table(
        [
            "Category",
            "Method",
            "Agriculture AnsRec",
            "Agriculture CAcc",
            "Agriculture Judge",
            "Legal AnsRec",
            "Legal CAcc",
            "Legal Judge",
            "Avg. Judge",
        ],
        rows,
        title="Answer quality on Agriculture-full and Legal-full.",
    )


def evidence_navigation_table(summaries: list[dict[str, Any]]) -> str:
    full = _summary_by_method(summaries).get("signpost.full", {})
    rows = [
        ["Hybrid RAG", "todo", "todo", "todo", "todo", "todo"],
        ["Clue-RAG", "todo", "todo", "todo", "todo", "todo"],
        ["AGRAG", "todo", "todo", "todo", "todo", "todo"],
        ["LinearRAG", "todo", "todo", "todo", "todo", "todo"],
        ["HiPRAG", "todo", "todo", "todo", "todo", "todo"],
        ["GraphRAG-R1", "todo", "todo", "todo", "todo", "todo"],
        [
            "Signpost",
            _fmt(full.get("target_unit_recall_at_5")),
            _fmt(full.get("silver_hit_at_5")),
            _fmt(full.get("silver_recall_at_5")),
            _fmt(full.get("evidence_mrr")),
            _fmt(full.get("claim_coverage_at_5")),
        ],
    ]
    return _markdown_table(
        ["Method", "TER@5", "SilverHit@5", "SilverRecall@5", "MRR", "ClaimCoverage@5"],
        rows,
        title="Silver evidence navigation metrics derived from source documents and reference answers.",
    )


def agent_process_table(summaries: list[dict[str, Any]]) -> str:
    full = _summary_by_method(summaries).get("signpost.full", {})
    rows = [
        ["HiPRAG", "todo", "todo", "todo", "todo"],
        ["GraphRAG-R1", "todo", "todo", "todo", "todo"],
        [
            "Signpost",
            _fmt(full.get("first_hit_step")),
            _fmt(full.get("tokens_to_first_hit")),
            _fmt(full.get("duplicate_read_ratio")),
            _fmt(full.get("miss_rate")),
        ],
    ]
    return _markdown_table(
        ["Method", "FirstHit", "Tok2Hit", "DupRead", "Miss"],
        rows,
        title="Agentic evidence-access behavior.",
    )


def online_efficiency_table(summaries: list[dict[str, Any]]) -> str:
    full = _summary_by_method(summaries).get("signpost.full", {})
    rows = [
        ["Vanilla LLM", "todo", "N/A", "N/A", "N/A", "todo", "todo", "todo"],
        ["Hybrid RAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["Clue-RAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["AGRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["LinearRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["HiPRAG", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        ["GraphRAG-R1", "todo", "todo", "todo", "todo", "todo", "todo", "todo"],
        [
            "Signpost",
            _fmt(full.get("llm_calls_mean")),
            _fmt(full.get("search_calls_mean")),
            _fmt(full.get("read_calls_mean")),
            _fmt(full.get("tool_calls_mean")),
            _fmt(full.get("total_tokens_mean")),
            _fmt(full.get("tokens_per_correct")),
            _fmt(full.get("llm_calls_per_correct")),
        ],
    ]
    return _markdown_table(
        ["Method", "LLM Calls", "Search Calls", "Read Calls", "Tool Calls", "Total Tok.", "Tok./Correct", "Calls/Correct"],
        rows,
        title="Online interaction cost under local H200 execution.",
    )


def index_efficiency_table(index_rows: list[dict[str, Any]]) -> str:
    signpost = index_rows[0] if index_rows else {}
    rows = [
        ["Hybrid RAG", "todo", "todo", "todo", "todo", "todo", "todo", "chunk index"],
        ["Clue-RAG", "todo", "todo", "todo", "todo", "todo", "todo", "multi-partite graph"],
        ["AGRAG", "todo", "todo", "todo", "todo", "todo", "todo", "entity-relation graph"],
        ["LinearRAG", "todo", "todo", "todo", "todo", "todo", "todo", "hierarchy/topology"],
        [
            "Signpost",
            _fmt(signpost.get("build_seconds")),
            _fmt(signpost.get("storage_mib")),
            _fmt_int(signpost.get("nodes")),
            _fmt_int(signpost.get("edges")),
            _fmt_int(signpost.get("retrievable_objects")),
            _fmt(signpost.get("throughput_objects_per_second")),
            "layered graph and sketches",
        ],
    ]
    return _markdown_table(
        ["Method", "Build", "Storage", "Nodes", "Edges", "Objects", "Throughput", "Scope"],
        rows,
        title="Method-specific index efficiency after shared preprocessing. Build is seconds; Storage is MiB.",
    )


def layer_breakdown_table(layer_rows: list[dict[str, Any]]) -> str:
    rows = [
        [
            row.get("component", ""),
            _fmt_int(row.get("nodes")),
            _fmt_int(row.get("edges")),
            _fmt(row.get("build_seconds")),
            _fmt(row.get("storage_bytes")),
        ]
        for row in layer_rows
    ]
    return _markdown_table(["Component", "Nodes", "Edges", "Build", "Storage"], rows, title="Layer-level index statistics for Signpost.")


def shared_stages_table(shared_rows: list[dict[str, Any]]) -> str:
    rows = [
        [row.get("stage", ""), row.get("scope", ""), _fmt(row.get("wall_time_seconds")), _fmt(row.get("disk_bytes")), row.get("note", "")]
        for row in shared_rows
    ]
    return _markdown_table(["Stage", "Scope", "Wall Time Seconds", "Disk Bytes", "Note"], rows, title="Shared or auxiliary stages recorded outside paper-facing method-specific build time.")


def ablation_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                f"`{row.get('variant', row.get('method', ''))}`",
                row.get("hidden_cues", ""),
                _fmt(row.get("answer_recall")),
                _fmt(row.get("llm_judge_accuracy")),
                _fmt(row.get("silver_hit_at_5")),
                _fmt(row.get("first_hit_step")),
                _fmt(row.get("duplicate_read_ratio")),
                _fmt(row.get("llm_calls_mean")),
                _fmt(row.get("total_tokens_mean")),
            ]
        )
    return _markdown_table(
        ["Variant", "Hidden cues", "AnsRec", "Judge", "SilverHit@5", "FirstHit", "DupRead", "LLM Calls", "Tokens"],
        table_rows,
        title="Visibility ablation under a fixed index. Cue families are masked at observation time.",
    )


def _markdown_table(headers: list[str], rows: list[list[Any]], *, title: str = "") -> str:
    lines = []
    if title:
        lines.extend([f"<!-- {title} -->", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _summary_by_method(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in summaries}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "todo"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "todo"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


def _mean(values: Iterable[Any]) -> float:
    nums = [float(value) for value in values]
    return sum(nums) / len(nums) if nums else 0.0


def _mean_defined(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def _number(value: Any, default: float) -> float:
    if value in (None, "", [], {}):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _span_from_tool_event(event: dict[str, Any]) -> tuple[str, int | None, int | None]:
    summary = event.get("output_summary") if isinstance(event.get("output_summary"), dict) else {}
    file_name = str(summary.get("file_name", "")).strip()
    resolved = summary.get("resolved") if isinstance(summary.get("resolved"), dict) else {}
    start_line = _to_int(resolved.get("start_line", summary.get("start_line")))
    end_line = _to_int(resolved.get("end_line", summary.get("end_line")))
    if file_name and start_line is not None and end_line is not None:
        return file_name, start_line, end_line
    tool_input = event.get("input") if isinstance(event.get("input"), dict) else {}
    locate = str(tool_input.get("locate", "")).strip()
    return _parse_locate(locate)


def _parse_locate(locate: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"(?P<file>.+):L(?P<start>\d+)(?:-L?(?P<end>\d+))?$", locate)
    if not match:
        return "", None, None
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    return match.group("file"), start, end


def _silver_chunks(silver_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not silver_row:
        return []
    return [item for item in _as_list(silver_row.get("silver_chunks")) if isinstance(item, dict)]


def _target_entities(entity_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not entity_row:
        return []
    return [item for item in _as_list(entity_row.get("target_entities")) if isinstance(item, dict)]


def _target_units(unit_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not unit_row:
        return []
    return [item for item in _as_list(unit_row.get("target_units")) if isinstance(item, dict)]


def _claims(claim_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not claim_row:
        return []
    return [item for item in _as_list(claim_row.get("claims")) if isinstance(item, dict)]


def _same_file_overlap(item: EvidenceItem, silver: dict[str, Any]) -> bool:
    if item.file_name != str(silver.get("file_name", "")):
        return False
    return _ranges_overlap(item.start_line, item.end_line, _to_int(silver.get("start_line")), _to_int(silver.get("end_line")))


def _ranges_overlap(start_a: int | None, end_a: int | None, start_b: int | None, end_b: int | None) -> bool:
    if start_a is None or end_a is None or start_b is None or end_b is None:
        return False
    return max(start_a, start_b) <= min(end_a, end_b)


def _unit_in_text(unit_norm: str, text_norm: str) -> bool:
    if not unit_norm or not text_norm:
        return False
    if unit_norm in text_norm:
        return True
    unit_tokens = set(unit_norm.split())
    text_tokens = set(text_norm.split())
    if not unit_tokens:
        return False
    return len(unit_tokens & text_tokens) / len(unit_tokens) >= 0.75


def discover_prediction_files(predictions_dir: Path) -> list[Path]:
    files = list(predictions_dir.glob("signpost.*.jsonl"))
    for baseline in ("vanilla_llm", "hybrid_rag"):
        path = predictions_dir / f"{baseline}.jsonl"
        if path.exists():
            files.append(path)
    cluerag = predictions_dir / "cluerag.jsonl"
    if cluerag.exists():
        files.append(cluerag)
    cluerag_prompt_normalized = predictions_dir / "cluerag_prompt_normalized.jsonl"
    if cluerag_prompt_normalized.exists():
        files.append(cluerag_prompt_normalized)
    by_method = {path.stem: path for path in files}
    ordered = [by_method[method] for method in METHOD_ORDER if method in by_method]
    remaining = sorted(path for path in files if path.stem not in METHOD_ORDER)
    return ordered + remaining


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate final Signpost ICDE metrics and paper tables.")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunks-file")
    parser.add_argument("--offline-stage-timing")
    parser.add_argument("--online-stage-timing")
    parser.add_argument("--judge-file")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    targets = load_targets(Path(args.targets_dir))
    chunk_index = load_chunk_index(Path(args.chunks_file) if args.chunks_file else None)
    judge_by_method = load_judge_labels(Path(args.judge_file) if args.judge_file else None)
    timing = summarize_timing(
        Path(args.offline_stage_timing) if args.offline_stage_timing else None,
        Path(args.online_stage_timing) if args.online_stage_timing else None,
    )

    summaries = []
    per_query_rows = []
    for path in discover_prediction_files(predictions_dir):
        method = path.stem
        rows = read_jsonl(path)
        judge_labels = judge_by_method.get(method, judge_by_method.get("*", {}))
        result = summarize_method(method, rows, targets, chunk_index, judge_labels)
        online_row = timing.get("latest_online", {}).get(method)
        if online_row:
            result["summary"]["batch_wall_time_seconds"] = float(online_row.get("wall_time_seconds", 0) or 0)
        summaries.append(result["summary"])
        per_query_rows.extend(result["per_query"])

    write_outputs(Path(args.output_dir), summaries, per_query_rows, timing)
    print(f"output={args.output_dir} methods={len(summaries)} queries={sum(row['num_queries'] for row in summaries)}")
    for row in summaries:
        print(
            "method={method} ansrec={answer_recall:.4f} cacc={contain_accuracy:.4f} "
            "silver_hit@5={silver_hit_at_5} llm_calls={llm_calls_mean:.4f}".format(
                **{**row, "silver_hit_at_5": _fmt(row.get("silver_hit_at_5"))}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
