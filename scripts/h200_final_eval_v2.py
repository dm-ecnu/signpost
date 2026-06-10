from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from signpost.evaluation.metrics import extract_answer_from_prediction, normalize_answer


DATASETS = {
    "agriculture": {
        "processed": "agriculture",
        "outputs": "agriculture",
        "ans": "agriculture",
    },
    "mixv0": {
        "processed": "mix",
        "outputs": "mixv0",
        "ans": "mixv0",
    },
}

METHODS = [
    "vanilla_llm",
    "hybrid_rag",
    "cluerag_prompt_normalized",
    "agrag",
    "linearrag",
    "hiprag",
    "graphrag_r1",
    "signpost.full",
    "signpost.no_offline",
    "signpost.no_online",
    "signpost.no_semantic_cues",
    "signpost.no_provenance_cues",
    "signpost.no_vertical_cues",
    "signpost.no_horizontal_cues",
]

METHOD_LABELS = {
    "vanilla_llm": "Vanilla LLM",
    "hybrid_rag": "Hybrid RAG",
    "cluerag_prompt_normalized": "ClueRAG-PN",
    "agrag": "AGRAG",
    "linearrag": "LinearRAG",
    "hiprag": "HiPRAG",
    "graphrag_r1": "GraphRAG-R1",
    "signpost.full": "Signpost",
    "signpost.no_offline": "Signpost no-offline",
    "signpost.no_online": "Signpost no-online",
    "signpost.no_semantic_cues": "Signpost no-semantic",
    "signpost.no_provenance_cues": "Signpost no-provenance",
    "signpost.no_vertical_cues": "Signpost no-vertical",
    "signpost.no_horizontal_cues": "Signpost no-horizontal",
}

BASELINE_METHODS = {
    "agrag",
    "linearrag",
    "hiprag",
    "graphrag_r1",
    "cluerag_prompt_normalized",
}

SILVER_EVIDENCE_METHODS = {
    "hiprag",
    "graphrag_r1",
    "signpost.full",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def final_answer(row: dict[str, Any]) -> str:
    text = extract_answer_from_prediction(str(row.get("prediction", ""))).strip()
    json_answer = extract_json_answer(text)
    if json_answer is not None:
        return json_answer
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text


def extract_json_answer(text: str) -> str | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    payloads = [candidate]
    if start >= 0 and end > start:
        payloads.append(candidate[start : end + 1])
    for payload in payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    return None


def answer_metrics(row: dict[str, Any]) -> dict[str, float]:
    pred = final_answer(row)
    golds = row.get("answer")
    if not isinstance(golds, list):
        golds = [golds]
    scored = [token_overlap_metrics(str(gold), pred) for gold in golds]
    return max(scored, key=lambda item: item["legacy_f1"]) if scored else zero_answer_metrics()


def zero_answer_metrics() -> dict[str, float]:
    return {
        "answer_recall": 0.0,
        "contain_accuracy": 0.0,
        "legacy_exact_match": 0.0,
        "legacy_precision": 0.0,
        "legacy_recall": 0.0,
        "legacy_f1": 0.0,
    }


def token_overlap_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    gold_tokens = gold_norm.split()
    pred_tokens = pred_norm.split()
    if not gold_tokens and not pred_tokens:
        precision = recall = f1 = 1.0
    elif not gold_tokens or not pred_tokens:
        precision = recall = f1 = 0.0
    else:
        overlap = Counter(gold_tokens) & Counter(pred_tokens)
        overlap_count = sum(overlap.values())
        precision = overlap_count / len(pred_tokens)
        recall = overlap_count / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "answer_recall": recall,
        "contain_accuracy": 1.0 if gold_norm and gold_norm in pred_norm else 0.0,
        "legacy_exact_match": 1.0 if gold_norm == pred_norm else 0.0,
        "legacy_precision": precision,
        "legacy_recall": recall,
        "legacy_f1": f1,
    }


def load_targets(processed_dir: Path) -> dict[str, Any]:
    units = {str(row["question_id"]): row for row in read_jsonl(processed_dir / "llm_target_units.jsonl")}
    silver = {str(row["question_id"]): row for row in read_jsonl(processed_dir / "llm_silver_chunks.jsonl")}
    return {"units": units, "silver": silver}


def evidence_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in as_list(row.get("evidence_chunks")):
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        file_name = str(item.get("file_name", "")).strip()
        if not chunk_id and not file_name:
            continue
        items.append(
            {
                "kind": "evidence_chunk",
                "chunk_id": chunk_id,
                "file_name": file_name,
                "start_line": to_int(item.get("start_line")),
                "end_line": to_int(item.get("end_line")),
                "round": to_int(item.get("round")),
                "rank": to_int(item.get("rank")),
            }
        )
    return items


def retrieved_chunk_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in as_list(row.get("retrieved_chunks")):
        if isinstance(item, dict):
            chunk_id = str(item.get("chunk_id", "")).strip()
            file_name = str(item.get("file_name", "")).strip()
            start_line = to_int(item.get("start_line"))
            end_line = to_int(item.get("end_line"))
        else:
            chunk_id = str(item).strip()
            file_name = ""
            start_line = end_line = None
        if chunk_id or file_name:
            items.append({"kind": "chunk", "chunk_id": chunk_id, "file_name": file_name, "start_line": start_line, "end_line": end_line})
    return items


def citation_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in as_list(row.get("citations")):
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "kind": "span",
                "chunk_id": "",
                "file_name": str(item.get("file_name", "")).strip(),
                "start_line": to_int(item.get("start_line")),
                "end_line": to_int(item.get("end_line")),
            }
        )
    return [item for item in items if item["file_name"]]


def read_file_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    step = 0
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    for event in trace:
        if not isinstance(event, dict) or event.get("event_type") != "tool_call":
            continue
        tool = str(event.get("tool", "")).lower()
        if "search" in tool or tool == "read_file":
            step += 1
        if tool != "read_file":
            continue
        file_name, start_line, end_line = span_from_event(event)
        if file_name:
            items.append({"kind": "span", "chunk_id": "", "file_name": file_name, "start_line": start_line, "end_line": end_line, "step": step})
    return items


def span_from_event(event: dict[str, Any]) -> tuple[str, int | None, int | None]:
    summary = event.get("output_summary") if isinstance(event.get("output_summary"), dict) else {}
    resolved = summary.get("resolved") if isinstance(summary.get("resolved"), dict) else {}
    file_name = str(summary.get("file_name", "")).strip()
    start_line = to_int(resolved.get("start_line", summary.get("start_line")))
    end_line = to_int(resolved.get("end_line", summary.get("end_line")))
    if file_name and start_line is not None and end_line is not None:
        return file_name, start_line, end_line
    tool_input = event.get("input") if isinstance(event.get("input"), dict) else {}
    locate = str(tool_input.get("locate", "")).strip()
    match = re.match(r"(?P<file>.+):L(?P<start>\d+)(?:-L?(?P<end>\d+))?$", locate)
    if not match:
        return "", None, None
    return match.group("file"), int(match.group("start")), int(match.group("end") or match.group("start"))


def evidence_metrics(row: dict[str, Any], target_row: dict[str, Any] | None, silver_row: dict[str, Any] | None, k: int = 5) -> dict[str, Any]:
    sequence = evidence_items(row)
    top_k = sequence[:k]
    silver_chunks = [item for item in as_list((silver_row or {}).get("silver_chunks")) if isinstance(item, dict)]
    all_silver_ids = {str(item.get("chunk_id", "")) for item in silver_chunks if item.get("chunk_id")}
    hit_ids = silver_hits(top_k, silver_chunks)
    all_hit_ranks = [idx for idx, item in enumerate(sequence, 1) if silver_hits([item], silver_chunks)]

    target_units = [item for item in as_list((target_row or {}).get("target_units")) if isinstance(item, dict)]
    target_unit_ids = {str(item.get("unit_id", "")) for item in target_units if item.get("unit_id")}
    supported_units = supported_units_for_hits(hit_ids, silver_chunks)
    facts = [item for item in as_list((target_row or {}).get("facts")) if isinstance(item, dict)]
    fact_hits = 0
    for fact in facts:
        required = {str(item) for item in as_list(fact.get("required_units"))}
        if required and required <= supported_units:
            fact_hits += 1

    return {
        "evidence_observed": 1 if sequence else 0,
        "num_evidence_chunks": len(sequence),
        "silver_scored": 1 if all_silver_ids else 0,
        "silver_hit_at_5": none_if_unscored(1.0 if hit_ids else 0.0, all_silver_ids),
        "silver_recall_at_5": none_if_unscored(len(hit_ids) / len(all_silver_ids) if all_silver_ids else 0.0, all_silver_ids),
        "evidence_mrr": none_if_unscored(1.0 / all_hit_ranks[0] if all_hit_ranks else 0.0, all_silver_ids),
        "first_hit_rank": none_if_unscored(float(all_hit_ranks[0]) if all_hit_ranks else None, all_silver_ids),
        "target_unit_scored": 1 if target_unit_ids else 0,
        "target_unit_recall_at_5": none_if_unscored(len(supported_units & target_unit_ids) / len(target_unit_ids) if target_unit_ids else 0.0, target_unit_ids),
        "claim_scored": 1 if facts else 0,
        "claim_coverage_at_5": none_if_unscored(fact_hits / len(facts) if facts else 0.0, facts),
        "duplicate_evidence_ratio": duplicate_ratio(sequence),
        "miss_rate": none_if_unscored(0.0 if hit_ids else 1.0, all_silver_ids),
    }


def silver_hits(sequence: list[dict[str, Any]], silver_chunks: list[dict[str, Any]]) -> set[str]:
    hits: set[str] = set()
    for item in sequence:
        for silver in silver_chunks:
            chunk_id = str(silver.get("chunk_id", ""))
            if item.get("chunk_id") and item.get("chunk_id") == chunk_id:
                hits.add(chunk_id)
                continue
            if item.get("file_name") and item.get("file_name") == str(silver.get("file_name", "")):
                if ranges_overlap(to_int(item.get("start_line")), to_int(item.get("end_line")), to_int(silver.get("start_line")), to_int(silver.get("end_line"))):
                    hits.add(chunk_id)
    return hits


def supported_units_for_hits(hit_ids: set[str], silver_chunks: list[dict[str, Any]]) -> set[str]:
    units: set[str] = set()
    for silver in silver_chunks:
        if str(silver.get("chunk_id", "")) not in hit_ids:
            continue
        units.update(str(item) for item in as_list(silver.get("supports")) if str(item).strip())
    return units


def online_cost(row: dict[str, Any]) -> dict[str, float]:
    return {
        "latency_seconds": number(row.get("latency_seconds")),
        "llm_calls": number(row.get("llm_calls")),
        "search_calls": number(row.get("knowledge_search_calls")),
        "read_calls": number(row.get("read_file_calls")),
        "tool_calls": number(row.get("tool_calls")),
        "embedding_calls": number(row.get("embedding_calls")),
        "rerank_calls": number(row.get("rerank_calls")),
        "input_tokens": number(row.get("input_tokens")),
        "output_tokens": number(row.get("output_tokens")),
        "total_tokens": number(row.get("total_tokens")),
    }


def parse_llm_scores(ans_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    by_method: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(ans_dir.glob("*.txt")):
        qid = path.stem.split("_", 1)[1] if "_" in path.stem else path.stem
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"^### 方法\s+", text, flags=re.M)
        for section in sections[1:]:
            first_line, _, rest = section.partition("\n")
            method = first_line.strip()
            dims = re.search(r"准确性\s*([0-9.]+)\s*/\s*4[，,]\s*完整性\s*([0-9.]+)\s*/\s*3[，,]\s*简洁性\s*([0-9.]+)\s*/\s*3", rest)
            if dims:
                accuracy = float(dims.group(1))
                completeness = float(dims.group(2))
                conciseness = float(dims.group(3))
            else:
                accuracy_match = re.search(r"准确性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*4", rest)
                completeness_match = re.search(r"完整性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3", rest)
                conciseness_match = re.search(r"简洁性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3", rest)
                if not (accuracy_match and completeness_match and conciseness_match):
                    continue
                accuracy = float(accuracy_match.group(1))
                completeness = float(completeness_match.group(1))
                conciseness = float(conciseness_match.group(1))
            by_method.setdefault(method, {})[qid] = {
                "llm_accuracy_score": accuracy,
                "llm_completeness_score": completeness,
                "llm_conciseness_score": conciseness,
                "llm_total_score": accuracy + completeness + conciseness,
            }
    return by_method


def summarize_dataset(root: Path, dataset_key: str, dataset_map: dict[str, dict[str, str]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = (dataset_map or DATASETS)[dataset_key]
    processed_dir = root / "datasets" / "processed" / cfg["processed"]
    output_dir = root / "outputs" / cfg["outputs"]
    targets = load_targets(processed_dir)
    llm_scores = parse_llm_scores(root / "ans" / cfg.get("ans", cfg["outputs"]))
    summaries = []
    per_query = []
    for method in METHODS:
        pred_path = output_dir / "predictions" / f"{method}.jsonl"
        rows = read_jsonl(pred_path)
        if not rows:
            continue
        method_per_query = []
        for row in rows:
            qid = str(row.get("question_id", ""))
            score = llm_scores.get(method, {}).get(qid, {})
            evidence = evidence_metrics(row, targets["units"].get(qid), targets["silver"].get(qid))
            if method not in SILVER_EVIDENCE_METHODS:
                for key in (
                    "silver_scored",
                    "silver_hit_at_5",
                    "silver_recall_at_5",
                    "evidence_mrr",
                    "first_hit_rank",
                    "target_unit_scored",
                    "target_unit_recall_at_5",
                    "claim_scored",
                    "claim_coverage_at_5",
                    "miss_rate",
                ):
                    evidence[key] = None
            item = {
                "dataset": dataset_key,
                "method": method,
                "question_id": qid,
                **answer_metrics(row),
                **score,
                **evidence,
                **online_cost(row),
            }
            method_per_query.append(item)
            per_query.append(item)
        summaries.append(summarize_method(dataset_key, method, method_per_query, output_dir))
    return summaries, per_query


def summarize_method(dataset: str, method: str, rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    llm_correct = [1.0 if row.get("llm_total_score", 0.0) >= 7.0 else 0.0 for row in rows if row.get("llm_total_score") is not None]
    summary = {
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "num_queries": len(rows),
        "answer_recall": mean(row.get("answer_recall") for row in rows),
        "contain_accuracy": mean(row.get("contain_accuracy") for row in rows),
        "legacy_exact_match": mean(row.get("legacy_exact_match") for row in rows),
        "legacy_precision": mean(row.get("legacy_precision") for row in rows),
        "legacy_recall": mean(row.get("legacy_recall") for row in rows),
        "legacy_f1": mean(row.get("legacy_f1") for row in rows),
        "llm_scored": len([row for row in rows if row.get("llm_total_score") is not None]),
        "llm_accuracy_score": mean_defined(row.get("llm_accuracy_score") for row in rows),
        "llm_completeness_score": mean_defined(row.get("llm_completeness_score") for row in rows),
        "llm_conciseness_score": mean_defined(row.get("llm_conciseness_score") for row in rows),
        "llm_total_score": mean_defined(row.get("llm_total_score") for row in rows),
        "llm_judge_accuracy_at_7": mean_defined(llm_correct),
        "silver_scored_queries": sum(int(row.get("silver_scored") or 0) for row in rows),
        "silver_hit_at_5": mean_defined(row.get("silver_hit_at_5") for row in rows),
        "silver_recall_at_5": mean_defined(row.get("silver_recall_at_5") for row in rows),
        "evidence_mrr": mean_defined(row.get("evidence_mrr") for row in rows),
        "target_unit_scored_queries": sum(int(row.get("target_unit_scored") or 0) for row in rows),
        "target_unit_recall_at_5": mean_defined(row.get("target_unit_recall_at_5") for row in rows),
        "claim_scored_queries": sum(int(row.get("claim_scored") or 0) for row in rows),
        "claim_coverage_at_5": mean_defined(row.get("claim_coverage_at_5") for row in rows),
        "first_hit_rank": mean_defined(row.get("first_hit_rank") for row in rows),
        "duplicate_evidence_ratio": mean_defined(row.get("duplicate_evidence_ratio") for row in rows),
        "miss_rate": mean_defined(row.get("miss_rate") for row in rows),
        "num_evidence_chunks": mean_defined(row.get("num_evidence_chunks") for row in rows),
    }
    for field in ("latency_seconds", "llm_calls", "search_calls", "read_calls", "tool_calls", "embedding_calls", "rerank_calls", "input_tokens", "output_tokens", "total_tokens"):
        summary[f"{field}_mean"] = mean(row.get(field) for row in rows)
        summary[f"{field}_sum"] = sum(number(row.get(field)) for row in rows)
    summary.update(run_metrics_summary(output_dir, method))
    return summary


def run_metrics_summary(output_dir: Path, method: str) -> dict[str, Any]:
    if method not in BASELINE_METHODS:
        return {
            "offline_wall_time_seconds": None,
            "online_wall_time_seconds": None,
            "total_wall_time_seconds": None,
            "offline_disk_bytes": None,
            "index_nodes": None,
            "index_edges": None,
            "retrievable_objects": None,
        }
    data = read_json(output_dir / "baselines" / method / "run_metrics.json")
    graph = data.get("graph_index") if isinstance(data.get("graph_index"), dict) else {}
    retrieval = data.get("retrieval_metadata") if isinstance(data.get("retrieval_metadata"), dict) else {}
    return {
        "offline_wall_time_seconds": data.get("offline_wall_time_seconds"),
        "online_wall_time_seconds": data.get("online_wall_time_seconds"),
        "total_wall_time_seconds": data.get("total_wall_time_seconds"),
        "offline_disk_bytes": data.get("disk_bytes", data.get("offline_disk_bytes")),
        "index_nodes": graph.get("nodes", retrieval.get("shared_knowledge_units")),
        "index_edges": graph.get("edges"),
        "retrievable_objects": graph.get("passage_nodes", graph.get("chunk_nodes", retrieval.get("shared_chunks"))),
    }


def write_tables(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    formal = [row for row in summaries if not row["method"].startswith("signpost.no_")]
    ablations = [row for row in summaries if row["method"].startswith("signpost.")]
    (tables / "table_answer_quality.md").write_text(markdown_table(
        ["Dataset", "Method", "AnsRec", "Contain", "LLM Acc", "LLM Comp", "LLM Conc", "LLM Total", "Judge@7", "Precision", "F1"],
        [[r["dataset"], r["method_label"], fmt(r["answer_recall"]), fmt(r["contain_accuracy"]), fmt(r["llm_accuracy_score"]), fmt(r["llm_completeness_score"]), fmt(r["llm_conciseness_score"]), fmt(r["llm_total_score"]), fmt(r["llm_judge_accuracy_at_7"]), fmt(r["legacy_precision"]), fmt(r["legacy_f1"])] for r in formal],
    ), encoding="utf-8")
    (tables / "table_evidence_navigation.md").write_text(markdown_table(
        ["Dataset", "Method", "EvidenceN", "TargetUnit@5", "SilverHit@5", "SilverRecall@5", "MRR", "ClaimCoverage@5"],
        [[r["dataset"], r["method_label"], fmt(r["num_evidence_chunks"]), fmt(r["target_unit_recall_at_5"]), fmt(r["silver_hit_at_5"]), fmt(r["silver_recall_at_5"]), fmt(r["evidence_mrr"]), fmt(r["claim_coverage_at_5"])] for r in formal],
    ), encoding="utf-8")
    (tables / "table_agent_process.md").write_text(markdown_table(
        ["Dataset", "Method", "FirstHitRank", "DupEvidence", "MissRate"],
        [[r["dataset"], r["method_label"], fmt(r["first_hit_rank"]), fmt(r["duplicate_evidence_ratio"]), fmt(r["miss_rate"])] for r in formal],
    ), encoding="utf-8")
    (tables / "table_online_efficiency.md").write_text(markdown_table(
        ["Dataset", "Method", "Latency", "LLM Calls", "Search", "Read", "Tool", "Emb", "Rerank", "Tokens"],
        [[r["dataset"], r["method_label"], fmt(r["latency_seconds_mean"]), fmt(r["llm_calls_mean"]), fmt(r["search_calls_mean"]), fmt(r["read_calls_mean"]), fmt(r["tool_calls_mean"]), fmt(r["embedding_calls_mean"]), fmt(r["rerank_calls_mean"]), fmt(r["total_tokens_mean"])] for r in formal],
    ), encoding="utf-8")
    (tables / "table_offline_index_efficiency.md").write_text(markdown_table(
        ["Dataset", "Method", "Offline s", "Online s", "Total s", "Disk bytes", "Nodes", "Edges", "Objects"],
        [[r["dataset"], r["method_label"], fmt(r["offline_wall_time_seconds"]), fmt(r["online_wall_time_seconds"]), fmt(r["total_wall_time_seconds"]), fmt(r["offline_disk_bytes"], 0), fmt(r["index_nodes"], 0), fmt(r["index_edges"], 0), fmt(r["retrievable_objects"], 0)] for r in formal],
    ), encoding="utf-8")
    (tables / "table_ablation.md").write_text(markdown_table(
        ["Dataset", "Variant", "AnsRec", "LLM Total", "SilverHit@5", "FirstHitRank", "DupEvidence", "LLM Calls", "Tokens"],
        [[r["dataset"], r["method_label"], fmt(r["answer_recall"]), fmt(r["llm_total_score"]), fmt(r["silver_hit_at_5"]), fmt(r["first_hit_rank"]), fmt(r["duplicate_evidence_ratio"]), fmt(r["llm_calls_mean"]), fmt(r["total_tokens_mean"])] for r in ablations],
    ), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def ranges_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    return max(a_start, b_start) <= min(a_end, b_end)


def duplicate_ratio(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    keys = []
    for item in items:
        if item.get("chunk_id"):
            keys.append(f"chunk:{item['chunk_id']}")
        else:
            keys.append(f"span:{item.get('file_name')}:{item.get('start_line')}:{item.get('end_line')}")
    return 1.0 - (len(set(keys)) / len(keys))


def none_if_unscored(value: Any, scored: Any) -> Any:
    return value if scored else None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def number(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        parsed = float(value)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values: Iterable[Any]) -> float:
    nums = [number(value) for value in values]
    return sum(nums) / len(nums) if nums else 0.0


def mean_defined(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="v2 project root, e.g. /home/srl/signpost_re_v2")
    parser.add_argument("--output-dir", default="outputs/final_eval_v2")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated dataset keys from DATASETS")
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Add a dynamic dataset mapping as key=processed:outputs[:ans], e.g. legal_q100=legal_q100:legal_q100.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    dataset_map = dict(DATASETS)
    for spec in args.dataset_spec:
        key, _, rest = spec.partition("=")
        parts = [part.strip() for part in rest.split(":")]
        key = key.strip()
        if len(parts) not in (2, 3) or not key or not parts[0] or not parts[1]:
            raise SystemExit(f"invalid --dataset-spec={spec!r}; expected key=processed:outputs[:ans]")
        dataset_map[key] = {"processed": parts[0], "outputs": parts[1], "ans": parts[2] if len(parts) == 3 and parts[2] else parts[1]}
    requested_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    all_summaries: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []
    for dataset in requested_datasets:
        if dataset not in dataset_map:
            raise SystemExit(f"unknown dataset key: {dataset}; choices={', '.join(dataset_map)}")
        summaries, per_query = summarize_dataset(root, dataset, dataset_map)
        all_summaries.extend(summaries)
        all_per_query.extend(per_query)

    summary_fields = list(all_summaries[0].keys()) if all_summaries else []
    per_query_fields = list(all_per_query[0].keys()) if all_per_query else []
    write_tsv(out_dir / "method_final_metrics.tsv", all_summaries, summary_fields)
    write_tsv(out_dir / "per_query_final_metrics.tsv", all_per_query, per_query_fields)
    write_json(out_dir / "method_final_metrics.json", all_summaries)
    write_tables(out_dir, all_summaries)
    print(f"wrote {out_dir} methods={len(all_summaries)} per_query={len(all_per_query)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
