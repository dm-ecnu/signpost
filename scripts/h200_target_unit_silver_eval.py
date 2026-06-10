from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from signpost.evaluation.metrics import extract_answer_from_prediction, normalize_answer


DATASETS = {
    "agriculture": {"processed": "agriculture", "outputs": "agriculture"},
    "mixv0": {"processed": "mix", "outputs": "mixv0"},
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

NO_EVIDENCE_SEQUENCE_METHODS = {"vanilla_llm", "hiprag", "graphrag_r1"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_targets(processed_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "units": {str(row.get("question_id")): row for row in read_jsonl(processed_dir / "llm_target_units.jsonl")},
        "silver": {str(row.get("question_id")): row for row in read_jsonl(processed_dir / "llm_silver_chunks.jsonl")},
    }


def final_answer(row: dict[str, Any]) -> str:
    text = extract_answer_from_prediction(str(row.get("prediction", ""))).strip()
    parsed = extract_json_answer(text)
    if parsed is not None:
        return parsed
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


def target_unit_recall(answer: str, unit_row: dict[str, Any] | None) -> float | None:
    units = [unit for unit in as_list((unit_row or {}).get("target_units")) if isinstance(unit, dict)]
    required = [unit for unit in units if unit.get("required", True)]
    if not required:
        return None
    answer_norm = normalize_answer(answer)
    covered = 0
    for unit in required:
        candidates = [str(unit.get("text", ""))]
        candidates.extend(str(alias) for alias in as_list(unit.get("aliases")))
        if any(unit_phrase_matches(candidate, answer_norm) for candidate in candidates):
            covered += 1
    return covered / len(required)


def unit_phrase_matches(candidate: str, answer_norm: str) -> bool:
    unit_norm = normalize_answer(candidate)
    if not unit_norm or not answer_norm:
        return False
    if unit_norm in answer_norm:
        return True
    unit_tokens = set(unit_norm.split())
    answer_tokens = set(answer_norm.split())
    if not unit_tokens:
        return False
    # Short units need exact phrase containment; longer units allow light paraphrase/word-order variation.
    if len(unit_tokens) <= 2:
        return unit_tokens <= answer_tokens
    return len(unit_tokens & answer_tokens) / len(unit_tokens) >= 0.75


def evidence_sequence(row: dict[str, Any]) -> list[dict[str, Any]]:
    reads = read_file_sequence(row)
    if reads:
        return reads
    citations = citation_sequence(row)
    if citations:
        return citations
    return retrieved_chunk_sequence(row)


def retrieved_chunk_sequence(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
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
            result.append({"chunk_id": chunk_id, "file_name": file_name, "start_line": start_line, "end_line": end_line})
    return result


def citation_sequence(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in as_list(row.get("citations")):
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name", "")).strip()
        if file_name:
            result.append({"chunk_id": "", "file_name": file_name, "start_line": to_int(item.get("start_line")), "end_line": to_int(item.get("end_line"))})
    return result


def read_file_sequence(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("trace") if isinstance(row.get("trace"), list) else []
    result = []
    for event in trace:
        if not isinstance(event, dict) or event.get("event_type") != "tool_call":
            continue
        if str(event.get("tool", "")).lower() != "read_file":
            continue
        file_name, start_line, end_line = span_from_tool_event(event)
        if file_name:
            result.append({"chunk_id": "", "file_name": file_name, "start_line": start_line, "end_line": end_line})
    return result


def span_from_tool_event(event: dict[str, Any]) -> tuple[str, int | None, int | None]:
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


def silver_metrics(row: dict[str, Any], silver_row: dict[str, Any] | None) -> dict[str, Any]:
    silver_chunks = [item for item in as_list((silver_row or {}).get("silver_chunks")) if isinstance(item, dict)]
    if not silver_chunks:
        return {"silver_hit_at_5": None, "silver_recall_at_5": None, "mrr": None, "claim_coverage_at_5": None}
    sequence = evidence_sequence(row)
    if not sequence:
        return {"silver_hit_at_5": None, "silver_recall_at_5": None, "mrr": None, "claim_coverage_at_5": None}
    top5 = sequence[:5]
    silver_ids = {str(item.get("chunk_id", "")) for item in silver_chunks if item.get("chunk_id")}
    hit_ids = silver_hits(top5, silver_chunks)
    ranks = [idx for idx, item in enumerate(sequence, start=1) if silver_hits([item], silver_chunks)]
    supported_units = set()
    for silver in silver_chunks:
        if str(silver.get("chunk_id", "")) in hit_ids:
            supported_units.update(str(unit) for unit in as_list(silver.get("supports")) if str(unit).strip())
    all_supported_units = set()
    for silver in silver_chunks:
        all_supported_units.update(str(unit) for unit in as_list(silver.get("supports")) if str(unit).strip())
    return {
        "silver_hit_at_5": 1.0 if hit_ids else 0.0,
        "silver_recall_at_5": len(hit_ids) / len(silver_ids) if silver_ids else None,
        "mrr": 1.0 / ranks[0] if ranks else 0.0,
        # With the new files, facts live in target_units rows, so this field is filled later.
        "_supported_units": supported_units,
        "_all_supported_units": all_supported_units,
    }


def add_claim_coverage(metrics: dict[str, Any], unit_row: dict[str, Any] | None) -> dict[str, Any]:
    if metrics.get("silver_hit_at_5") is None:
        metrics["claim_coverage_at_5"] = None
        return metrics
    facts = [item for item in as_list((unit_row or {}).get("facts")) if isinstance(item, dict)]
    if not facts:
        metrics["claim_coverage_at_5"] = None
        return metrics
    supported_units = metrics.pop("_supported_units", set())
    metrics.pop("_all_supported_units", None)
    covered = 0
    for fact in facts:
        required = {str(unit) for unit in as_list(fact.get("required_units")) if str(unit).strip()}
        if required and required <= supported_units:
            covered += 1
    metrics["claim_coverage_at_5"] = covered / len(facts)
    return metrics


def silver_hits(sequence: list[dict[str, Any]], silver_chunks: list[dict[str, Any]]) -> set[str]:
    hits = set()
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_queries": len(rows),
        "target_unit_recall": mean_defined(row.get("target_unit_recall") for row in rows),
        "silver_hit_at_5": mean_defined(row.get("silver_hit_at_5") for row in rows),
        "silver_recall_at_5": mean_defined(row.get("silver_recall_at_5") for row in rows),
        "mrr": mean_defined(row.get("mrr") for row in rows),
        "claim_coverage_at_5": mean_defined(row.get("claim_coverage_at_5") for row in rows),
        "target_unit_scored": sum(1 for row in rows if row.get("target_unit_recall") is not None),
        "silver_scored": sum(1 for row in rows if row.get("silver_hit_at_5") is not None),
    }


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


def ranges_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    return max(a_start, b_start) <= min(a_end, b_end)


def mean_defined(values: Iterable[Any]) -> float | None:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.4f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Method", "TargetUnitRecall", "SilverHit@5", "SilverRecall@5", "MRR", "ClaimCoverage@5", "TU scored", "Silver scored"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    fmt(row.get("target_unit_recall")),
                    fmt(row.get("silver_hit_at_5")),
                    fmt(row.get("silver_recall_at_5")),
                    fmt(row.get("mrr")),
                    fmt(row.get("claim_coverage_at_5")),
                    str(row.get("target_unit_scored", "")),
                    str(row.get("silver_scored", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525")
    parser.add_argument("--output-dir", default="/home/ruolinsu/signpost/local_backup_before_h200_merge_20260525/target_unit_silver_eval_v1")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    per_query = []
    summaries = []
    for dataset, cfg in DATASETS.items():
        processed_dir = root / "datasets" / "processed" / cfg["processed"]
        outputs_dir = root / "outputs" / cfg["outputs"]
        targets = load_targets(processed_dir)
        for method in METHODS:
            pred_path = outputs_dir / "predictions" / f"{method}.jsonl"
            rows = read_jsonl(pred_path)
            if not rows:
                continue
            method_rows = []
            for row in rows:
                qid = str(row.get("question_id", ""))
                unit_row = targets["units"].get(qid)
                silver_row = targets["silver"].get(qid)
                answer = final_answer(row)
                evidence = silver_metrics(row, silver_row)
                evidence = add_claim_coverage(evidence, unit_row)
                if method in NO_EVIDENCE_SEQUENCE_METHODS:
                    evidence = {"silver_hit_at_5": None, "silver_recall_at_5": None, "mrr": None, "claim_coverage_at_5": None}
                item = {
                    "dataset": dataset,
                    "method": method,
                    "question_id": qid,
                    "target_unit_recall": target_unit_recall(answer, unit_row),
                    **evidence,
                }
                per_query.append(item)
                method_rows.append(item)
            summary = {"dataset": dataset, "method": method, **summarize(method_rows)}
            summaries.append(summary)

    fields = ["dataset", "method", "num_queries", "target_unit_recall", "silver_hit_at_5", "silver_recall_at_5", "mrr", "claim_coverage_at_5", "target_unit_scored", "silver_scored"]
    per_fields = ["dataset", "method", "question_id", "target_unit_recall", "silver_hit_at_5", "silver_recall_at_5", "mrr", "claim_coverage_at_5"]
    write_tsv(out_dir / "method_target_unit_silver_metrics.tsv", summaries, fields)
    write_tsv(out_dir / "per_query_target_unit_silver_metrics.tsv", per_query, per_fields)
    write_json(out_dir / "method_target_unit_silver_metrics.json", summaries)
    table = markdown_table(summaries)
    (out_dir / "table_target_unit_silver_metrics.md").write_text(table, encoding="utf-8")
    print(f"wrote {out_dir} methods={len(summaries)} per_query={len(per_query)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
