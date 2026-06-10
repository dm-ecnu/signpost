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
    "legal": {"processed": "legal", "outputs": "legal"},
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

SILVER_METHODS = {
    "hiprag",
    "graphrag_r1",
    "signpost.full",
}


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


def target_unit_details(answer: str, unit_row: dict[str, Any] | None) -> dict[str, Any]:
    units = [unit for unit in as_list((unit_row or {}).get("target_units")) if isinstance(unit, dict)]
    required = [unit for unit in units if unit.get("required", True)]
    if not required:
        return {"target_unit_recall": None, "covered_units": [], "missed_units": []}
    answer_norm = normalize_answer(answer)
    covered = []
    missed = []
    for unit in required:
        candidates = [str(unit.get("text", ""))]
        candidates.extend(str(alias) for alias in as_list(unit.get("aliases")))
        item = {
            "unit_id": str(unit.get("unit_id") or unit.get("id") or unit.get("text") or ""),
            "text": str(unit.get("text") or ""),
        }
        if any(unit_phrase_matches(candidate, answer_norm) for candidate in candidates):
            covered.append(item)
        else:
            missed.append(item)
    return {
        "target_unit_recall": len(covered) / len(required),
        "covered_units": covered,
        "missed_units": missed,
    }


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
    if len(unit_tokens) <= 2:
        return unit_tokens <= answer_tokens
    return len(unit_tokens & answer_tokens) / len(unit_tokens) >= 0.75


def evidence_chunks(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in as_list(row.get("evidence_chunks")) if isinstance(item, dict)]


def silver_metrics(row: dict[str, Any], silver_row: dict[str, Any] | None, unit_row: dict[str, Any] | None) -> dict[str, Any]:
    silver_chunks = [item for item in as_list((silver_row or {}).get("silver_chunks")) if isinstance(item, dict)]
    if not silver_chunks:
        return empty_silver()
    sequence = evidence_chunks(row)
    if not sequence:
        return empty_silver(num_evidence_chunks=0)
    top5 = sequence[:5]
    silver_ids = {str(item.get("chunk_id", "")) for item in silver_chunks if item.get("chunk_id")}
    hit_ids = silver_hits(top5, silver_chunks)
    ranks = [idx for idx, item in enumerate(sequence, start=1) if silver_hits([item], silver_chunks)]
    supported_units = set()
    for silver in silver_chunks:
        if str(silver.get("chunk_id", "")) in hit_ids:
            supported_units.update(str(unit) for unit in as_list(silver.get("supports")) if str(unit).strip())
    facts = [item for item in as_list((unit_row or {}).get("facts")) if isinstance(item, dict)]
    claim_coverage = None
    if facts:
        covered = 0
        for fact in facts:
            required = {str(unit) for unit in as_list(fact.get("required_units")) if str(unit).strip()}
            if required and required <= supported_units:
                covered += 1
        claim_coverage = covered / len(facts)
    return {
        "silver_hit_at_5": 1.0 if hit_ids else 0.0,
        "silver_recall_at_5": len(hit_ids) / len(silver_ids) if silver_ids else None,
        "mrr": 1.0 / ranks[0] if ranks else 0.0,
        "claim_coverage_at_5": claim_coverage,
        "num_evidence_chunks": len(sequence),
        "hit_silver_chunks": sorted(hit_ids),
        "missed_silver_chunks": sorted(silver_ids - hit_ids),
    }


def empty_silver(num_evidence_chunks: int | None = None) -> dict[str, Any]:
    return {
        "silver_hit_at_5": None,
        "silver_recall_at_5": None,
        "mrr": None,
        "claim_coverage_at_5": None,
        "num_evidence_chunks": num_evidence_chunks,
        "hit_silver_chunks": [],
        "missed_silver_chunks": [],
    }


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
        "num_evidence_chunks": mean_defined(row.get("num_evidence_chunks") for row in rows),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="v2 project root; used for both targets and outputs unless overridden")
    parser.add_argument("--targets-root", default=None, help="root containing datasets/processed/<dataset>/llm_target_units.jsonl and llm_silver_chunks.jsonl")
    parser.add_argument("--outputs-root", default=None, help="root containing outputs/<dataset>/predictions/<method>.jsonl")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated dataset keys from DATASETS, e.g. mixv0")
    parser.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Add a dynamic dataset mapping as key=processed:outputs, e.g. legal_q100=legal_q100:legal_q100.",
    )
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method names; missing prediction files are skipped")
    parser.add_argument("--output-dir", default="outputs/target_unit_silver_eval_v2")
    args = parser.parse_args()

    root = Path(args.root)
    targets_root = Path(args.targets_root) if args.targets_root else root
    outputs_root = Path(args.outputs_root) if args.outputs_root else root
    out_dir = Path(args.output_dir)
    dataset_map = dict(DATASETS)
    for spec in args.dataset_spec:
        key, _, rest = spec.partition("=")
        processed, _, outputs = rest.partition(":")
        key = key.strip()
        processed = processed.strip()
        outputs = outputs.strip()
        if not key or not processed or not outputs:
            raise SystemExit(f"invalid --dataset-spec={spec!r}; expected key=processed:outputs")
        dataset_map[key] = {"processed": processed, "outputs": outputs}
    requested_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    requested_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    per_query = []
    summaries = []
    skipped: list[dict[str, Any]] = []
    for dataset in requested_datasets:
        if dataset not in dataset_map:
            raise SystemExit(f"unknown dataset key: {dataset}; choices={', '.join(dataset_map)}")
        cfg = dataset_map[dataset]
        processed_dir = targets_root / "datasets" / "processed" / cfg["processed"]
        outputs_dir = outputs_root / "outputs" / cfg["outputs"]
        targets = load_targets(processed_dir)
        if not targets["units"]:
            skipped.append({"dataset": dataset, "method": "*", "reason": f"missing target units under {processed_dir}"})
        if not targets["silver"]:
            skipped.append({"dataset": dataset, "method": "*", "reason": f"missing silver chunks under {processed_dir}"})
        for method in requested_methods:
            pred_path = outputs_dir / "predictions" / f"{method}.jsonl"
            rows = read_jsonl(pred_path)
            if not rows:
                skipped.append({"dataset": dataset, "method": method, "reason": f"missing or empty {pred_path}"})
                continue
            method_rows = []
            for row in rows:
                qid = str(row.get("question_id", ""))
                unit_row = targets["units"].get(qid)
                silver_row = targets["silver"].get(qid)
                answer = final_answer(row)
                target = target_unit_details(answer, unit_row)
                evidence = silver_metrics(row, silver_row, unit_row) if method in SILVER_METHODS else empty_silver()
                item = {
                    "dataset": dataset,
                    "method": method,
                    "question_id": qid,
                    **target,
                    **evidence,
                }
                per_query.append(item)
                method_rows.append(item)
            summaries.append({"dataset": dataset, "method": method, **summarize(method_rows)})

    fields = [
        "dataset",
        "method",
        "num_queries",
        "target_unit_recall",
        "silver_hit_at_5",
        "silver_recall_at_5",
        "mrr",
        "claim_coverage_at_5",
        "num_evidence_chunks",
        "target_unit_scored",
        "silver_scored",
    ]
    per_fields = [
        "dataset",
        "method",
        "question_id",
        "target_unit_recall",
        "silver_hit_at_5",
        "silver_recall_at_5",
        "mrr",
        "claim_coverage_at_5",
        "num_evidence_chunks",
    ]
    write_tsv(out_dir / "method_target_unit_silver_metrics.tsv", summaries, fields)
    write_tsv(out_dir / "per_query_target_unit_silver_metrics.tsv", per_query, per_fields)
    write_json(out_dir / "method_target_unit_silver_metrics.json", summaries)
    write_json(out_dir / "per_query_target_unit_silver_details.json", per_query)
    write_json(out_dir / "skipped_inputs.json", skipped)
    table = target_silver_table(summaries)
    (out_dir / "table_target_unit_silver_metrics.md").write_text(table, encoding="utf-8")
    print(f"wrote {out_dir} methods={len(summaries)} per_query={len(per_query)} skipped={len(skipped)}")
    return 0


def target_silver_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Dataset", "Method", "TargetUnitRecall", "SilverHit@5", "SilverRecall@5", "MRR", "ClaimCoverage@5", "EvidenceN", "Queries"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("method", "")),
                    fmt(row.get("target_unit_recall")),
                    fmt(row.get("silver_hit_at_5")),
                    fmt(row.get("silver_recall_at_5")),
                    fmt(row.get("mrr")),
                    fmt(row.get("claim_coverage_at_5")),
                    fmt(row.get("num_evidence_chunks")),
                    str(row.get("num_queries", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
