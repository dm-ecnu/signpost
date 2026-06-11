from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "datasets" / "processed"
TARGETS_ROOT = ROOT / "outputs" / "suffer10_20260609_final_metrics_targets"

SPECS = {
    "agriculture_suffer10_20260609": Path("/data/srl/handoffs/h200_eval_formal5_20260604/work/target_silver_root/datasets/processed/agriculture"),
    "mix_suffer10_20260609": Path("/data/srl/handoffs/h200_eval_formal5_20260604/work/target_silver_root/datasets/processed/mix"),
    "legal_suffer10_20260609": Path("/data/srl/handoffs/h200_eval_formal5_20260604/work/target_silver_root/datasets/processed/legal_q100"),
    "graphrag_bench_medical_suffer10_20260609": Path("/data/srl/handoffs/h200_eval_formal5_20260604/work/target_silver_root/datasets/processed/graphrag-bench-medical_q100"),
    "graphrag_bench_novel_suffer10_20260609": Path("/data/srl/handoffs/h200_eval_formal5_20260604/work/target_silver_root/datasets/processed/graphrag-bench-novel_q100"),
    "musique_suffer10_20260609": Path("/data/srl/0607musique/work/signpost_re_v2/datasets/processed/musique_q100"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_question_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("question_id")): row for row in rows}


def subset_rows(source_dir: Path, file_name: str, question_ids: list[str]) -> list[dict[str, Any]]:
    source = by_question_id(read_jsonl(source_dir / file_name))
    missing = [qid for qid in question_ids if qid not in source]
    if missing:
        raise RuntimeError(f"{source_dir / file_name} missing {len(missing)} ids: {missing[:5]}")
    return [source[qid] for qid in question_ids]


def question_ids_for_dataset(processed_dir: Path) -> list[str]:
    for name in ("queries.jsonl", "test.jsonl", "questions.jsonl"):
        path = processed_dir / name
        if path.exists():
            return [str(row.get("question_id")) for row in read_jsonl(path)]
    raise RuntimeError(f"cannot find query file under {processed_dir}")


def final_metrics_target_rows(target_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_units = []
    claim_units = []
    target_entities = []
    for row in target_rows:
        qid = str(row.get("question_id"))
        units = []
        for unit in row.get("target_units") or []:
            if not isinstance(unit, dict):
                continue
            item = dict(unit)
            units.append(item)
        target_units.append({"question_id": qid, "target_units": units})

        claims = []
        unit_by_id = {str(unit.get("unit_id")): unit for unit in units}
        for fact in row.get("facts") or []:
            if not isinstance(fact, dict):
                continue
            required = [str(unit_id) for unit_id in fact.get("required_units") or []]
            claims.append(
                {
                    "claim_id": fact.get("fact_id"),
                    "description": fact.get("description", ""),
                    "target_units": [str(unit_by_id.get(unit_id, {}).get("text") or unit_id) for unit_id in required],
                    "target_unit_ids": required,
                }
            )
        claim_units.append({"question_id": qid, "claims": claims})

        target_entities.append({"question_id": qid, "target_entities": []})
    return target_units, claim_units, target_entities


def silver_for_final_metrics(silver_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units_by_qid = {
        str(row.get("question_id")): {
            str(unit.get("unit_id")): str(unit.get("text", ""))
            for unit in row.get("target_units") or []
            if isinstance(unit, dict)
        }
        for row in target_rows
    }
    converted = []
    for row in silver_rows:
        qid = str(row.get("question_id"))
        unit_map = units_by_qid.get(qid, {})
        silver_chunks = []
        for chunk in row.get("silver_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            item = dict(chunk)
            supports = [str(unit_id) for unit_id in item.get("supports") or []]
            item.setdefault("matched_units", [unit_map.get(unit_id, unit_id) for unit_id in supports])
            item.setdefault("matched_numbers", [])
            silver_chunks.append(item)
        converted.append({"question_id": qid, "silver_chunks": silver_chunks})
    return converted


def main() -> int:
    for dataset_dir_name, source_dir in SPECS.items():
        processed_dir = PROCESSED / dataset_dir_name
        question_ids = question_ids_for_dataset(processed_dir)
        target_rows = subset_rows(source_dir, "llm_target_units.jsonl", question_ids)
        silver_rows = subset_rows(source_dir, "llm_silver_chunks.jsonl", question_ids)
        combined_rows = subset_rows(source_dir, "llm_targets_silver.jsonl", question_ids)

        write_jsonl(processed_dir / "llm_target_units.jsonl", target_rows)
        write_jsonl(processed_dir / "llm_silver_chunks.jsonl", silver_rows)
        write_jsonl(processed_dir / "llm_targets_silver.jsonl", combined_rows)

        targets_dir = TARGETS_ROOT / dataset_dir_name
        final_target_units, claim_units, target_entities = final_metrics_target_rows(target_rows)
        write_jsonl(targets_dir / "target_units.jsonl", final_target_units)
        write_jsonl(targets_dir / "claim_units.jsonl", claim_units)
        write_jsonl(targets_dir / "target_entities.jsonl", target_entities)
        write_jsonl(targets_dir / "silver_evidence_chunks.jsonl", silver_for_final_metrics(silver_rows, target_rows))

        print(f"{dataset_dir_name}: wrote {len(question_ids)} target/silver rows")
    print(f"final_metrics_targets={TARGETS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
