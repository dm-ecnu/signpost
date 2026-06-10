from __future__ import annotations

"""Build method summary rows for cost-quality analysis."""

import argparse
import json
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def build_method_summary(
    *,
    method: str,
    dataset: str,
    query_metrics_path: Path,
    stage_log_path: Path | None = None,
    offline_stages: list[str] | None = None,
) -> dict[str, Any]:
    query_metrics = json.loads(query_metrics_path.read_text(encoding="utf-8"))
    offline = summarize_offline_from_stage_log(stage_log_path, offline_stages or []) if stage_log_path else {}
    return {
        "method": method,
        "dataset": dataset,
        "num_queries": query_metrics.get("num_queries", 0),
        "quality": query_metrics.get("quality", {}),
        "quality_counts": query_metrics.get("quality_counts", {}),
        "cost": query_metrics.get("cost", {}),
        "retrieval": query_metrics.get("retrieval", {}),
        "offline": offline,
    }


def summarize_offline_from_stage_log(path: Path | None, stages: list[str]) -> dict[str, float]:
    if path is None or not path.exists():
        return {"wall_time_seconds": 0.0, "llm_calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "disk_bytes": 0.0}
    rows = list(read_jsonl(path))
    selected = [
        row
        for row in rows
        if (not stages or str(row.get("stage")) in set(stages)) and str(row.get("status", "ok")) == "ok"
    ]
    return {
        "wall_time_seconds": sum(_offline_metric(row, "wall_time_seconds", "offline_wall_time_seconds") for row in selected),
        "llm_calls": sum(_offline_metric(row, "llm_calls", "offline_llm_calls") for row in selected),
        "input_tokens": sum(_offline_metric(row, "input_tokens", "offline_input_tokens") for row in selected),
        "output_tokens": sum(_offline_metric(row, "output_tokens", "offline_output_tokens") for row in selected),
        "disk_bytes": sum(_offline_metric(row, "disk_bytes", "offline_disk_bytes") for row in selected),
    }


def _offline_metric(row: dict[str, Any], field: str, override_field: str) -> float:
    extra = row.get("extra_metrics") if isinstance(row.get("extra_metrics"), dict) else {}
    if override_field in extra:
        return float(extra.get(override_field, 0.0) or 0.0)
    return float(row.get(field, 0.0) or 0.0)


def upsert_summary(path: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            existing = [row for row in payload if isinstance(row, dict)]
    key = (summary.get("dataset"), summary.get("method"))
    rows = [row for row in existing if (row.get("dataset"), row.get("method")) != key]
    rows.append(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update one method summary row for cost_quality.py.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query-metrics", required=True)
    parser.add_argument("--stage-log")
    parser.add_argument("--offline-stage", action="append", default=[])
    parser.add_argument("--output", required=True, help="JSON list path, e.g. outputs/<dataset>/metrics/method_summaries.json")
    args = parser.parse_args()

    summary = build_method_summary(
        method=args.method,
        dataset=args.dataset,
        query_metrics_path=resolve_project_path(args.query_metrics),
        stage_log_path=resolve_project_path(args.stage_log) if args.stage_log else None,
        offline_stages=args.offline_stage,
    )
    rows = upsert_summary(resolve_project_path(args.output), summary)
    print(f"output={resolve_project_path(args.output)} methods={len(rows)} updated={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
