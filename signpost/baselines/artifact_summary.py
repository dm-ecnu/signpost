from __future__ import annotations

"""Write baseline-owned run_metrics.json and run_status.json files."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from signpost.config.context import resolve_project_path
from signpost.parsing.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Write baseline run artifact summaries.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--query-metrics", required=True)
    parser.add_argument("--stage-log", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--graph-metrics")
    args = parser.parse_args()

    artifact_dir = resolve_project_path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    query_metrics = _read_json(resolve_project_path(args.query_metrics))
    graph_metrics = _read_json(resolve_project_path(args.graph_metrics)) if args.graph_metrics else {}
    stages = _matching_stages(resolve_project_path(args.stage_log), args.method)
    measured_online_wall = sum(float(row.get("wall_time_seconds", 0.0) or 0.0) for row in stages if row.get("method_scope") == "online_query")
    offline_wall = float(graph_metrics.get("offline_wall_time_seconds", 0.0) or 0.0)
    online_wall = max(0.0, measured_online_wall - offline_wall)
    eval_wall = sum(float(row.get("wall_time_seconds", 0.0) or 0.0) for row in stages if row.get("method_scope") == "evaluation")
    cost_totals = query_metrics.get("cost", {}).get("totals", {}) if isinstance(query_metrics.get("cost"), dict) else {}
    run_metrics = {
        "dataset": args.dataset,
        "method": args.method,
        "num_queries": query_metrics.get("num_queries", 0),
        "offline_wall_time_seconds": offline_wall,
        "online_wall_time_seconds": online_wall,
        "evaluation_wall_time_seconds": eval_wall,
        "total_wall_time_seconds": offline_wall + online_wall + eval_wall,
        "offline_embedding_calls": graph_metrics.get("offline_embedding_calls", 0.0),
        "online_llm_calls": cost_totals.get("online_llm_calls", 0.0),
        "online_input_tokens": cost_totals.get("input_tokens", 0.0),
        "online_output_tokens": cost_totals.get("output_tokens", 0.0),
        "online_total_tokens": cost_totals.get("total_tokens", 0.0),
        "tool_calls": cost_totals.get("tool_calls", 0.0),
        "graph_ppr_calls": cost_totals.get("graph_ppr_calls", 0.0),
        "retrieved_chunks": cost_totals.get("retrieved_chunks", 0.0),
        "graph_index": graph_metrics,
        "disk_bytes": _disk_bytes(artifact_dir),
    }
    status = {
        "dataset": args.dataset,
        "method": args.method,
        "status": "completed",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {
            "run_metrics": str(artifact_dir / "run_metrics.json"),
            "run_status": str(artifact_dir / "run_status.json"),
            "graph_metrics": str(resolve_project_path(args.graph_metrics)) if args.graph_metrics else None,
        },
    }
    (artifact_dir / "run_metrics.json").write_text(json.dumps(run_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={artifact_dir / 'run_metrics.json'} status={artifact_dir / 'run_status.json'}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _matching_stages(path: Path, method: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [row for row in read_jsonl(path) if row.get("method") == method and str(row.get("status", "ok")) == "ok"]


def _disk_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
