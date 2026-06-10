from __future__ import annotations

"""Cost-quality derived metrics for ICDE experiment tables."""

import argparse
import json
from pathlib import Path
from typing import Any

from signpost.benchmark.stats import safe_div
from signpost.config.context import resolve_project_path


DEFAULT_WORKLOAD_SIZES = [10, 50, 100, 500, 1000, 5000, 10000]


def summarize_methods(methods: list[dict[str, Any]], *, workload_sizes: list[int] | None = None) -> dict[str, Any]:
    sizes = workload_sizes or DEFAULT_WORKLOAD_SIZES
    normalized = [_normalize_method(row) for row in methods]
    return {
        "methods": normalized,
        "pareto": pareto_frontier(normalized),
        "amortized": {row["method"]: amortized_costs(row, sizes) for row in normalized},
        "pairwise": pairwise_cost_quality(normalized),
    }


def amortized_costs(method: dict[str, Any], workload_sizes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for n in workload_sizes:
        rows.append(
            {
                "queries": n,
                "amortized_time_seconds": method["offline_wall_time_seconds"] / n + method["online_latency_seconds_mean"],
                "amortized_tokens": method["offline_tokens"] / n + method["online_tokens_mean"],
                "amortized_llm_calls": method["offline_llm_calls"] / n + method["online_llm_calls_mean"],
            }
        )
    return rows


def pairwise_cost_quality(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for baseline in methods:
        for method in methods:
            if method["method"] == baseline["method"]:
                continue
            quality_delta = method["quality_score"] - baseline["quality_score"]
            offline_delta = method["offline_tokens"] - baseline["offline_tokens"]
            online_delta = method["online_tokens_mean"] - baseline["online_tokens_mean"]
            total_delta_at_measured_n = offline_delta + method["num_queries"] * online_delta
            rows.append(
                {
                    "method": method["method"],
                    "baseline": baseline["method"],
                    "delta_quality": quality_delta,
                    "delta_offline_tokens": offline_delta,
                    "delta_online_tokens_per_query": online_delta,
                    "cost_per_extra_correct_tokens": safe_div(total_delta_at_measured_n, method["num_queries"] * quality_delta),
                    "break_even_queries_tokens": break_even_queries(
                        method["offline_tokens"],
                        baseline["offline_tokens"],
                        method["online_tokens_mean"],
                        baseline["online_tokens_mean"],
                    ),
                    "break_even_queries_time": break_even_queries(
                        method["offline_wall_time_seconds"],
                        baseline["offline_wall_time_seconds"],
                        method["online_latency_seconds_mean"],
                        baseline["online_latency_seconds_mean"],
                    ),
                }
            )
    return rows


def pareto_frontier(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in methods:
        dominated = False
        for other in methods:
            if other["method"] == candidate["method"]:
                continue
            no_worse = other["quality_score"] >= candidate["quality_score"] and other["online_tokens_mean"] <= candidate["online_tokens_mean"]
            strictly_better = other["quality_score"] > candidate["quality_score"] or other["online_tokens_mean"] < candidate["online_tokens_mean"]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append({"method": candidate["method"], "quality_score": candidate["quality_score"], "online_tokens_mean": candidate["online_tokens_mean"]})
    return sorted(frontier, key=lambda row: (row["online_tokens_mean"], -row["quality_score"]))


def break_even_queries(method_offline: float, baseline_offline: float, method_online: float, baseline_online: float) -> float | None:
    denominator = baseline_online - method_online
    if denominator <= 0:
        return None
    return (method_offline - baseline_offline) / denominator


def _normalize_method(row: dict[str, Any]) -> dict[str, Any]:
    quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
    cost = row.get("cost", {}) if isinstance(row.get("cost"), dict) else {}
    means = cost.get("means", {}) if isinstance(cost.get("means"), dict) else {}
    totals = cost.get("totals", {}) if isinstance(cost.get("totals"), dict) else {}
    offline = row.get("offline", {}) if isinstance(row.get("offline"), dict) else {}
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    method = {
        "method": str(row.get("method") or row.get("name") or metadata.get("method") or ""),
        "dataset": str(row.get("dataset") or metadata.get("dataset") or ""),
        "num_queries": int(row.get("num_queries") or row.get("quality_counts", {}).get("num_scored") or 0),
        "quality_score": float(row.get("quality_score", quality.get("exact_match", quality.get("f1", 0.0))) or 0.0),
        "online_tokens_mean": float(row.get("online_tokens_mean", means.get("total_tokens", 0.0)) or 0.0),
        "online_latency_seconds_mean": float(row.get("online_latency_seconds_mean", means.get("latency_seconds", 0.0)) or 0.0),
        "online_llm_calls_mean": float(row.get("online_llm_calls_mean", means.get("online_llm_calls", means.get("llm_calls", 0.0))) or 0.0),
        "online_tokens_total": float(row.get("online_tokens_total", totals.get("total_tokens", 0.0)) or 0.0),
        "offline_tokens": float(row.get("offline_tokens", offline.get("input_tokens", 0.0) + offline.get("output_tokens", 0.0)) or 0.0),
        "offline_llm_calls": float(row.get("offline_llm_calls", offline.get("llm_calls", 0.0)) or 0.0),
        "offline_wall_time_seconds": float(row.get("offline_wall_time_seconds", offline.get("wall_time_seconds", 0.0)) or 0.0),
    }
    if not method["method"]:
        raise ValueError(f"method row is missing method/name: {row}")
    return method


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute amortized cost, break-even, and Pareto metrics from method summaries.")
    parser.add_argument("--methods", required=True, help="JSON file containing a list of method summary objects.")
    parser.add_argument("--workload-sizes", type=int, nargs="*", default=DEFAULT_WORKLOAD_SIZES)
    parser.add_argument("--output")
    args = parser.parse_args()

    methods = json.loads(resolve_project_path(args.methods).read_text(encoding="utf-8"))
    if not isinstance(methods, list):
        raise ValueError("--methods must point to a JSON list")
    result = summarize_methods(methods, workload_sizes=args.workload_sizes)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"output={output} methods={len(result['methods'])}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
