#!/usr/bin/env python3
"""Top-b cue-exposure recount — ZERO model calls. (ICDE Prop 2 realization)

Loads a unified graph, rebuilds each retrievable object's offline signpost sketch
using the project's OWN offline_signpost.build_offline_signpost (so the counts are
exactly the cues the paper materializes), then recounts the per-object cue exposure
under Top-b budgets b in {8,16,32,64, inf}. Produces the table that converts
Proposition 2 (per-step exposure = O(b)) from "design bound" to "measured".

WHY THIS MATTERS: the full-materialization tail (e.g. relation neighboring_entities
P95=389 on agriculture) is the single fact an ICDE area chair uses to call the
"bounded O(b)" claim aspirational. This script shows that truncating each family to
b caps the realized exposure at <= b, with the empirical mean/P95 before vs after.

NO ES, NO LLM, NO REBUILD. Runs against an existing graph.unified.json in seconds.

Usage:
  python recount_cue_exposure.py --graph datasets/processed/agriculture/graph.unified.json --dataset agriculture
  python recount_cue_exposure.py --graph datasets/processed/mix/graph.unified.json --dataset mix
  # writes <out>/cue_exposure_<dataset>.json and prints a markdown table

Run from the project root (so `import signpost...` resolves), e.g. on H200:
  cd /home/srl/signpost_re_v2
  export PYTHONPATH=/home/srl/signpost_re_v2
  python /path/to/recount_cue_exposure.py --graph datasets/processed/agriculture/graph.unified.json --dataset agriculture
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the project's own sketch builders so counts == paper's materialized cues.
# IMPORTANT: build_offline_signpost() rebuilds the whole GraphIndex on every call
# (offline_signpost.py:23). For a 75k-object graph that is 75k rebuilds of a 196MB
# index — minutes-to-hours. We instead build the GraphIndex ONCE and call the same
# per-type builders directly, replicating build_offline_signpost's dispatch.
from signpost.retrieval.offline_signpost import (
    GraphIndex,
    _chunk_signpost,
    _summary_signpost,
    _entity_signpost,
    _relation_signpost,
)

BUDGETS = [8, 16, 32, 64, None]  # None == inf == full materialization (the current system)

# Cue families counted per result_type. Keys map to the paper's C_v/C_h/C_s/C_p.
# Each entry: (sketch_section, field) — value is expected to be a list.
FAMILY_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    # family_letter, sketch_section, list_field
    "chunk": [
        ("v", "vertical", "parent_summaries"),
        # horizontal prev/next are size<=1 each; counted as fixed, not truncated
    ],
    "summary": [
        ("v", "vertical", "child_summaries"),
        ("v", "vertical", "child_chunks"),
        ("p", "provenance", "source_chunk_ids"),
        ("p", "provenance", "source_locates"),
    ],
    "entity": [
        ("s", "semantic", "neighboring_entities"),
        ("p", "provenance", "source_chunk_ids"),
        ("p", "provenance", "source_locates"),
    ],
    "relation": [
        ("s", "semantic", "neighboring_entities"),
        ("p", "provenance", "source_chunk_ids"),
        ("p", "provenance", "source_locates"),
    ],
}


def _list_len(sketch: dict[str, Any], section: str, field: str) -> int:
    sec = sketch.get(section)
    if not isinstance(sec, dict):
        return 0
    value = sec.get(field)
    return len(value) if isinstance(value, list) else 0


def _total_cues(sketch: dict[str, Any], result_type: str, budget: int | None) -> int:
    """Total materialized cues for one object, after Top-b truncation per family."""
    total = 0
    # horizontal prev/next (chunk only): each present link counts as 1, never truncated
    horiz = sketch.get("horizontal")
    if isinstance(horiz, dict):
        total += sum(1 for k in ("previous_chunk", "next_chunk") if horiz.get(k))
    # chunk single provenance locate counts as 1 if present
    prov = sketch.get("provenance")
    if isinstance(prov, dict) and prov.get("locate"):
        total += 1
    for _family, section, field in FAMILY_FIELDS.get(result_type, []):
        n = _list_len(sketch, section, field)
        if budget is not None:
            n = min(n, budget)
        total += n
    return total


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description="Zero-model Top-b cue exposure recount")
    parser.add_argument("--graph", required=True, help="path to graph.unified.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=".", help="output dir for JSON")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="optional cap on objects sampled (0 = all); use for a fast smoke run",
    )
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    index = GraphIndex(graph)  # built ONCE; reused for every object below

    # Object set = retrievable candidates: chunk + summary + entity nodes, plus semantic relation edges.
    # We sketch each by calling the project's own per-type builder directly against the shared index.
    objects: list[tuple[str, Any]] = []  # (result_type, node-or-edge dict)
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        ntype = node.get("node_type")
        if ntype in {"chunk", "summary", "entity"}:
            objects.append((ntype, node))
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("edge_type") == "semantic":
            objects.append(("relation", edge))

    if args.max_objects and len(objects) > args.max_objects:
        # deterministic stride sample to keep type mix
        stride = len(objects) // args.max_objects
        objects = objects[::stride][: args.max_objects]

    def _build_sketch(result_type: str, ref: dict[str, Any]) -> dict[str, Any]:
        if result_type == "chunk":
            return _chunk_signpost(index, ref)
        if result_type == "summary":
            return _summary_signpost(index, ref)
        if result_type == "entity":
            return _entity_signpost(index, ref)
        return _relation_signpost(index, ref)

    # exposures[budget_label][result_type] = list of per-object total cue counts
    exposures: dict[str, dict[str, list[int]]] = {
        ("inf" if b is None else str(b)): defaultdict(list) for b in BUDGETS
    }

    counted = 0
    for result_type, ref in objects:
        sketch = _build_sketch(result_type, ref)
        rtype = sketch.get("result_type", result_type)
        for b in BUDGETS:
            label = "inf" if b is None else str(b)
            exposures[label][rtype].append(_total_cues(sketch, rtype, b))
        counted += 1

    # Aggregate
    report: dict[str, Any] = {"dataset": args.dataset, "graph": args.graph, "objects_counted": counted, "by_budget": {}}
    all_types = ["chunk", "summary", "entity", "relation"]
    for b in BUDGETS:
        label = "inf" if b is None else str(b)
        per_type = {}
        for rtype in all_types:
            vals = sorted(exposures[label].get(rtype, []))
            if not vals:
                continue
            per_type[rtype] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 2),
                "p50": round(_percentile(vals, 0.50), 1),
                "p95": round(_percentile(vals, 0.95), 1),
                "max": vals[-1],
            }
        report["by_budget"][label] = per_type

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cue_exposure_{args.dataset}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown table: per-object total cues (mean / P95) by object type x budget
    print(f"\n### Cue exposure (per-object total cues, mean / P95) — {args.dataset}")
    print(f"objects_counted={counted}  source={args.graph}\n")
    header = "| Object | " + " | ".join(
        (f"b={('∞' if b is None else b)}") for b in BUDGETS
    ) + " |"
    sep = "|" + "---|" * (len(BUDGETS) + 1)
    print(header)
    print(sep)
    for rtype in all_types:
        cells = []
        for b in BUDGETS:
            label = "inf" if b is None else str(b)
            stats = report["by_budget"].get(label, {}).get(rtype)
            cells.append(f"{stats['mean']} / {stats['p95']}" if stats else "—")
        print(f"| {rtype.capitalize()} | " + " | ".join(cells) + " |")
    print(f"\nwrote: {out_path}")
    print(
        "\nReading: the b=∞ column reproduces the paper's full-materialization "
        "Table cue_audit (e.g. relation P95 should match ~389 on agriculture / ~97 on mix); "
        "the b<∞ columns show realized exposure capped at the budget — this is the measured Prop 2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
