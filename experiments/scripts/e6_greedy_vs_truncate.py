#!/usr/bin/env python3
"""E6: greedy (submodular) vs. truncation (positional prefix) depth selection.

ZERO model calls, ZERO Elasticsearch. Turns the (1-1/e) guarantee of the depth
axis from a theorem into a measurement, and extends the depth-budget sweep down
to b in {1,2,4}.

For every retrievable object the script rebuilds sigma(o) with the project's own
offline builders (so the entries are exactly the ones the paper materializes),
then for every truncatable entry list and every budget b compares two selection
rules at IDENTICAL storage (both keep exactly b entries):

  truncate : value[:b], the positional prefix of the offline order (the default,
             SIGNPOST_CUE_SELECT=truncate)
  greedy   : the marginal-coverage prefix of Alg. 2's budgeted branch
             (SIGNPOST_CUE_SELECT=greedy), which carries the (1-1/e) bound

and reports the coverage f_{o,x} each attains as a fraction of the untruncated
f_{o,x}(all). The candidate model (Phi and omega) is copied verbatim from
signpost/retrieval/signpost_variants.py::_greedy_select_list so that what is
measured here is what the serving path actually does.

Usage (on the H200, from the project root so `import signpost...` resolves):
  export PYTHONPATH=/data/srl/signpost_re_v2
  python3 e6_greedy_vs_truncate.py \
      --graph /data/srl/signpost_re_v2/datasets/processed/agriculture/graph.unified.json \
      --dataset agriculture --out ./out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from signpost.retrieval.offline_signpost import (
    GraphIndex,
    _chunk_signpost,
    _summary_signpost,
    _entity_signpost,
    _relation_signpost,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cue_coverage import CueCandidate, coverage_value, select_budgeted_greedy  # noqa: E402

BUDGETS = [1, 2, 4, 8, 16, 32, 64, None]  # None == inf == full materialization

# (family_letter, sketch_section, list_field) per result type -- same table as
# recount_cue_exposure.py, which is what produced the b in {8..64} sweep.
FAMILY_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    "chunk": [("v", "vertical", "parent_summaries")],
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


def candidates_of(value: list[Any]) -> list[CueCandidate]:
    """Verbatim port of signpost_variants._greedy_select_list's candidate model."""
    out: list[CueCandidate] = []
    for pos, entry in enumerate(value):
        if isinstance(entry, dict):
            locate = entry.get("locate") or entry.get("node_id") or entry.get("chunk_id") or str(pos)
            chunks = entry.get("source_chunk_ids") or []
            stable = str(entry.get("node_id") or entry.get("chunk_id") or pos)
        else:
            locate, chunks, stable = str(entry), [], str(entry)
        weight = 1.0 / (1.0 + pos)
        phi = {locate: weight}
        for cid in chunks:
            phi[cid] = max(phi.get(cid, 0.0), weight)
        out.append(CueCandidate(phi=phi, score=weight, stable_id=stable, payload=entry))
    return out


def _list_of(sketch: dict[str, Any], section: str, field: str) -> list[Any]:
    sec = sketch.get(section)
    if not isinstance(sec, dict):
        return []
    value = sec.get(field)
    return value if isinstance(value, list) else []


def _total_cues(sketch: dict[str, Any], result_type: str, budget: int | None) -> int:
    total = 0
    horiz = sketch.get("horizontal")
    if isinstance(horiz, dict):
        total += sum(1 for k in ("previous_chunk", "next_chunk") if horiz.get(k))
    prov = sketch.get("provenance")
    if isinstance(prov, dict) and prov.get("locate"):
        total += 1
    for _family, section, field in FAMILY_FIELDS.get(result_type, []):
        n = len(_list_of(sketch, section, field))
        if budget is not None:
            n = min(n, budget)
        total += n
    return total


def _percentile(sorted_vals: list[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="./out")
    ap.add_argument("--max-objects", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    index = GraphIndex(graph)
    print(f"[{args.dataset}] graph loaded + indexed in {time.time() - t0:.1f}s", flush=True)

    objects: list[tuple[str, Any]] = []
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and node.get("node_type") in {"chunk", "summary", "entity"}:
            objects.append((node["node_type"], node))
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("edge_type") == "semantic":
            objects.append(("relation", edge))
    if args.max_objects and len(objects) > args.max_objects:
        stride = max(1, len(objects) // args.max_objects)
        objects = objects[::stride][: args.max_objects]

    def build(result_type: str, ref: dict[str, Any]) -> dict[str, Any]:
        if result_type == "chunk":
            return _chunk_signpost(index, ref)
        if result_type == "summary":
            return _summary_signpost(index, ref)
        if result_type == "entity":
            return _entity_signpost(index, ref)
        return _relation_signpost(index, ref)

    labels = ["inf" if b is None else str(b) for b in BUDGETS]
    exposures: dict[str, dict[str, list[int]]] = {lb: defaultdict(list) for lb in labels}
    # cov[label][family] = [sum f_greedy, sum f_trunc, sum f_full, n_lists, n_differ]
    cov: dict[str, dict[str, list[float]]] = {
        lb: defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0]) for lb in labels if lb != "inf"
    }

    counted = 0
    for result_type, ref in objects:
        sketch = build(result_type, ref)
        rtype = sketch.get("result_type", result_type)
        for b, lb in zip(BUDGETS, labels):
            exposures[lb][rtype].append(_total_cues(sketch, rtype, b))
        for family, section, field in FAMILY_FIELDS.get(rtype, []):
            value = _list_of(sketch, section, field)
            if len(value) <= 1:
                continue
            cands = candidates_of(value)
            f_full = coverage_value(cands)
            if f_full <= 0.0:
                continue
            for b, lb in zip(BUDGETS, labels):
                if b is None or len(value) <= b:
                    continue
                trunc = cands[:b]
                greedy = select_budgeted_greedy(cands, b)
                if not greedy:
                    greedy = trunc
                slot = cov[lb][family]
                slot[0] += coverage_value(greedy)
                slot[1] += coverage_value(trunc)
                slot[2] += f_full
                slot[3] += 1
                if [c.stable_id for c in greedy] != [c.stable_id for c in trunc]:
                    slot[4] += 1
        counted += 1
        if counted % 20000 == 0:
            print(f"  ...{counted}/{len(objects)} objects ({time.time() - t0:.0f}s)", flush=True)

    report: dict[str, Any] = {
        "dataset": args.dataset,
        "graph": args.graph,
        "objects_counted": counted,
        "by_budget": {},
        "coverage": {},
    }
    for lb in labels:
        per_type = {}
        for rtype in ("chunk", "summary", "entity", "relation"):
            vals = sorted(exposures[lb].get(rtype, []))
            if not vals:
                continue
            per_type[rtype] = {
                "n": len(vals),
                "mean": round(sum(vals) / len(vals), 2),
                "p50": round(_percentile(vals, 0.50), 1),
                "p95": round(_percentile(vals, 0.95), 1),
                "max": vals[-1],
            }
        report["by_budget"][lb] = per_type
    for lb, fams in cov.items():
        report["coverage"][lb] = {
            fam: {
                "greedy_over_full": round(v[0] / v[2], 4) if v[2] else None,
                "truncate_over_full": round(v[1] / v[2], 4) if v[2] else None,
                "greedy_over_truncate": round(v[0] / v[1], 4) if v[1] else None,
                "lists": v[3],
                "lists_differing": v[4],
            }
            for fam, v in sorted(fams.items())
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"e6_{args.dataset}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n### E6 {args.dataset}  (objects={counted}, {time.time() - t0:.0f}s)")
    print("\ncoverage attained / untruncated coverage   (greedy | truncate | ratio | lists | differ)")
    for lb in [x for x in labels if x != "inf"]:
        for fam, d in report["coverage"].get(lb, {}).items():
            if not d["lists"]:
                continue
            print(f"  b={lb:<3} {fam}: {d['greedy_over_full']:.4f} | {d['truncate_over_full']:.4f} | "
                  f"{d['greedy_over_truncate']:.4f} | {d['lists']} | {d['lists_differing']}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
