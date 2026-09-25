#!/usr/bin/env python3
"""T0-2 (cost-model calibration) + T0-6 (object-access skew) from frozen topb8 predictions.

Pure log analysis: no ES, no LLM, no GPU. Reads the 2026-06-06 topb8 ablation
prediction JSONL and reports, per dataset x variant:
  - per-query cost: llm_calls, tokens, latency decomposition  (feeds the cost model)
  - object-access distribution over retrieved_chunks / evidence_chunks (feeds skew)
  - trace event_type histogram (tells us what a cue-follow analysis could use)
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data/topb8-outputs"


def load(dataset: str, variant: str) -> list[dict]:
    rows = []
    parts = DATA / dataset / "predictions" / "parts"
    for f in sorted(parts.glob(f"{variant}.part_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def cost_row(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "llm_calls": mean(r.get("llm_calls") for r in rows),
        "ks_calls": mean(r.get("knowledge_search_calls") for r in rows),
        "read_calls": mean(r.get("read_file_calls") for r in rows),
        "in_tok": mean(r.get("input_tokens") for r in rows),
        "out_tok": mean(r.get("output_tokens") for r in rows),
        "tot_tok": mean(r.get("total_tokens") for r in rows),
        "lat": mean(r.get("latency_seconds") for r in rows),
        "retr_lat": mean(r.get("retrieval_latency_seconds") for r in rows),
        "read_lat": mean(r.get("read_file_latency_seconds") for r in rows),
        "gen_lat": mean(r.get("agent_reasoning_latency_seconds") for r in rows),
    }


def obj_keys(r: dict, field: str) -> list[str]:
    out = []
    for c in r.get(field) or []:
        if not isinstance(c, dict):
            continue
        # prefer a stable source-object identity: file+line span, else chunk/doc id
        fn, s, e = c.get("file_name"), c.get("start_line"), c.get("end_line")
        if fn and s is not None:
            out.append(f"{fn}:{s}-{e}")
        elif c.get("chunk_id"):
            out.append(str(c["chunk_id"]))
        elif c.get("doc_id"):
            out.append(str(c["doc_id"]))
    return out


def skew(rows: list[dict], field: str) -> dict:
    per_query = [set(obj_keys(r, field)) for r in rows]
    cnt = Counter()
    for s in per_query:
        cnt.update(s)  # distinct objects per query; repeats within a query not double counted
    if not cnt:
        return {}
    accesses = sum(cnt.values())
    freqs = sorted(cnt.values(), reverse=True)
    n_obj = len(freqs)

    def head_share(p: float) -> float:
        k = max(1, int(round(n_obj * p)))
        return sum(freqs[:k]) / accesses

    return {
        "queries": len(rows),
        "distinct_objects": n_obj,
        "total_accesses": accesses,
        "accesses_per_query": accesses / len(rows),
        "mean_queries_per_object": accesses / n_obj,
        "max_queries_per_object": freqs[0],
        "share_top1pct": head_share(0.01),
        "share_top5pct": head_share(0.05),
        "share_top10pct": head_share(0.10),
        "share_top20pct": head_share(0.20),
        "objects_touched_once_frac": sum(1 for f in freqs if f == 1) / n_obj,
        "reused_objects_frac": sum(1 for f in freqs if f > 1) / n_obj,
        "accesses_from_reused_frac": sum(f for f in freqs if f > 1) / accesses,
    }


def main() -> None:
    datasets = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    variants = ["signpost.full", "signpost.no_offline", "signpost.no_provenance_cues"]

    print("=" * 110)
    print("T0-2  per-query cost (frozen topb8 run, 2026-06-06)")
    print("=" * 110)
    hdr = f"{'dataset':<28}{'variant':<30}{'n':>4}{'calls':>7}{'KS':>6}{'read':>6}{'in_tok':>9}{'out_tok':>9}{'lat':>8}{'retr':>8}{'gen':>7}"
    print(hdr)
    cache: dict[tuple[str, str], list[dict]] = {}
    for ds in datasets:
        for v in variants:
            rows = load(ds, v)
            if not rows:
                continue
            cache[(ds, v)] = rows
            c = cost_row(rows)
            print(f"{ds:<28}{v:<30}{c['n']:>4}{c['llm_calls']:>7.2f}{c['ks_calls']:>6.1f}"
                  f"{c['read_calls']:>6.1f}{c['in_tok']:>9.0f}{c['out_tok']:>9.0f}"
                  f"{c['lat']:>8.1f}{c['retr_lat']:>8.1f}{c['gen_lat']:>7.1f}")

    print()
    print("=" * 110)
    print("T0-6  object-access skew  (signpost.full; identity = file:start-end, else chunk/doc id)")
    print("=" * 110)
    for field in ("retrieved_chunks", "evidence_chunks"):
        print(f"\n--- field: {field} ---")
        print(f"{'dataset':<28}{'q':>4}{'objs':>7}{'acc':>7}{'acc/q':>7}{'q/obj':>7}{'max':>5}"
              f"{'top1%':>7}{'top5%':>7}{'top10%':>7}{'reuse_obj':>10}{'acc_reuse':>10}")
        for ds in datasets:
            rows = cache.get((ds, "signpost.full"))
            if not rows:
                continue
            s = skew(rows, field)
            if not s:
                print(f"{ds:<28} (no {field})")
                continue
            print(f"{ds:<28}{s['queries']:>4}{s['distinct_objects']:>7}{s['total_accesses']:>7}"
                  f"{s['accesses_per_query']:>7.1f}{s['mean_queries_per_object']:>7.2f}"
                  f"{s['max_queries_per_object']:>5}{s['share_top1pct']:>7.1%}"
                  f"{s['share_top5pct']:>7.1%}{s['share_top10pct']:>7.1%}"
                  f"{s['reused_objects_frac']:>10.1%}{s['accesses_from_reused_frac']:>10.1%}")

    print()
    print("=" * 110)
    print("trace event_type histogram (signpost.full, agriculture)")
    print("=" * 110)
    rows = cache.get(("agriculture", "signpost.full")) or []
    ev = Counter()
    for r in rows:
        for t in r.get("trace") or []:
            if isinstance(t, dict):
                ev[t.get("event_type")] += 1
    for k, v in ev.most_common():
        print(f"  {k:<34}{v:>8}  ({v / max(1, len(rows)):.2f}/query)")


if __name__ == "__main__":
    main()
