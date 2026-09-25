#!/usr/bin/env python3
"""Compare the original name-labelled list-wise judging with the blind re-judge."""
from __future__ import annotations

import csv, glob, json, random, statistics, sys
from collections import defaultdict

SEED = 20260919
ARMS = ["no_vertical_cues", "no_horizontal_cues", "no_semantic_cues",
        "no_online", "no_provenance_cues", "no_offline"]
BASELINES = ["hybrid_rag", "linearrag", "hiprag", "cluerag", "cluerag_prompt_normalized",
             "agrag", "graphrag_r1", "graphrag_r1_original_offline", "memgraphrag", "vanilla_llm"]
DSMAP = {"agriculture": "agriculture", "mixv0": "mixv0",
         "graphrag-bench-medical_q100": "medicalq100",
         "graphrag-bench-novel_q100": "novelq100", "legal_q100": "legalq100"}


def load_blind(root="data/judge/blind"):
    scores, positions = {}, {}
    for path in glob.glob(root + "/*/*.json"):
        row = json.load(open(path, encoding="utf-8"))
        ds = DSMAP.get(row["dataset"], row["dataset"])
        for method, s in row["scores"].items():
            scores[(method, ds, row["question_id"])] = s["llm_total_score"]
            positions[(method, ds, row["question_id"])] = row["position"].get(method)
    return scores, positions


def load_original(path="data/perquery/formal5_per_query.tsv"):
    scores = {}
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        v = r.get("llm_total_score", "")
        if v not in ("", "None", "nan"):
            scores[(r["method"], r["dataset"], r["question_id"])] = float(v)
    return scores


def paired(scores, a, b, rng):
    diffs = [scores[k] - scores[(b,) + k[1:]]
             for k in scores if k[0] == a and (b,) + k[1:] in scores]
    if not diffs:
        return None
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
    w = sum(1 for d in diffs if d > 0)
    l = sum(1 for d in diffs if d < 0)
    return len(diffs), statistics.mean(diffs), boot[99], boot[-100], w, l


def main() -> int:
    blind, positions = load_blind()
    orig = load_original()
    if not blind:
        print("no blind scores yet", file=sys.stderr)
        return 1
    rng = random.Random(SEED)
    nq = len({k[1:] for k in blind})
    print(f"blind: {nq} questions, {len({k[0] for k in blind})} methods, {len(blind)} scores\n")

    print("=== signpost.full minus ablation arm ===")
    print(f"{'arm':24s} {'original (named, fixed order)':>30s}   {'blind (anon, shuffled)':>28s}")
    for arm in ARMS:
        o = paired(orig, "signpost.full", "signpost." + arm, rng)
        b = paired(blind, "signpost.full", "signpost." + arm, rng)
        fo = f"{o[1]:+.2f} [{o[2]:+.2f},{o[3]:+.2f}] {o[4]}/{o[5]}" if o else "-"
        fb = f"{b[1]:+.2f} [{b[2]:+.2f},{b[3]:+.2f}] {b[4]}/{b[5]}" if b else "-"
        print(f"{arm:24s} {fo:>30s}   {fb:>28s}")

    print("\n=== signpost.full minus baseline ===")
    for m in BASELINES:
        o = paired(orig, "signpost.full", m, rng)
        b = paired(blind, "signpost.full", m, rng)
        fo = f"{o[1]:+.2f} [{o[2]:+.2f},{o[3]:+.2f}] {o[4]}/{o[5]}" if o else "-"
        fb = f"{b[1]:+.2f} [{b[2]:+.2f},{b[3]:+.2f}] {b[4]}/{b[5]}" if b else "-"
        print(f"{m:24s} {fo:>30s}   {fb:>28s}")

    print("\n=== mean total score per method ===")
    print(f"{'method':32s} {'original':>9s} {'blind':>9s} {'delta':>8s}")
    methods = sorted({k[0] for k in blind})
    for m in methods:
        ob = [v for k, v in orig.items() if k[0] == m]
        bb = [v for k, v in blind.items() if k[0] == m]
        if not ob or not bb:
            continue
        mo, mb = statistics.mean(ob), statistics.mean(bb)
        print(f"{m:32s} {mo:9.2f} {mb:9.2f} {mb - mo:+8.2f}")

    print("\n=== position effect in the blind run (order was randomized) ===")
    bypos = defaultdict(list)
    for k, v in blind.items():
        p = positions.get(k)
        if p is not None:
            bypos[p].append(v)
    for p in sorted(bypos):
        v = bypos[p]
        print(f"  slot {p:2d}  n={len(v):4d}  mean={statistics.mean(v):.3f}")
    first = bypos.get(0, [])
    rest = [v for p, vs in bypos.items() if p > 0 for v in vs]
    if first and rest:
        print(f"  slot 0 minus slots 1+ : {statistics.mean(first) - statistics.mean(rest):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
