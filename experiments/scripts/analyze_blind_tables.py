#!/usr/bin/env python3
"""Per-dataset blind-judge tables: S_LLM and J@7, original vs blind."""
from __future__ import annotations
import csv, glob, json, random, statistics
from collections import defaultdict

SEED = 20260919
DSMAP = {"agriculture": "agriculture", "mixv0": "mixv0",
         "graphrag-bench-medical_q100": "medicalq100",
         "graphrag-bench-novel_q100": "novelq100", "legal_q100": "legalq100"}
ORDER = ["agriculture", "medicalq100", "novelq100", "mixv0", "legalq100"]
SHOW = ["vanilla_llm", "hiprag", "memgraphrag", "graphrag_r1", "graphrag_r1_original_offline",
        "agrag", "linearrag", "cluerag", "cluerag_prompt_normalized", "hybrid_rag",
        "signpost.full", "signpost.no_vertical_cues", "signpost.no_horizontal_cues",
        "signpost.no_semantic_cues", "signpost.no_online", "signpost.no_provenance_cues",
        "signpost.no_offline"]

blind, seen = {}, defaultdict(set)
for p in glob.glob("data/judge/blind/*/*.json"):
    r = json.load(open(p, encoding="utf-8"))
    ds = DSMAP.get(r["dataset"], r["dataset"])
    for m, s in r["scores"].items():
        blind[(m, ds, r["question_id"])] = s["llm_total_score"]
        seen[ds].add(r["question_id"])
orig = {}
for r in csv.DictReader(open("data/perquery/formal5_per_query.tsv", encoding="utf-8"), delimiter="\t"):
    v = r.get("llm_total_score", "")
    if v not in ("", "None", "nan"):
        orig[(r["method"], r["dataset"], r["question_id"])] = float(v)

print("questions per dataset (blind):", {d: len(q) for d, q in sorted(seen.items())})

def cell(store, m, ds):
    v = [x for k, x in store.items() if k[0] == m and k[1] == ds]
    if not v: return None, None
    return statistics.mean(v), 100.0 * sum(1 for x in v if x >= 7.0) / len(v)

for metric, idx in (("S_LLM", 0), ("J@7", 1)):
    print(f"\n=== {metric}: original -> blind ===")
    print(f"{'method':32s} " + " ".join(f"{d:>17s}" for d in ORDER) + f" {'macro4':>8s}")
    for m in SHOW:
        cells, macro_o, macro_b = [], [], []
        for d in ORDER:
            o = cell(orig, m, d)[idx]; b = cell(blind, m, d)[idx]
            if o is None and b is None:
                cells.append(f"{'--':>17s}"); continue
            cells.append(f"{(o if o is not None else float('nan')):7.2f}->{(b if b is not None else float('nan')):6.2f}" if metric=="S_LLM"
                         else f"{(o if o is not None else float('nan')):7.1f}->{(b if b is not None else float('nan')):6.1f}")
            if d != "legalq100":
                if o is not None: macro_o.append(o)
                if b is not None: macro_b.append(b)
        mo = statistics.mean(macro_o) if macro_o else float("nan")
        mb = statistics.mean(macro_b) if macro_b else float("nan")
        print(f"{m:32s} " + " ".join(cells) + f" {mo:.2f}->{mb:.2f}")

# paired full-vs-baseline per dataset, blind
rng = random.Random(SEED)
print("\n=== blind: signpost.full minus baseline, per dataset ===")
print(f"{'method':30s} " + " ".join(f"{d:>22s}" for d in ORDER))
for m in SHOW[:10]:
    out = []
    for d in ORDER:
        diffs = [blind[("signpost.full", d, q)] - blind[(m, d, q)]
                 for q in seen[d] if ("signpost.full", d, q) in blind and (m, d, q) in blind]
        if not diffs:
            out.append(f"{'--':>22s}"); continue
        boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(4000))
        out.append(f"{statistics.mean(diffs):+5.2f} [{boot[99]:+5.2f},{boot[-100]:+5.2f}]")
    print(f"{m:30s} " + " ".join(out))
