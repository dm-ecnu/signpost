#!/usr/bin/env python3
"""Depth-budget sweep: storage share and per-step exposure as a function of b.

Pure re-analysis of the frozen 2026-06-06 exposure recount
(`data/exposure/cue_exposure_*.json`), which holds, for every corpus, every
budget b in {8,16,32,64,inf} and every object type, the per-object entry count
distribution (n / mean / p50 / p95 / max).

Storage share at budget b is the total number of materialized entries relative
to the untruncated view:  sum_t n_t * mean_t(b)  /  sum_t n_t * mean_t(inf).
The b=8 column reproduces Table `breadthdepth` in the paper, which is how we
check that this aggregation matches the one used there.
"""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "results/exposure"
OUT = Path(__file__).resolve().parents[1] / "out"
BUDGETS = ["8", "16", "32", "64", "inf"]
NAME = {"agriculture": "Agriculture", "medical": "Medical", "novel": "Novel",
        "legal": "Legal", "mix": "Mix"}
ORDER = ["Agriculture", "Medical", "Novel", "Mix", "Legal"]


def load() -> dict[str, dict]:
    out = {}
    for f in sorted(EXP.glob("cue_exposure_*.json")):
        d = json.loads(f.read_text())
        out[NAME.get(d["dataset"], d["dataset"])] = d
    return out


def entries(d: dict, b: str) -> float:
    return sum(v["n"] * v["mean"] for v in d["by_budget"][b].values())


def main() -> None:
    data = load()
    rows = {}
    for corpus in ORDER:
        d = data[corpus]
        full = entries(d, "inf")
        rows[corpus] = {
            "objects": d["objects_counted"],
            "storage": {b: entries(d, b) / full for b in BUDGETS},
            "rel_p95": {b: d["by_budget"][b]["relation"]["p95"] for b in BUDGETS},
            "ent_p95": {b: d["by_budget"][b]["entity"]["p95"] for b in BUDGETS},
        }

    hdr = "".join(f"{('b=' + b):>10}" for b in BUDGETS)
    print("storage share of the untruncated view")
    print(f"{'corpus':<13}{'objects':>9}{hdr}")
    for c in ORDER:
        r = rows[c]
        print(f"{c:<13}{r['objects']:>9}"
              + "".join(f"{r['storage'][b] * 100:>9.1f}%" for b in BUDGETS))

    print("\nrelation-entry P95 per-step exposure")
    print(f"{'corpus':<13}{'':>9}{hdr}")
    for c in ORDER:
        r = rows[c]
        print(f"{c:<13}{'':>9}" + "".join(f"{r['rel_p95'][b]:>10.0f}" for b in BUDGETS))

    print("\nP95 growth factor per doubling of b (8->16->32->64), and the jump to inf")
    for c in ORDER:
        p = rows[c]["rel_p95"]
        steps = [p["16"] / p["8"], p["32"] / p["16"], p["64"] / p["32"]]
        print(f"{c:<13}" + "".join(f"{s:>8.2f}x" for s in steps)
              + f"   inf/64 = {p['inf'] / p['64']:>6.1f}x")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bsweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT / 'bsweep.json'}")


if __name__ == "__main__":
    main()
