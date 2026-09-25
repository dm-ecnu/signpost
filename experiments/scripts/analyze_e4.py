#!/usr/bin/env python3
"""E4 -> LaTeX. Reads data/e4/e4_*.json and emits the maintenance table rows."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent / "results" / "maintenance"

# display name, json stem, relation-entry P95 (from tab:breadthdepth)
CORPORA = [
    ("Agriculture", "agriculture", 388),
    ("Medical", "graphrag-bench-medical", 151),
    ("Novel", "graphrag-bench-novel", 156),
    ("Mix", "mix", 97),
    ("Legal", "legal", 1789),
]
FRACS = [0.001, 0.005, 0.01, 0.05, 0.10]

rows, summary = [], []
for name, stem, p95 in CORPORA:
    r = json.loads((D / f"e4_{stem}.json").read_text())
    by = {e["edit_fraction"]: e for e in r["edits"] if e["granularity"] == "chunk"}
    cells, hubs = [], []
    for f in FRACS:
        e = by[f]
        cells.append(f"{e['dirty_object_share']*100:.1f}\\,/\\,{e['share_of_full']*100:.0f}")
        if e.get("hub_bias"):
            hubs.append(e["hub_bias"])
    rows.append(f"{name} & {p95:,} & " + " & ".join(cells) + r" \\")
    summary.append(dict(
        name=name, p95=p95, objects=r["objects"], chunks=r["chunks"],
        full_s=r["full_rebuild_seconds"],
        d001=by[0.001]["dirty_object_share"], d01=by[0.01]["dirty_object_share"],
        d05=by[0.05]["dirty_object_share"], d10=by[0.10]["dirty_object_share"],
        t01=by[0.01]["share_of_full"], t05=by[0.05]["share_of_full"],
        t10=by[0.10]["share_of_full"],
        amp01=by[0.01]["amplification"], amp001=by[0.001]["amplification"],
        hub_lo=min(hubs) if hubs else None, hub_hi=max(hubs) if hubs else None,
    ))

print("% --- table rows: dirty-set share / rebuild-time share, both in percent")
for r in rows:
    print(r)

print("\n% --- prose anchors")
for s in summary:
    print(f"% {s['name']:12s} N={s['objects']:6d} chunks={s['chunks']:5d} full={s['full_s']:5.2f}s "
          f"P95={s['p95']:5d}  dirty 0.1%={s['d001']:.3%} 1%={s['d01']:.1%} 5%={s['d05']:.1%} 10%={s['d10']:.1%} "
          f"| t 1%={s['t01']:.0%} 5%={s['t05']:.0%} 10%={s['t10']:.0%} "
          f"| amp@1%={s['amp01']}x  hub {s['hub_lo']}-{s['hub_hi']}x")

d01 = [s["d01"] for s in summary]
print(f"\n% dirty@1%  range: {min(d01):.1%}-{max(d01):.1%}")
print(f"% amp@1%    range: {min(s['amp01'] for s in summary)}x-{max(s['amp01'] for s in summary)}x")
print(f"% t@5%      range: {min(s['t05'] for s in summary):.0%}-{max(s['t05'] for s in summary):.0%}")
print(f"% t@10%     range: {min(s['t10'] for s in summary):.0%}-{max(s['t10'] for s in summary):.0%}")
hubs = [s["hub_hi"] for s in summary if s["hub_hi"]]
los = [s["hub_lo"] for s in summary if s["hub_lo"]]
print(f"% hub-bias  range: {min(los)}x-{max(hubs)}x")

# Spearman between relation P95 and dirty share at 1%
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for pos, i in enumerate(order):
        rk[i] = pos + 1.0
    return rk
a, b = rank([s["p95"] for s in summary]), rank(d01)
n = len(a)
rho = 1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1))
print(f"% Spearman(rel-P95, dirty@1%) = {rho:.2f}  (n={n})")
