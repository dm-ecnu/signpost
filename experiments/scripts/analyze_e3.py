#!/usr/bin/env python3
"""E3 -> LaTeX: per-entry-class attribution of the spans the agent actually read."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "attribution"
CORPORA = [("Agri.", "agriculture"), ("Med.", "graphrag-bench-medical"),
           ("Novel", "graphrag-bench-novel"), ("Mix", "mix"), ("Legal", "legal")]
CLASSES = [("Zoom $C_v$", "zoom"), ("Read $C_h$", "read"),
           ("Jump $C_s$", "jump"), ("Verify $C_p$", "verify")]

R = {stem: json.loads((D / f"e3_{stem}.json").read_text()) for _, stem in CORPORA}

print("% rows: entries/query (mean), then direct-attribution share per corpus, then one-hop range, unique")
for label, key in CLASSES:
    ent = sum(R[s]["per_class"][key]["entries_surfaced"] / R[s]["queries"] for _, s in CORPORA) / len(CORPORA)
    direct = [R[s]["per_class"][key]["reads_direct"] / R[s]["reads_total"] for _, s in CORPORA]
    onehop = [R[s]["per_class"][key]["reads_onehop"] / R[s]["reads_total"] for _, s in CORPORA]
    uniq = sum(R[s]["per_class"][key]["reads_unique_onehop"] for _, s in CORPORA)
    cells = " & ".join(f"{d*100:.0f}" for d in direct)
    lo, hi = min(onehop) * 100, max(onehop) * 100
    print(f"{label} & {ent:.0f} & {cells} & {lo:.0f}--{hi:.0f} & {uniq} \\\\")

print("\n% anchors")
tot_reads = sum(R[s]["reads_total"] for _, s in CORPORA)
tot_q = sum(R[s]["queries"] for _, s in CORPORA)
print(f"% {tot_q} queries, {tot_reads} distinct read_file calls, "
      f"{sum(R[s]['reads_unattributed'] for _, s in CORPORA)} unattributed")
for name, s in CORPORA:
    r = R[s]
    pc = r["per_class"]
    print(f"% {name:6s} q={r['queries']:3d} reads={r['reads_total']:4d} served={r['served_objects_resolved']:5d} "
          + " ".join(f"{k}:{pc[k]['reads_direct']/r['reads_total']:.0%}/{pc[k]['reads_onehop']/r['reads_total']:.0%}"
                     for _, k in CLASSES))
for label, key in CLASSES:
    d = [R[s]["per_class"][key]["reads_direct"] / R[s]["reads_total"] for _, s in CORPORA]
    print(f"% {key:7s} direct range {min(d):.0%}-{max(d):.0%}")
