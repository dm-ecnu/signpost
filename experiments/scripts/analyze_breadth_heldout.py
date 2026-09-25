#!/usr/bin/env python3
"""Held-out breadth: does ANY history-fitted object set serve future queries?

Why this exists
---------------
The in-sample curve (analyze_t0b.py, T0-7b) ranks objects by access count within
the workload being served and asks how many queries keep their whole evidence set.
Two problems with using it as the paper's headline:

  1. Its denominator is the *evidence universe of the workload* (445-566 objects),
     not the corpus (28,966-101,136 objects).  "Top 30%" is therefore 0.17-0.46%
     of the corpus, while the depth axis (b=8) is quoted as 5.4-40.7% of the FULL
     view over ALL objects.  Plotting both on one "storage budget" axis compares
     different denominators; at equal storage, breadth would serve 100%.
  2. Ranking by access count is an oracle on the WRONG objective.  Problem 2's
     objective is evidence-set coverage, not frequency, so the in-sample curve
     does not bound "any history-driven policy" the way the paper claims.

The generalization framing has neither problem: fit on train queries, serve test
queries, and give the policy an UNLIMITED budget -- materialize the entire
evidence universe observed in training.  If that still fails, "materialize every
object" is justified by generalization rather than by storage arithmetic.

Result (20 random 50/50 splits, same `evidence_chunks` field the paper uses):
  corpus        test servable    test evidence objects never seen in training
  Agriculture       0.4%                      84.1%
  Medical          19.0%                      47.9%
  Novel             2.3%                      80.2%
  Legal             0.0%                      84.2%
  Mix               4.5%                      71.7%

Validation: this script's in-sample@30% reproduces the paper's Fig.1 exactly
(19.0 / 22.0 / 16.0 / 17.0 / 13.8), so it reads the same data and field.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data/topb8-outputs"
SPLITS = 20


def obj_id(c: dict) -> str | None:
    fn, s, e = c.get("file_name"), c.get("start_line"), c.get("end_line")
    if fn and s is not None:
        return f"{fn}:{s}-{e}"
    return str(c.get("chunk_id") or c.get("doc_id") or "") or None


def evidence_sets(ds: str) -> list[set[str]]:
    rows = []
    for f in sorted((DATA / ds / "predictions" / "parts").glob("signpost.full.part_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        ids = {oid for c in (r.get("evidence_chunks") or []) if isinstance(c, dict) and (oid := obj_id(c))}
        if ids:
            out.append(ids)
    return out


def in_sample_at(ev: list[set[str]], x: float) -> float:
    cnt: Counter = Counter()
    for e in ev:
        cnt.update(e)
    hot = {o for o, _ in cnt.most_common(max(1, round(len(cnt) * x)))}
    return sum(1 for e in ev if e <= hot) / len(ev)


def held_out(ev: list[set[str]]) -> tuple[float, float, int]:
    servs, unseen, usizes = [], [], []
    for seed in range(SPLITS):
        idx = list(range(len(ev)))
        random.Random(seed).shuffle(idx)
        h = len(idx) // 2
        tr, te = idx[:h], idx[h:]
        U = set().union(*[ev[i] for i in tr])
        servs.append(sum(1 for i in te if ev[i] <= U) / len(te))
        all_te = set().union(*[ev[i] for i in te])
        unseen.append(len(all_te - U) / len(all_te))
        usizes.append(len(U))
    mean = lambda v: sum(v) / len(v)
    return mean(servs), mean(unseen), round(mean(usizes))


def main() -> None:
    print(f"{'corpus':<28}{'|E*| univ':>10}{'|E*_q|':>8}{'in-samp@30%':>13}"
          f"{'held-out serv':>15}{'unseen objs':>13}{'|U_train|':>11}")
    for ds in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        ev = evidence_sets(ds)
        if not ev:
            continue
        univ = len(set().union(*ev))
        mean_sz = sum(len(e) for e in ev) / len(ev)
        serv, unseen, usz = held_out(ev)
        print(f"{ds:<28}{univ:>10}{mean_sz:>8.1f}{in_sample_at(ev,0.30):>12.1%}"
              f"{serv:>15.1%}{unseen:>13.1%}{usz:>11}")
    print("\nHeld-out policy has NO budget: it materializes the whole training "
          "evidence universe.\nIt still fails, so the design rule rests on "
          "generalization, not on storage arithmetic.")


if __name__ == "__main__":
    raise SystemExit(main())
