#!/usr/bin/env python3
"""T0-2 calibration table + T0-7 materialization-budget curve.

Two outputs:
  (1) cost-model calibration from frozen artifacts (online: calls/tokens from the
      topb8 predictions; offline: build-time decomposition and bytes/object from
      the p2 scaling TSVs).
  (2) the denominator-free curve: materialize the top-x% most-accessed objects of
      the observed workload -> what fraction of all accesses is served from the
      materialized set.  This is the partial-materialization ("partial index")
      result and needs no corpus-size assumption.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data/topb8-outputs"
DRIVE = Path(__file__).resolve().parents[1] / "data/icde-exp"  # build-time CSVs of the prior submission
OUT = Path(__file__).resolve().parents[1] / "out"


def load(dataset: str, variant: str = "signpost.full") -> list[dict]:
    rows = []
    for f in sorted((DATA / dataset / "predictions" / "parts").glob(f"{variant}.part_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def obj_id(c: dict) -> str | None:
    fn, s, e = c.get("file_name"), c.get("start_line"), c.get("end_line")
    if fn and s is not None:
        return f"{fn}:{s}-{e}"
    return str(c.get("chunk_id") or c.get("doc_id") or "") or None


def access_counter(rows: list[dict], field: str) -> Counter:
    cnt = Counter()
    for r in rows:
        seen = {oid for c in (r.get(field) or []) if isinstance(c, dict) and (oid := obj_id(c))}
        cnt.update(seen)
    return cnt


def budget_curve(cnt: Counter, grid=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)):
    freqs = sorted(cnt.values(), reverse=True)
    total = sum(freqs)
    n = len(freqs)
    out = []
    for x in grid:
        k = max(1, int(round(n * x)))
        out.append((x, k, sum(freqs[:k]) / total))
    return out


def query_coverage(rows: list[dict], cnt: Counter, field: str, x: float) -> float:
    """Fraction of queries whose every *evidence* object is in the top-x% materialized set."""
    hot = {o for o, _ in cnt.most_common(max(1, int(round(len(cnt) * x))))}
    ok = 0
    for r in rows:
        ids = {oid for c in (r.get(field) or []) if isinstance(c, dict) and (oid := obj_id(c))}
        if ids and ids <= hot:
            ok += 1
    return ok / len(rows)


def read_tsv(name: str) -> list[dict]:
    with open(DRIVE / name, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = sorted(p.name for p in DATA.iterdir() if p.is_dir())

    # ---------- (1) online side of the cost model ----------
    print("=" * 104)
    print("T0-2a  ONLINE cost constants per query (frozen topb8, signpost.full)")
    print("=" * 104)
    print(f"{'dataset':<28}{'n':>4}{'c_call':>8}{'c_ks':>7}{'c_read':>8}{'tok_in':>9}{'tok_out':>9}{'tok/read':>10}")
    online = {}
    for ds in datasets:
        rows = load(ds)
        if not rows:
            continue
        n = len(rows)
        calls = sum(r.get("llm_calls") or 0 for r in rows) / n
        ks = sum(r.get("knowledge_search_calls") or 0 for r in rows) / n
        rd = sum(r.get("read_file_calls") or 0 for r in rows) / n
        ti = sum(r.get("input_tokens") or 0 for r in rows) / n
        to = sum(r.get("output_tokens") or 0 for r in rows) / n
        online[ds] = dict(n=n, calls=calls, ks=ks, read=rd, tin=ti, tout=to)
        print(f"{ds:<28}{n:>4}{calls:>8.2f}{ks:>7.1f}{rd:>8.1f}{ti:>9.0f}{to:>9.0f}"
              f"{(ti / rd if rd else float('nan')):>10.0f}")

    # ---------- (2) offline side ----------
    print()
    print("=" * 104)
    print("T0-2b  OFFLINE cost per retrievable object (p2 scaling TSVs)")
    print("=" * 104)
    print(f"{'corpus':<10}{'frac':>6}{'objects':>9}{'build_s':>10}{'nav_s':>8}{'es_sync_s':>11}"
          f"{'nav_ms/obj':>12}{'sync_ms/obj':>13}{'bytes/obj':>11}{'nav_share':>11}")
    for fn, tag in (("p2_scaling_agriculture.tsv", "agri"), ("p2_scaling_mix.tsv", "mix")):
        for r in read_tsv(fn):
            objs = int(r["retrievable_objects"])
            build = float(r["build_seconds_f7_f10"])
            nav = float(r["seconds_f7_structure"]) + float(r["seconds_f8_sequence"]) + float(r["seconds_f9_unified"])
            sync = float(r["seconds_f10_es_sync"])
            by = int(r["graph_json_bytes"])
            print(f"{tag:<10}{float(r['fraction']):>6.2f}{objs:>9}{build:>10.1f}{nav:>8.2f}{sync:>11.1f}"
                  f"{nav / objs * 1000:>12.4f}{sync / objs * 1000:>13.2f}{by / objs:>11.0f}{nav / build:>11.2%}")

    # ---------- (3) denominator-free materialization-budget curve ----------
    print()
    print("=" * 104)
    print("T0-7  materialize top-x% of ACCESSED objects -> share of accesses served (retrieved_chunks)")
    print("=" * 104)
    grid = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
    print(f"{'dataset':<28}" + "".join(f"{int(g * 100):>7}%" for g in grid))
    curves = {}
    for ds in datasets:
        rows = load(ds)
        cnt = access_counter(rows, "retrieved_chunks")
        if not cnt:
            continue
        cur = budget_curve(cnt, grid)
        curves[ds] = cur
        print(f"{ds:<28}" + "".join(f"{c[2]:>8.1%}" for c in cur))

    print()
    print("same, but on evidence_chunks (what actually entered the context)")
    print(f"{'dataset':<28}" + "".join(f"{int(g * 100):>7}%" for g in grid))
    for ds in datasets:
        rows = load(ds)
        cnt = access_counter(rows, "evidence_chunks")
        if not cnt:
            continue
        print(f"{ds:<28}" + "".join(f"{c[2]:>8.1%}" for c in budget_curve(cnt, grid)))

    # ---------- (4) query-level completeness under a materialization budget ----------
    print()
    print("=" * 104)
    print("T0-7b  fraction of queries whose ENTIRE evidence set lies in the top-x% materialized set")
    print("=" * 104)
    print(f"{'dataset':<28}" + "".join(f"{int(g * 100):>7}%" for g in grid))
    for ds in datasets:
        rows = load(ds)
        cnt = access_counter(rows, "evidence_chunks")
        if not cnt:
            continue
        print(f"{ds:<28}" + "".join(f"{query_coverage(rows, cnt, 'evidence_chunks', g):>8.1%}" for g in grid))

    with open(OUT / "t0_curves.json", "w") as fh:
        json.dump({"online": online, "curves": {k: v for k, v in curves.items()}}, fh, indent=2)
    print(f"\nwrote {OUT / 't0_curves.json'}")


if __name__ == "__main__":
    main()
