#!/usr/bin/env python3
"""Parse the re-judged MuSiQue answers and recompute S_LLM / J@7 per method.

Compares against the frozen 2026-06-09 judge output, whose baseline sections were
truncated away in 54 of 100 files.
"""
from __future__ import annotations

import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PQ = HERE.parent / "data/perquery/musique_per_query.tsv"
RNG = np.random.default_rng(20260919)

DIM = re.compile(r"准确性\s*([0-9.]+)\s*/\s*4[，,]\s*完整性\s*([0-9.]+)\s*/\s*3[，,]\s*简洁性\s*([0-9.]+)\s*/\s*3")
DIM_A = re.compile(r"准确性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*4")
DIM_C = re.compile(r"完整性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3")
DIM_S = re.compile(r"简洁性(?:\*\*)?\s*[：:]\s*([0-9.]+)\s*/\s*3")


def parse_file(text: str) -> dict[str, float]:
    out = {}
    for sec in re.split(r"^### 方法\s+", text, flags=re.M)[1:]:
        name, _, rest = sec.partition("\n")
        m = DIM.search(rest)
        if m:
            vals = tuple(float(x) for x in m.groups())
        else:
            a, c, s = DIM_A.search(rest), DIM_C.search(rest), DIM_S.search(rest)
            if not (a and c and s):
                continue
            vals = (float(a.group(1)), float(c.group(1)), float(s.group(1)))
        out[name.strip()] = sum(vals)
    return out


def boot(x: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(x), size=(10000, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    new: dict[str, dict[str, float]] = defaultdict(dict)
    for path in sorted((HERE / "musique_ans").glob("*.txt")):
        qid = path.stem.split("_", 1)[1]
        for method, score in parse_file(path.read_text(encoding="utf-8")).items():
            new[method][qid] = score

    old: dict[str, dict[str, float]] = defaultdict(dict)
    with open(PQ) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                old[r["method"]][r["question_id"]] = float(r["llm_total_score"])
            except (TypeError, ValueError):
                pass

    print(f"{'method':<34}{'old n':>6}{'old mean':>10}{'new n':>7}{'new S_LLM [95% CI]':>26}{'J@7':>7}")
    for m in sorted(new):
        d = new[m]
        x = np.array(list(d.values()))
        mu, lo, hi = boot(x)
        j7 = float((x >= 7).mean() * 100)
        o = old.get(m, {})
        om = st.mean(o.values()) if o else float("nan")
        print(f"{m:<34}{len(o):>6}{om:>10.2f}{len(d):>7}   {mu:5.2f} [{lo:5.2f},{hi:5.2f}]{j7:>9.1f}")

    full = new.get("signpost.full", {})
    if full:
        print("\npaired  signpost.full - X  (all questions both were scored on)")
        from scipy.stats import wilcoxon
        for m in sorted(new):
            if m == "signpost.full":
                continue
            qs = sorted(set(full) & set(new[m]))
            d = np.array([full[q] - new[m][q] for q in qs])
            mu, lo, hi = boot(d)
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            print(f"  vs {m:<34} n={len(d):3d}  {mu:+5.2f} [{lo:+5.2f},{hi:+5.2f}]  p={p:.2e}")


if __name__ == "__main__":
    main()
