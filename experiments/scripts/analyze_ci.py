#!/usr/bin/env python3
"""Bootstrap CIs and paired tests on the frozen per-query judge scores (ARR R5).

Sources (read-only copies of the server's eval outputs):
  data/perquery/formal5_per_query.tsv   -- 4 domain corpora, all methods
  data/perquery/musique_per_query.tsv   -- MuSiQue, all methods
Both carry llm_total_score in [0,10] and answer_recall per (dataset, method, qid).

Outputs
  (a) per-corpus mean S_LLM with a 95% percentile-bootstrap CI for every method
  (b) paired SignPost-vs-X differences on the common question set, with a
      bootstrap CI over paired differences and a two-sided Wilcoxon signed-rank p
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1] / "results/perquery"
RNG = np.random.default_rng(20260919)
B = 10000

CORPORA = [
    ("agriculture", "Agriculture"),
    ("medicalq100", "Medical"),
    ("novelq100", "Novel"),
    ("mixv0", "Mix"),
    ("legalq100", "Legal"),
    ("musique_q100", "MuSiQue"),
]
BASELINES = ["vanilla_llm", "hybrid_rag", "cluerag_prompt_normalized", "agrag",
             "linearrag", "memgraphrag", "hiprag", "graphrag_r1_original_offline"]
ABLATIONS = ["signpost.no_vertical_cues", "signpost.no_horizontal_cues",
             "signpost.no_semantic_cues", "signpost.no_provenance_cues",
             "signpost.no_offline"]


def load() -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """(dataset, method) -> qid -> {metric: value}"""
    out: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for name in ("formal5_per_query.tsv", "musique_per_query.tsv"):
        with open(ROOT / name) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rec = {}
                for m in ("llm_total_score", "answer_recall"):
                    try:
                        rec[m] = float(r[m])
                    except (TypeError, ValueError):
                        pass
                if "llm_total_score" in rec:
                    out[(r["dataset"], r["method"])][r["question_id"]] = rec
    return out


def boot_mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a: dict, b: dict, metric: str = "llm_total_score"):
    qs = sorted(set(a) & set(b))
    xa = np.array([a[q][metric] for q in qs if metric in a[q] and metric in b[q]])
    xb = np.array([b[q][metric] for q in qs if metric in a[q] and metric in b[q]])
    d = xa - xb
    m, lo, hi = boot_mean_ci(d)
    try:
        p = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue)
    except ValueError:      # all differences zero
        p = 1.0
    return len(d), m, lo, hi, p


def main() -> None:
    data = load()

    print("=" * 96)
    print("(a) per-corpus S_LLM mean [95% bootstrap CI], n queries")
    print("=" * 96)
    methods = ["signpost.full"] + BASELINES + ABLATIONS
    for key, label in CORPORA:
        print(f"\n-- {label} ({key})")
        for m in methods:
            d = data.get((key, m))
            if not d:
                continue
            x = np.array([v["llm_total_score"] for v in d.values()])
            mu, lo, hi = boot_mean_ci(x)
            print(f"   {m:<34} {mu:5.2f}  [{lo:5.2f}, {hi:5.2f}]  n={len(x)}")

    print()
    print("=" * 96)
    print("(b) paired SignPost - X on the common question set: mean diff [95% CI], Wilcoxon p")
    print("=" * 96)
    for key, label in CORPORA:
        full = data.get((key, "signpost.full"))
        if not full:
            continue
        print(f"\n-- {label}")
        for m in BASELINES + ABLATIONS:
            other = data.get((key, m))
            if not other:
                continue
            n, mu, lo, hi, p = paired(full, other)
            star = "n.s." if p >= 0.05 else ("*" if p >= 1e-3 else "***")
            print(f"   vs {m:<34} n={n:3d}  {mu:+5.2f} [{lo:+5.2f}, {hi:+5.2f}]  p={p:.2e} {star}")

    print()
    print("=" * 96)
    print("(c) macro over the four domain corpora: paired diff pooled across corpora")
    print("=" * 96)
    four = ["agriculture", "medicalq100", "novelq100", "mixv0"]
    for m in BASELINES + ABLATIONS:
        ds = []
        for key in four:
            a, b = data.get((key, "signpost.full")), data.get((key, m))
            if not a or not b:
                continue
            qs = sorted(set(a) & set(b))
            ds.extend(a[q]["llm_total_score"] - b[q]["llm_total_score"] for q in qs)
        if not ds:
            continue
        d = np.array(ds)
        mu, lo, hi = boot_mean_ci(d)
        p = float(wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
        print(f"   vs {m:<34} n={len(d):4d}  {mu:+5.2f} [{lo:+5.2f}, {hi:+5.2f}]  p={p:.2e}")


if __name__ == "__main__":
    main()
