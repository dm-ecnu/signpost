# Reproducing the paper

This file maps each table, figure, and in-text result of the PVLDB submission
to the script that produces it and the frozen result files that script reads.

All scripts named below are in `experiments/scripts/` and their frozen
outputs in `experiments/results/` (see `experiments/README.md`), except where
the last column says otherwise.

**Builder revision.** The paper's numbers were produced with a server copy of
`signpost/retrieval/offline_signpost.py` whose four per-type builders are
identical to the ones committed here; it differs only in memoizing the graph
index (`_INDEX_CACHE`), which changes latency, not output. The b = 8 cap
(`_apply_cue_topb_item`) matches the committed version under the default
`SIGNPOST_CUE_SELECT=truncate`. The RQ6 pilot additionally used two serving
variants, `online_cues` and `provenance_only`, that are not in the committed
`signpost_variants.py`.

## Requirements by experiment family

| Family | Model endpoint | Elasticsearch | Input |
|---|---|---|---|
| Physical design (Tables 2–3, serialization order, cost model) | no | no | `datasets/processed/<corpus>/graph.unified.json` |
| Query-dependent ranking (Table 7) | no | no | graph + frozen search subqueries |
| End-to-end QA (Tables 4–6, RQ6 pilot) | yes | yes | full T1 deployment (see `README.md`) |

The physical-design scripts make no model, Elasticsearch, or network calls.
Corpus identifiers: `graphrag-bench-medical_q100` (Medical), `mix`,
`graphrag-bench-novel_q100` (Novel), `agriculture`, `legal_q100` (Legal),
`musique_q100` (MuSiQue). See `DATASETS.md` for how to obtain each corpus.

## Tables and figures

| Paper item | What it reports | Script | Frozen results | In repo |
|---|---|---|---|---|
| Table 1 (`tab:signpost_ref`) | Entry classes and their fields | — (definitional; see `offline_signpost.py`) | — | yes |
| Figure 1 (`fig:motivation`) | Held-out residency of history-fitted object sets | `analyze_breadth_heldout.py`, `f1_breadth_heldout.py` | raw prediction logs (`data/topb8-outputs/`, 57 MB) | scripts yes; logs not bundled |
| Table 2 (`tab:physical-cache`) | Eager vs lazy+cache vs no cache; LRU 1%/10%; cold P95; storage share | `benchmark.py` (run), `gen_physical_table.py` (table) | `results/efficiency/<corpus>.{runs.jsonl,manifest.json,complete.json,trace.json}` | yes (six corpora) |
| Table 3 (`tab:maintenance`) | Dirty-set recompute vs full rebuild, chunk edit batches 0.1–10% | `e4_incremental.py` (run), `analyze_e4.py` (rows) | `results/maintenance/e4_<corpus>.json` | yes |
| Table 4 (`tab:online`) | Per-query latency, model calls, tokens | `analyze_t0.py`, `analyze_t0b.py` | raw prediction logs (`data/topb8-outputs/`) | scripts yes; logs not bundled |
| Table 5 (`tab:quality`) | Blind answer quality | `blind_judge.py`, `analyze_blind_tables.py`, `analyze_ci.py` | `results/perquery/formal5_per_query.tsv`, `musique_per_query.tsv` | yes |
| Table 6 (`tab:ablation`) | Serve-time removals under blind judging | `blind_judge.py`, `analyze_blind_tables.py` | `results/perquery/` (raw judge bundles, 22 MB, not bundled) | yes |
| Table 7 (`tab:e1divergence`) | Query-dependent order/set divergence | `gen_div_table.py` | — | **script not located** |

Table numbers are those of the submitted PDF; the LaTeX labels in parentheses
are stable across revisions.

## In-text results

| Result (section) | Script | Frozen results | In repo |
|---|---|---|---|
| Depth sweep, b ∈ {8,16,32,64,∞}: storage share and relation/entity P95 (RQ1) | `analyze_bsweep.py` | `results/exposure/cue_exposure_<corpus>.json` → `bsweep.json` | yes |
| Depth sweep, all points b ∈ {1,2,4,8,16,32,64,∞}, six corpora (RQ1) | `x5_depth_sweep.py` | `results/depth/x5_<corpus>.json` | script yes; results added when the run completes |
| Coverage-greedy vs score-prefix selection at b = 8 (RQ1) | `e6_greedy_vs_truncate.py`, `cue_coverage.py` | `results/coverage/` | yes |
| Relation-entry P95 for MuSiQue, Table 3 column (RQ3) | `recount_cue_exposure.py` | `cue_exposure_musique.json` | yes |
| Which entry classes expose the 3,160 spans read (RQ6) | `e3_follow_rate.py`, `analyze_e3.py` | `results/attribution/` | yes |
| Maintenance cost model t̂/t_full = a·Σ_{o∈D}\|σ(o)\| / Σ_o\|σ(o)\| (RQ3) | `x4_features.py`, `x4_fit.py` | `x4_<corpus>.json`, `fit5.json`, `fit6.json`, `PREREGISTERED.md` | yes |
| Serialization-order invariance with canonical load order (RQ6) | `x3_canonical.py` | `results/order/x3_canonical.json` | script yes; results added when the Legal run completes |
| Distinct-object growth over the traces; questions to build every view (Discussion) | `trace_growth.py` | `results/efficiency/*.trace.json` | yes |
| RQ6 construction-time / depth pilot (Medical) | `analyze_e1e2.py` | `results/e1e2/*.judged.jsonl`, `audit.json` | results yes; **analysis script not located** |

## Protocol notes that affect the numbers

- **Timing medians.** Table 2 reports medians over five repeats; warm service is
  the last of three replays; cold P95 is the median over repeats of the
  first replay's per-question P95.
- **Byte equality is checked, not assumed.** Every lazy or cached run row
  carries `equal_to_eager`; `gen_physical_table.py` refuses input with any
  `false` row, and checks the runs file against the SHA-256 in
  `<corpus>.complete.json`.
- **Edit batches** use seed `20260919` and chunk fractions
  {0.001, 0.005, 0.01, 0.05, 0.10}. `x4_features.py` re-draws the same batches
  and asserts that its dirty-set sizes match the frozen `e4_<corpus>.json`.
- **Cost model.** The constant a is fitted on the five non-MuSiQue corpora; its
  form was fixed (`PREREGISTERED.md`) before the MuSiQue maintenance run.
  Evaluation is leave-one-corpus-out plus MuSiQue as a held-out corpus.
- **Canonical load order** sorts nodes by (`node_id`, canonical JSON) and
  edges by (`edge_type`, `source`, `target`, canonical JSON), where canonical
  JSON is `json.dumps(sort_keys=True, separators=(',', ':'))`.
- **Depth b = 8** is applied with `SIGNPOST_CUE_TOPB=8` (all four classes), i.e.
  `_apply_cue_topb_item(item, dict(v=8, h=8, s=8, p=8))`.
