# Signpost — implemented gaps & how to re-run (2026-06-09)

This snapshot fills the code↔paper gaps found in the audit
(`rag/notes/signpost-code-gap-2026-06-09.md`). All additions are real and
unit-tested (66/66). **The experiments must be re-run** — the numbers in the
current paper predate these changes.

## What changed

| Area | Files | What it does |
|---|---|---|
| **Sketch chaining** (Alg.3 core) | `signpost/agent/sketch_chaining.py`, `signpost/agent/supervisor.py` | Researcher now follows zoom/read/jump cues to successor objects across multiple hops, maintains the visited set `H_t`, applies priority `verify > read > zoom > jump`, and reads verify cues via ReadFile. **Deterministic** cue-following — no extra LLM call per hop. |
| **Iso-call baseline** | `signpost/baselines/iso_call.py`, `scripts/baselines/run_iso_call.py` | ReAct over **untyped** neighbors at the same LLM-call budget, for attribution (isolates the benefit of typed cues vs. budget). Registered in `scripts/baselines/run_baseline_method.sh`. |
| **Confidence intervals** | `signpost/benchmark/stats.py` | `bootstrap_ci`, `paired_bootstrap_diff` (CI + bootstrap p-value), `summarize_with_ci`. Stdlib-only, deterministic (fixed seed). |

## Run the unit tests first (no ES/LLM/corpus needed)

```bash
python3 -m pytest tests/test_sketch_chaining.py tests/test_stats_ci.py tests/test_iso_call_baseline.py -q
# expect: 66 passed
```

## Re-run the experiments (needs the H200/ES/LLM corpus setup)

1. **Sketch chaining is ON by default** (`AgentConfig.enable_sketch_chaining=True`).
   To reproduce the old simplified 2-call agent for comparison, set it `False`.
2. Re-run the full eval (`scripts/h200_final_eval_v2.py` and the F-stage
   pipeline) for **all** methods. Quantities that change with chaining ON:
   evidence reached, ReadFile/tool-call counts, latency, and all quality/silver
   numbers (RQ1–RQ5, Tab.1–6).
3. **Add the iso-call baseline** to the comparison:
   `scripts/baselines/run_baseline_method.sh iso_call <dataset>` (set
   `ISO_CALL_CALL_BUDGET` to Signpost's measured call count).
4. **Report CIs**: use `summarize_with_ci` / `paired_bootstrap_diff` on the
   per-question scores (the Tab.1 macro margin is small — ~0.24 — and needs a
   significance test).

## Important notes

- **"2 LLM calls" headline:** sketch chaining here is *deterministic* (cue
  following, no LLM per hop), so the 2-LLM-call profile is preserved. Frame the
  claim as deterministic navigation. (A *full ReAct* port — LLM per hop — would
  instead make it ~11–13 calls/query; that is a different design and is NOT what
  this implements.)
- **Silver-evidence construction is still external** (`extract_llm_targets_silver.py`
  on the H200) — not reconstructed here, because rebuilding it could produce a
  *different* method than what generated the silver targets. Bring the real
  script into the repo and audit the chunk-level-vs-span self-reference concern.
- **Paper text edits (separate from code):** relation objects `V_r` are stored as
  edges, not graph nodes; sketches are computed at query time, not stored inline
  in ES; greedy submodular selection is gated by `SIGNPOST_CUE_SELECT=greedy`
  (default is truncate); provenance locators are chunk-level. Reconcile these in
  the paper or the code.

This changes ruolinsu's agent and invalidates the current numbers — please
review before the re-run.
