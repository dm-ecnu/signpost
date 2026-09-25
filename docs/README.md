# docs/

## Reference documentation (English)

| File | Contents |
|---|---|
| `REPRODUCING.md` | Paper table/figure → script → frozen result files |
| `environment_setup.md` | Backing services, Elasticsearch, and model endpoint for the full pipeline |
| `dev/REACT_PORT_RUNBOOK.md` | Notes on the ReAct controller port |

Top-level `DATASETS.md`, `BASELINES.md`, `METHOD_MAP.md`, and
`GAPS_IMPLEMENTED.md` cover corpora, compared systems, the method-to-code map,
and the end-to-end QA runbook.

## Internal development notes (Chinese)

Files named `*.zh.md` or `*_zh.md` in this directory and in `ablations/` and
`baselines/` are the authors' internal working notes: server runbooks, refactor
plans, rerun plans, and metric audits written during development. They are kept
for provenance. They are not maintained, may describe superseded
configurations, and use the earlier action vocabulary (zoom/read/jump/verify)
rather than the paper's entry classes (see the terminology table in the
top-level `README.md`). They are not needed to reproduce the paper;
`REPRODUCING.md` is.
