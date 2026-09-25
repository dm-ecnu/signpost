# experiments/

Scripts and frozen outputs behind the paper's tables and in-text numbers.
`../docs/REPRODUCING.md` maps each paper item to the files here.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Measurement scripts (run against a built corpus) and analysis scripts (read `results/`) |
| `results/efficiency/` | Table 2: per-corpus `runs.jsonl`, `manifest.json`, `complete.json` (SHA-256 of the runs file), `trace.json` (request sequence) |
| `results/maintenance/` | Table 3: `e4_<corpus>.json` |
| `results/costmodel/` | Maintenance cost model: `x4_<corpus>.json` features, `fit5.json` (five corpora), `fit6.json` (with MuSiQue held out) |
| `results/exposure/` | Per-budget entry counts (`cue_exposure_<corpus>.json`) and `bsweep.json` |
| `results/coverage/` | Coverage-greedy vs score-prefix selection at b = 8 |
| `results/attribution/` | Which entry classes expose the spans read |
| `results/perquery/` | Per-question judge scores used by Tables 5–6 and the bootstrap CIs |
| `results/e1e2/` | RQ6 pilot: judged answers per arm, `audit.json` |

## Measurement scripts (no model, Elasticsearch, or network calls)

Each takes a project root containing `signpost/` and
`datasets/processed/<corpus>/graph.unified.json`.

```bash
python scripts/benchmark.py --root . --dataset mix --out results/efficiency     # Table 2
python scripts/e4_incremental.py --graph datasets/processed/mix/graph.unified.json --dataset mix --out results/maintenance  # Table 3
python scripts/x4_features.py --graph ... --dataset mix --e4 results/maintenance/e4_mix.json --out results/costmodel
python scripts/x4_fit.py results/costmodel [results/maintenance/e4_musique.json]
python scripts/x3_canonical.py --root . --datasets mix --out results/order        # load-order invariance
python scripts/x5_depth_sweep.py --root . --dataset mix --out results/depth       # b in {1,...,64,inf}
python scripts/recount_cue_exposure.py --graph ... --dataset mix --out results/exposure
```

`benchmark.py` refuses to overwrite an existing runs file. Measurement outputs
record the SHA-256 of the graph and of the script; the copies here are
byte-identical to the ones that produced `results/`.

## Analysis scripts

`gen_physical_table.py` (Table 2), `analyze_e4.py` (Table 3),
`analyze_bsweep.py`, `analyze_e3.py`, `analyze_ci.py`, `analyze_blind*.py`
read `results/` directly. `analyze_t0*.py`, `analyze_breadth_heldout.py`, and
`f1_breadth_heldout.py` read the raw prediction logs of the end-to-end runs
(`data/topb8-outputs/`, 57 MB), which are not bundled.

## Scripts that call a model

`blind_judge.py` and `rejudge.py` re-score answers with an OpenAI-compatible
endpoint; they read the key from `LITELLM_MASTER_KEY`. Their outputs are
bundled in `results/perquery/` and `results/e1e2/`, so the tables can be
checked without rerunning them.
