# SignPost

[![CI](https://github.com/junjie-yao/signpost/actions/workflows/ci.yml/badge.svg)](https://github.com/junjie-yao/signpost/actions/workflows/ci.yml)

SignPost is a materialized action interface for agentic graph RAG serving.
Offline, it builds source-backed objects (chunks, summaries, entities,
relations) over a document corpus and compiles, for every retrievable object, a
*signpost sketch*: four ranked cue lists for the actions **zoom** (broader
scope), **read** (adjacent source text), **jump** (semantic relation), and
**verify** (file-line provenance). Online, a fixed two-LLM-call controller
retrieves objects already carrying sketches and follows typed cues
deterministically — no per-hop LLM calls, no per-query graph reconstruction.

## Repository layout

| Path | What it is |
|---|---|
| `signpost/chunking/` | Heading detection, source-line tree, tree-guided chunker |
| `signpost/graph/` | Summary hierarchy, entity/relation extraction, four edge families |
| `signpost/indexing/` | Dual Elasticsearch index (BM25 + dense), graph sync |
| `signpost/retrieval/` | Sketch compilation, cue records, submodular cue coverage, RRF fusion |
| `signpost/agent/` | Serving controller; `sketch_chaining.py` = deterministic multi-hop cue following (visited set, priority `verify > read > zoom > jump`) |
| `signpost/baselines/` | Baseline adapters, incl. `iso_call.py` (untyped-neighbor ReAct at an equal LLM-call budget) |
| `signpost/benchmark/` | Metrics and statistics, incl. bootstrap CIs and paired bootstrap tests |
| `signpost/llm/` | OpenAI-compatible client (ECNU or any compatible endpoint) |
| `scripts/` | Dataset preparation, build-and-score pipelines, evaluation suites |
| `tests/` | Unit tests; the three suites below run fully offline |
| `GAPS_IMPLEMENTED.md` | What was added on top of the original system + how to re-run experiments |

Vendored third-party baseline repositories (ClueRAG, HiPRAG, …) and benchmark
corpora are **not** included; fetch them from their upstreams (see
`docs/baseline_harness.zh.md`).

## Setup

Requires Python 3.11–3.12.

```bash
pip install -e '.[test]'
cp conf/service_conf.example.yaml conf/service_conf.yaml   # then fill it in
```

Credentials are read from the environment — never commit them:

- `ECNU_API_KEY` (or `OPENAI_API_KEY`) — chat/embedding endpoint key
- `ECNU_EMBEDDING_API_KEY` / `OPENAI_EMBEDDING_API_KEY` — optional separate embedding key

The full pipeline additionally needs Elasticsearch (and optionally
Postgres/Redis/MinIO, see `conf/service_conf.example.yaml` and
`docs/environment_setup.md`).

## Offline tests (no ES / LLM / corpus needed)

```bash
python -m pytest tests/test_sketch_chaining.py tests/test_stats_ci.py tests/test_iso_call_baseline.py -q
# expect: 66 passed
```

## Running experiments

See `GAPS_IMPLEMENTED.md` for the experiment runbook: sketch chaining is on by
default (`AgentConfig.enable_sketch_chaining=True`); the iso-call baseline runs
via `scripts/baselines/run_baseline_method.sh iso_call <dataset>`; report
confidence intervals with `signpost.benchmark.stats.summarize_with_ci` /
`paired_bootstrap_diff`.

## License

Apache-2.0 (see `LICENSE`).
