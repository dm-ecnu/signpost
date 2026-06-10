# SignPost

[![CI](https://github.com/dm-ecnu/signpost/actions/workflows/ci.yml/badge.svg)](https://github.com/dm-ecnu/signpost/actions/workflows/ci.yml)

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
| `tests/` | Unit tests; the four suites in the quickstart run fully offline |
| `GAPS_IMPLEMENTED.md` | What was added on top of the original system + how to re-run experiments |
| `METHOD_MAP.md` | Paper concept → code file:symbol map, with paper-vs-code deltas |

Vendored third-party baseline repositories (ClueRAG, HiPRAG, …) and benchmark
corpora are **not** included; fetch them from their upstreams (see
`docs/baseline_harness.zh.md`).

## Quick start (reviewers start here)

Zero-setup check on a fresh clone — no Elasticsearch, no LLM, no corpus, runs in
seconds (Python 3.11–3.12):

```bash
git clone https://github.com/dm-ecnu/signpost.git && cd signpost
pip install -e '.[test]'      # offline-test deps only (pyyaml + pytest)
make test                     # or: python -m pytest tests/test_sketch_chaining.py \
                              #         tests/test_stats_ci.py tests/test_iso_call_baseline.py \
                              #         tests/test_silver_builder.py -q
```

Expected: `79 passed`. These suites exercise the serving mechanisms directly —
deterministic sketch chaining (Alg. 3), the iso-call attribution baseline,
bootstrap CIs, and the in-repo silver-evidence builder — without any external
service.

## Deployment tiers

| Tier | What runs | Needs |
|---|---|---|
| **T0 — offline tests** | the quickstart above; serving-mechanism + builder logic | nothing beyond `pip install -e '.[test]'` |
| **T1 — full pipeline** | offline construction + online serving over a real corpus | `pip install -r requirements.txt`, the backing services, an Elasticsearch instance, and an OpenAI-compatible LLM key |

For **T1**:

```bash
pip install -r requirements.txt
cp conf/service_conf.example.yaml conf/service_conf.yaml      # then fill it in
make services-up                                              # Postgres / Valkey / MinIO via docker/docker-compose.yml
# bring up your own Elasticsearch (see docs/environment_setup.md)
```

Credentials are read from the environment — never commit them:

- `ECNU_API_KEY` (or `OPENAI_API_KEY`) — chat/embedding endpoint key
- `ECNU_EMBEDDING_API_KEY` / `OPENAI_EMBEDDING_API_KEY` — optional separate embedding key

`docker/docker-compose.yml` starts Postgres, Valkey (Redis), and MinIO;
Elasticsearch and the LLM endpoint are external (see `docs/environment_setup.md`).

## Paper ↔ code

`METHOD_MAP.md` maps each paper concept (section / algorithm / equation) to its
implementing file and symbol, with an honest-scope section on paper-vs-code
deltas. `GAPS_IMPLEMENTED.md` records what was added on top of the original
system and how to re-run the experiments.

## Running experiments

See `GAPS_IMPLEMENTED.md` for the experiment runbook: sketch chaining is on by
default (`AgentConfig.enable_sketch_chaining=True`); the iso-call baseline runs
via `scripts/baselines/run_baseline_method.sh iso_call <dataset>`; report
confidence intervals with `signpost.benchmark.stats.summarize_with_ci` /
`paired_bootstrap_diff`.

## License

Apache-2.0 (see `LICENSE`).
