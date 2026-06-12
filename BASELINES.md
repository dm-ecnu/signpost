# Baselines

The project benchmark compares
Signpost against **eight** baselines, grouped to make the comparison current
rather than canonical: a closed-book control, a flat-retrieval control, an
index-centric graph method, two graph-optimization methods, a single-round
graph-memory retriever, and two multi-round agentic graph retrievers.

Every external citation key below was verified to resolve in
`references.bib`; verbatim entries are in `CITATIONS.bib`. The two in-house
controls (Vanilla LLM, HybridRAG) have no project citation — they are project
controls.

## Grouping (as the benchmark setup presents it)

| Category | Method | Role (one line) | Config compared | Citation key | Baseline code |
|---|---|---|---|---|---|
| Closed-book control | **Vanilla LLM** | No-retrieval parametric-knowledge floor | direct generation, no retrieval | *(in-house, none)* | in-repo: `signpost/baselines/vanilla_llm.py` |
| Flat-retrieval control | **HybridRAG** | Single-round BM25+dense over chunks | `MODE=hybrid USE_ES=1`, hybrid top-`k`=5 fusion | *(in-house, none)* | in-repo: `signpost/baselines/hybrid_rag.py` |
| Index-centric | **LinearRAG** | Linear-structure index: entity activation + PageRank over a relation-free Tri-Graph | offline index, per-query selection | `zhuang2025linearrag` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/linearrag.py` |
| Graph optimization | **AGRAG** | Solves a query-specific subgraph (PPR / MCMI high-order association) at request time | offline graph build, online subgraph retrieval | `agrag2026` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/agrag.py` |
| Graph optimization | **Clue-RAG** | Query-driven beam search over a knapsack-budgeted multi-partite graph | normalized-prompt run over its own retrieval (quality row) | `cluerag2026` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/cluerag.py` |
| Graph memory (single-round) | **MemGraphRAG** | Single-round memory-walk retriever | reported with single-round baselines; **not** in the multi-step evidence-navigation comparison | `wu2026memgraphrag` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/memgraphrag.py` |
| Multi-round agentic | **HiPRAG** | Hierarchical-process-reward agentic retrieval (multi-round) | as-is (did not complete MuSiQue — context overflow) | `wu2026hiprag` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/hiprag.py` |
| Multi-round agentic | **GraphRAG-R1** | RL-trained (process-constrained) agentic graph traversal (multi-round) | original-offline configuration | `graphragr12026` | vendored `baselines/` (gitignored), fetch upstream; adapter `signpost/baselines/graphrag_r1.py` |

The two **multi-round agentic** methods (HiPRAG, GraphRAG-R1) are the most direct
comparisons for online cost and evidence navigation; they are the only external
methods reported in the silver-navigation table (`tab:silver`) and the
per-query online cost table (`tab:online`). MemGraphRAG, although a graph-memory
method, is deliberately reported as single-round and excluded from the
multi-step navigation comparison.

## Vendored vs in-house code

- **In-house controls** (Vanilla LLM, HybridRAG) and **all baseline adapters**
  live in this repo under `signpost/baselines/` with run scripts under
  `scripts/baselines/`. The adapters constrain every method to one experiment
  interface (`datasets/processed/<dataset>/{documents,chunks,questions}.jsonl`
  in; `outputs/<dataset>/...` out) — see `docs/baseline_harness.zh.md`.
- **Vendored third-party baseline repositories** (ClueRAG, HiPRAG, LinearRAG,
  AGRAG, GraphRAG-R1, MemGraphRAG official code/weights) are **not** included in
  the repo: the `/baselines/` directory is gitignored (see `.gitignore`). Fetch
  each from its upstream and place it there. Upstream pointers (from
  `docs/baselines/final_baseline_selection_zh.md`):
  - LinearRAG — `https://github.com/DEEP-PolyU/LinearRAG`
  - Clue-RAG — `https://github.com/Feesuu/ClueRAG`
  - AGRAG — `https://github.com/Wyb0627/AGRAG`
  - HiPRAG — `https://github.com/qualidea1217/HiPRAG`
  - GraphRAG-R1 — `https://github.com/ycygit/GraphRAG-R1` (weights at `https://huggingface.co/yuchuanyue/GraphRAG-R1`)
  - MemGraphRAG — arXiv 2606.00610 (see `CITATIONS.bib`)
  See `docs/baseline_harness.zh.md` and the per-method runbooks under
  `docs/baselines/<method>/` for the adaptation procedure.

## Metrics context

All methods share the same generator family (`Llama-3.3-70B-FP8`) and the same
temperature-zero **ECNU-Plus** answer judge. Quality is scored as
`S_LLM ∈ [0,10]`, J@7 (acceptable-answer rate), and AnsRec (token-overlap answer
recall). Multi-round methods are additionally compared on LLM calls, tokens,
latency, and silver-evidence navigation. Signpost itself runs a fixed two-call
budgeted controller (Table `tab:quality`/`tab:online`) and is also stress-tested
under a multi-step ReAct controller over the same materialized cues (Table
`tab:dualpolicy`); the in-repo iso-call attribution baseline
(`signpost/baselines/iso_call.py`) runs untyped-neighbor ReAct at an equal
LLM-call budget.
