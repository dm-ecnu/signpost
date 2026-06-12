# Signpost: Method-to-Code Map

Maps the Signpost concepts, symbols, and algorithms to the
implementation in `signpost/`. Stable method labels are in the left column; verified `file:symbol` locations are in the middle. All paths are
relative to `signpost/`.

The method has two stages: **OFFLINE** construct source-backed objects + typed
links and compile per-object signpost views `σ(o)=⟨C_v,C_h,C_s,C_p⟩`
(`MaterializeSignposts`, Alg. 2); **ONLINE** serve a query by following typed cues
with bounded source reads (`ServeSignpostQuery`, Alg. 3) under an O(b) per-step
exposure bound.

## Core abstractions (§3)

| Method concept (§/Alg/Eq) | Code location (file:symbol) | Note |
| --- | --- | --- |
| Evidence-bounded query execution, Problem 1 (§3.1, eq:cost) | (formulation; no single symbol) — cost accounting in `agent/batch.py` (`llm_calls`, token sums) | Objective is realized by the fixed-controller cost contract, not a solver. |
| Signpost sketch `σ(o)=⟨C_v,C_h,C_s,C_p⟩` (§3.2, eq:approach-sketch) | `retrieval/offline_signpost.py:build_offline_signpost` | Returns `{vertical, horizontal, semantic, provenance}` ≙ `C_v/C_h/C_s/C_p` (zoom/read/jump/verify). |
| Cue record `c=⟨x,target,label,locator,score,cost⟩` (§3.2, eq:cue-record) | `retrieval/cue_record.py:enrich_cue`, `enrich_offline_signpost` | `x`/`target`/`label`/`score`/`cost` are added **online**; offline stores target/locator/ids only. `score = 1/(1+rank)`, `cost = _estimate_tokens(...)`. |
| Four action families v/h/s/p = zoom/read/jump/verify (§3.2 action semantics, §4.1 operators) | `cue_record.py:_FAMILY_LABEL`, `sketch_chaining.py:_FAMILY_NAME` | `{"v":"zoom","h":"read","s":"jump","p":"verify"}`. |
| Navigation state `s_t=⟨q_i,F_t,H_t,R_t,B_t⟩` (§3.2, eq:nav-state) | `agent/sketch_chaining.py:SketchChainer` (`H_t`, `R_t`, `frontier`, `read_budget`) | `B_t` = `read_budget` (ReadFile call budget); `F_t` = `self.frontier`. |
| Object substrate `G_D=(V,E,τ,μ)` (§3.3, Alg. 1 BuildSignpostIndex) | `graph/unified.py:build_unified_graph` (merge/validate) | Nodes `{chunk, summary, entity}` only; see delta on `V_r`. |

## Offline construction (§3.3 substrate, §4 materialization)

| Method concept (§/Alg) | Code location (file:symbol) | Note |
| --- | --- | --- |
| Heading recognition (Alg. 1) | `chunking/headers.py:recognize_headers` | Deterministic Markdown/Chinese/English/dotted patterns + optional LLM path. |
| Source-line-preserving tree `T_d` (Alg. 1) | `chunking/tree.py:build_document_tree` | Stack parser. |
| Tree-guided chunker → chunks `V_c` (Alg. 1) | `chunking/chunker.py:chunk_document` | Fold short sections / split oversized nodes under `b_chunk`; keeps locators + prev/next. |
| Structural nodes `V_s` + edges `E_str`, bottom-up summaries (§3.3, Alg. 1) | `graph/structure.py:build_structure_graph` | RAPTOR-style fallback when no hierarchy. Backs `C_v` (zoom). |
| Sequential edges `E_seq` (§3.3) | `graph/sequence.py:build_sequence_graph` | Adjacent-chunk prev/next; backs `C_h` (read). |
| Semantic substrate: entities `V_e`, relations, `E_sem`, `E_prov` (§3.3) | `graph/semantic.py:build_semantic_graph` | Entity normalization + majority-vote type; relation aggregation by endpoint/label. Backs `C_s` (jump) + `C_p` (verify). |
| Physical schema: chunk index `I_C` (§3.3) | `indexing/chunk_schema.py:chunk_index_mapping`, `chunk_index_name` | Lexical + dense + locator + sketch payload. |
| Physical schema: graph-object index `I_G` (§3.3) | `indexing/graph_schema.py:graph_index_mapping`, `graph_index_name` | Summaries/entities/relations; `|I_C|+|I_G|=|O|`. |
| Candidate generation per object/operator (§4.1) | `retrieval/offline_signpost.py:_chunk_signpost`, `_summary_signpost`, `_entity_signpost`, `_relation_signpost` | Deterministic, uses only `G_D` + locators. |
| Execution fields (display vs. execution, Fig. sketch) (§4.1) | `retrieval/cue_record.py:enrich_cue` (label/target/locator/score/cost) | Label is serialized to the model; ids/line-ranges/cost stay in runtime state. |
| Source-evidence units `Φ(c)`, family relevance `ω_x` (§4.2, eq:coverage) | `retrieval/cue_coverage.py:omega_vertical/horizontal/semantic/provenance`, `CueCandidate.phi` | zoom `(1+hop)^-1`, read `(1+|Δline|)^-1`, jump `w·log(1+|S|)`, verify `cov`. |
| Coverage value `f_{o,x}(S)`, marginal gain `Δ_{o,x}` (§4.2, eq:marginal) | `cue_coverage.py:coverage_value`, `_marginal_gain` | Weighted set-coverage; each unit counted once (monotone submodular). |
| `MaterializeSignposts` complete branch (Alg. 2) | `cue_coverage.py:select_complete` | Sort by (score desc, stable id), no truncation. |
| `MaterializeSignposts` budgeted greedy branch + Prop. 1 `(1-1/e)` (Alg. 2) | `cue_coverage.py:select_budgeted_greedy`, `select_family` | Greedy marginal-coverage prefix `≤ b_x`; cues with `Φ(c)=∅` never selected. |

## Online serving (§5)

| Method concept (§/Alg) | Code location (file:symbol) | Note |
| --- | --- | --- |
| `ServeSignpostQuery` (Alg. 3), budgeted controller (§5.1) | `agent/supervisor.py:Supervisor`, `research_with_chaining` | Decompose → per-subquestion search + chain → synthesize. **2 LLM calls total.** |
| Supervisor decomposition into ≤3 subquestions (Alg. 3 line 1, §5.1) | `agent/supervisor.py:decompose`, `deterministic_decompose` | LLM with deterministic fallback. |
| Frontier `F = TopK(I_C ∪ I_G, q_i, k)` by RRF lexical/dense fusion (Alg. 3 line 4) | `retrieval/chunk_search.py:search_chunks` + `retrieval/graph_search.py:search_graph` | Hybrid fusion seeds `F_0`. |
| Sketch chaining as query-time routing (§5.1, Alg. 3 loop) | `agent/sketch_chaining.py:run_sketch_chaining`, `SketchChainer.run` | `d`-hop path = object lookups + reads, no per-hop LLM call. |
| Context adaptation: filter `H_t`, dedup vs `R_t`, family priority (§5.1) | `sketch_chaining.py:_adapt_cues`, `_filter_nav_cues` | Removes visited targets, dedups locates in `R_t`. |
| Family priority verify > read > zoom > jump (§5.1) | `sketch_chaining.py:_FAMILY_PRIORITY` = `{"p":0,"h":1,"v":2,"s":3}`; `_follow_nav_cues` order `("h","v","s")` | Verify executed first in `run()`, then nav cues. |
| Verify cues → `R_t` via ReadFile (Alg. 3 line 7) | `sketch_chaining.py:_follow_verify_cues`; `agent/tools.py:ReadFileTool`; `retrieval/read_file.py` | Each read decrements `B_t`; locate added to `H_t`. |
| Follow zoom/read/jump → successors ∉ `H_t` (Alg. 3 line 8) | `sketch_chaining.py:_follow_nav_cues`, `_resolve_successor` | Successor re-resolves its own `σ(o)` via `build_offline_signpost`. |
| Visited set `H_t` cycle/repeat suppression (§3.2, §5.1) | `sketch_chaining.py:self.H_t`, `_mark_visited` | Holds both object ids and read locates. |
| Top-`b_x` prefix per family / O(b) exposure bound, Prop. 2 (§5.3) | `sketch_chaining.py:_apply_budget`; `retrieval/signpost_variants.py:apply_cue_topb` | `cue_budget_per_family` caps each family. |
| Budgeted vs. complete modes; cue-family ablations (RQ5, §5) | `retrieval/signpost_variants.py:apply_signpost_variant`, `apply_cue_topb`, `_cue_select_mode` | `SIGNPOST_CUE_TOPB*` budgets; `SIGNPOST_CUE_SELECT` greedy/truncate. |
| Insufficient-evidence contract + final synthesis from `R_t` only (§5.1, §5.4) | `agent/supervisor.py:SYNTHESIS_SYSTEM_PROMPT`, synthesis path | Outputs cited answer or exactly "Insufficient evidence". |
| ReAct controller over the same interface (§5 dual-policy) | `react/react_adapter.py`, `baselines/iso_call.py` | Same materialized cue interface, more model calls. |
| Auxiliary online jump recommender (PPR) | `retrieval/online_signpost.py:compute_online_signpost` | Not the core loop; ablated by `NO_ONLINE`. |

## Evaluation harness (§5 experiments)

| Method concept | Code location (file:symbol) | Note |
| --- | --- | --- |
| Frozen answer-supporting (silver) evidence `E_q^*` (§5 setup) | `evaluation/silver_builder.py` (decompose → ground) | In-repo reference impl of target-units + silver-chunks. |
| Iso-call attribution control (equal LLM-call budget, untyped neighbors) | `baselines/iso_call.py` | Rules out "more calls" as the source of gains. |
| Metric aggregation / bootstrap stats | `benchmark/stats.py`, `benchmark/final_metrics.py`, `evaluation/metrics.py` | — |

## Honest scope / method-vs-implementation deltas

These are real, verified reconciliations between the method description and
the artifact. They do not change measured behavior but a careful reader should
know them.

1. **Relation objects `V_r` are stored as EDGES, not graph nodes.** The method description
   defines `V = V_c ∪ V_s ∪ V_e ∪ V_r` with reified relation *nodes*. In code,
   `graph/unified.py` admits node types `{chunk, summary, entity}` only (see the
   `node_type not in {"chunk","summary","entity"}` validation), and relation
   aggregates are emitted as semantic edges (`graph/semantic.py`, `edge_type
   "semantic_relation"`). `offline_signpost.py:_relation_signpost` therefore
   resolves an **edge** (not a node) into a `result_type:"relation"` cue. The
   jump/verify cue families and provenance behave as the benchmark setup describes; only
   the storage form of `V_r` differs.

2. **Sketches are COMPUTED AT QUERY TIME, not stored inline in ES.** The method description's
   physical-schema prose says each object record holds "four cue arrays" and a
   serialized sketch payload. In the artifact, `build_offline_signpost(graph, o)`
   recomputes `σ(o)` deterministically from the unified graph at
   retrieval/serving time (and successor resolution in `sketch_chaining.py` calls
   it again per hop). The ES schema reserves a sketch field, but the headline
   runs materialize cues on read from `graph.unified.json`. Because construction
   is deterministic and graph-only, this is equivalent in output to a stored
   sketch; it differs only in *when* the bytes are produced.

3. **Greedy submodular cue selection is GATED; the default is truncation.** The
   `(1-1/e)` budgeted-coverage greedy of §4.2 / Alg. 2 lives in
   `cue_coverage.py:select_budgeted_greedy`, but `signpost_variants.py` only
   invokes it when `SIGNPOST_CUE_SELECT=greedy`. The default (`truncate`) takes a
   positional `value[:b]` prefix of the offline-ordered list. So the budgeted-mode
   experiments use submodular selection only when that env var is set; otherwise
   they use a rank prefix. Both respect the same `b_x` exposure cap.

4. **Provenance locators are CHUNK-LEVEL.** `Φ(c)` and verify cues resolve to
   `file:Lstart-Lend` ranges derived from chunk/section/relation source
   locators (`offline_signpost.py:_locate`, `_merge_locates`;
   `cue_coverage.py:_parse_line_span`). They are as fine-grained as the stored
   chunk/occurrence line ranges, not sub-line or token-offset spans. The method description's
   "occurrence-level source lines" for relations are the per-chunk line ranges of
   each occurrence, merged per file.

5. **Sketch chaining is DETERMINISTIC — no extra LLM call per hop.** Multi-hop
   navigation (`sketch_chaining.py`) follows materialized cues, resolves
   successors via the graph index, and reads source spans with no language-model
   call inside the loop. The only LLM calls per query are the Supervisor's one
   decomposition and one synthesis (`supervisor.py:research_with_chaining`
   docstring; `decompose` + synthesis). This determinism is exactly what
   preserves the method description's **"2 LLM calls"** systems claim for the budgeted
   controller; the ReAct controller (`react/`, `baselines/iso_call.py`) trades
   more calls for adaptivity over the same interface.
