# Legal F10 Graph-Index Embedding Auto-Recovery

This note records the H200 recovery mechanism for the Legal dataset F10 graph
Elasticsearch sync. It is an engineering recovery path for the graph-object
embedding service, not an additional LLM component.

## Problem

During Legal F10 (`F10_graph_es_sync`), the local H200 embedding service
(`http://localhost:8001/v1/embeddings`, model `/data/srl/nemotron-8b`) can return
HTTP 500 and terminate with a vLLM `EngineDeadError`. The failure happens during
graph-object embedding after F6/F7/F8/F9 have already completed.

Restarting F10 with the previous script is wasteful because the original F10
path used `--recreate` and did not checkpoint successful graph objects. A failed
run therefore rebuilt the graph ES index from the beginning.

## Recovery Policy

The recovery path keeps the default Signpost F10 behavior unchanged unless the
H200 auto-recovery runner is explicitly used.

Default behavior:

```text
one graph object -> one ES document -> one content_vector
no truncation
no splitting
```

Auto-recovery behavior for Legal:

1. F10 writes progress for every vector document to
   `outputs/legal/logs/F10_graph_es_sync.progress.jsonl`.
2. On failure, the runner restarts the `embed` tmux service using the same H200
   command:

   ```bash
   CUDA_VISIBLE_DEVICES=2 VLLM_USE_DEEP_GEMM=0 python -m vllm.entrypoints.openai.api_server --model /data/srl/nemotron-8b --runner pooling --port 8001 --trust-remote-code
   ```

3. F10 resumes from the checkpoint instead of rebuilding already indexed graph
   objects.
4. If the same original graph object fails repeatedly, only that graph object is
   converted to multi-vector ES subdocuments. Normal graph objects remain in the
   original one-document, one-vector format.
5. Multi-vector fallback does not truncate content. It splits the failed graph
   object's full `content` into equal non-overlapping windows and indexes each
   window as a vector subdocument. Retrieval de-duplicates subdocument hits back
   to the original `graph_parent_id`.

## Metrics and Audit Artifacts

Recovery decisions are recorded in:

```text
outputs/legal/logs/F10_graph_es_sync.progress.jsonl
outputs/legal/logs/F10_graph_es_sync.state.json
outputs/legal/logs/F10_graph_es_sync.multivector_parts.json
outputs/legal/logs/F10_graph_es_sync.recovery_decisions.jsonl
outputs/legal/logs/F10_graph_es_sync.multivector_objects.jsonl
```

`stage_timing.jsonl` retains each F10 attempt, including failed attempts. The
paper metrics should use successful-stage aggregation for final cost summaries,
while the failed rows remain available for audit.

## Scope

This recovery path is explicitly enabled by the Legal H200 runner. Existing
Agriculture and Mix results are not modified. Future runs of other datasets keep
the original F10 behavior unless the recovery runner is explicitly invoked.
