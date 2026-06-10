#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_signpost_dataset_pipeline.sh <dataset> [namespace]}"
NAMESPACE="${2:-$DATASET}"

SEMANTIC_EXTRACTOR="${SEMANTIC_EXTRACTOR:-llm}"
GLEANING_ROUNDS="${GLEANING_ROUNDS:-0}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
MAX_TOKENS="${MAX_TOKENS:-512}"
OVERLAP_TOKENS="${OVERLAP_TOKENS:-64}"
SUMMARIZER="${SUMMARIZER:-deterministic}"

PROCESSED="datasets/processed/${DATASET}"
OUT="outputs/${DATASET}"
LOG="${OUT}/logs/stage_timing.jsonl"

mkdir -p "${OUT}/logs/stage_metrics" "${OUT}/predictions" "${OUT}/metrics" "${PROCESSED}"

echo "[signpost-pipeline] dataset=${DATASET} namespace=${NAMESPACE} semantic=${SEMANTIC_EXTRACTOR} embedding=${EMBEDDING_PROVIDER}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F3_data_prepare \
  --method-scope shared_preprocess \
  --log "${LOG}" \
  --output-path "${PROCESSED}/raw_corpus.jsonl" \
  --disk-path "${PROCESSED}" \
  --auto-metrics \
  -- \
  python -m signpost.data.prepare --datasets "${DATASET}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F3_5_parse_documents \
  --method-scope shared_preprocess \
  --log "${LOG}" \
  --input-path "${PROCESSED}/raw_corpus.jsonl" \
  --output-path "${PROCESSED}/documents.jsonl" \
  --disk-path "${PROCESSED}/documents.jsonl" \
  --auto-metrics \
  -- \
  python -m signpost.parsing.parse_documents \
    --input "${PROCESSED}/raw_corpus.jsonl" \
    --output "${PROCESSED}/documents.jsonl"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F4_chunk_tree \
  --method-scope shared_preprocess \
  --log "${LOG}" \
  --input-path "${PROCESSED}/documents.jsonl" \
  --output-path "${PROCESSED}/chunks.jsonl" \
  --disk-path "${PROCESSED}" \
  --auto-metrics \
  -- \
  python -m signpost.chunking.run \
    --input "${PROCESSED}/documents.jsonl" \
    --output "${PROCESSED}/chunks.jsonl" \
    --tree-output "${PROCESSED}/document_trees.jsonl" \
    --max-tokens "${MAX_TOKENS}" \
    --overlap-tokens "${OVERLAP_TOKENS}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F5_chunk_index \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --input-path "${PROCESSED}/chunks.jsonl" \
  --output-path "${OUT}/logs/F5_chunk_index.done" \
  --auto-metrics \
  -- \
  python -m signpost.indexing.chunk_index \
    --namespace "${NAMESPACE}" \
    --dataset-id "${DATASET}" \
    --chunks "${PROCESSED}/chunks.jsonl" \
    --embedding-provider "${EMBEDDING_PROVIDER}" \
    --recreate

if [[ "${SEMANTIC_EXTRACTOR}" == "llm" ]]; then
  SEMANTIC_OUTPUT="${PROCESSED}/graph.semantic.llm.json"
  SEMANTIC_STAGE="F6_semantic_graph_llm"
else
  SEMANTIC_OUTPUT="${PROCESSED}/graph.semantic.json"
  SEMANTIC_STAGE="F6_semantic_graph_det"
fi

SEMANTIC_CMD=(python -m signpost.indexing.semantic_graph
  --namespace "${NAMESPACE}"
  --chunks "${PROCESSED}/chunks.jsonl"
  --output "${SEMANTIC_OUTPUT}"
  --extractor "${SEMANTIC_EXTRACTOR}")

if [[ "${SEMANTIC_EXTRACTOR}" == "llm" ]]; then
  SEMANTIC_CMD+=(--gleaning-rounds "${GLEANING_ROUNDS}"
    --progress-file "${PROCESSED}/semantic_llm.progress.jsonl"
    --extractions-cache "${PROCESSED}/semantic_llm.extractions.jsonl"
    --llm-retries "${LLM_RETRIES:-3}"
    --llm-timeout "${LLM_TIMEOUT:-300}"
    --retry-sleep "${RETRY_SLEEP:-5}")
fi

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "${SEMANTIC_STAGE}" \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --input-path "${PROCESSED}/chunks.jsonl" \
  --output-path "${SEMANTIC_OUTPUT}" \
  --disk-path "${SEMANTIC_OUTPUT}" \
  --auto-metrics \
  -- \
  "${SEMANTIC_CMD[@]}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F7_structure_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --input-path "${PROCESSED}/chunks.jsonl" \
  --output-path "${PROCESSED}/graph.structure.json" \
  --disk-path "${PROCESSED}/graph.structure.json" \
  --auto-metrics \
  -- \
  python -m signpost.indexing.structure_graph \
    --namespace "${NAMESPACE}" \
    --chunks "${PROCESSED}/chunks.jsonl" \
    --document-trees "${PROCESSED}/document_trees.jsonl" \
    --output "${PROCESSED}/graph.structure.json" \
    --summarizer "${SUMMARIZER}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F8_sequence_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --input-path "${PROCESSED}/chunks.jsonl" \
  --output-path "${PROCESSED}/graph.sequence.json" \
  --disk-path "${PROCESSED}/graph.sequence.json" \
  --auto-metrics \
  -- \
  python -m signpost.indexing.sequence_graph \
    --namespace "${NAMESPACE}" \
    --chunks "${PROCESSED}/chunks.jsonl" \
    --output "${PROCESSED}/graph.sequence.json"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F9_unified_graph \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --output-path "${PROCESSED}/graph.unified.json" \
  --disk-path "${PROCESSED}/graph.unified.json" \
  --auto-metrics \
  -- \
  python -m signpost.graph.merge \
    --namespace "${NAMESPACE}" \
    --semantic "${SEMANTIC_OUTPUT}" \
    --structure "${PROCESSED}/graph.structure.json" \
    --sequence "${PROCESSED}/graph.sequence.json" \
    --output "${PROCESSED}/graph.unified.json"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage F10_graph_es_sync \
  --method-scope method_offline_index \
  --method signpost \
  --log "${LOG}" \
  --input-path "${PROCESSED}/graph.unified.json" \
  --output-path "${OUT}/logs/F10_graph_es_sync.done" \
  --auto-metrics \
  -- \
  python -m signpost.indexing.graph_es_sync \
    --namespace "${NAMESPACE}" \
    --graph "${PROCESSED}/graph.unified.json" \
    --embedding-provider "${EMBEDDING_PROVIDER}" \
    --recreate \
    --update-chunk-parents

python -m signpost.benchmark.index_metrics \
  --stage-log "${LOG}" \
  --semantic-cache "${PROCESSED}/semantic_llm.extractions.jsonl" \
  --graph "${PROCESSED}/graph.unified.json" \
  --gleaning-rounds "${GLEANING_ROUNDS}" \
  --output "${OUT}/metrics/index_metrics.json"

echo "[signpost-pipeline] done dataset=${DATASET}"
