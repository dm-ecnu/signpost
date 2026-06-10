#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_baseline_method.sh <vanilla_llm|vanilla_rag|hybrid_rag|agrag|memgraphrag|linearrag|hiprag|graphrag_r1|graphrag_r1_hipporag2|iso_call> <dataset> [namespace]}"
DATASET="${2:?usage: run_baseline_method.sh <vanilla_llm|vanilla_rag|hybrid_rag|agrag|memgraphrag|linearrag|hiprag|graphrag_r1|graphrag_r1_hipporag2|iso_call> <dataset> [namespace]}"
NAMESPACE="${3:-$DATASET}"
PROCESSED_DATASET="${PROCESSED_DATASET:-$DATASET}"

LIMIT="${LIMIT:-}"
BASELINE_QUERY_WORKERS="${BASELINE_QUERY_WORKERS:-${V2_QUERY_WORKERS:-8}}"
USE_ES="${USE_ES:-1}"
MODE="${MODE:-hybrid}"
TOP_K="${TOP_K:-5}"
GRAPH_TOP_K="${GRAPH_TOP_K:-5}"
LINK_TOP_K="${LINK_TOP_K:-8}"
PPR_ALPHA="${PPR_ALPHA:-0.85}"
MCMI_STEPS="${MCMI_STEPS:-20}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-3500}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
BASELINE_CHUNK_INDEX="${BASELINE_CHUNK_INDEX:-baseline-v2-${DATASET}-chunks}"
BASELINE_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE:-32}"
BASELINE_EMBED_RETRIES="${BASELINE_EMBED_RETRIES:-3}"
BASELINE_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP:-5}"
export BASELINE_EMBED_BATCH_SIZE BASELINE_EMBED_RETRIES BASELINE_EMBED_RETRY_SLEEP
REUSE_BASELINE_INDEX="${REUSE_BASELINE_INDEX:-0}"
EFFECTIVE_REUSE_BASELINE_INDEX="${REUSE_BASELINE_INDEX}"
BASELINE_SOURCE_DIR="${BASELINE_SOURCE_DIR:-${V2_BASELINE_SOURCE_DIR:-}}"
AGRAG_EMBED_BATCH_SIZE="${AGRAG_EMBED_BATCH_SIZE:-${BASELINE_EMBED_BATCH_SIZE}}"
MEMGRAPHRAG_RETRIEVAL_TOP_K="${MEMGRAPHRAG_RETRIEVAL_TOP_K:-200}"
MEMGRAPHRAG_QA_TOP_K="${MEMGRAPHRAG_QA_TOP_K:-5}"
MEMGRAPHRAG_LINKING_TOP_K="${MEMGRAPHRAG_LINKING_TOP_K:-5}"
MEMGRAPHRAG_PPR_DAMPING="${MEMGRAPHRAG_PPR_DAMPING:-0.5}"
MEMGRAPHRAG_PPR_ITERATIONS="${MEMGRAPHRAG_PPR_ITERATIONS:-20}"
MEMGRAPHRAG_PASSAGE_NODE_WEIGHT="${MEMGRAPHRAG_PASSAGE_NODE_WEIGHT:-0.05}"
MEMGRAPHRAG_SCHEMA_MIN_COUNT="${MEMGRAPHRAG_SCHEMA_MIN_COUNT:-2}"
MEMGRAPHRAG_EMBED_BATCH_SIZE="${MEMGRAPHRAG_EMBED_BATCH_SIZE:-${BASELINE_EMBED_BATCH_SIZE}}"
MEMGRAPHRAG_SYNONYMY_EDGES="${MEMGRAPHRAG_SYNONYMY_EDGES:-1}"
MEMGRAPHRAG_SYNONYMY_EDGE_SIM_THRESHOLD="${MEMGRAPHRAG_SYNONYMY_EDGE_SIM_THRESHOLD:-0.8}"
MEMGRAPHRAG_SYNONYMY_EDGE_MAX_NEIGHBORS="${MEMGRAPHRAG_SYNONYMY_EDGE_MAX_NEIGHBORS:-100}"
export MEMGRAPHRAG_EMBED_RETRIES="${MEMGRAPHRAG_EMBED_RETRIES:-${BASELINE_EMBED_RETRIES}}"
export MEMGRAPHRAG_EMBED_RETRY_SLEEP="${MEMGRAPHRAG_EMBED_RETRY_SLEEP:-${BASELINE_EMBED_RETRY_SLEEP}}"
LINEARRAG_EMBED_BATCH_SIZE="${LINEARRAG_EMBED_BATCH_SIZE:-${BASELINE_EMBED_BATCH_SIZE}}"
export LINEARRAG_EMBED_RETRIES="${LINEARRAG_EMBED_RETRIES:-${BASELINE_EMBED_RETRIES}}"
export LINEARRAG_EMBED_RETRY_SLEEP="${LINEARRAG_EMBED_RETRY_SLEEP:-${BASELINE_EMBED_RETRY_SLEEP}}"
LINEARRAG_RETRIEVAL_TOP_K="${LINEARRAG_RETRIEVAL_TOP_K:-5}"
LINEARRAG_HYBRID_TOP_K="${LINEARRAG_HYBRID_TOP_K:-5}"
LINEARRAG_SEED_TOP_K="${LINEARRAG_SEED_TOP_K:-8}"
LINEARRAG_TOP_K_SENTENCE="${LINEARRAG_TOP_K_SENTENCE:-1}"
LINEARRAG_MAX_ITERATIONS="${LINEARRAG_MAX_ITERATIONS:-3}"
LINEARRAG_ITERATION_THRESHOLD="${LINEARRAG_ITERATION_THRESHOLD:-0.5}"
LINEARRAG_PASSAGE_RATIO="${LINEARRAG_PASSAGE_RATIO:-1.5}"
LINEARRAG_PASSAGE_NODE_WEIGHT="${LINEARRAG_PASSAGE_NODE_WEIGHT:-0.05}"
LINEARRAG_DAMPING="${LINEARRAG_DAMPING:-0.5}"
HIPRAG_SEARCH_TOP_K="${HIPRAG_SEARCH_TOP_K:-3}"
HIPRAG_MAX_STEPS="${HIPRAG_MAX_STEPS:-4}"
HIPRAG_EMBED_BATCH_SIZE="${HIPRAG_EMBED_BATCH_SIZE:-${BASELINE_EMBED_BATCH_SIZE}}"
export HIPRAG_EMBED_RETRIES="${HIPRAG_EMBED_RETRIES:-${BASELINE_EMBED_RETRIES}}"
export HIPRAG_EMBED_RETRY_SLEEP="${HIPRAG_EMBED_RETRY_SLEEP:-${BASELINE_EMBED_RETRY_SLEEP}}"
GRAPHRAG_R1_GRAPH_TOP_K="${GRAPHRAG_R1_GRAPH_TOP_K:-5}"
GRAPHRAG_R1_CHUNK_TOP_K="${GRAPHRAG_R1_CHUNK_TOP_K:-5}"
GRAPHRAG_R1_LINK_TOP_K="${GRAPHRAG_R1_LINK_TOP_K:-8}"
GRAPHRAG_R1_MAX_STEPS="${GRAPHRAG_R1_MAX_STEPS:-4}"
GRAPHRAG_R1_PPR_ALPHA="${GRAPHRAG_R1_PPR_ALPHA:-0.85}"
GRAPHRAG_R1_PPR_ITERATIONS="${GRAPHRAG_R1_PPR_ITERATIONS:-20}"
GRAPHRAG_R1_EMBED_BATCH_SIZE="${GRAPHRAG_R1_EMBED_BATCH_SIZE:-${BASELINE_EMBED_BATCH_SIZE}}"
export GRAPHRAG_R1_EMBED_RETRIES="${GRAPHRAG_R1_EMBED_RETRIES:-${BASELINE_EMBED_RETRIES}}"
export GRAPHRAG_R1_EMBED_RETRY_SLEEP="${GRAPHRAG_R1_EMBED_RETRY_SLEEP:-${BASELINE_EMBED_RETRY_SLEEP}}"
GRAPHRAG_R1_HIPPORAG2_URL="${GRAPHRAG_R1_HIPPORAG2_URL:-http://127.0.0.1:8090}"
GRAPHRAG_R1_HIPPORAG2_RETRIEVAL_NUM="${GRAPHRAG_R1_HIPPORAG2_RETRIEVAL_NUM:-5}"
GRAPHRAG_R1_HIPPORAG2_MAX_STEPS="${GRAPHRAG_R1_HIPPORAG2_MAX_STEPS:-${GRAPHRAG_R1_MAX_STEPS}}"
GRAPHRAG_R1_HIPPORAG2_TIMEOUT="${GRAPHRAG_R1_HIPPORAG2_TIMEOUT:-300}"
ISO_CALL_CALL_BUDGET="${ISO_CALL_CALL_BUDGET:-2}"
ISO_CALL_SEARCH_TOP_K="${ISO_CALL_SEARCH_TOP_K:-5}"
ISO_CALL_GRAPH_TOP_K="${ISO_CALL_GRAPH_TOP_K:-5}"

PROCESSED="datasets/processed/${PROCESSED_DATASET}"
OUT="outputs/${DATASET}"
LOG="${OUT}/logs/stage_timing.jsonl"
PREDICTIONS="${OUT}/predictions/${METHOD}.jsonl"
QUERY_LOG="${OUT}/logs/${METHOD}.query.jsonl"

mkdir -p "${OUT}/logs" "${OUT}/predictions" "${OUT}/metrics"

CMD=(python -m "scripts.baselines.run_${METHOD}"
  --dataset "${DATASET}"
  --namespace "${NAMESPACE}"
  --questions "${PROCESSED}/questions.jsonl"
  --output "${PREDICTIONS}"
  --query-log "${QUERY_LOG}"
  --workers "${BASELINE_QUERY_WORKERS}")

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

OFFLINE_STAGES=()
if [[ "${METHOD}" == "vanilla_rag" || "${METHOD}" == "hybrid_rag" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --mode "${MODE}"
    --top-k "${TOP_K}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}")
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es --chunk-index-name "${BASELINE_CHUNK_INDEX}")
    OFFLINE_STAGES=(F5_chunk_index)
  fi
elif [[ "${METHOD}" == "agrag" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --mode "${MODE}"
    --top-k "${TOP_K}"
    --graph-top-k "${GRAPH_TOP_K}"
    --link-top-k "${LINK_TOP_K}"
    --ppr-alpha "${PPR_ALPHA}"
    --mcmi-steps "${MCMI_STEPS}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --embedding-batch-size "${AGRAG_EMBED_BATCH_SIZE}")
  if [[ "${REUSE_BASELINE_INDEX}" == "1" ]]; then
    CMD+=(--reuse-index)
    if [[ -n "${BASELINE_SOURCE_DIR}" ]]; then
      CMD+=(--reuse-index-dir "${BASELINE_SOURCE_DIR}/${METHOD}")
    fi
  fi
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es)
    OFFLINE_STAGES=(F5_chunk_index "baseline_${METHOD}")
  else
    OFFLINE_STAGES=("baseline_${METHOD}")
  fi
elif [[ "${METHOD}" == "memgraphrag" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --retrieval-top-k "${MEMGRAPHRAG_RETRIEVAL_TOP_K}"
    --qa-top-k "${MEMGRAPHRAG_QA_TOP_K}"
    --linking-top-k "${MEMGRAPHRAG_LINKING_TOP_K}"
    --ppr-damping "${MEMGRAPHRAG_PPR_DAMPING}"
    --ppr-iterations "${MEMGRAPHRAG_PPR_ITERATIONS}"
    --passage-node-weight "${MEMGRAPHRAG_PASSAGE_NODE_WEIGHT}"
    --schema-min-count "${MEMGRAPHRAG_SCHEMA_MIN_COUNT}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --embedding-batch-size "${MEMGRAPHRAG_EMBED_BATCH_SIZE}"
    --synonymy-edge-sim-threshold "${MEMGRAPHRAG_SYNONYMY_EDGE_SIM_THRESHOLD}"
    --synonymy-edge-max-neighbors "${MEMGRAPHRAG_SYNONYMY_EDGE_MAX_NEIGHBORS}")
  if [[ "${MEMGRAPHRAG_SYNONYMY_EDGES}" == "1" ]]; then
    CMD+=(--synonymy-edges)
  else
    CMD+=(--no-synonymy-edges)
  fi
  if [[ "${REUSE_BASELINE_INDEX}" == "1" ]]; then
    CMD+=(--reuse-index)
    if [[ -n "${BASELINE_SOURCE_DIR}" ]]; then
      CMD+=(--reuse-index-dir "${BASELINE_SOURCE_DIR}/${METHOD}")
    fi
  fi
  OFFLINE_STAGES=("baseline_${METHOD}")
elif [[ "${METHOD}" == "linearrag" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --mode "${MODE}"
    --retrieval-top-k "${LINEARRAG_RETRIEVAL_TOP_K}"
    --hybrid-top-k "${LINEARRAG_HYBRID_TOP_K}"
    --seed-top-k "${LINEARRAG_SEED_TOP_K}"
    --top-k-sentence "${LINEARRAG_TOP_K_SENTENCE}"
    --max-iterations "${LINEARRAG_MAX_ITERATIONS}"
    --iteration-threshold "${LINEARRAG_ITERATION_THRESHOLD}"
    --passage-ratio "${LINEARRAG_PASSAGE_RATIO}"
    --passage-node-weight "${LINEARRAG_PASSAGE_NODE_WEIGHT}"
    --damping "${LINEARRAG_DAMPING}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --embedding-batch-size "${LINEARRAG_EMBED_BATCH_SIZE}")
  if [[ "${REUSE_BASELINE_INDEX}" == "1" ]]; then
    CMD+=(--reuse-index)
    if [[ -n "${BASELINE_SOURCE_DIR}" ]]; then
      CMD+=(--reuse-index-dir "${BASELINE_SOURCE_DIR}/${METHOD}")
    fi
  fi
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es)
    OFFLINE_STAGES=(F5_chunk_index "baseline_${METHOD}")
  else
    OFFLINE_STAGES=("baseline_${METHOD}")
  fi
elif [[ "${METHOD}" == "hiprag" ]]; then
  EFFECTIVE_REUSE_BASELINE_INDEX="${REUSE_HIPRAG_INDEX:-${REUSE_BASELINE_INDEX}}"
  if [[ "${USE_ES}" == "1" && -z "${REUSE_HIPRAG_INDEX+x}" ]]; then
    EFFECTIVE_REUSE_BASELINE_INDEX=0
  fi
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --mode "${MODE}"
    --search-top-k "${HIPRAG_SEARCH_TOP_K}"
    --max-steps "${HIPRAG_MAX_STEPS}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --embedding-batch-size "${HIPRAG_EMBED_BATCH_SIZE}")
  if [[ "${EFFECTIVE_REUSE_BASELINE_INDEX}" == "1" ]]; then
    CMD+=(--reuse-index)
    if [[ -n "${BASELINE_SOURCE_DIR}" ]]; then
      CMD+=(--reuse-index-dir "${BASELINE_SOURCE_DIR}/${METHOD}")
    fi
  fi
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es)
    OFFLINE_STAGES=(F5_chunk_index "baseline_${METHOD}")
  else
    OFFLINE_STAGES=("baseline_${METHOD}")
  fi
elif [[ "${METHOD}" == "graphrag_r1" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --mode "${MODE}"
    --graph-top-k "${GRAPHRAG_R1_GRAPH_TOP_K}"
    --chunk-top-k "${GRAPHRAG_R1_CHUNK_TOP_K}"
    --link-top-k "${GRAPHRAG_R1_LINK_TOP_K}"
    --max-steps "${GRAPHRAG_R1_MAX_STEPS}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --ppr-alpha "${GRAPHRAG_R1_PPR_ALPHA}"
    --ppr-iterations "${GRAPHRAG_R1_PPR_ITERATIONS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --embedding-batch-size "${GRAPHRAG_R1_EMBED_BATCH_SIZE}")
  if [[ "${REUSE_BASELINE_INDEX}" == "1" ]]; then
    CMD+=(--reuse-index)
    if [[ -n "${BASELINE_SOURCE_DIR}" ]]; then
      CMD+=(--reuse-index-dir "${BASELINE_SOURCE_DIR}/${METHOD}")
    fi
  fi
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es)
    OFFLINE_STAGES=(F5_chunk_index "baseline_${METHOD}")
  else
    OFFLINE_STAGES=("baseline_${METHOD}")
  fi
elif [[ "${METHOD}" == "graphrag_r1_hipporag2" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --artifact-dir "${OUT}/baselines/${METHOD}"
    --hipporag-url "${GRAPHRAG_R1_HIPPORAG2_URL}"
    --retrieval-num "${GRAPHRAG_R1_HIPPORAG2_RETRIEVAL_NUM}"
    --max-steps "${GRAPHRAG_R1_HIPPORAG2_MAX_STEPS}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --server-timeout-seconds "${GRAPHRAG_R1_HIPPORAG2_TIMEOUT}")
  OFFLINE_STAGES=("baseline_${METHOD}_server")
elif [[ "${METHOD}" == "iso_call" ]]; then
  CMD+=(--chunks "${PROCESSED}/chunks.jsonl"
    --call-budget "${ISO_CALL_CALL_BUDGET}"
    --search-top-k "${ISO_CALL_SEARCH_TOP_K}"
    --graph-top-k "${ISO_CALL_GRAPH_TOP_K}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --mode "${MODE}")
  if [[ -f "${PROCESSED}/graph.unified.json" ]]; then
    CMD+=(--graph "${PROCESSED}/graph.unified.json")
  fi
  if [[ "${USE_ES}" == "1" ]]; then
    CMD+=(--use-es)
    OFFLINE_STAGES=(F5_chunk_index)
  fi
elif [[ "${METHOD}" != "vanilla_llm" ]]; then
  echo "unknown method=${METHOD}; expected vanilla_llm, vanilla_rag, hybrid_rag, agrag, memgraphrag, linearrag, hiprag, graphrag_r1, graphrag_r1_hipporag2, or iso_call" >&2
  exit 2
fi

if [[ "${EFFECTIVE_REUSE_BASELINE_INDEX}" == "1" && ( "${METHOD}" == "agrag" || "${METHOD}" == "memgraphrag" || "${METHOD}" == "linearrag" || "${METHOD}" == "hiprag" || "${METHOD}" == "graphrag_r1" ) ]]; then
  OFFLINE_STAGES=()
fi

echo "[baseline] dataset=${DATASET} namespace=${NAMESPACE} method=${METHOD}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "baseline_${METHOD}" \
  --method-scope online_query \
  --method "${METHOD}" \
  --log "${LOG}" \
  --input-path "${PROCESSED}/questions.jsonl" \
  --output-path "${PREDICTIONS}" \
  --disk-path "${PREDICTIONS}" \
  --auto-metrics \
  -- \
  "${CMD[@]}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "baseline_eval_${METHOD}" \
  --method-scope evaluation \
  --method "${METHOD}" \
  --log "${LOG}" \
  --input-path "${PREDICTIONS}" \
  --output-path "${OUT}/metrics/${METHOD}.basic_eval.json" \
  --auto-metrics \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input "${PREDICTIONS}" \
    --output "${OUT}/metrics/${METHOD}.basic_eval.json" \
    --normalize

python -m signpost.benchmark.query_metrics \
  --input "${PREDICTIONS}" \
  --output "${OUT}/metrics/${METHOD}.query_metrics.json" \
  --normalize \
  --top-k 5 10

METHOD_SUMMARY_CMD=(python -m signpost.benchmark.method_summary
  --method "${METHOD}"
  --dataset "${DATASET}"
  --query-metrics "${OUT}/metrics/${METHOD}.query_metrics.json"
  --stage-log "${LOG}"
  --output "${OUT}/metrics/method_summaries.json")

for stage in "${OFFLINE_STAGES[@]}"; do
  METHOD_SUMMARY_CMD+=(--offline-stage "${stage}")
done

"${METHOD_SUMMARY_CMD[@]}"

python -m signpost.benchmark.cost_quality \
  --methods "${OUT}/metrics/method_summaries.json" \
  --output "${OUT}/metrics/cost_quality.json"

if [[ "${METHOD}" == "agrag" || "${METHOD}" == "memgraphrag" || "${METHOD}" == "linearrag" || "${METHOD}" == "hiprag" || "${METHOD}" == "graphrag_r1" || "${METHOD}" == "graphrag_r1_hipporag2" ]]; then
  GRAPH_METRICS="${OUT}/baselines/${METHOD}/graph.json"
  if [[ "${METHOD}" == "hiprag" ]]; then
    GRAPH_METRICS="${OUT}/baselines/${METHOD}/retrieval_index.json"
  fi
  python -m signpost.baselines.artifact_summary \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --query-metrics "${OUT}/metrics/${METHOD}.query_metrics.json" \
    --stage-log "${LOG}" \
    --artifact-dir "${OUT}/baselines/${METHOD}" \
    --graph-metrics "${GRAPH_METRICS}"
fi

echo "[baseline] done dataset=${DATASET} method=${METHOD}"
