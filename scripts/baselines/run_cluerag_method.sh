#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_cluerag_method.sh <dataset> [namespace]}"
NAMESPACE="${2:-$DATASET}"
PROCESSED_DATASET="${PROCESSED_DATASET:-$DATASET}"

LIMIT="${LIMIT:-}"
REPO_PATH="${CLUERAG_REPO_PATH:-baselines/ClueRAG}"
ALPHA="${ALPHA:-1.0}"
SELECT_METRIC="${SELECT_METRIC:-COSINE}"
RUN_OFFICIAL="${RUN_OFFICIAL:-0}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
OFFICIAL_OUTPUT_DIR="${OFFICIAL_OUTPUT_DIR:-}"
BACKEND="${CLUERAG_BACKEND:-shared_es}"
USE_ES="${USE_ES:-1}"
MODE="${CLUERAG_SEARCH_MODE:-${MODE:-hybrid}}"
DIRECT_TOP_K="${DIRECT_TOP_K:-10}"
KU_TOP_K="${KU_TOP_K:-3}"
GRAPH_TOP_K="${GRAPH_TOP_K:-5}"
TOP_N="${TOP_N:-5}"
DEPTH="${DEPTH:-3}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
REUSE_GRAPH="${REUSE_GRAPH:-0}"
LLM_PROCESSES="${LLM_PROCESSES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
RERANK_URL="${RERANK_URL:-${CLUERAG_RERANK_URL:-http://127.0.0.1:8033/v1/rerank}}"
RERANK_MODEL="${RERANK_MODEL:-${CLUERAG_RERANK_MODEL:-/data/srl/llama-nemotron-rerank-1b-v2}}"
PROMPT_STYLE="${CLUERAG_PROMPT_STYLE:-${PROMPT_STYLE:-adapter}}"
GENERATION_ONLY="${CLUERAG_GENERATION_ONLY:-${GENERATION_ONLY:-0}}"
METHOD_NAME="${CLUERAG_METHOD_NAME:-${METHOD_NAME:-cluerag}}"
if [[ "${PROMPT_STYLE}" == "signpost_fewshot" && "${METHOD_NAME}" == "cluerag" ]]; then
  METHOD_NAME="cluerag_prompt_normalized"
fi
SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:-${CLUERAG_SOURCE_OUTPUT_DIR:-outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00}}"
GENERATION_OUTPUT_DIR="${GENERATION_OUTPUT_DIR:-${CLUERAG_GENERATION_OUTPUT_DIR:-outputs/${DATASET}/baselines/${METHOD_NAME}/shared_outputs/COSINE_1.00}}"

if [[ "${RUN_OFFICIAL}" == "1" ]]; then
  BACKEND="official_oceanbase"
fi

PROCESSED="datasets/processed/${PROCESSED_DATASET}"
OUT="outputs/${DATASET}"
LOG="${OUT}/logs/stage_timing.jsonl"
PREDICTIONS="${OUT}/predictions/${METHOD_NAME}.jsonl"
QUERY_LOG="${OUT}/logs/${METHOD_NAME}.query.jsonl"
RUN_METRICS="${OUT}/baselines/${METHOD_NAME}/run_metrics.json"

mkdir -p "${OUT}/logs" "${OUT}/predictions" "${OUT}/metrics" "${OUT}/baselines/${METHOD_NAME}"

PREPARE_CMD=(python -m scripts.baselines.run_cluerag
  --dataset "${DATASET}"
  --namespace "${NAMESPACE}"
  --repo-path "${REPO_PATH}"
  --documents "${PROCESSED}/documents.jsonl"
  --questions "${PROCESSED}/questions.jsonl"
  --prepare-only)

if [[ -n "${LIMIT}" ]]; then
  PREPARE_CMD+=(--limit "${LIMIT}")
fi

echo "[cluerag] dataset=${DATASET} namespace=${NAMESPACE} backend=${BACKEND} repo=${REPO_PATH} method=${METHOD_NAME} prompt=${PROMPT_STYLE}"

if [[ "${CONVERT_ONLY}" != "1" && "${GENERATION_ONLY}" != "1" && "${REUSE_GRAPH}" != "1" ]]; then
  python -m signpost.benchmark.time_stage \
    --dataset "${DATASET}" \
    --stage "baseline_prepare_cluerag" \
    --method-scope preprocessing \
    --method cluerag \
    --log "${LOG}" \
    --input-path "${PROCESSED}/documents.jsonl" \
    --input-path "${PROCESSED}/questions.jsonl" \
    --output-path "${OUT}/baselines/cluerag/manifest.json" \
    --auto-metrics \
    -- \
    "${PREPARE_CMD[@]}"
fi

if [[ "${GENERATION_ONLY}" == "1" ]]; then
  GENERATE_CMD=(python -m scripts.baselines.run_cluerag
    --dataset "${DATASET}"
    --namespace "${NAMESPACE}"
    --repo-path "${REPO_PATH}"
    --documents "${PROCESSED}/documents.jsonl"
    --questions "${PROCESSED}/questions.jsonl"
    --chunks "${PROCESSED}/chunks.jsonl"
    --output "${PREDICTIONS}"
    --query-log "${QUERY_LOG}"
    --prompt-style "${PROMPT_STYLE}"
    --method-name "${METHOD_NAME}"
    --source-output-dir "${SOURCE_OUTPUT_DIR}"
    --generation-output-dir "${GENERATION_OUTPUT_DIR}"
    --generation-only)

  python -m signpost.benchmark.time_stage \
    --dataset "${DATASET}" \
    --stage "baseline_${METHOD_NAME}_generation" \
    --method-scope external_pipeline \
    --method "${METHOD_NAME}" \
    --log "${LOG}" \
    --input-path "${SOURCE_OUTPUT_DIR}/retrieval_results.json" \
    --output-path "${PREDICTIONS}" \
    --disk-path "${PREDICTIONS}" \
    --disk-path "${GENERATION_OUTPUT_DIR}" \
    --metrics-json "${RUN_METRICS}" \
    --auto-metrics \
    -- \
    "${GENERATE_CMD[@]}"
elif [[ "${CONVERT_ONLY}" == "1" ]]; then
  CONVERT_CMD=(python -m scripts.baselines.run_cluerag
    --dataset "${DATASET}"
    --namespace "${NAMESPACE}"
    --repo-path "${REPO_PATH}"
    --documents "${PROCESSED}/documents.jsonl"
    --questions "${PROCESSED}/questions.jsonl"
    --output "${PREDICTIONS}"
    --query-log "${QUERY_LOG}"
    --method-name "${METHOD_NAME}"
    --convert-only)

  if [[ -n "${OFFICIAL_OUTPUT_DIR}" ]]; then
    CONVERT_CMD+=(--official-output-dir "${OFFICIAL_OUTPUT_DIR}")
  fi

  python -m signpost.benchmark.time_stage \
    --dataset "${DATASET}" \
    --stage "baseline_convert_cluerag" \
    --method-scope output_conversion \
    --method "${METHOD_NAME}" \
    --log "${LOG}" \
    --output-path "${PREDICTIONS}" \
    --disk-path "${PREDICTIONS}" \
    --auto-metrics \
    -- \
    "${CONVERT_CMD[@]}"
elif [[ "${BACKEND}" == "shared_es" || "${BACKEND}" == "shared_local" ]]; then
  RUN_CMD=(python -m scripts.baselines.run_cluerag
    --dataset "${DATASET}"
    --namespace "${NAMESPACE}"
    --repo-path "${REPO_PATH}"
    --documents "${PROCESSED}/documents.jsonl"
    --questions "${PROCESSED}/questions.jsonl"
    --chunks "${PROCESSED}/chunks.jsonl"
    --semantic-extractions "${PROCESSED}/semantic_llm.extractions.jsonl"
    --output "${PREDICTIONS}"
    --query-log "${QUERY_LOG}"
    --prompt-style "${PROMPT_STYLE}"
    --method-name "${METHOD_NAME}"
    --mode "${MODE}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --direct-top-k "${DIRECT_TOP_K}"
    --ku-top-k "${KU_TOP_K}"
    --graph-top-k "${GRAPH_TOP_K}"
    --top-n "${TOP_N}"
    --depth "${DEPTH}"
    --run-shared)

  if [[ "${BACKEND}" == "shared_es" && "${USE_ES}" == "1" ]]; then
    RUN_CMD+=(--use-es)
  fi
  if [[ "${REUSE_GRAPH}" == "1" ]]; then
    RUN_CMD+=(--reuse-graph)
  fi
  if [[ -n "${LIMIT}" ]]; then
    RUN_CMD+=(--limit "${LIMIT}")
  fi
  if [[ -n "${RERANK_URL}" ]]; then
    RUN_CMD+=(--rerank-url "${RERANK_URL}")
  fi
  if [[ -n "${RERANK_MODEL}" ]]; then
    RUN_CMD+=(--rerank-model "${RERANK_MODEL}")
  fi

  python -m signpost.benchmark.time_stage \
    --dataset "${DATASET}" \
    --stage "baseline_cluerag_full" \
    --method-scope external_pipeline \
    --method "${METHOD_NAME}" \
    --log "${LOG}" \
    --input-path "${OUT}/baselines/cluerag/manifest.json" \
    --input-path "${PROCESSED}/chunks.jsonl" \
    --input-path "${PROCESSED}/semantic_llm.extractions.jsonl" \
    --output-path "${PREDICTIONS}" \
    --disk-path "${PREDICTIONS}" \
    --disk-path "${OUT}/baselines/cluerag/shared_outputs" \
    --metrics-json "${RUN_METRICS}" \
    --auto-metrics \
    -- \
    "${RUN_CMD[@]}"
elif [[ "${BACKEND}" == "official_oceanbase" ]]; then
  RUN_CMD=(python -m scripts.baselines.run_cluerag
    --dataset "${DATASET}"
    --namespace "${NAMESPACE}"
    --repo-path "${REPO_PATH}"
    --documents "${PROCESSED}/documents.jsonl"
    --questions "${PROCESSED}/questions.jsonl"
    --output "${PREDICTIONS}"
    --query-log "${QUERY_LOG}"
    --prompt-style "${PROMPT_STYLE}"
    --method-name "${METHOD_NAME}"
    --alpha "${ALPHA}"
    --select-metric "${SELECT_METRIC}"
    --llm-processes "${LLM_PROCESSES}"
    --num-processes "${NUM_PROCESSES}"
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
    --run-official)

  if [[ -n "${LIMIT}" ]]; then
    RUN_CMD+=(--limit "${LIMIT}")
  fi
  if [[ -n "${RERANK_URL}" ]]; then
    RUN_CMD+=(--rerank-url "${RERANK_URL}")
  fi

  python -m signpost.benchmark.time_stage \
    --dataset "${DATASET}" \
    --stage "baseline_cluerag_full" \
    --method-scope external_pipeline \
    --method "${METHOD_NAME}" \
    --log "${LOG}" \
    --input-path "${OUT}/baselines/cluerag/manifest.json" \
    --output-path "${PREDICTIONS}" \
    --disk-path "${PREDICTIONS}" \
    --metrics-json "${RUN_METRICS}" \
    --auto-metrics \
    -- \
    "${RUN_CMD[@]}"
else
  echo "[cluerag] unknown CLUERAG_BACKEND=${BACKEND}. Use shared_es, shared_local, or official_oceanbase." >&2
  exit 2
fi

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "baseline_eval_cluerag" \
  --method-scope evaluation \
  --method "${METHOD_NAME}" \
  --log "${LOG}" \
  --input-path "${PREDICTIONS}" \
  --output-path "${OUT}/metrics/${METHOD_NAME}.basic_eval.json" \
  --auto-metrics \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input "${PREDICTIONS}" \
    --output "${OUT}/metrics/${METHOD_NAME}.basic_eval.json" \
    --normalize

python -m signpost.benchmark.query_metrics \
  --input "${PREDICTIONS}" \
  --output "${OUT}/metrics/${METHOD_NAME}.query_metrics.json" \
  --normalize \
  --top-k 5 10

METHOD_SUMMARY_CMD=(python -m signpost.benchmark.method_summary
  --method "${METHOD_NAME}"
  --dataset "${DATASET}"
  --query-metrics "${OUT}/metrics/${METHOD_NAME}.query_metrics.json"
  --stage-log "${LOG}"
  --output "${OUT}/metrics/method_summaries.json")

if [[ "${METHOD_NAME}" == "cluerag" ]]; then
  METHOD_SUMMARY_CMD+=(--offline-stage baseline_prepare_cluerag --offline-stage baseline_cluerag_full)
elif [[ "${GENERATION_ONLY}" == "1" ]]; then
  METHOD_SUMMARY_CMD+=(--offline-stage "baseline_${METHOD_NAME}_generation")
fi

"${METHOD_SUMMARY_CMD[@]}"

python -m signpost.benchmark.cost_quality \
  --methods "${OUT}/metrics/method_summaries.json" \
  --output "${OUT}/metrics/cost_quality.json"

echo "[cluerag] done dataset=${DATASET}"
