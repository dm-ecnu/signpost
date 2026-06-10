#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_signpost_method.sh <dataset> [variant] [namespace]}"
VARIANT="${2:-full}"
NAMESPACE="${3:-$DATASET}"

EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
LIMIT="${LIMIT:-}"
SIGNPOST_QUERY_WORKERS="${SIGNPOST_QUERY_WORKERS:-${V2_QUERY_WORKERS:-1}}"
USE_ES="${USE_ES:-1}"
USE_LLM="${USE_LLM:-1}"
OFFLINE_STAGES="${OFFLINE_STAGES:-F7_structure_graph F8_sequence_graph F9_unified_graph F10_graph_es_sync}"

PROCESSED="datasets/processed/${DATASET}"
OUT="outputs/${DATASET}"
LOG="${OUT}/logs/stage_timing.jsonl"
METHOD="signpost"
METHOD_ID="signpost.${VARIANT}"

mkdir -p "${OUT}/logs" "${OUT}/predictions" "${OUT}/metrics"

CMD=(python -m signpost.agent.batch
  --namespace "${NAMESPACE}"
  --dataset "${DATASET}"
  --questions "${PROCESSED}/questions.jsonl"
  --output "${OUT}/predictions/${METHOD_ID}.jsonl"
  --embedding-provider "${EMBEDDING_PROVIDER}"
  --signpost-variant "${VARIANT}"
  --query-log "${OUT}/logs/${METHOD_ID}.query.jsonl"
  --workers "${SIGNPOST_QUERY_WORKERS}")

if [[ "${USE_ES}" == "1" ]]; then
  CMD+=(--use-es)
fi
if [[ "${USE_LLM}" == "1" ]]; then
  CMD+=(--use-llm)
else
  CMD+=(--no-use-llm)
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "[signpost-method] dataset=${DATASET} namespace=${NAMESPACE} variant=${VARIANT} embedding=${EMBEDDING_PROVIDER}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "F15_agent_batch_${METHOD_ID}" \
  --method-scope online_query \
  --method "${METHOD_ID}" \
  --log "${LOG}" \
  --input-path "${PROCESSED}/questions.jsonl" \
  --output-path "${OUT}/predictions/${METHOD_ID}.jsonl" \
  --disk-path "${OUT}/predictions/${METHOD_ID}.jsonl" \
  --auto-metrics \
  -- \
  "${CMD[@]}"

python -m signpost.benchmark.time_stage \
  --dataset "${DATASET}" \
  --stage "F16_basic_eval_${METHOD_ID}" \
  --method-scope evaluation \
  --method "${METHOD_ID}" \
  --log "${LOG}" \
  --input-path "${OUT}/predictions/${METHOD_ID}.jsonl" \
  --output-path "${OUT}/metrics/${METHOD_ID}.basic_eval.json" \
  --auto-metrics \
  -- \
  python -m signpost.evaluation.evaluate_basic \
    --input "${OUT}/predictions/${METHOD_ID}.jsonl" \
    --output "${OUT}/metrics/${METHOD_ID}.basic_eval.json" \
    --normalize

python -m signpost.benchmark.query_metrics \
  --input "${OUT}/predictions/${METHOD_ID}.jsonl" \
  --output "${OUT}/metrics/${METHOD_ID}.query_metrics.json" \
  --normalize \
  --top-k 5 10

METHOD_SUMMARY_CMD=(python -m signpost.benchmark.method_summary
  --method "${METHOD_ID}"
  --dataset "${DATASET}"
  --query-metrics "${OUT}/metrics/${METHOD_ID}.query_metrics.json"
  --stage-log "${LOG}"
  --output "${OUT}/metrics/method_summaries.json")

for stage in ${OFFLINE_STAGES}; do
  METHOD_SUMMARY_CMD+=(--offline-stage "${stage}")
done

"${METHOD_SUMMARY_CMD[@]}"

python -m signpost.benchmark.cost_quality \
  --methods "${OUT}/metrics/method_summaries.json" \
  --output "${OUT}/metrics/cost_quality.json"

echo "[signpost-method] done dataset=${DATASET} variant=${VARIANT}"
