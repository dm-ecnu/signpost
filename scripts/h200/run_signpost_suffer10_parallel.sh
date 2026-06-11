#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/srl/signpost_re_FULL_20260609_run}"
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"
EXPECTED_ROWS="${EXPECTED_ROWS:-10}"

DATASETS=(
  "agriculture_suffer10_20260609:agriculture"
  "mix_suffer10_20260609:mix"
  "legal_suffer10_20260609:legal"
  "graphrag_bench_medical_suffer10_20260609:graphrag-bench-medical"
  "graphrag_bench_novel_suffer10_20260609:graphrag-bench-novel"
  "musique_suffer10_20260609:musique"
)

VARIANTS=(
  full
  no_offline
  no_online
  no_semantic_cues
  no_provenance_cues
  no_vertical_cues
  no_horizontal_cues
)

cd "${PROJECT_DIR}"

for conda_sh in \
  /home/srl/miniforge3/etc/profile.d/conda.sh \
  /home/srl/miniconda3/etc/profile.d/conda.sh \
  /home/srl/anaconda3/etc/profile.d/conda.sh \
  /opt/anaconda3/etc/profile.d/conda.sh \
  /opt/conda/etc/profile.d/conda.sh; do
  if [[ -f "${conda_sh}" ]]; then
    source "${conda_sh}"
    break
  fi
done
conda activate signpost-re

export PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}"
export RAG_PROJECT_BASE="${PROJECT_DIR}"
export ECNU_API_BASE="${ECNU_API_BASE:-http://localhost:8000/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"
export ECNU_API_KEY="${ECNU_API_KEY:-EMPTY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export ECNU_CHAT_MODEL="${ECNU_CHAT_MODEL:-/data/srl/Llama-3.3-70B-FP8}"
export ECNU_EMBEDDING_API_BASE="${ECNU_EMBEDDING_API_BASE:-http://localhost:8001/v1/embeddings}"
export OPENAI_EMBEDDING_API_BASE="${OPENAI_EMBEDDING_API_BASE:-http://localhost:8001/v1/embeddings}"
export ECNU_EMBEDDING_API_KEY="${ECNU_EMBEDDING_API_KEY:-EMPTY}"
export OPENAI_EMBEDDING_API_KEY="${OPENAI_EMBEDDING_API_KEY:-EMPTY}"
export ECNU_EMBEDDING_MODEL="${ECNU_EMBEDDING_MODEL:-/data/srl/nemotron-8b}"
export ECNU_RERANK_MODEL="${ECNU_RERANK_MODEL:-/data/srl/llama-nemotron-rerank-1b-v2}"
export SIGNPOST_RERANK_URL="${SIGNPOST_RERANK_URL:-http://localhost:8033/v1/rerank}"
export SIGNPOST_RERANK_MODEL="${SIGNPOST_RERANK_MODEL:-/data/srl/llama-nemotron-rerank-1b-v2}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
export LLM_RETRIES="${LLM_RETRIES:-5}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
export RETRY_SLEEP="${RETRY_SLEEP:-5}"
export SIGNPOST_QUERY_WORKERS=1
export V2_QUERY_WORKERS=1
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p run_logs

rows_in_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo 0
    return
  fi
  wc -l < "${path}"
}

run_prediction_eval_query() {
  local dataset="$1"
  local namespace="$2"
  local variant="$3"
  local method_id="signpost.${variant}"
  local out="outputs/${dataset}"
  local processed="datasets/processed/${dataset}"
  local pred="${out}/predictions/${method_id}.jsonl"
  local log="${out}/logs/stage_timing.jsonl"
  local task_log="run_logs/${dataset}_${method_id}.log"

  mkdir -p "${out}/logs" "${out}/predictions" "${out}/metrics"
  if [[ "$(rows_in_file "${pred}")" -eq "${EXPECTED_ROWS}" ]]; then
    echo "[skip] ${dataset} ${method_id} already has ${EXPECTED_ROWS} rows" | tee -a "${task_log}"
  else
    echo "[run] ${dataset} ${method_id} namespace=${namespace}" | tee "${task_log}"
    python -m signpost.benchmark.time_stage \
      --dataset "${dataset}" \
      --stage "F15_agent_batch_${method_id}" \
      --method-scope online_query \
      --method "${method_id}" \
      --log "${log}" \
      --input-path "${processed}/questions.jsonl" \
      --output-path "${pred}" \
      --disk-path "${pred}" \
      --auto-metrics \
      -- \
      python -m signpost.agent.batch \
        --namespace "${namespace}" \
        --dataset "${dataset}" \
        --questions "${processed}/questions.jsonl" \
        --output "${pred}" \
        --embedding-provider "${EMBEDDING_PROVIDER}" \
        --signpost-variant "${variant}" \
        --query-log "${out}/logs/${method_id}.query.jsonl" \
        --workers 1 \
        --use-es \
        --use-llm >> "${task_log}" 2>&1
  fi

  python -m signpost.benchmark.time_stage \
    --dataset "${dataset}" \
    --stage "F16_basic_eval_${method_id}" \
    --method-scope evaluation \
    --method "${method_id}" \
    --log "${log}" \
    --input-path "${pred}" \
    --output-path "${out}/metrics/${method_id}.basic_eval.json" \
    --auto-metrics \
    -- \
    python -m signpost.evaluation.evaluate_basic \
      --input "${pred}" \
      --output "${out}/metrics/${method_id}.basic_eval.json" \
      --normalize >> "${task_log}" 2>&1

  python -m signpost.benchmark.query_metrics \
    --input "${pred}" \
    --output "${out}/metrics/${method_id}.query_metrics.json" \
    --normalize \
    --top-k 5 10 >> "${task_log}" 2>&1
  echo "[done] ${dataset} ${method_id}" | tee -a "${task_log}"
}

summarize_dataset() {
  local dataset="$1"
  local out="outputs/${dataset}"
  local log="${out}/logs/stage_timing.jsonl"
  rm -f "${out}/metrics/method_summaries.json" "${out}/metrics/cost_quality.json"
  for variant in "${VARIANTS[@]}"; do
    local method_id="signpost.${variant}"
    if [[ "$(rows_in_file "${out}/predictions/${method_id}.jsonl")" -ne "${EXPECTED_ROWS}" ]]; then
      echo "[summarize] missing ${dataset} ${method_id}" >&2
      return 1
    fi
    python -m signpost.benchmark.method_summary \
      --method "${method_id}" \
      --dataset "${dataset}" \
      --query-metrics "${out}/metrics/${method_id}.query_metrics.json" \
      --stage-log "${log}" \
      --output "${out}/metrics/method_summaries.json" \
      --offline-stage F7_structure_graph \
      --offline-stage F8_sequence_graph \
      --offline-stage F9_unified_graph \
      --offline-stage F10_graph_es_sync
  done
  python -m signpost.benchmark.cost_quality \
    --methods "${out}/metrics/method_summaries.json" \
    --output "${out}/metrics/cost_quality.json"
}

active=0
failures=0
pids=()
for spec in "${DATASETS[@]}"; do
  dataset="${spec%%:*}"
  namespace="${spec##*:}"
  for variant in "${VARIANTS[@]}"; do
    run_prediction_eval_query "${dataset}" "${namespace}" "${variant}" &
    pids+=("$!")
    active=$((active + 1))
    if [[ "${active}" -ge "${PARALLEL_JOBS}" ]]; then
      wait -n || failures=$((failures + 1))
      active=$((active - 1))
    fi
  done
done

while [[ "${active}" -gt 0 ]]; do
  wait -n || failures=$((failures + 1))
  active=$((active - 1))
done

if [[ "${failures}" -ne 0 ]]; then
  echo "[parallel] failed tasks=${failures}" >&2
  exit 1
fi

for spec in "${DATASETS[@]}"; do
  summarize_dataset "${spec%%:*}"
done

echo "[parallel] done datasets=${#DATASETS[@]} variants=${#VARIANTS[@]} parallel=${PARALLEL_JOBS}"
