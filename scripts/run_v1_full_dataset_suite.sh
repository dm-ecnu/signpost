#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_v1_full_dataset_suite.sh <dataset> [namespace]}"
NAMESPACE="${2:-$DATASET}"
METHOD_PREFIX="${METHOD_PREFIX:-signpost.full_rerank_v1}"

export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
export USE_ES="${USE_ES:-1}"
export USE_LLM="${USE_LLM:-1}"
export MODE="${MODE:-hybrid}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-3500}"
export BASELINE_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE:-32}"
export BASELINE_EMBED_RETRIES="${BASELINE_EMBED_RETRIES:-3}"
export BASELINE_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP:-5}"
export CLUERAG_BACKEND="${CLUERAG_BACKEND:-shared_es}"
export CLUERAG_RERANK_URL="${CLUERAG_RERANK_URL:-http://127.0.0.1:8033/v1/rerank}"
export CLUERAG_RERANK_MODEL="${CLUERAG_RERANK_MODEL:-/data/srl/llama-nemotron-rerank-1b-v2}"

run_baseline_if_missing() {
  local method="$1"
  if [[ -s "outputs/${DATASET}/predictions/${method}.jsonl" ]]; then
    echo "[skip] ${DATASET} ${method} exists"
  else
    scripts/baselines/run_baseline_method.sh "${method}" "${DATASET}" "${NAMESPACE}"
  fi
}

run_cluerag_if_missing() {
  if [[ ! -s "outputs/${DATASET}/predictions/cluerag.jsonl" || ! -s "outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00/retrieval_results.json" ]]; then
    unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR
    scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"
  else
    echo "[skip] ${DATASET} cluerag exists"
  fi

  if [[ -s "outputs/${DATASET}/predictions/cluerag_prompt_normalized.jsonl" ]]; then
    echo "[skip] ${DATASET} cluerag_prompt_normalized exists"
  else
    export CLUERAG_GENERATION_ONLY=1
    export CLUERAG_PROMPT_STYLE=signpost_fewshot
    export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
    export CLUERAG_SOURCE_OUTPUT_DIR="outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00"
    scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"
    unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR
  fi
}

date
scripts/run_signpost_ablation_suite_variant.sh "${DATASET}" "${NAMESPACE}" "${METHOD_PREFIX}"

for method in vanilla_llm vanilla_rag hybrid_rag agrag linearrag hiprag graphrag_r1; do
  run_baseline_if_missing "${method}"
done

run_cluerag_if_missing

python scripts/build_all_and_score.py \
  --root "${RAG_PROJECT_BASE:-$(pwd)}" \
  --project-dir "${RAG_PROJECT_BASE:-$(pwd)}" \
  --dataset "${DATASET}" \
  --clean-all \
  --clean-ans

wc -l "outputs/${DATASET}/predictions/${METHOD_PREFIX}.jsonl"
find "ans/${DATASET}" -maxdepth 1 -name "*.txt" | wc -l
date
