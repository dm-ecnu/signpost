#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-legal}"
NAMESPACE="${2:-$DATASET}"
PROJECT_DIR="${RAG_PROJECT_BASE:-/data/srl/signpost_re}"
STAMP="$(date +%Y%m%d_%H%M)"
RUN_LOG="/home/srl/${DATASET}_after_f9_auto_recovery_${STAMP}.log"

exec > >(tee -a "${RUN_LOG}") 2>&1

cd "${PROJECT_DIR}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "signpost-re" ]]; then
  echo "[env] expected active conda env signpost-re, got ${CONDA_DEFAULT_ENV:-none}" >&2
  exit 1
fi

set -a
source .env.h200
set +a

export PYTHONPATH="${PROJECT_DIR}"
export RAG_PROJECT_BASE="${PROJECT_DIR}"
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES=3
export LLM_TIMEOUT=300
export RETRY_SLEEP=5
export BASELINE_EMBED_BATCH_SIZE=32
export BASELINE_EMBED_RETRIES=3
export BASELINE_EMBED_RETRY_SLEEP=5

PROCESSED="datasets/processed/${DATASET}"
OUT="outputs/${DATASET}"
LOG="${OUT}/logs/stage_timing.jsonl"

echo "[start] dataset=${DATASET} namespace=${NAMESPACE} log=${RUN_LOG}"
date

test -s "${PROCESSED}/chunks.jsonl"
test -s "${PROCESSED}/questions.jsonl"
test -s "${PROCESSED}/semantic_llm.extractions.jsonl"
test -s "${PROCESSED}/graph.semantic.llm.json"
test -s "${PROCESSED}/graph.structure.json"
test -s "${PROCESSED}/graph.sequence.json"
test -s "${PROCESSED}/graph.unified.json"

python -m scripts.h200.run_f10_auto_recovery \
  --dataset "${DATASET}" \
  --namespace "${NAMESPACE}" \
  --batch-size 1 \
  --max-attempts "${F10_AUTO_MAX_ATTEMPTS:-20}" \
  --repeat-threshold "${F10_AUTO_REPEAT_THRESHOLD:-3}"

python -m signpost.benchmark.index_metrics \
  --stage-log "${LOG}" \
  --semantic-cache "${PROCESSED}/semantic_llm.extractions.jsonl" \
  --graph "${PROCESSED}/graph.unified.json" \
  --gleaning-rounds "${GLEANING_ROUNDS}" \
  --output "${OUT}/metrics/index_metrics.json"

scripts/run_signpost_ablation_suite.sh "${DATASET}" "${NAMESPACE}"

scripts/baselines/run_baseline_method.sh vanilla_llm "${DATASET}" "${NAMESPACE}"
scripts/baselines/run_baseline_method.sh vanilla_rag "${DATASET}" "${NAMESPACE}"
scripts/baselines/run_baseline_method.sh hybrid_rag "${DATASET}" "${NAMESPACE}"

export CLUERAG_BACKEND=shared_es
export CLUERAG_EMBED_BATCH_SIZE=32
export CLUERAG_EMBED_RETRIES=3
export CLUERAG_EMBED_RETRY_SLEEP=5
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://127.0.0.1:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"

export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
export CLUERAG_SOURCE_OUTPUT_DIR="outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00"
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"
unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR

export USE_ES=1 MODE=hybrid MAX_CONTEXT_TOKENS=3500
scripts/baselines/run_baseline_method.sh agrag "${DATASET}" "${NAMESPACE}"
scripts/baselines/run_baseline_method.sh linearrag "${DATASET}" "${NAMESPACE}"

export USE_ES=1 MODE=hybrid MAX_CONTEXT_TOKENS=2500
export HIPRAG_API_BASE="${HIPRAG_API_BASE:-http://127.0.0.1:8003/v1}"
export HIPRAG_CHAT_MODEL="${HIPRAG_CHAT_MODEL:-/data/srl/HiPRAG-7B}"
scripts/baselines/run_baseline_method.sh hiprag "${DATASET}" "${NAMESPACE}"
export GRAPHRAG_R1_API_BASE="${GRAPHRAG_R1_API_BASE:-http://127.0.0.1:8002/v1}"
export GRAPHRAG_R1_CHAT_MODEL="${GRAPHRAG_R1_CHAT_MODEL:-/data/srl/GraphRAG-R1}"
scripts/baselines/run_baseline_method.sh graphrag_r1 "${DATASET}" "${NAMESPACE}"

echo "[metrics] recompute all basic/query/method/cost metrics"

for pred in "${OUT}"/predictions/*.jsonl; do
  method="$(basename "$pred" .jsonl)"
  python -m signpost.evaluation.evaluate_basic \
    --input "$pred" \
    --output "${OUT}/metrics/${method}.basic_eval.json" \
    --normalize
  python -m signpost.benchmark.query_metrics \
    --input "$pred" \
    --output "${OUT}/metrics/${method}.query_metrics.json" \
    --normalize --top-k 5 10
done

printf '[]\n' > "${OUT}/metrics/method_summaries.json"

for qm in "${OUT}"/metrics/*.query_metrics.json; do
  method="$(basename "$qm" .query_metrics.json)"
  cmd=(python -m signpost.benchmark.method_summary
    --method "$method"
    --dataset "$DATASET"
    --query-metrics "$qm"
    --stage-log "$LOG"
    --output "${OUT}/metrics/method_summaries.json")

  case "$method" in
    signpost.*)
      cmd+=(--offline-stage F5_chunk_index --offline-stage F6_semantic_graph_llm --offline-stage F7_structure_graph --offline-stage F8_sequence_graph --offline-stage F9_unified_graph --offline-stage F10_graph_es_sync)
      ;;
    vanilla_rag|hybrid_rag)
      cmd+=(--offline-stage F5_chunk_index)
      ;;
    cluerag)
      cmd+=(--offline-stage baseline_prepare_cluerag --offline-stage baseline_cluerag_full)
      ;;
    cluerag_prompt_normalized)
      cmd+=(--offline-stage baseline_cluerag_prompt_normalized_generation)
      ;;
    agrag|linearrag|hiprag|graphrag_r1)
      cmd+=(--offline-stage "baseline_${method}")
      ;;
  esac

  "${cmd[@]}"
done

python -m signpost.benchmark.cost_quality \
  --methods "${OUT}/metrics/method_summaries.json" \
  --workload-sizes 10 50 100 500 1000 5000 10000 \
  --output "${OUT}/metrics/cost_quality.json"

ANALYSIS_DIR="analysis_${DATASET}"
if [[ -s "${ANALYSIS_DIR}/targets/silver_evidence_chunks.jsonl" \
   && -s "${ANALYSIS_DIR}/targets/target_entities.jsonl" \
   && -s "${ANALYSIS_DIR}/targets/target_units.jsonl" \
   && -s "${ANALYSIS_DIR}/targets/claim_units.jsonl" ]]; then
  python -m signpost.benchmark.final_metrics \
    --predictions-dir "${OUT}/predictions" \
    --targets-dir "${ANALYSIS_DIR}/targets" \
    --output-dir "${ANALYSIS_DIR}" \
    --chunks-file "${PROCESSED}/chunks.jsonl" \
    --offline-stage-timing "${LOG}" \
    --online-stage-timing "${LOG}"
else
  echo "[metrics] skip final_metrics: ${ANALYSIS_DIR}/targets incomplete"
fi

echo "[done] dataset=${DATASET}"
date
