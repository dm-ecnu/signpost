#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_v2_dataset_all.sh <output-dataset> [namespace] [processed-dataset]}"
NAMESPACE="${2:-$DATASET}"
PROCESSED_DATASET="${3:-$DATASET}"

PROJECT_DIR="${PROJECT_DIR:-/home/srl/signpost_re_v2}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="${LOG_FILE:-/home/srl/${DATASET}_v2_all_${STAMP}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[v2-all] start dataset=${DATASET} namespace=${NAMESPACE} processed=${PROCESSED_DATASET} log=${LOG_FILE}"
date

cd "${PROJECT_DIR}"
for conda_sh in \
  /home/srl/miniforge3/etc/profile.d/conda.sh \
  /home/srl/miniconda3/etc/profile.d/conda.sh \
  /home/srl/anaconda3/etc/profile.d/conda.sh \
  /opt/conda/etc/profile.d/conda.sh; do
  if [[ -f "${conda_sh}" ]]; then
    source "${conda_sh}"
    break
  fi
done
conda activate signpost-re
if [[ -f .env.h200 ]]; then
  set -a
  source .env.h200
  set +a
else
  echo "[v2-all] .env.h200 not found; use explicit H200 local service defaults"
fi

export PYTHONPATH="${PROJECT_DIR}"
export RAG_PROJECT_BASE="${PROJECT_DIR}"
export ECNU_API_BASE=http://localhost:8000/v1
export OPENAI_API_BASE=http://localhost:8000/v1
export ECNU_CHAT_MODEL=/data/srl/Llama-3.3-70B-FP8
export ECNU_EMBEDDING_API_BASE=http://localhost:8001/v1/embeddings
export OPENAI_EMBEDDING_API_BASE=http://localhost:8001/v1/embeddings
export ECNU_EMBEDDING_MODEL=/data/srl/nemotron-8b
export ECNU_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export SIGNPOST_RERANK_URL=http://localhost:8033/v1/rerank
export SIGNPOST_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export CLUERAG_RERANK_URL=http://localhost:8033/v1/rerank
export CLUERAG_RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
export EMBEDDING_PROVIDER=ecnu
export SEMANTIC_EXTRACTOR=llm
export GLEANING_ROUNDS=0
export LLM_RETRIES="${LLM_RETRIES:-5}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
export RETRY_SLEEP="${RETRY_SLEEP:-5}"
export V2_QUERY_WORKERS="${V2_QUERY_WORKERS:-1}"
export SIGNPOST_QUERY_WORKERS="${SIGNPOST_QUERY_WORKERS:-${V2_QUERY_WORKERS}}"
export BASELINE_QUERY_WORKERS="${BASELINE_QUERY_WORKERS:-${V2_QUERY_WORKERS}}"
export BASELINE_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE:-32}"
export BASELINE_EMBED_RETRIES="${BASELINE_EMBED_RETRIES:-3}"
export BASELINE_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP:-5}"

export PROCESSED="datasets/processed/${PROCESSED_DATASET}"
export OUT="outputs/${DATASET}"
export BASELINE_CHUNK_INDEX="${BASELINE_CHUNK_INDEX:-baseline-v2-${DATASET}-chunks}"
export PROCESSED_DATASET

if [[ -n "${V2_PROCESSED_SOURCE_DIR:-}" ]]; then
  echo "[v2-all] link processed artifacts from ${V2_PROCESSED_SOURCE_DIR}"
  mkdir -p "datasets/processed"
  rm -rf "${PROCESSED}"
  ln -s "${V2_PROCESSED_SOURCE_DIR}" "${PROCESSED}"
fi

test -f "${PROCESSED}/questions.jsonl"
test -f "${PROCESSED}/chunks.jsonl"
test -f "${PROCESSED}/semantic_llm.extractions.jsonl"
test -f "${PROCESSED}/graph.unified.json"

curl -fsS http://127.0.0.1:9200 >/tmp/v2_es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/v2_chat_models.ok
curl -fsS http://localhost:8001/v1/models >/tmp/v2_embed_models.ok
curl -fsS http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["embedding health check"]}' >/tmp/v2_embed_smoke.ok
curl -fsS http://localhost:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"test","documents":["test document"]}' >/tmp/v2_rerank_smoke.ok

echo "[v2-all] services ok"

mkdir -p "${OUT}/logs" "${OUT}/predictions" "${OUT}/metrics" "${OUT}/baselines"

if [[ -n "${V2_BASELINE_SOURCE_DIR:-}" && "${REUSE_GRAPH:-${REUSE_CLUERAG_GRAPH:-${REUSE_BASELINE_INDEX:-0}}}" == "1" ]]; then
  if [[ -d "${V2_BASELINE_SOURCE_DIR}/cluerag/shared_graph" && ! -e "${OUT}/baselines/cluerag/shared_graph" ]]; then
    echo "[v2-all] link read-only ClueRAG shared_graph from ${V2_BASELINE_SOURCE_DIR}/cluerag/shared_graph"
    mkdir -p "${OUT}/baselines/cluerag"
    ln -s "${V2_BASELINE_SOURCE_DIR}/cluerag/shared_graph" "${OUT}/baselines/cluerag/shared_graph"
  fi
fi

if [[ "${REUSE_BASELINE_CHUNK_INDEX:-${REUSE_BASELINE_INDEX:-0}}" == "1" ]]; then
  echo "[v2-all] reuse mode: keep existing independent baseline chunk index: ${BASELINE_CHUNK_INDEX}"
else
  echo "[v2-all] build independent baseline chunk index: ${BASELINE_CHUNK_INDEX}"
  python -m signpost.indexing.chunk_index \
    --namespace "${NAMESPACE}" \
    --dataset-id "${DATASET}" \
    --chunks "${PROCESSED}/chunks.jsonl" \
    --index-name "${BASELINE_CHUNK_INDEX}" \
    --embedding-provider ecnu \
    --batch-size "${BASELINE_EMBED_BATCH_SIZE}" \
    --embedding-retries "${BASELINE_EMBED_RETRIES}" \
    --retry-sleep "${BASELINE_EMBED_RETRY_SLEEP}" \
    --recreate
fi

echo "[v2-all] Signpost full and ablations"
scripts/run_signpost_ablation_suite.sh "${PROCESSED_DATASET}" "${NAMESPACE}"
if [[ "${PROCESSED_DATASET}" != "${DATASET}" ]]; then
  mkdir -p "outputs/${DATASET}"
  rsync -a "outputs/${PROCESSED_DATASET}/" "outputs/${DATASET}/"
fi

echo "[v2-all] baselines"
LIMIT="${LIMIT:-}" USE_ES=0 scripts/baselines/run_baseline_method.sh vanilla_llm "${DATASET}" "${NAMESPACE}"

LIMIT="${LIMIT:-}" USE_ES=1 MODE=hybrid TOP_K=5 MAX_CONTEXT_TOKENS=3500 BASELINE_CHUNK_INDEX="${BASELINE_CHUNK_INDEX}" \
  scripts/baselines/run_baseline_method.sh hybrid_rag "${DATASET}" "${NAMESPACE}"

export CLUERAG_BACKEND=shared_es
export REUSE_GRAPH="${REUSE_GRAPH:-${REUSE_CLUERAG_GRAPH:-${REUSE_BASELINE_INDEX:-0}}}"
export CLUERAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}"
export CLUERAG_EMBED_RETRIES="${BASELINE_EMBED_RETRIES}"
export CLUERAG_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP}"
export USE_ES=1
export CLUERAG_SEARCH_MODE=hybrid
export DIRECT_TOP_K=10
export KU_TOP_K=3
export GRAPH_TOP_K=5
export TOP_N=5
export DEPTH=3
export RERANK_URL=http://localhost:8033/v1/rerank
export RERANK_MODEL=/data/srl/llama-nemotron-rerank-1b-v2
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"

export CLUERAG_GENERATION_ONLY=1
export CLUERAG_PROMPT_STYLE=signpost_fewshot
export CLUERAG_METHOD_NAME=cluerag_prompt_normalized
if [[ -n "${V2_BASELINE_SOURCE_DIR:-}" && "${REUSE_CLUERAG_OUTPUTS:-0}" == "1" && -d "${V2_BASELINE_SOURCE_DIR}/cluerag/shared_outputs/COSINE_1.00" ]]; then
  export CLUERAG_SOURCE_OUTPUT_DIR="${V2_BASELINE_SOURCE_DIR}/cluerag/shared_outputs/COSINE_1.00"
else
  export CLUERAG_SOURCE_OUTPUT_DIR=outputs/${DATASET}/baselines/cluerag/shared_outputs/COSINE_1.00
fi
scripts/baselines/run_cluerag_method.sh "${DATASET}" "${NAMESPACE}"
unset CLUERAG_GENERATION_ONLY CLUERAG_PROMPT_STYLE CLUERAG_METHOD_NAME CLUERAG_SOURCE_OUTPUT_DIR

export USE_ES=1 MODE=hybrid TOP_K=5 GRAPH_TOP_K=5 LINK_TOP_K=8 PPR_ALPHA=0.85 MCMI_STEPS=20 MAX_CONTEXT_TOKENS=3500
export AGRAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}"
REUSE_BASELINE_INDEX="${REUSE_AGRAG_INDEX:-${REUSE_BASELINE_INDEX:-0}}" scripts/baselines/run_baseline_method.sh agrag "${DATASET}" "${NAMESPACE}"

export USE_ES=1 MODE=hybrid MAX_CONTEXT_TOKENS=3500
export LINEARRAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}"
export LINEARRAG_RETRIEVAL_TOP_K=5 LINEARRAG_HYBRID_TOP_K=5 LINEARRAG_SEED_TOP_K=8 LINEARRAG_TOP_K_SENTENCE=1
export LINEARRAG_MAX_ITERATIONS=3 LINEARRAG_ITERATION_THRESHOLD=0.5 LINEARRAG_PASSAGE_RATIO=1.5 LINEARRAG_PASSAGE_NODE_WEIGHT=0.05 LINEARRAG_DAMPING=0.5
REUSE_BASELINE_INDEX="${REUSE_LINEARRAG_INDEX:-${REUSE_BASELINE_INDEX:-0}}" scripts/baselines/run_baseline_method.sh linearrag "${DATASET}" "${NAMESPACE}"

export USE_ES=1 MODE=hybrid MAX_CONTEXT_TOKENS=2500
export HIPRAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}" HIPRAG_SEARCH_TOP_K=3 HIPRAG_MAX_STEPS=4
export HIPRAG_API_BASE="${HIPRAG_API_BASE:-http://127.0.0.1:8003/v1}"
export HIPRAG_CHAT_MODEL="${HIPRAG_CHAT_MODEL:-/data/srl/HiPRAG-7B}"
REUSE_BASELINE_INDEX="${REUSE_HIPRAG_INDEX:-${REUSE_BASELINE_INDEX:-0}}" scripts/baselines/run_baseline_method.sh hiprag "${DATASET}" "${NAMESPACE}"

export USE_ES=1 MODE=hybrid MAX_CONTEXT_TOKENS=2500
export GRAPHRAG_R1_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}" GRAPHRAG_R1_GRAPH_TOP_K=5 GRAPHRAG_R1_CHUNK_TOP_K=5
export GRAPHRAG_R1_LINK_TOP_K=8 GRAPHRAG_R1_MAX_STEPS=4 GRAPHRAG_R1_PPR_ALPHA=0.85 GRAPHRAG_R1_PPR_ITERATIONS=20
export GRAPHRAG_R1_API_BASE="${GRAPHRAG_R1_API_BASE:-http://127.0.0.1:8002/v1}"
export GRAPHRAG_R1_CHAT_MODEL="${GRAPHRAG_R1_CHAT_MODEL:-/data/srl/GraphRAG-R1}"
REUSE_BASELINE_INDEX="${REUSE_GRAPHRAG_R1_INDEX:-${REUSE_BASELINE_INDEX:-0}}" scripts/baselines/run_baseline_method.sh graphrag_r1 "${DATASET}" "${NAMESPACE}"

echo "[v2-all] v2 target/silver metrics"
python scripts/h200_target_unit_silver_eval_v2.py \
  --root "${PROJECT_DIR}" \
  --datasets "${DATASET}" \
  --dataset-spec "${DATASET}=${PROCESSED_DATASET}:${DATASET}" \
  --output-dir "${OUT}/metrics/target_unit_silver_eval_v2"

echo "[v2-all] v2 final metrics"
python scripts/h200_final_eval_v2.py \
  --root "${PROJECT_DIR}" \
  --datasets "${DATASET}" \
  --dataset-spec "${DATASET}=${PROCESSED_DATASET}:${DATASET}:${DATASET}" \
  --output-dir "${OUT}/metrics/final_eval_v2"

echo "[v2-all] completed dataset=${DATASET} namespace=${NAMESPACE} processed=${PROCESSED_DATASET}"
date
