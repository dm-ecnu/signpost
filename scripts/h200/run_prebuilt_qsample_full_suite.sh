#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${1:?usage: run_prebuilt_qsample_full_suite.sh <source-dataset> <namespace> <prebuilt-dir> <output-dataset> <expected-questions>}"
NAMESPACE="${2:?usage: run_prebuilt_qsample_full_suite.sh <source-dataset> <namespace> <prebuilt-dir> <output-dataset> <expected-questions>}"
PREBUILT_DIR="${3:?usage: run_prebuilt_qsample_full_suite.sh <source-dataset> <namespace> <prebuilt-dir> <output-dataset> <expected-questions>}"
OUT_DATASET="${4:?usage: run_prebuilt_qsample_full_suite.sh <source-dataset> <namespace> <prebuilt-dir> <output-dataset> <expected-questions>}"
EXPECTED_QUESTIONS="${5:?usage: run_prebuilt_qsample_full_suite.sh <source-dataset> <namespace> <prebuilt-dir> <output-dataset> <expected-questions>}"

PROJECT_DIR="${PROJECT_DIR:-/home/srl/signpost_re_v2}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="${LOG_FILE:-/home/srl/${OUT_DATASET}_prebuilt_full_suite_${STAMP}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[prebuilt-suite] start source=${SOURCE_DATASET} namespace=${NAMESPACE} prebuilt=${PREBUILT_DIR} output=${OUT_DATASET} expected=${EXPECTED_QUESTIONS} log=${LOG_FILE}"
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

export PROJECT_DIR
export RAG_PROJECT_BASE="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}"
export ECNU_API_BASE="${ECNU_API_BASE:-http://localhost:8000/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"
export ECNU_CHAT_MODEL="${ECNU_CHAT_MODEL:-/data/srl/Llama-3.3-70B-FP8}"
export ECNU_EMBEDDING_API_BASE="${ECNU_EMBEDDING_API_BASE:-http://localhost:8001/v1/embeddings}"
export OPENAI_EMBEDDING_API_BASE="${OPENAI_EMBEDDING_API_BASE:-http://localhost:8001/v1/embeddings}"
export ECNU_EMBEDDING_MODEL="${ECNU_EMBEDDING_MODEL:-/data/srl/nemotron-8b}"
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ecnu}"
export V2_QUERY_WORKERS="${V2_QUERY_WORKERS:-1}"
export SIGNPOST_QUERY_WORKERS="${SIGNPOST_QUERY_WORKERS:-${V2_QUERY_WORKERS}}"
export BASELINE_QUERY_WORKERS="${BASELINE_QUERY_WORKERS:-${V2_QUERY_WORKERS}}"
export BASELINE_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE:-32}"
export BASELINE_EMBED_RETRIES="${BASELINE_EMBED_RETRIES:-3}"
export BASELINE_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP:-5}"
export GRAPHRAG_R1_API_BASE="${GRAPHRAG_R1_API_BASE:-http://127.0.0.1:8002/v1}"
export GRAPHRAG_R1_CHAT_MODEL="${GRAPHRAG_R1_CHAT_MODEL:-/data/srl/GraphRAG-R1}"
export HIPRAG_API_BASE="${HIPRAG_API_BASE:-http://127.0.0.1:8003/v1}"
export HIPRAG_CHAT_MODEL="${HIPRAG_CHAT_MODEL:-/data/srl/HiPRAG-7B}"

test -s "datasets/processed/${SOURCE_DATASET}/questions.jsonl"
test -s "datasets/processed/${SOURCE_DATASET}/chunks.jsonl"
test -s "datasets/processed/${SOURCE_DATASET}/semantic_llm.extractions.jsonl"
test -s "datasets/processed/${SOURCE_DATASET}/graph.unified.json"
test -s "${PREBUILT_DIR}/questions.jsonl"

curl -fsS http://127.0.0.1:9200 >/tmp/prebuilt_es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/prebuilt_chat.ok
curl -fsS http://localhost:8001/v1/models >/tmp/prebuilt_embed.ok
curl -fsS "${GRAPHRAG_R1_API_BASE%/}/models" >/tmp/prebuilt_gr1.ok
curl -fsS "${HIPRAG_API_BASE%/}/models" >/tmp/prebuilt_hiprag.ok

echo "[prebuilt-suite] dataset scale for full source"
python scripts/h200/report_dataset_scale.py \
  --root "${PROJECT_DIR}" \
  --dataset "${SOURCE_DATASET}" \
  --top-docs "${SCALE_TOP_DOCS:--1}"

install_args=(
  python scripts/h200/install_prebuilt_qsample.py
  --root "${PROJECT_DIR}"
  --source-dataset "${SOURCE_DATASET}"
  --prebuilt-dir "${PREBUILT_DIR}"
  --output-dataset "${OUT_DATASET}"
  --expected-questions "${EXPECTED_QUESTIONS}"
  --overwrite
  --require-target-silver
)

if [[ "${EXPECTED_QUESTIONS}" -ge "${REQUIRE_TARGET_SILVER_MIN_QUESTIONS:-100}" ]]; then
  install_args+=(--copy-static)
fi

echo "[prebuilt-suite] install prebuilt qsample"
"${install_args[@]}"

echo "[prebuilt-suite] run full v2 suite"
LOG_FILE="/home/srl/${OUT_DATASET}_v2_all_${STAMP}.log" \
  scripts/h200/run_v2_dataset_all.sh "${OUT_DATASET}" "${NAMESPACE}" "${OUT_DATASET}"

echo "[prebuilt-suite] verify outputs"
methods=(
  vanilla_llm
  hybrid_rag
  cluerag_prompt_normalized
  agrag
  linearrag
  hiprag
  graphrag_r1
  signpost.full
  signpost.no_offline
  signpost.no_online
  signpost.no_semantic_cues
  signpost.no_provenance_cues
  signpost.no_vertical_cues
  signpost.no_horizontal_cues
)

for method in "${methods[@]}"; do
  pred="outputs/${OUT_DATASET}/predictions/${method}.jsonl"
  test -s "${pred}"
  lines="$(wc -l < "${pred}")"
  echo "${method} rows=${lines}"
  if [[ "${lines}" -ne "${EXPECTED_QUESTIONS}" ]]; then
    echo "[prebuilt-suite] ${method} expected ${EXPECTED_QUESTIONS} rows, got ${lines}" >&2
    exit 1
  fi
done

echo "[prebuilt-suite] done output=${OUT_DATASET}"
date
