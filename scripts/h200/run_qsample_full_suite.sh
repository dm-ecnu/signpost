#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET="${1:?usage: run_qsample_full_suite.sh <source-dataset> <namespace> <sample-size> [output-dataset]}"
NAMESPACE="${2:?usage: run_qsample_full_suite.sh <source-dataset> <namespace> <sample-size> [output-dataset]}"
SAMPLE_SIZE="${3:?usage: run_qsample_full_suite.sh <source-dataset> <namespace> <sample-size> [output-dataset]}"
OUT_DATASET="${4:-${SOURCE_DATASET}_q${SAMPLE_SIZE}}"

PROJECT_DIR="${PROJECT_DIR:-/home/srl/signpost_re_v2}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="${LOG_FILE:-/home/srl/${OUT_DATASET}_qsample_full_suite_${STAMP}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[qsample-suite] start source=${SOURCE_DATASET} namespace=${NAMESPACE} sample=${SAMPLE_SIZE} output=${OUT_DATASET} log=${LOG_FILE}"
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

curl -fsS http://127.0.0.1:9200 >/tmp/qsample_es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/qsample_chat.ok
curl -fsS http://localhost:8001/v1/models >/tmp/qsample_embed.ok
curl -fsS "${GRAPHRAG_R1_API_BASE%/}/models" >/tmp/qsample_gr1.ok
curl -fsS "${HIPRAG_API_BASE%/}/models" >/tmp/qsample_hiprag.ok

echo "[qsample-suite] dataset scale"
python scripts/h200/report_dataset_scale.py \
  --root "${PROJECT_DIR}" \
  --dataset "${SOURCE_DATASET}" \
  --top-docs "${SCALE_TOP_DOCS:--1}"

subset_args=(
  python scripts/h200/make_question_length_subset.py
  --root "${PROJECT_DIR}"
  --source-dataset "${SOURCE_DATASET}"
  --output-dataset "${OUT_DATASET}"
  --sample-size "${SAMPLE_SIZE}"
  --overwrite
)

if [[ "${COPY_STATIC:-auto}" == "1" || ( "${COPY_STATIC:-auto}" == "auto" && "${SAMPLE_SIZE}" -ge 100 ) ]]; then
  subset_args+=(--copy-static)
fi

echo "[qsample-suite] make question-length subset"
"${subset_args[@]}"

actual_questions="$(wc -l < "datasets/processed/${OUT_DATASET}/questions.jsonl")"
echo "[qsample-suite] selected questions=${actual_questions}"
if [[ "${actual_questions}" -ne "${SAMPLE_SIZE}" ]]; then
  echo "[qsample-suite] expected ${SAMPLE_SIZE} questions, got ${actual_questions}" >&2
  exit 1
fi

target_units="datasets/processed/${OUT_DATASET}/llm_target_units.jsonl"
silver_chunks="datasets/processed/${OUT_DATASET}/llm_silver_chunks.jsonl"
target_lines=0
silver_lines=0
[[ -s "${target_units}" ]] && target_lines="$(wc -l < "${target_units}")"
[[ -s "${silver_chunks}" ]] && silver_lines="$(wc -l < "${silver_chunks}")"
if [[ "${target_lines}" -ne "${SAMPLE_SIZE}" || "${silver_lines}" -ne "${SAMPLE_SIZE}" ]]; then
  echo "[qsample-suite] target/silver missing for ${OUT_DATASET}: target_lines=${target_lines} silver_lines=${silver_lines}; expected=${SAMPLE_SIZE}" >&2
  echo "[qsample-suite] build qsample and LLM target/silver locally, sync it to H200, then rerun." >&2
  exit 1
else
  echo "[qsample-suite] target/silver already present target_lines=${target_lines} silver_lines=${silver_lines}"
fi

echo "[qsample-suite] run full v2 suite"
LOG_FILE="/home/srl/${OUT_DATASET}_v2_all_${STAMP}.log" \
  scripts/h200/run_v2_dataset_all.sh "${OUT_DATASET}" "${NAMESPACE}" "${OUT_DATASET}"

echo "[qsample-suite] verify outputs"
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
  if [[ "${lines}" -ne "${SAMPLE_SIZE}" ]]; then
    echo "[qsample-suite] ${method} expected ${SAMPLE_SIZE} rows, got ${lines}" >&2
    exit 1
  fi
done

python - <<PY
import json
from pathlib import Path
out = "${OUT_DATASET}"
for method, rel in [("graphrag_r1", "graph.json"), ("hiprag", "retrieval_index.json")]:
    p = Path("outputs") / out / "baselines" / method / rel
    if not p.exists():
        print(method, "missing", p)
        continue
    d = json.loads(p.read_text())
    print(method, "chat_model_used=", d.get("chat_model_used"), "run_mode=", d.get(method + "_run_mode"), "use_es=", d.get("uses_shared_signpost_chunk_es_index"))
PY

echo "[qsample-suite] done output=${OUT_DATASET}"
date
