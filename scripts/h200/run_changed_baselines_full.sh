#!/usr/bin/env bash
set -euo pipefail

OUT_DATASET="${1:?usage: run_changed_baselines_full.sh <output-dataset> <namespace> [processed-dataset]}"
NAMESPACE="${2:?usage: run_changed_baselines_full.sh <output-dataset> <namespace> [processed-dataset]}"
PROCESSED_DATASET="${3:-$OUT_DATASET}"

PROJECT_DIR="${PROJECT_DIR:-/home/srl/signpost_re_v2}"
STAMP="$(date +%Y%m%d_%H%M)"
LOG_FILE="${LOG_FILE:-/home/srl/${OUT_DATASET}_changed_baselines_full_${STAMP}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[changed-baselines] start output=${OUT_DATASET} namespace=${NAMESPACE} processed=${PROCESSED_DATASET} log=${LOG_FILE}"
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
export USE_ES=1
export MODE="${MODE:-hybrid}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-3500}"
export V2_QUERY_WORKERS="${V2_QUERY_WORKERS:-1}"
export BASELINE_QUERY_WORKERS="${BASELINE_QUERY_WORKERS:-${V2_QUERY_WORKERS}}"
export BASELINE_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE:-32}"
export BASELINE_EMBED_RETRIES="${BASELINE_EMBED_RETRIES:-3}"
export BASELINE_EMBED_RETRY_SLEEP="${BASELINE_EMBED_RETRY_SLEEP:-5}"
export GRAPHRAG_R1_API_BASE="${GRAPHRAG_R1_API_BASE:-http://127.0.0.1:8002/v1}"
export GRAPHRAG_R1_CHAT_MODEL="${GRAPHRAG_R1_CHAT_MODEL:-/data/srl/GraphRAG-R1}"
export HIPRAG_API_BASE="${HIPRAG_API_BASE:-http://127.0.0.1:8003/v1}"
export HIPRAG_CHAT_MODEL="${HIPRAG_CHAT_MODEL:-/data/srl/HiPRAG-7B}"

PROCESSED="datasets/processed/${PROCESSED_DATASET}"
test -s "${PROCESSED}/questions.jsonl"
test -s "${PROCESSED}/chunks.jsonl"
test -s "${PROCESSED}/semantic_llm.extractions.jsonl"

curl -fsS http://127.0.0.1:9200 >/tmp/changed_baselines_es.ok
curl -fsS http://localhost:8000/v1/models >/tmp/changed_baselines_chat.ok
curl -fsS http://localhost:8001/v1/models >/tmp/changed_baselines_embed.ok
curl -fsS "${GRAPHRAG_R1_API_BASE%/}/models" >/tmp/changed_baselines_gr1.ok
curl -fsS "${HIPRAG_API_BASE%/}/models" >/tmp/changed_baselines_hiprag.ok

run_method () {
  local method="$1"
  echo "[changed-baselines] run ${method}"
  case "${method}" in
    agrag)
      PROCESSED_DATASET="${PROCESSED_DATASET}" AGRAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}" REUSE_BASELINE_INDEX=0 \
        scripts/baselines/run_baseline_method.sh agrag "${OUT_DATASET}" "${NAMESPACE}"
      ;;
    graphrag_r1)
      PROCESSED_DATASET="${PROCESSED_DATASET}" GRAPHRAG_R1_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}" REUSE_BASELINE_INDEX=0 \
        scripts/baselines/run_baseline_method.sh graphrag_r1 "${OUT_DATASET}" "${NAMESPACE}"
      ;;
    hiprag)
      PROCESSED_DATASET="${PROCESSED_DATASET}" HIPRAG_EMBED_BATCH_SIZE="${BASELINE_EMBED_BATCH_SIZE}" REUSE_HIPRAG_INDEX=0 \
        scripts/baselines/run_baseline_method.sh hiprag "${OUT_DATASET}" "${NAMESPACE}"
      ;;
    *)
      echo "unknown changed baseline method=${method}" >&2
      exit 2
      ;;
  esac
  test -s "outputs/${OUT_DATASET}/predictions/${method}.jsonl"
  wc -l "outputs/${OUT_DATASET}/predictions/${method}.jsonl"
}

run_method agrag
run_method graphrag_r1
run_method hiprag

echo "[changed-baselines] verify model metadata"
python - <<PY
import json
from pathlib import Path
base = Path("outputs/${OUT_DATASET}/baselines")
for method, rel in [("graphrag_r1", "graph.json"), ("hiprag", "retrieval_index.json")]:
    p = base / method / rel
    if not p.exists():
        print(method, "missing", p)
        continue
    d = json.loads(p.read_text())
    print(method, "chat_model_used=", d.get("chat_model_used"), "run_mode=", d.get(method + "_run_mode"), "use_es=", d.get("uses_shared_signpost_chunk_es_index"))
PY

echo "[changed-baselines] done output=${OUT_DATASET}"
date
