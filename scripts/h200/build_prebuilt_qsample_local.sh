#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: build_prebuilt_qsample_local.sh <root> <source-dataset> <output-dataset> <sample-size> <prebuilt-dir>}"
SOURCE_DATASET="${2:?usage: build_prebuilt_qsample_local.sh <root> <source-dataset> <output-dataset> <sample-size> <prebuilt-dir>}"
OUT_DATASET="${3:?usage: build_prebuilt_qsample_local.sh <root> <source-dataset> <output-dataset> <sample-size> <prebuilt-dir>}"
SAMPLE_SIZE="${4:?usage: build_prebuilt_qsample_local.sh <root> <source-dataset> <output-dataset> <sample-size> <prebuilt-dir>}"
PREBUILT_DIR="${5:?usage: build_prebuilt_qsample_local.sh <root> <source-dataset> <output-dataset> <sample-size> <prebuilt-dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/home/ruolinsu/signpost/signpost_re_v2}"
EXTRACTOR="${TARGET_SILVER_EXTRACTOR:-/home/ruolinsu/signpost/extract/extract_llm_targets_silver.py}"
INCLUDE_TARGET_SILVER="${INCLUDE_TARGET_SILVER:-${EXTRACT_TARGET_SILVER:-1}}"

cd "${PROJECT_DIR}"

echo "[local-qsample] root=${ROOT} source=${SOURCE_DATASET} output=${OUT_DATASET} sample=${SAMPLE_SIZE} prebuilt=${PREBUILT_DIR}"

subset_args=(
  python "${SCRIPT_DIR}/make_question_length_subset.py"
  --root "${ROOT}" \
  --source-dataset "${SOURCE_DATASET}" \
  --output-dataset "${OUT_DATASET}" \
  --sample-size "${SAMPLE_SIZE}" \
  --overwrite
)
if [[ "${COPY_STATIC:-0}" == "1" ]]; then
  subset_args+=(--copy-static)
fi
"${subset_args[@]}"

if [[ "${EXTRACT_TARGET_SILVER:-1}" == "1" ]]; then
  if [[ ! -s "${EXTRACTOR}" ]]; then
    echo "[local-qsample] extractor not found: ${EXTRACTOR}" >&2
    exit 1
  fi
  echo "[local-qsample] ensure LLM target/silver extraction (--resume skips already extracted questions)"
  python "${EXTRACTOR}" \
    --root "${ROOT}" \
    --project-dir "${PROJECT_DIR}" \
    --dataset "${OUT_DATASET}" \
    --workers "${TARGET_SILVER_WORKERS:-2}" \
    --candidate-top-k "${TARGET_SILVER_CANDIDATE_TOP_K:-20}" \
    --resume
fi

mkdir -p "${PREBUILT_DIR}"
files=(
  questions.jsonl
  question_length_subset_manifest.json
)
if [[ "${INCLUDE_TARGET_SILVER}" == "1" ]]; then
  files+=(
    llm_targets_silver.jsonl
    llm_target_units.jsonl
    llm_silver_chunks.jsonl
  )
else
  rm -f \
    "${PREBUILT_DIR}/llm_targets_silver.jsonl" \
    "${PREBUILT_DIR}/llm_target_units.jsonl" \
    "${PREBUILT_DIR}/llm_silver_chunks.jsonl"
fi

for f in "${files[@]}"; do
  if [[ -s "${ROOT}/datasets/processed/${OUT_DATASET}/${f}" ]]; then
    cp -av "${ROOT}/datasets/processed/${OUT_DATASET}/${f}" "${PREBUILT_DIR}/"
  else
    echo "[local-qsample] missing optional ${f}"
  fi
done

test -s "${PREBUILT_DIR}/questions.jsonl"
question_lines="$(wc -l < "${PREBUILT_DIR}/questions.jsonl")"
echo "[local-qsample] questions=${question_lines}"
if [[ "${question_lines}" -ne "${SAMPLE_SIZE}" ]]; then
  echo "[local-qsample] expected ${SAMPLE_SIZE} questions, got ${question_lines}" >&2
  exit 1
fi

if [[ "${INCLUDE_TARGET_SILVER}" == "1" ]]; then
  for f in llm_targets_silver.jsonl llm_target_units.jsonl llm_silver_chunks.jsonl; do
    test -s "${PREBUILT_DIR}/${f}"
    lines="$(wc -l < "${PREBUILT_DIR}/${f}")"
    echo "[local-qsample] ${f}=${lines}"
    if [[ "${lines}" -ne "${SAMPLE_SIZE}" ]]; then
      echo "[local-qsample] expected ${SAMPLE_SIZE} lines in ${f}, got ${lines}" >&2
      exit 1
    fi
  done
fi

echo "[local-qsample] done prebuilt=${PREBUILT_DIR}"
