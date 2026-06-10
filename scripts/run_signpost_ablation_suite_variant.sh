#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_signpost_ablation_suite_variant.sh <dataset> [namespace] [method_prefix]}"
NAMESPACE="${2:-$DATASET}"
METHOD_PREFIX="${3:-signpost.full_rerank_v1}"

declare -A METHOD_IDS=(
  [full]="${METHOD_PREFIX}"
  [no_offline]="${METHOD_PREFIX}.no_offline"
  [no_online]="${METHOD_PREFIX}.no_online"
  [no_semantic_cues]="${METHOD_PREFIX}.no_semantic_cues"
  [no_provenance_cues]="${METHOD_PREFIX}.no_provenance_cues"
  [no_vertical_cues]="${METHOD_PREFIX}.no_vertical_cues"
  [no_horizontal_cues]="${METHOD_PREFIX}.no_horizontal_cues"
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

for variant in "${VARIANTS[@]}"; do
  scripts/run_signpost_method_variant.sh "${DATASET}" "${variant}" "${NAMESPACE}" "${METHOD_IDS[$variant]}"
done
