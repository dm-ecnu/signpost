#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_signpost_ablation_suite.sh <dataset> [namespace]}"
NAMESPACE="${2:-$DATASET}"

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
  scripts/run_signpost_method.sh "${DATASET}" "${variant}" "${NAMESPACE}"
done
