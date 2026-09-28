#!/usr/bin/env bash
set -euo pipefail

# Compatibility test only; not evidence of the new C1 -> C2 training lineage.
if (( $# != 2 )); then
  echo "Usage: bash $0 /absolute/Iter25-gs24-complete-checkpoint /absolute/new-output" >&2
  exit 2
fi
if [[ "$1" != /* || ! -d "$1" || "$2" != /* || -e "$2" || -L "$2" ]]; then
  echo "Require an existing absolute checkpoint directory and a new absolute output path." >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/scripts/pacman_paths.sh"
export CURRICULUM1_CHECKPOINT="$(cd "$1" && pwd -P)"
export CONFIG="${REPO_ROOT}/configs/level1/train/curriculum2.yaml"
export RUN_ID="legacy-iter25-gs24-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export ARTIFACT_ROOT="$2"
export DATASET_OUTPUT_ROOT="${ARTIFACT_ROOT}/dataset"
unset TRAIN_EPISODES VALIDATION_EPISODES DATASET_MAX_STEPS MODEL_PATH
echo "Legacy Iter25 gs24 compatibility smoke only; not a new-C1 lineage acceptance test."
echo "Verify the supplied checkpoint provenance is Iter25 globalstep24 before launching."
echo "checkpoint=${CURRICULUM1_CHECKPOINT}"
exec bash "${REPO_ROOT}/scripts/level1/train/run_level1_training.sh" --smoke-updates 2
