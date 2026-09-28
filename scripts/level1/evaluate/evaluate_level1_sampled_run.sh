#!/usr/bin/env bash
set -euo pipefail

# Shared runner: matching recipe, provenance, GPU safety, and no win-rate threshold.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/scripts/pacman_paths.sh"
export SAMPLED_ONLY=1
export EVAL_ROOT="${EVAL_ROOT:-${SOURCE_RUN:?SOURCE_RUN is required}/sampled_test}"
exec bash "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1_run.sh" "$@"
