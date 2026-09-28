#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/scripts/pacman_paths.sh"
SOURCE_RUN="${SOURCE_RUN:?SOURCE_RUN is required}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-${HOME}/.conda/envs/maapacman-rl}}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:?PACMAN_PYTHON_ROOT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_RUN}/ab_demo_latest}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?CHECKPOINT_ROOT is required}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-18269}"
SEED="${SEED:-0}"

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model is missing: ${BASE_MODEL}" >&2
  exit 2
fi
if [[ ! -d "${PACMAN_PYTHON_ROOT}" ]]; then
  echo "pacman-python is missing: ${PACMAN_PYTHON_ROOT}" >&2
  exit 2
fi

active="$(
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "${active}" ]]; then
  echo "GPU ${GPU_ID} is busy; refusing to preempt PIDs: ${active}" >&2
  exit 3
fi

mapfile -t COMPLETE_CHECKPOINTS < <(
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name 'epoch*globalstep*' -print |
    while read -r checkpoint; do
      [[ -s "${checkpoint}/model.safetensors" ]] || continue
      [[ -s "${checkpoint}/config.json" ]] || continue
      printf '%s\n' "${checkpoint}"
    done |
    sort -V
)
if [[ "${#COMPLETE_CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No complete checkpoints found below ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
LATEST_CHECKPOINT="${COMPLETE_CHECKPOINTS[-1]}"
LATEST_LABEL="$(basename "${LATEST_CHECKPOINT}")"
mkdir -p "${OUTPUT_ROOT}/servers" "${OUTPUT_ROOT}/base" "${OUTPUT_ROOT}/latest"

COMPLETE_MODEL="${OUTPUT_ROOT}/complete_checkpoint_${LATEST_LABEL}"
if [[ ! -s "${COMPLETE_MODEL}/merge_manifest.json" ]]; then
  "${PYTHON}" \
    "${REPO_ROOT}/scripts/level1/report/build_complete_vlm_checkpoint.py" \
    --trained-dir "${LATEST_CHECKPOINT}" \
    --base-dir "${BASE_MODEL}" \
    --output-dir "${COMPLETE_MODEL}" \
    --restore-all-missing \
    >"${OUTPUT_ROOT}/merge.log" 2>&1
fi

SERVER_PID=""
stop_server() {
  local count
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    count="$(
      ps -eo sid= | awk -v sid="${SERVER_PID}" \
        '$1 == sid {n++} END {print n + 0}'
    )"
    [[ "${count}" -eq 0 ]] && break
    sleep 1
  done
  count="$(
    ps -eo sid= | awk -v sid="${SERVER_PID}" \
      '$1 == sid {n++} END {print n + 0}'
  )"
  if [[ "${count}" -gt 0 ]]; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

start_server() {
  local label="$1"
  local model="$2"
  local log="${OUTPUT_ROOT}/servers/${label}.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" setsid "${PYTHON}" \
    -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model "${model}" \
    --served-model-name "${label}" \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${OUTPUT_ROOT}/servers/${label}.pid"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" \
      >"${OUTPUT_ROOT}/servers/${label}.models.json" 2>/dev/null; then
      return
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -100 "${log}" >&2
      echo "vLLM server exited before readiness: ${label}" >&2
      exit 4
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM server: ${label}" >&2
  exit 4
}

evaluate_one() {
  local label="$1"
  local model="$2"
  local output_dir="${OUTPUT_ROOT}/${label}"
  start_server "${label}" "${model}"
  "${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1.py" \
    --model "${label}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --episodes 1 \
    --seed "${SEED}" \
    --max-steps 2000 \
    --concurrency 1 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-completion-tokens 3 \
    --prompt-style live_state_v3 \
    --open-action-mask \
    --tokenizer-path "${model}" \
    --wall-clock-limit-seconds 600 \
    --stuck-no-progress-steps 100 \
    --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
    --trajectory-dir "${output_dir}/trajectories" \
    --output "${output_dir}/summary.json" \
    >"${output_dir}/evaluate.log" 2>&1
  stop_server
}

evaluate_one base "${BASE_MODEL}"
evaluate_one latest "${COMPLETE_MODEL}"

BASE_TRAJECTORY="$(find "${OUTPUT_ROOT}/base/trajectories" -type f -name '*.json' | head -n 1)"
LATEST_TRAJECTORY="$(find "${OUTPUT_ROOT}/latest/trajectories" -type f -name '*.json' | head -n 1)"

"${PYTHON}" \
  "${REPO_ROOT}/scripts/level1/report/export_level1_ab_demo_video.py" \
  --trajectory "${BASE_TRAJECTORY}" \
  --output "${OUTPUT_ROOT}/qwen_base_seed${SEED}.mp4" \
  --model-label "Initial Qwen3.5-9B (no RL)" \
  --checkpoint-label "Base model | greedy | no thinking | dynamic open-action mask" \
  >"${OUTPUT_ROOT}/base_video.json"

"${PYTHON}" \
  "${REPO_ROOT}/scripts/level1/report/export_level1_ab_demo_video.py" \
  --trajectory "${LATEST_TRAJECTORY}" \
  --output "${OUTPUT_ROOT}/qwen_rl_${LATEST_LABEL}_seed${SEED}.mp4" \
  --model-label "Qwen3.5-9B + AReaL RL" \
  --checkpoint-label "${LATEST_LABEL} | greedy | no thinking | dynamic open-action mask" \
  >"${OUTPUT_ROOT}/latest_video.json"

"${PYTHON}" - "${OUTPUT_ROOT}" "${LATEST_CHECKPOINT}" "${LATEST_LABEL}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "base" / "summary.json").read_text())
latest = json.loads((root / "latest" / "summary.json").read_text())
manifest = {
    "comparison_contract": {
        "same_seed": int(sys.argv[4]),
        "same_environment": True,
        "same_prompt_style": "live_state_v3",
        "same_decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
            "dynamic_open_action_mask": True,
        },
        "normal_terminal_conditions": ["death", "level_cleared"],
        "safety_limits": {
            "max_steps": 2000,
            "wall_clock_seconds": 600,
            "no_score_or_pellet_progress_steps": 100,
        },
    },
    "base_model": base["episode_results"][0],
    "latest_checkpoint": {
        "path": sys.argv[2],
        "label": sys.argv[3],
        **latest["episode_results"][0],
    },
}
(root / "comparison_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "ab_demo_complete=${OUTPUT_ROOT}"
echo "latest_checkpoint=${LATEST_CHECKPOINT}"
