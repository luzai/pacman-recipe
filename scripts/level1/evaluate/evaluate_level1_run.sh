#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/scripts/pacman_paths.sh"
ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-${HOME}/.conda/envs/maapacman-rl}}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:?PACMAN_PYTHON_ROOT is required}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
SOURCE_RUN="${SOURCE_RUN:?SOURCE_RUN is required}"
EVAL_ROOT="${EVAL_ROOT:-${SOURCE_RUN}/corrected_eval}"
GPU_ID="${GPU_ID:-0}"
PORT_BASE="${PORT_BASE:-18150}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
CHECKPOINT_LIST="${CHECKPOINT_LIST:-}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/level1/train/curriculum2.yaml}"
EVAL_SEED="${EVAL_SEED:-108}"
EVAL_EPISODES="${EVAL_EPISODES:-4}"
EVAL_SAMPLES_PER_SEED="${EVAL_SAMPLES_PER_SEED:-12}"
EVAL_PURPOSE="${EVAL_PURPOSE:-validation}"
SAMPLED_ONLY="${SAMPLED_ONLY:-0}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if ! git -C "${PACMAN_PYTHON_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Pinned pacman-python checkout is missing: ${PACMAN_PYTHON_ROOT}" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_LIST}" && ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root is missing: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
if [[ -e "${EVAL_ROOT}/comparison.json" ]]; then
  echo "Evaluation is already complete: ${EVAL_ROOT}/comparison.json" >&2
  exit 2
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be numeric: ${GPU_ID}" >&2
  exit 2
fi

active="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)" || {
  echo "Cannot verify GPU availability; refusing to start." >&2
  exit 3
}
if [[ -n "${active}" ]]; then
  echo "GPU ${GPU_ID} is busy; refusing to preempt PIDs: ${active}" >&2
  exit 3
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1

if [[ -n "${CHECKPOINT_LIST}" ]]; then
  mapfile -t TRAINED_DIRS < "${CHECKPOINT_LIST}"
else
  mapfile -t TRAINED_DIRS < <(
    find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' | sort -V
  )
fi
if [[ "${#TRAINED_DIRS[@]}" -eq 0 ]]; then
  echo "At least one trained checkpoint is required." >&2
  exit 2
fi
echo "checkpoints_selected=${#TRAINED_DIRS[@]} (all listed checkpoints will be exported/evaluated; use CHECKPOINT_LIST for a subset)"

mkdir -p "${EVAL_ROOT}/complete_checkpoints" "${EVAL_ROOT}/servers"
LABELS=(base)
MODEL_PATHS=("${BASE_MODEL}")
for index in "${!TRAINED_DIRS[@]}"; do
  label="$(basename "${TRAINED_DIRS[${index}]}")"
  if [[ ! -d "${TRAINED_DIRS[${index}]}" || ! "${label}" =~ ^[A-Za-z0-9._-]+$ || " ${LABELS[*]} " == *" ${label} "* ]]; then
    echo "Invalid or duplicate checkpoint directory: ${TRAINED_DIRS[${index}]}" >&2
    exit 2
  fi
  output="${EVAL_ROOT}/complete_checkpoints/${label}"
  if [[ ! -f "${output}/merge_manifest.json" ]]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/level1/report/build_complete_vlm_checkpoint.py" \
      --trained-dir "${TRAINED_DIRS[${index}]}" \
      --base-dir "${BASE_MODEL}" \
      --output-dir "${output}"
  fi
  # An existing manifest is not a validity check: always verify the selected
  # bundle's architecture, offline metadata and export file hashes before any
  # model server starts, including freshly exported checkpoints.
  "${PYTHON}" "${REPO_ROOT}/scripts/level1/train/validate_model_checkpoint.py" \
    "${output}"
  LABELS+=("${label}")
  MODEL_PATHS+=("${output}")
done

SERVER_PID=""
stop_server() {
  local count
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    count="$(ps -eo sid= | awk -v sid="${SERVER_PID}" '$1 == sid {n++} END {print n + 0}')"
    if [[ "${count}" -eq 0 ]]; then
      break
    fi
    sleep 1
  done
  count="$(ps -eo sid= | awk -v sid="${SERVER_PID}" '$1 == sid {n++} END {print n + 0}')"
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
  local port="$3"
  local served_model_name="${4:-${label}}"
  local log="${EVAL_ROOT}/servers/${label}.log"
  local current_pids
  current_pids="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)" || return 3
  if [[ -n "${current_pids}" ]]; then
    echo "GPU became busy; refusing to preempt: ${current_pids}" >&2
    return 3
  fi
  "${PYTHON}" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${port}" || {
    echo "Service port is unavailable; refusing to use an unrelated server." >&2
    return 3
  }
  "${PYTHON}" -c 'import json,sys; print(json.dumps({"model_path":sys.argv[1],"served_model_id":sys.argv[2],"gpu_id":sys.argv[3],"port":int(sys.argv[4])},sort_keys=True))' \
    "${model}" "${served_model_name}" "${GPU_ID}" "${port}" >"${EVAL_ROOT}/servers/${label}.launch.json"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" setsid "${PYTHON}" \
    -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${model}" \
    --served-model-name "${served_model_name}" \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${EVAL_ROOT}/servers/${label}.pid"

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${log}" >&2
      echo "vLLM server exited before readiness: ${label}" >&2
      exit 4
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM server: ${label}" >&2
  exit 4
}

for index in "${!LABELS[@]}"; do
  if [[ "${SAMPLED_ONLY}" == 1 ]]; then break; fi
  label="${LABELS[${index}]}"
  port=$((PORT_BASE + index))
  mkdir -p "${EVAL_ROOT}/${label}"
  if [[ -f "${EVAL_ROOT}/${label}/greedy.json" ]]; then
    echo "greedy_already_complete=${label}"
    continue
  fi
  start_server "${label}" "${MODEL_PATHS[${index}]}" "${port}"
  "${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1.py" \
    --model "${label}" \
    --config "${CONFIG}" \
    --checkpoint-path "${MODEL_PATHS[${index}]}" \
    --tokenizer-path "${MODEL_PATHS[${index}]}" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --episodes "${EVAL_EPISODES}" --samples-per-seed 1 --seed "${EVAL_SEED}" --purpose "${EVAL_PURPOSE}" \
    --concurrency 1 \
    --temperature 0.0 \
    --top-p 1.0 \
    --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
    --output "${EVAL_ROOT}/${label}/greedy.json" \
    >"${EVAL_ROOT}/${label}/greedy.log" 2>&1
  stop_server
done

SAMPLED_LABELS=("${LABELS[@]}")
for sampled_offset in "${!SAMPLED_LABELS[@]}"; do
  sampled_label="${SAMPLED_LABELS[${sampled_offset}]}"
  if [[ -f "${EVAL_ROOT}/${sampled_label}/sampled12.json" ]]; then
    echo "sampled12_already_complete=${sampled_label}"
    continue
  fi
  sampled_index=-1
  for index in "${!LABELS[@]}"; do
    if [[ "${LABELS[${index}]}" == "${sampled_label}" ]]; then
      sampled_index="${index}"
      break
    fi
  done
  if [[ "${sampled_index}" -lt 0 ]]; then
    echo "Sampled label is not in the checkpoint list: ${sampled_label}" >&2
    exit 5
  fi
  sampled_port=$((PORT_BASE + ${#LABELS[@]} + sampled_offset))
  start_server \
    "${sampled_label}-sampled12" \
    "${MODEL_PATHS[${sampled_index}]}" \
    "${sampled_port}" \
    "${sampled_label}"
  "${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1.py" \
    --model "${sampled_label}" \
    --config "${CONFIG}" \
    --checkpoint-path "${MODEL_PATHS[${sampled_index}]}" \
    --tokenizer-path "${MODEL_PATHS[${sampled_index}]}" \
    --base-url "http://127.0.0.1:${sampled_port}/v1" \
    --episodes "${EVAL_EPISODES}" --samples-per-seed "${EVAL_SAMPLES_PER_SEED}" --seed "${EVAL_SEED}" --purpose "${EVAL_PURPOSE}" \
    --concurrency 4 \
    --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
    --output "${EVAL_ROOT}/${sampled_label}/sampled12.json" \
    >"${EVAL_ROOT}/${sampled_label}/sampled12.log" 2>&1
  stop_server
done

COMPARE_FLAGS=(--require-complete-dual)
if [[ "${SAMPLED_ONLY}" == 1 ]]; then COMPARE_FLAGS=(--sampled-only); fi
"${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/compare_level1_run.py" \
  --eval-root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/comparison.json" \
  "${COMPARE_FLAGS[@]}"
BEST_SAMPLED_LABEL="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_sampled_label"])' \
    "${EVAL_ROOT}/comparison.json"
)"
BEST_GREEDY_LABEL="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_greedy_label"])' \
    "${EVAL_ROOT}/comparison.json"
)"

echo "evaluation_completed=true (process completion is not a win-rate gate; see each summary's full_completions and win_rate)"
echo "best_sampled_label=${BEST_SAMPLED_LABEL}"
echo "best_greedy_label=${BEST_GREEDY_LABEL}"
echo "comparison=${EVAL_ROOT}/comparison.json"
