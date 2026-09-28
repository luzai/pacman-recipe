#!/usr/bin/env bash
set -euo pipefail

SMOKE_UPDATES=""
SMOKE_UPDATES_SET=0
REWARD_ABLATION=""
REWARD_ABLATION_SET=0
while (( $# )); do
  case "$1" in
    --reward-ablation|--reward-ablation=*)
      if (( REWARD_ABLATION_SET )); then
        echo "--reward-ablation may be specified only once." >&2
        exit 2
      fi
      REWARD_ABLATION_SET=1
      if [[ "$1" == --reward-ablation ]]; then
        if (( $# < 2 )); then
          echo "--reward-ablation requires fixed-distance or binary-outcome." >&2
          exit 2
        fi
        REWARD_ABLATION="$2"
        shift 2
      else
        REWARD_ABLATION="${1#*=}"
        shift
      fi
      ;;
    --smoke-updates)
      if (( SMOKE_UPDATES_SET )); then
        echo "--smoke-updates may be specified only once." >&2
        exit 2
      fi
      if (( $# < 2 )); then
        echo "--smoke-updates requires a positive integer." >&2
        exit 2
      fi
      SMOKE_UPDATES="$2"
      SMOKE_UPDATES_SET=1
      shift 2
      ;;
    --smoke-updates=*)
      if (( SMOKE_UPDATES_SET )); then
        echo "--smoke-updates may be specified only once." >&2
        exit 2
      fi
      SMOKE_UPDATES="${1#*=}"
      SMOKE_UPDATES_SET=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
if (( SMOKE_UPDATES_SET )) && [[ ! "${SMOKE_UPDATES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--smoke-updates requires a positive integer." >&2
  exit 2
fi
REWARD_ABLATION_ARGS=()
if (( REWARD_ABLATION_SET )); then
  case "${REWARD_ABLATION}" in
    fixed-distance)
      if [[ "${SMOKE_UPDATES}" != 4 ]]; then
        echo "--reward-ablation fixed-distance requires --smoke-updates 4." >&2
        exit 2
      fi
      ;;
    binary-outcome)
      # Terminal win signal only. The recipe fixes the run length, so this
      # ablation deliberately does not force a smoke budget.
      ;;
    *)
      echo "--reward-ablation supports only fixed-distance or binary-outcome." >&2
      exit 2
      ;;
  esac
  REWARD_ABLATION_ARGS=(--reward-ablation "${REWARD_ABLATION}")
fi
SMOKE_ARGS=()
if (( SMOKE_UPDATES_SET )); then
  SMOKE_ARGS=(--smoke-updates "${SMOKE_UPDATES}")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${REPO_ROOT}/scripts/pacman_paths.sh"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
OWNER_ROOT="${OWNER_ROOT:-${HOME:?HOME must be set}}"
ENV_ROOT="${ENV_ROOT:-${OWNER_ROOT}/miniconda/envs/maapacman-rl}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
AREAL_ROOT="${AREAL_ROOT:-${WORKSPACE_ROOT}/AReaL}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/level1/train/curriculum1.yaml}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RECIPE_NAME="$(basename "${CONFIG}" .yaml)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OWNER_ROOT}/run_artifacts/maapacman-rl/${RECIPE_NAME}-${RUN_ID}}"
DATASET_OUTPUT_ROOT="${DATASET_OUTPUT_ROOT:-${ARTIFACT_ROOT}/dataset}"
# Dataset defaults are read from the selected recipe, not shared shell defaults.
DATASET_ARGS=()
[[ -z "${TRAIN_EPISODES:-}" ]] || DATASET_ARGS+=(--train-episodes "${TRAIN_EPISODES}")
[[ -z "${VALIDATION_EPISODES:-}" ]] || DATASET_ARGS+=(--validation-episodes "${VALIDATION_EPISODES}")
[[ -z "${DATASET_MAX_STEPS:-}" ]] || DATASET_ARGS+=(--max-steps "${DATASET_MAX_STEPS}")

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
# vLLM 0.22.1 uses VLLM_RPC_BASE_PATH (default tempfile.gettempdir())
# followed by '/' + a 36-character UUID for its Unix-domain socket path.
# Check this before importing model/GPU libraries or creating run artifacts.
"${PYTHON}" - --pacman-vllm-ipc-preflight <<'PY'
import json
import os
import socket
import sys
import tempfile
import uuid

def fail(message):
    print(f"vllm_ipc_preflight=failed: {message}", file=sys.stderr)
    raise SystemExit(2)

override = os.environ.get("VLLM_RPC_BASE_PATH")
explicit_temp = next((name for name in ("TMPDIR", "TEMP", "TMP") if os.environ.get(name)), None)
base = (override if override is not None else
        os.environ[explicit_temp] if explicit_temp else tempfile.gettempdir())
source = "VLLM_RPC_BASE_PATH" if override is not None else (explicit_temp or "tempfile default")
if not base or not os.path.isabs(base):
    fail(f"{source} must identify an existing absolute directory")
endpoint = f"{base}/{uuid.uuid4()}"
endpoint_bytes = len(os.fsencode(endpoint))
if endpoint_bytes > 107:
    fail(
        f"{source} gives a {endpoint_bytes}-byte Unix socket path (maximum 107; "
        "base + '/' + 36-character UUID). Set VLLM_RPC_BASE_PATH to an existing "
        "short task-owned directory; TMPDIR and artifact paths need not change."
    )
if not os.path.isdir(base):
    fail(f"{source} directory does not exist: {base}")
bound = False
if override is not None or explicit_temp:
    # Probe only the explicitly selected directory. Do not create a directory,
    # alter TMPDIR, or introduce an implicit short path elsewhere.
    if not hasattr(socket, "AF_UNIX"):
        fail("selected Python does not support AF_UNIX; run this launcher on Linux")
    probe = None
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.bind(endpoint)
        bound = True
        inode = os.lstat(endpoint)
    except OSError as error:
        fail(f"cannot bind a Unix socket in {source}: {error}")
    finally:
        if probe is not None:
            probe.close()
        if bound:
            current = os.lstat(endpoint)
            if (current.st_dev, current.st_ino) != (inode.st_dev, inode.st_ino):
                fail("probe socket was replaced; refusing to remove another file")
            os.unlink(endpoint)
print("vllm_ipc_preflight=ok " + json.dumps({
    "source": source, "base_path": base, "endpoint_bytes": endpoint_bytes,
    "max_endpoint_bytes": 107, "socket_bind_verified": bound,
}, sort_keys=True))
PY
if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config does not exist: ${CONFIG}" >&2
  exit 2
fi
ACTOR_PATH_SPEC="$(
  awk '
    /^actor:/ { in_actor = 1; next }
    in_actor && /^[^[:space:]]/ { exit }
    in_actor && /^[[:space:]]+path:/ { print $2; exit }
  ' "${CONFIG}"
)"
if [[ -z "${ACTOR_PATH_SPEC}" ]]; then
  echo "Training config has no actor.path: ${CONFIG}" >&2
  exit 2
fi
MODEL_PATH="${MODEL_PATH:-${OWNER_ROOT}/models/Qwen3.5-9B}"
if ! "${PYTHON}" "${REPO_ROOT}/scripts/level1/train/validate_model_checkpoint.py" \
  "${MODEL_PATH}"; then
  echo "Model checkpoint is incomplete: ${MODEL_PATH}" >&2
  exit 2
fi
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd -P)"
if [[ ! -f "${AREAL_ROOT}/areal/__init__.py" ]]; then
  echo "Selected AReaL checkout is incomplete: ${AREAL_ROOT}" >&2
  exit 2
fi
if [[ -e "${DATASET_OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite immutable v4 dataset: ${DATASET_OUTPUT_ROOT}" >&2
  exit 2
fi

# Keep the MaaPacman recipe independent from the historical robotics fork.
# The selected AReaL worktree must win import resolution even when the Conda
# environment still contains an older editable AReaL installation.
export PYTHONPATH="${AREAL_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" - "${REPO_ROOT}" <<'PY'
import sys
from pathlib import Path

import pacman_recipe
import pacman_env

repo_root = Path(sys.argv[1]).resolve()
recipe_package = Path(pacman_recipe.__file__).resolve().parent
environment_package = Path(pacman_env.__file__).resolve().parent
if recipe_package.parent != repo_root:
    raise SystemExit(
        f"pacman_recipe import escaped active checkout: {recipe_package}"
    )
if environment_package.parent != repo_root:
    raise SystemExit(
        "pacman_env import escaped active bundled checkout; uninstall the old "
        f"standalone maapacman distribution: {environment_package}"
    )
print(f"pacman_recipe_import={recipe_package}")
print(f"pacman_env_import={environment_package}")
PY
AREAL_IMPORT_PATH="$(
  "${PYTHON}" -c 'import pathlib, areal; print(pathlib.Path(areal.__file__).resolve())'
)"
AREAL_ROOT_REAL="$(
  "${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' \
    "${AREAL_ROOT}"
)"
case "${AREAL_IMPORT_PATH}" in
  "${AREAL_ROOT_REAL}"/*) ;;
  *)
    echo "AReaL import escaped selected checkout: ${AREAL_IMPORT_PATH}" >&2
    exit 2
    ;;
esac
echo "areal_import=${AREAL_IMPORT_PATH}"

# AReaL's proxy workers bind the node-private interface so sibling worker
# processes can reach them. Use one unpredictable, process-lifetime key for
# both controller generations and never print or persist its resolved value.
if [[ -z "${AREAL_ADMIN_API_KEY:-}" ]]; then
  AREAL_ADMIN_API_KEY="$(
    "${PYTHON}" -c 'import secrets; print(secrets.token_urlsafe(32))'
  )"
fi
if [[ "${AREAL_ADMIN_API_KEY}" == "areal-admin-key" ]]; then
  echo "AREAL_ADMIN_API_KEY must not use AReaL's public default." >&2
  exit 2
fi
export AREAL_ADMIN_API_KEY

CONFIG_GPU_COUNT="$(
  awk '/^[[:space:]]+n_gpus_per_node:/ { print $2; exit }' "${CONFIG}"
)"
if [[ ! "${CONFIG_GPU_COUNT}" =~ ^(6|8)$ ]]; then
  echo "Config must request an accepted 6- or 8-GPU topology: ${CONFIG_GPU_COUNT}" >&2
  exit 2
fi

ACTOR_BACKEND="$(
  awk '
    /^actor:/ { in_actor = 1; next }
    in_actor && /^[^[:space:]]/ { exit }
    in_actor && /^[[:space:]]+backend:/ {
      gsub(/"/, "", $2); print $2; exit
    }
  ' "${CONFIG}"
)"
if [[ ! "${ACTOR_BACKEND}" =~ ^[^:]+:d([0-9]+)p[0-9]+t[0-9]+$ ]]; then
  echo "Cannot determine actor data-parallel size: ${ACTOR_BACKEND}" >&2
  exit 2
fi
ACTOR_DP_SIZE="${BASH_REMATCH[1]}"
MAAPACMAN_LOGP_RPC_CHUNK_SIZE="${MAAPACMAN_LOGP_RPC_CHUNK_SIZE:-${ACTOR_DP_SIZE}}"
if [[ ! "${MAAPACMAN_LOGP_RPC_CHUNK_SIZE}" =~ ^[0-9]+$ ]] \
  || (( MAAPACMAN_LOGP_RPC_CHUNK_SIZE < ACTOR_DP_SIZE )) \
  || (( MAAPACMAN_LOGP_RPC_CHUNK_SIZE % ACTOR_DP_SIZE != 0 )); then
  echo "MAAPACMAN_LOGP_RPC_CHUNK_SIZE must be a positive multiple of actor dp_size=${ACTOR_DP_SIZE}." >&2
  exit 2
fi
export MAAPACMAN_LOGP_RPC_CHUNK_SIZE
echo "logp_rpc_chunk_size=${MAAPACMAN_LOGP_RPC_CHUNK_SIZE}"

CONFIG_OFFLOAD="$(
  awk '/^enable_offload:/ { print $2; exit }' "${CONFIG}"
)"
if [[ "${CONFIG_OFFLOAD}" == "true" ]]; then
  "${PYTHON}" -c '
from pathlib import Path
import torch_memory_saver
from areal.utils.offload import get_tms_env_vars

preload = Path(get_tms_env_vars()["LD_PRELOAD"])
if not preload.is_file():
    raise SystemExit(f"missing torch-memory-saver preload library: {preload}")
print(f"torch_memory_saver_preload={preload}")
'
fi
if grep -q '^[[:space:]]*offload_params:[[:space:]]*true' "${CONFIG}"; then
  "${PYTHON}" -c '
import inspect
from areal.engine.fsdp_engine import FSDPEngine

source = inspect.getsource(FSDPEngine.initialize)
marker = "CPUOffloadPolicy has already moved the persistent FSDP parameter"
if marker not in source:
    raise SystemExit(
        "AReaL is missing patches/areal_fsdp_cpu_offload_empty_cache.patch"
    )
print("areal_fsdp_cpu_offload_empty_cache_patch=ok")
'
fi
if grep -Eq '^(open_action_mask|edward_options):[[:space:]]*true' "${CONFIG}"; then
  "${PYTHON}" - "${CONFIG}" <<'PY'
import inspect
import sys
from pathlib import Path

from areal.api.cli_args import vLLMConfig
from areal.engine.fsdp_engine import FSDPEngine

config_text = Path(sys.argv[1]).read_text(encoding="utf-8")
source = inspect.getsource(FSDPEngine)
if "_apply_pacman_action_mask" not in source:
    raise SystemExit(
        "AReaL is missing patches/areal_pacman_action_logprobs.patch"
    )
if not hasattr(vLLMConfig, "logprobs_mode"):
    raise SystemExit(
        "AReaL vLLMConfig cannot select processed rollout log-probabilities"
    )
if "logprobs_mode: processed_logprobs" not in config_text:
    raise SystemExit("open-action-mask training requires processed_logprobs")
if "top_p: 0.95" in config_text:
    raise SystemExit("constrained Pacman training requires top_p=1.0")
if "edward_options: true" in config_text:
    if "pacman_allowed_token_ids" not in source:
        raise SystemExit(
            "AReaL is missing generic Edward option-token log-prob support"
        )
    if "action_token_choice: false" not in config_text:
        raise SystemExit("Edward options must disable atomic action choices")
    if "open_action_mask: false" not in config_text:
        raise SystemExit("Edward options must disable the legacy action mask")
print("pacman_recipe_action_logprobs_patch=ok")
PY
fi
if grep -Eq '^(open_action_mask|edward_options|action_token_choice):[[:space:]]*true' "${CONFIG}"; then
  # Constrained sampling relies on vLLM's allowed_token_ids mask. vLLM 0.22.1
  # (and upstream main) swaps that mask between batch slots with the Python
  # tuple-swap idiom, which is a no-op on 2-D tensor rows, so a reordered
  # request silently samples with another request's constraint. The failure is
  # silent -- shapes and dtypes stay valid and vLLM raises nothing -- so a
  # rebuilt venv that loses the patch would quietly corrupt rollouts instead of
  # failing. Refuse to launch rather than produce such a run.
  "${PYTHON}" - <<'PY'
import inspect

from vllm.v1.worker.gpu_input_batch import InputBatch

swap_source = inspect.getsource(InputBatch.swap_states)
if "i1_mask" not in swap_source:
    raise SystemExit(
        "vLLM is missing patches/vllm_allowed_token_ids_mask_swap.patch: "
        "InputBatch.swap_states would assign the wrong allowed_token_ids mask "
        "to a reordered request, silently sampling outside its allowed set"
    )
condense_source = inspect.getsource(InputBatch.condense)
if "allowed_token_ids_mask_cpu_tensor[last_req_index].fill_" not in condense_source:
    raise SystemExit(
        "vLLM is missing the condense() half of "
        "patches/vllm_allowed_token_ids_mask_swap.patch: a vacated batch slot "
        "keeps a stale mask that a later unconstrained request would inherit"
    )
print("vllm_allowed_token_ids_mask_patch=ok")
PY
fi
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(seq -s, 0 $((CONFIG_GPU_COUNT - 1)))"
fi

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${CONFIG_GPU_COUNT}" ]]; then
  echo "Config requires ${CONFIG_GPU_COUNT} unique GPUs, got ${GPU_IDS}." >&2
  exit 2
fi

declare -A SEEN_GPUS=()
# A monitoring failure is not evidence of an idle GPU.
if ! nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv; then
  echo "Cannot verify live GPU availability; refusing to launch." >&2
  exit 3
fi
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain unique numeric GPU IDs: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS[${gpu}]=1
  if ! active_pids="$(
    nvidia-smi -i "${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"; then
    echo "Cannot verify GPU ${gpu}; refusing to launch." >&2
    exit 3
  fi
  if [[ -n "${active_pids}" ]]; then
    echo "GPU ${gpu} is busy; refusing to preempt PIDs: ${active_pids}" >&2
    exit 3
  fi
done

mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_ROOT}/training/trajectories"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HOME="${HF_HOME:-${OWNER_ROOT}/hf-cache}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1
# PyTorch 2.11 + CUDA 13 can select cuDNN SDPA for Qwen3.5 shapes that
# cuDNN cannot plan. sitecustomize.py applies this before every controller
# and RPC worker imports AReaL/torch, preserving SDPA with another backend.
export MAAPACMAN_DISABLE_CUDNN_SDPA="${MAAPACMAN_DISABLE_CUDNN_SDPA:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MAAPACMAN_PACMAN_PYTHON_ROOT="${MAAPACMAN_PACMAN_PYTHON_ROOT:-${WORKSPACE_ROOT}/pacman-python}"
export PACMAN_TRAJECTORY_DIR="${ARTIFACT_ROOT}/training/trajectories"

if grep -Eq '^edward_options:[[:space:]]*true|^action_protocol:[[:space:]]*direct-open-action-token-v1' "${CONFIG}"; then
  MAX_MODEL_LEN="$(
    awk '/^[[:space:]]+max_model_len:/ { print $2; exit }' "${CONFIG}"
  )"
  if [[ ! "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]]; then
    echo "Formal image config must define numeric vllm.max_model_len." >&2
    exit 2
  fi
  BUDGET_CHECK=scripts/level1/train/check_edward_prompt_budget.py
  BUDGET_ARGS=(--config "${CONFIG}")
  if grep -q '^action_protocol:[[:space:]]*direct-open-action-token-v1' "${CONFIG}"; then
    BUDGET_CHECK=scripts/level1/train/check_direct_prompt_budget.py
    BUDGET_ARGS=()
  fi
  "${PYTHON}" "${BUDGET_CHECK}" \
    --model-path "${MODEL_PATH}" \
    --max-input-tokens "${MAX_MODEL_LEN}" "${BUDGET_ARGS[@]}"
fi

if [[ "$(command -v python3)" != "$(dirname "${PYTHON}")/python3" ]]; then
  echo "python3 does not resolve inside the selected Conda environment: $(command -v python3)" >&2
  exit 2
fi

"${PYTHON}" scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root "${DATASET_OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  "${DATASET_ARGS[@]}" \
  --write-hf
"${PYTHON}" train_areal.py \
  --config "${CONFIG}" \
  "${SMOKE_ARGS[@]}" \
  "${REWARD_ABLATION_ARGS[@]}" \
  --dry-run \
  --validate-areal \
  "train_dataset.path=${DATASET_OUTPUT_ROOT}/train_hf" \
  "valid_dataset.path=${DATASET_OUTPUT_ROOT}/validation_hf"
cp "${CONFIG}" "${ARTIFACT_ROOT}/config.yaml"
"${PYTHON}" scripts/level1/dataset/write_level1_manifest.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --model-revision "${MODEL_PATH}" \
  --dataset-manifest "${DATASET_OUTPUT_ROOT}/manifest.json" \
  --config "${CONFIG}" \
  "${SMOKE_ARGS[@]}" \
  "${REWARD_ABLATION_ARGS[@]}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight=ok"
  echo "areal_root=${AREAL_ROOT_REAL}"
  echo "gpu_ids=${GPU_IDS}"
  echo "artifacts=${ARTIFACT_ROOT}"
  exit 0
fi

echo "Starting AReaL level-1 training"
echo "  run_id=${RUN_ID}"
echo "  recipe=${RECIPE_NAME}"
if (( SMOKE_UPDATES_SET )); then
  echo "  smoke_updates=${SMOKE_UPDATES}"
fi
echo "  areal=${AREAL_ROOT_REAL}"
echo "  gpu_ids=${GPU_IDS}"
echo "  model=${MODEL_PATH}"
echo "  artifacts=${ARTIFACT_ROOT}"

exec "${PYTHON}" train_areal.py \
  --config "${CONFIG}" \
  "${SMOKE_ARGS[@]}" \
  "${REWARD_ABLATION_ARGS[@]}" \
  "artifact_root=${ARTIFACT_ROOT}" \
  "cluster.fileroot=${ARTIFACT_ROOT}/training" \
  "cluster.name_resolve.nfs_record_root=${ARTIFACT_ROOT}/name_resolve" \
  "actor.path=${MODEL_PATH}" \
  "train_dataset.path=${DATASET_OUTPUT_ROOT}/train_hf" \
  "valid_dataset.path=${DATASET_OUTPUT_ROOT}/validation_hf" \
  "experiment_name=maapacman-level1" \
  "trial_name=${RECIPE_NAME}-${RUN_ID}"
