#!/usr/bin/env bash
# Source after setting REPO_ROOT. Retain aliases used by existing deployments.
pacman_path_alias() {
  local canonical="$1" name value selected=""
  shift
  for name in "$canonical" "$@"; do
    value="${!name:-}"
    [[ -z "$value" ]] && continue
    value="$(realpath -m -- "$value")" || return
    if [[ -n "$selected" && "$value" != "$selected" ]]; then
      echo "Conflicting path variables for ${canonical}: ${name}" >&2
      return 2
    fi
    selected="$value"
  done
  [[ -z "$selected" ]] && return 0
  for name in "$canonical" "$@"; do
    export "$name=$selected"
  done
}

pacman_path_alias PACMAN_RECIPE_ROOT AREAL_PACMAN_ROOT || return $?
export PACMAN_RECIPE_ROOT="${PACMAN_RECIPE_ROOT:-${REPO_ROOT:?REPO_ROOT is required}}"
if [[ "$(realpath -m -- "$PACMAN_RECIPE_ROOT")" != "$(realpath -m -- "$REPO_ROOT")" ]]; then
  echo "PACMAN_RECIPE_ROOT must point to the checkout containing this launcher." >&2
  return 2
fi
export AREAL_PACMAN_ROOT="$PACMAN_RECIPE_ROOT"
pacman_path_alias PACMAN_PYTHON_ROOT MAAPACMAN_PACMAN_ROOT MAAPACMAN_PACMAN_PYTHON_ROOT || return $?
export PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-$(dirname "$REPO_ROOT")/pacman-python}"
export MAAPACMAN_PACMAN_ROOT="$PACMAN_PYTHON_ROOT"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$PACMAN_PYTHON_ROOT"

# The AReaL fork consumes the historical names; export both explicitly.
for suffix in LOGP_RPC_CHUNK_SIZE DISABLE_CUDNN_SDPA; do
  canonical="PACMAN_${suffix}"
  legacy="MAAPACMAN_${suffix}"
  if [[ -n "${!canonical:-}" && -n "${!legacy:-}" && "${!canonical}" != "${!legacy}" ]]; then
    echo "Conflicting settings: ${canonical} and ${legacy}" >&2
    return 2
  fi
  value="${!canonical:-${!legacy:-}}"
  if [[ -n "$value" ]]; then
    export "$canonical=$value" "$legacy=$value"
  fi
done
unset canonical legacy suffix value
