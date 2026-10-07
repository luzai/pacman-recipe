"""CPU-only configuration contract shared by data preparation and training."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from pacman_env.env.ghost_modes import validate_ghost_mode
from .vision_prompt import VISION_LAYOUT_VERSION


def normalize_episode_life_mode(mode: str) -> str:
    """Normalize legacy alias names to canonical episode life modes."""
    return "original_three_lives" if mode == "three_lives" else mode


DIRECT_ACTION_PROTOCOL = "direct-open-action-token-v1"
EDWARD_OPTION_PROTOCOL = "edward-option-code-v1"

DIRECT_PROMPT_VERSION = f"live-state-direct-action-v3+{VISION_LAYOUT_VERSION}"
EDWARD_PROMPT_VERSION = f"edward-option-code-v2+{VISION_LAYOUT_VERSION}"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def json_safe_number(value: Any) -> Any:
    """Represent non-finite config bounds without emitting invalid JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN is not a valid release recipe value")
        return "inf" if value > 0 else "-inf"
    return value


def json_safe_value(value: Any) -> Any:
    """Recursively make config metadata strict JSON (not task rewards)."""
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return json_safe_number(value)


def actual_prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    """Hash the exact two text fields sent beside one observation image."""

    return _canonical_sha256(
        {"system_prompt": system_prompt, "user_prompt": user_prompt}
    )


def prompt_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable template identity for one stage's real harness prompt."""

    from .prompts import prompt_contract_metadata

    prompt_style = str(raw.get("image_prompt_style", "minimal_v1"))
    edward_options = bool(raw.get("edward_options", False))
    actual = prompt_contract_metadata(
        prompt_style, edward_options=edward_options,
        fallback_mode=raw.get("edward_fallback_mode", "refuse"),
    )
    version = str(raw.get("prompt_version", "legacy"))
    if version != "legacy" and version != actual["prompt_version"]:
        raise ValueError("prompt_version does not match the actual prompt renderer")
    return {
        **actual,
        "version": version,
        "image_prompt_style": prompt_style,
        "template_sha256": actual["prompt_template_sha256"],
        "expected_version": actual["prompt_version"],
    }


def recipe_contract_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build JSON-safe, stage-specific provenance for data/run manifests."""

    environment = raw.get("environment") or {}
    generation = raw.get("dataset_generation") or {}
    actor = raw.get("actor") or {}
    reward_norm = actor.get("reward_norm")
    if isinstance(reward_norm, Mapping):
        reward_norm = dict(reward_norm)
        if reward_norm.get("group_size") == "${gconfig.n_samples}":
            reward_norm["group_size"] = int((raw.get("gconfig") or {})["n_samples"])
    reward_fields = (
        "use_base_reward",
        "normal_pellet_reward",
        "power_pellet_reward",
        "ghost_reward",
        "fruit_reward",
        "completion_reward",
        "death_penalty",
        "safety_refusal_penalty",
        "step_penalty",
        "step_penalty_cleared_ratio_scale",
        "wall_penalty",
        "nearest_pellet_alpha",
        "nearest_pellet_remaining_ratio_threshold",
        "nearest_pellet_scale_by_cleared_ratio",
        "nearest_pellet_skip_on_eat",
    )
    dataset_seed = int(generation.get("seed", 0))
    train_rows = int(generation.get("train_episodes", 0))
    validation_rows = int(generation.get("validation_episodes", 0))
    validation_seed_start = generation.get("validation_seed_start")
    if validation_seed_start is None:
        validation_seed_start = dataset_seed + train_rows
    else:
        validation_seed_start = int(validation_seed_start)
    train_batch = int((raw.get("train_dataset") or {}).get("batch_size", 0))
    return {
        "schema": "pacman-recipe-contract-v1",
        "project": "pacman-recipe",
        "training_backend": "areal",
        "ghost_mode": environment.get("ghost_mode"),
        "episode_life_mode": environment.get(
            "episode_life_mode", "single_death"
        ),
        "harness": {
            "action_protocol": raw.get("action_protocol", "legacy"),
            "edward_options": bool(raw.get("edward_options", False)),
            **({"edward_fallback_mode": raw["edward_fallback_mode"]}
               if raw.get("edward_fallback_mode", "refuse") != "refuse" else {}),
            "action_token_choice": bool(raw.get("action_token_choice", False)),
            "open_action_mask": bool(raw.get("open_action_mask", False)),
            "objective_encoding": raw.get("objective_encoding", "legacy"),
        },
        "prompt": prompt_contract(raw),
        "decoding": {
            "enable_thinking": raw.get("enable_thinking"),
            "train": {key: (raw.get("gconfig") or {}).get(key) for key in (
                "n_samples", "min_new_tokens", "max_new_tokens", "temperature", "top_p", "greedy"
            )},
            "validation": {key: (raw.get("eval_gconfig") or {}).get(key) for key in (
                "n_samples", "min_new_tokens", "max_new_tokens", "temperature", "top_p", "greedy"
            )},
        },
        "reward": {
            "formula_version": raw.get("reward_recipe_version"),
            "objective_contract": raw.get("reward_objective_contract"),
            "clip": json_safe_number(actor.get("reward_clip")),
            "normalization": reward_norm,
            "advantage_normalization": actor.get("adv_norm"),
            "coefficients": {field: raw.get(field) for field in reward_fields},
        },
        "data": {
            "training_rng_seed": raw.get("seed"),
            "dataset_seed": dataset_seed,
            "validation_seed_start": generation.get("validation_seed_start"),
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "train_seeds": list(range(dataset_seed, dataset_seed + train_rows)),
            "validation_seeds": list(
                range(validation_seed_start, validation_seed_start + validation_rows)
            ),
            "environment_steps_per_episode": environment.get("max_steps"),
            "train_batch_size": train_batch,
            "validation_batch_size": (raw.get("valid_dataset") or {}).get(
                "batch_size"
            ),
            "samples_per_row": (raw.get("gconfig") or {}).get("n_samples"),
            "updates_per_epoch": train_rows // train_batch if train_batch else None,
            "epochs": raw.get("total_train_epochs"),
        },
        "model": {
            "actor_initialization": actor.get("path"),
            "reference_initialization": (raw.get("ref") or {}).get("path"),
        },
    }


def load_recipe_document(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training recipe must be a mapping")
    return raw


@dataclass
class EnvironmentConfig:
    ghost_mode: str = "normal"
    max_steps: int = 256
    episode_life_mode: str = "single_death"

    def __post_init__(self):
        self.episode_life_mode = normalize_episode_life_mode(self.episode_life_mode)
        validate_ghost_mode(self.ghost_mode)
        if self.episode_life_mode not in {
            "single_death",
            "original_three_lives",
        }:
            raise ValueError(
                "environment.episode_life_mode must be single_death or "
                "original_three_lives"
            )
        if type(self.max_steps) is not int or self.max_steps not in {
            32,
            256,
            512,
            2000,
        }:
            raise ValueError("environment.max_steps must be 32, 256, 512, or 2000")


@dataclass
class DatasetGenerationConfig:
    train_episodes: int = 8
    validation_episodes: int = 2
    seed: int = 0
    validation_seed_start: int | None = None

    def __post_init__(self):
        for name in ("train_episodes", "validation_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(
                    f"dataset_generation.{name} must be a positive integer"
                )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("dataset_generation.seed must be a non-negative integer")
        if (
            self.validation_seed_start is not None
            and (
                type(self.validation_seed_start) is not int
                or self.validation_seed_start < 0
            )
        ):
            raise ValueError(
                "dataset_generation.validation_seed_start must be a non-negative integer or null"
            )


@dataclass
class PlannerAuditConfig:
    max_steps: int = 512

    def __post_init__(self):
        if type(self.max_steps) is not int or self.max_steps not in {32, 256, 512, 2000}:
            raise ValueError("planner_audit.max_steps must be 32, 256, 512, or 2000")


def load_recipe_settings(
    path: Path,
) -> tuple[EnvironmentConfig, DatasetGenerationConfig]:
    raw = load_recipe_document(path)
    return (
        EnvironmentConfig(**raw.get("environment", {})),
        DatasetGenerationConfig(**raw.get("dataset_generation", {})),
    )


def load_planner_audit_settings(path: Path) -> PlannerAuditConfig:
    return PlannerAuditConfig(**load_recipe_document(path).get("planner_audit", {}))
