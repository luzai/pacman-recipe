"""Deterministic episode specifications for the production level-1 recipe."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Iterator, Mapping

from maapacman.env import (
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
)
from maapacman.env.ghost_modes import validate_ghost_mode
from maapacman.planner import EdwardPlanner

from .recipe import (
    DIRECT_ACTION_PROTOCOL,
    DIRECT_PROMPT_VERSION,
    EDWARD_OPTION_PROTOCOL,
    EDWARD_PROMPT_VERSION,
)


ENV_NAME = "pacman-python-level1-ghostdoor-v3"
ENV_API_VERSION = "3.0"
ENV_BACKEND = "original-pygame"
DATASET_CONTRACT_VERSION = "maapacman-level1-dataset-v4"
PREFIX_AUDIT_CONTRACT_VERSION = "planner-preterminal-prefix-audit-v1"
# Production horizons are selected by each immutable dataset row.  The v4
# contract deliberately has no implicit 287-step episode assumption.
PRODUCTION_MAX_STEPS = 512
SHORT_HORIZON_MAX_STEPS = 32
LONG_HORIZON_MAX_STEPS = 256
STRESS_MAX_STEPS = 512
DEMO_SAFETY_MAX_STEPS = 2000
SUPPORTED_MAX_STEPS = frozenset(
    {
        SHORT_HORIZON_MAX_STEPS,
        LONG_HORIZON_MAX_STEPS,
        STRESS_MAX_STEPS,
        DEMO_SAFETY_MAX_STEPS,
    }
)
REPOSITORY_NAMES = ("pacman-python", "areal-pacman", "AReaL")


def audit_anchor_semantics(action_protocol: str) -> dict[str, Any]:
    if action_protocol not in {DIRECT_ACTION_PROTOCOL, EDWARD_OPTION_PROTOCOL}:
        raise ValueError("unsupported episode action_protocol")
    return {
        "scope": (
            "initial_state_one_direct_step"
            if action_protocol == DIRECT_ACTION_PROTOCOL
            else "initial_state_one_edward_step"
        ),
        "model_rollout": False,
        "training_sample": False,
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def repository_revisions() -> dict[str, dict[str, Any]]:
    """Return the three-repository provenance required by each v4 record."""

    recipe_root = Path(__file__).resolve().parents[2]
    workspace_root = recipe_root.parent
    pacman_python_root = Path(
        os.getenv("MAAPACMAN_PACMAN_ROOT")
        or os.getenv("MAAPACMAN_PACMAN_PYTHON_ROOT")
        or workspace_root / "pacman-python"
    ).resolve()
    areal_root = Path(
        os.getenv("AREAL_ROOT") or workspace_root / "AReaL"
    ).resolve()
    repositories = {
        "pacman-python": pacman_python_root,
        "areal-pacman": recipe_root,
        "AReaL": areal_root,
    }
    revisions: dict[str, dict[str, Any]] = {}
    for name in REPOSITORY_NAMES:
        repository = repositories[name]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if len(commit) != 40:
            raise RuntimeError(f"{name} git commit is invalid")
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        revisions[name] = {"commit": commit, "dirty": dirty}
    return revisions


def _validate_source_revisions(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(REPOSITORY_NAMES):
        raise ValueError("source_revisions must contain all three repositories")
    for name, revision in value.items():
        if not isinstance(revision, Mapping):
            raise ValueError(f"source_revisions.{name} must be an object")
        commit = revision.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not isinstance(revision.get("dirty"), bool)
        ):
            raise ValueError(f"source_revisions.{name} is invalid")


@lru_cache(maxsize=2)
def environment_metadata(ghost_mode: str = "normal") -> dict[str, Any]:
    env = PygamePacmanEnv(PygamePacmanEnvConfig(ghost_mode=ghost_mode))
    try:
        info = env.provenance
        revisions = repository_revisions()
        recipe_revision = revisions["areal-pacman"]
        if (
            info["maapacman_commit"] != recipe_revision["commit"]
            or bool(info["maapacman_dirty"]) is not recipe_revision["dirty"]
        ):
            raise RuntimeError(
                "maapacman must be bundled in the active areal-pacman checkout"
            )
        return {
            "name": env.spec.env_id,
            "api_version": env.spec.api_version,
            "backend": ENV_BACKEND,
            "ghost_mode": ghost_mode,
            "pacman_python_revision": info["pacman_python_commit"],
            "pacman_python_source_sha256": info[
                "pacman_python_source_sha256"
            ],
            "pacman_python_dirty": bool(info["pacman_python_dirty"]),
            "maapacman_revision": recipe_revision["commit"],
            "maapacman_env_source_sha256": info[
                "maapacman_env_source_sha256"
            ],
            "maapacman_dirty": recipe_revision["dirty"],
            "level": int(info["level"]),
            "level_revision": env.spec.level_revision,
            "renderer_revision": env.spec.renderer_revision,
            "ruleset_revision": env.spec.ruleset_revision,
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "source_revisions": revisions,
        }
    finally:
        env.close()


def environment_revision() -> str:
    return environment_metadata()["level_revision"]


def validate_episode_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("episode id must be a non-empty string")
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    env = row.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("env must be an object")
    installed = environment_metadata(validate_ghost_mode(env.get("ghost_mode")))
    if env.get("name") != ENV_NAME or env.get("name") != installed["name"]:
        raise ValueError(f"env.name must be {ENV_NAME!r}")
    if (
        env.get("api_version") != ENV_API_VERSION
        or env.get("api_version") != installed["api_version"]
    ):
        raise ValueError(f"env.api_version must be {ENV_API_VERSION!r}")
    if env.get("backend") != ENV_BACKEND:
        raise ValueError(f"env.backend must be {ENV_BACKEND!r}")
    if env.get("pacman_python_revision") != installed["pacman_python_revision"]:
        raise ValueError(
            "env.pacman_python_revision does not match installed pacman-python"
        )
    if env.get("pacman_python_source_sha256") != installed[
        "pacman_python_source_sha256"
    ]:
        raise ValueError(
            "env.pacman_python_source_sha256 does not match installed source"
        )
    if env.get("pacman_python_dirty") is not installed["pacman_python_dirty"]:
        raise ValueError(
            "env.pacman_python_dirty does not match installed source"
        )
    if env.get("maapacman_revision") != installed["maapacman_revision"]:
        raise ValueError(
            "env.maapacman_revision does not match bundled maapacman"
        )
    if env.get("maapacman_env_source_sha256") != installed[
        "maapacman_env_source_sha256"
    ]:
        raise ValueError(
            "env.maapacman_env_source_sha256 does not match installed source"
        )
    if env.get("maapacman_dirty") is not installed["maapacman_dirty"]:
        raise ValueError("env.maapacman_dirty does not match installed source")
    if env.get("level_revision") != installed["level_revision"]:
        raise ValueError("env.level_revision does not match bundled maapacman")
    if env.get("ruleset_revision") != installed["ruleset_revision"]:
        raise ValueError("env.ruleset_revision does not match bundled maapacman")
    if env.get("renderer_revision") != installed["renderer_revision"]:
        raise ValueError("env.renderer_revision does not match bundled maapacman")
    if row.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
        raise ValueError(
            f"dataset_contract_version must be {DATASET_CONTRACT_VERSION!r}"
        )
    _validate_source_revisions(row.get("source_revisions"))
    if row.get("source_revisions") != repository_revisions():
        raise ValueError("source_revisions do not match installed repositories")
    if env.get("level") != 1 or isinstance(env.get("level"), bool):
        raise ValueError("env.level must be integer 1")
    seed = env.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("env.seed must be an integer")
    max_steps = env.get("max_steps")
    if max_steps not in SUPPORTED_MAX_STEPS or isinstance(max_steps, bool):
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_MAX_STEPS))
        raise ValueError(f"env.max_steps must be one of: {supported}")
    if env.get("observation_mode") != "rgb":
        raise ValueError("env.observation_mode must be 'rgb'")
    action_protocol = row.get("action_protocol", EDWARD_OPTION_PROTOCOL)
    if "action_protocol" in row:
        audit_anchor_semantics(action_protocol)
        expected_prompt = (
            DIRECT_PROMPT_VERSION
            if action_protocol == DIRECT_ACTION_PROTOCOL
            else EDWARD_PROMPT_VERSION
        )
        if row.get("prompt_version") != expected_prompt:
            raise ValueError("episode prompt_version does not match action_protocol")
    restart_fields = ("restart_state_path", "restart_state_sha256", "restart_state_id")
    if any(row.get(name) is not None for name in restart_fields):
        if not all(isinstance(row.get(name), str) and row[name] for name in restart_fields):
            raise ValueError("restart rows require path, sha256, and id together")
        digest = row["restart_state_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("restart_state_sha256 must be a lowercase SHA256")
        if action_protocol != DIRECT_ACTION_PROTOCOL:
            raise ValueError("restart learning requires primitive direct actions")
        if row.get("state_prefix_actions") or row.get("decision_steps") is not None:
            raise ValueError("restart state cannot be combined with a prefix or one-step row")
    prefix_actions = row.get("state_prefix_actions", [])
    if (
        not isinstance(prefix_actions, list)
        or any(action not in {"U", "D", "L", "R"} for action in prefix_actions)
    ):
        raise ValueError(
            "state_prefix_actions must be a list containing only U, D, L, or R"
        )
    decision_steps = row.get("decision_steps")
    if decision_steps is not None and decision_steps != 1:
        raise ValueError("decision_steps, when present, must be exactly 1")
    prefix_audit = row.get("state_prefix_audit")
    if prefix_actions or decision_steps is not None:
        if action_protocol == DIRECT_ACTION_PROTOCOL:
            raise ValueError("direct episode cannot contain Edward state prefixes")
        if not isinstance(prefix_audit, Mapping):
            raise ValueError("state-prefix rows require state_prefix_audit")
        if prefix_audit.get("contract_version") != PREFIX_AUDIT_CONTRACT_VERSION:
            raise ValueError(
                "state_prefix_audit.contract_version must be "
                f"{PREFIX_AUDIT_CONTRACT_VERSION!r}"
            )
        if prefix_audit.get("source") != "edward-planner-preterminal-replay":
            raise ValueError("state_prefix_audit.source is invalid")
        if prefix_audit.get("source_seed") != seed:
            raise ValueError("state prefix source seed must match env.seed")
        if prefix_audit.get("prefix_state_index") != len(prefix_actions):
            raise ValueError("state prefix index must match its action count")
        source_steps = prefix_audit.get("source_episode_steps")
        if (
            not isinstance(source_steps, int)
            or isinstance(source_steps, bool)
            or source_steps <= len(prefix_actions)
        ):
            raise ValueError("state prefix must precede the source terminal step")
        if prefix_audit.get("prefix_verified_nonterminal") is not True:
            raise ValueError("state prefix must be verified nonterminal")
        if prefix_audit.get("prefix_wall_collisions") != 0:
            raise ValueError("state prefix must have zero wall collisions")
        terminal_reason = prefix_audit.get("source_terminal_reason")
        if terminal_reason not in {"death", "all_normal_pellets"}:
            raise ValueError("state prefix source must have an audited terminal")
        cleared = prefix_audit.get("source_cleared_level")
        if cleared is not (terminal_reason == "all_normal_pellets"):
            raise ValueError("state prefix source clear flag is inconsistent")
        if prefix_audit.get("successful_baseline") is not cleared:
            raise ValueError("successful_baseline must exactly match level clear")
    anchor = row.get("audit_anchor")
    if anchor is not None:
        if not isinstance(anchor, Mapping):
            raise ValueError("audit_anchor must be an object")
        anchor_digest = row.get("audit_anchor_sha256")
        if (
            not isinstance(anchor_digest, str)
            or len(anchor_digest) != 64
            or anchor_digest != _canonical_sha256(anchor)
        ):
            raise ValueError("audit_anchor canonical hash mismatch")
        if (
            anchor.get("audit_role") != "episode_spec_audit_anchor"
            or anchor.get("audit_contract_version")
            != "maapacman-level1-planner-audit-v4"
            or anchor.get("dataset_contract_version") != DATASET_CONTRACT_VERSION
            or anchor.get("seed") != seed
            or anchor.get("step") != 1
        ):
            raise ValueError("audit_anchor identity does not match episode spec")
        if anchor.get("audit_anchor_semantics") != audit_anchor_semantics(action_protocol):
            raise ValueError("audit_anchor semantics are invalid")
        if anchor.get("action_protocol", EDWARD_OPTION_PROTOCOL) != action_protocol:
            raise ValueError("audit_anchor action_protocol does not match episode")
        if row.get("source_revisions") != anchor.get("source_revisions"):
            raise ValueError("episode source revisions differ from audit_anchor")
        selected = anchor.get("selected_option")
        action = anchor.get("executed_primitive_action")
        if action_protocol == DIRECT_ACTION_PROTOCOL:
            if selected is not None or anchor.get("planner_candidates") != []:
                raise ValueError("direct audit_anchor cannot contain Edward options")
            if action not in {"U", "D", "L", "R"}:
                raise ValueError("direct audit_anchor must execute U/D/L/R")
        elif not isinstance(selected, Mapping) or selected.get("first_action") != action:
            raise ValueError("audit_anchor selected option/action mismatch")
        anchor_env = anchor.get("env")
        if not isinstance(anchor_env, Mapping):
            raise ValueError("audit_anchor env must be an object")
        identity_fields = {
            "ghost_mode",
            "name",
            "api_version",
            "backend",
            "pacman_python_revision",
            "pacman_python_source_sha256",
            "pacman_python_dirty",
            "maapacman_revision",
            "maapacman_env_source_sha256",
            "maapacman_dirty",
            "level_revision",
            "renderer_revision",
            "ruleset_revision",
            "max_steps",
        }
        if any(anchor_env.get(field) != env.get(field) for field in identity_fields):
            raise ValueError("audit_anchor provenance differs from episode spec")


def make_episode_row(
    index: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
    ghost_mode: str = "normal",
    action_protocol: str | None = None,
) -> dict[str, Any]:
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ValueError("index must be a positive integer")
    installed = environment_metadata(ghost_mode)
    row = {
        "id": f"level1-{ghost_mode}-seed{seed}-{split}-{index:04d}",
        "split": split,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "source_revisions": repository_revisions(),
        "env": {
            "ghost_mode": ghost_mode,
            "name": ENV_NAME,
            "api_version": ENV_API_VERSION,
            "backend": ENV_BACKEND,
            "pacman_python_revision": installed["pacman_python_revision"],
            "pacman_python_source_sha256": installed[
                "pacman_python_source_sha256"
            ],
            "pacman_python_dirty": installed["pacman_python_dirty"],
            "maapacman_revision": installed["maapacman_revision"],
            "maapacman_env_source_sha256": installed[
                "maapacman_env_source_sha256"
            ],
            "maapacman_dirty": installed["maapacman_dirty"],
            "level_revision": installed["level_revision"],
            "renderer_revision": installed["renderer_revision"],
            "ruleset_revision": installed["ruleset_revision"],
            "level": 1,
            "seed": seed,
            "max_steps": max_steps,
            "observation_mode": "rgb",
        },
    }
    if action_protocol is not None:
        row["action_protocol"] = action_protocol
        row["prompt_version"] = (
            DIRECT_PROMPT_VERSION
            if action_protocol == DIRECT_ACTION_PROTOCOL
            else EDWARD_PROMPT_VERSION
        )
    validate_episode_row(row)
    return row


def generate_episode_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
    ghost_mode: str = "normal",
    action_protocol: str | None = None,
) -> Iterator[dict[str, Any]]:
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    for index in range(1, count + 1):
        yield make_episode_row(
            index, split=split, seed=seed, max_steps=max_steps, ghost_mode=ghost_mode,
            action_protocol=action_protocol,
        )


def planner_baseline_state_prefixes(*, seed: int = 0) -> list[list[str]]:
    """Return prefixes from a verified successful Edward baseline replay."""

    return [
        record["prefix"]
        for record in planner_baseline_state_records(seed=seed)
    ]


def planner_baseline_state_records(*, seed: int = 0) -> list[dict[str, Any]]:
    """Return deterministic planner states only after a successful replay."""
    records = planner_audit_state_records(seed=seed)
    audit = records[0]["prefix_audit"] if records else None
    if not records:
        raise RuntimeError("level-1 planner baseline produced no decision states")
    if not audit["successful_baseline"]:
        raise RuntimeError(
            "Edward planner baseline did not clear level 1: "
            f"terminal_reason={audit['source_terminal_reason']!r}"
        )
    return records


def planner_audit_state_prefixes(*, seed: int = 0) -> list[list[str]]:
    """Return verified nonterminal prefixes without claiming level success."""

    return [record["prefix"] for record in planner_audit_state_records(seed=seed)]


def planner_audit_state_records(*, seed: int = 0) -> list[dict[str, Any]]:
    """Return collision-free states before an audited terminal outcome.

    A death-terminated replay is valid evidence that each earlier prefix was
    nonterminal and collision-free.  It is explicitly not a successful
    baseline and cannot pass :func:`planner_baseline_state_records`.
    """
    env = PygamePacmanEnv()
    planner = EdwardPlanner()
    try:
        _, info = env.reset(seed=seed)
        actions: list[str] = []
        records: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            snapshot = env.snapshot()
            records.append(
                {
                    "state_index": len(actions),
                    "prefix": list(actions),
                    "open_actions": [
                        token
                        for token in ("U", "D", "L", "R")
                        if token in set(snapshot.get("open") or [])
                    ],
                }
            )
            decision = planner.decide(snapshot)
            _, _, terminated, truncated, info = env.step(decision.action)
            if info["wall_collision"]:
                raise RuntimeError("Edward planner prefix collided with a wall")
            actions.append(decision.action)
        if not records:
            raise RuntimeError("level-1 planner audit produced no decision states")
        terminal_reason = info.get("terminal_reason")
        if not terminated or truncated or terminal_reason not in {
            "death",
            "all_normal_pellets",
        }:
            raise RuntimeError(
                "Edward planner audit did not reach an accepted terminal: "
                f"terminated={terminated!r}, truncated={truncated!r}, "
                f"terminal_reason={terminal_reason!r}"
            )
        cleared = terminal_reason == "all_normal_pellets"
        audit = {
            "contract_version": PREFIX_AUDIT_CONTRACT_VERSION,
            "source": "edward-planner-preterminal-replay",
            "source_seed": seed,
            "source_terminal_reason": terminal_reason,
            "source_cleared_level": cleared,
            "successful_baseline": cleared,
            "prefix_verified_nonterminal": True,
            "prefix_wall_collisions": 0,
        }
        for record in records:
            record["prefix_audit"] = {
                **audit,
                "prefix_state_index": int(record["state_index"]),
                "source_episode_steps": len(actions),
            }
        return records
    finally:
        env.close()


def oracle_state_prefixes(*, seed: int = 0) -> list[list[str]]:
    """Compatibility alias; this is a verified planner baseline, not an oracle."""

    return planner_baseline_state_prefixes(seed=seed)


def oracle_state_records(*, seed: int = 0) -> list[dict[str, Any]]:
    """Compatibility alias; this is a verified planner baseline, not an oracle."""

    return planner_baseline_state_records(seed=seed)


def generate_single_step_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
    offset: int = 0,
) -> Iterator[dict[str, Any]]:
    """Generate varied screenshot -> one action -> immediate reward samples."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    records = planner_audit_state_records(seed=seed)
    prefixes = [record["prefix"] for record in records]
    if offset < 0 or offset + count > len(prefixes):
        raise ValueError(
            f"requested prefix range [{offset}, {offset + count}) exceeds "
            f"{len(prefixes)} available planner-audit states"
        )
    for index, record in enumerate(records[offset : offset + count], start=1):
        prefix = record["prefix"]
        row = make_episode_row(
            index,
            split=split,
            seed=seed,
            max_steps=max_steps,
        )
        row["id"] = f"level1-wall-{split}-{offset + index:04d}"
        row["state_prefix_actions"] = prefix
        row["decision_steps"] = 1
        row["state_prefix_audit"] = dict(record["prefix_audit"])
        validate_episode_row(row)
        yield row


def _evenly_spaced(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"requested {count} states from only {len(records)}")
    return [
        records[((2 * index + 1) * len(records)) // (2 * count)]
        for index in range(count)
    ]


def generate_balanced_corridor_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
) -> Iterator[dict[str, Any]]:
    """Generate a 50/50 horizontal-vs-vertical held-out corridor task.

    Every direction is open in exactly half of the selected states. This
    prevents a constant action such as always-R from earning positive reward
    without reading the screenshot.
    """
    if split not in {"train", "validation"}:
        raise ValueError("balanced corridor split must be train or validation")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if count % 2:
        raise ValueError("balanced corridor row count must be even")

    by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = {
        ("L", "R"): [],
        ("U", "D"): [],
    }
    for record in planner_audit_state_records(seed=seed):
        mask = tuple(record["open_actions"])
        if mask in by_mask:
            by_mask[mask].append(record)

    needed_per_mask = 24
    selected = {
        mask: _evenly_spaced(records, needed_per_mask)
        for mask, records in by_mask.items()
    }
    records_by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for mask in (("L", "R"), ("U", "D")):
        records = selected[mask]
        candidates = [
            record
            for index, record in enumerate(records)
            if (index % 3 == 2) == (split == "validation")
        ]
        records_by_mask[mask] = candidates[: count // 2]
    split_records = [
        record
        for pair in zip(
            records_by_mask[("L", "R")],
            records_by_mask[("U", "D")],
            strict=True,
        )
        for record in pair
    ]

    for index, record in enumerate(split_records, start=1):
        row = make_episode_row(
            index,
            split=split,
            seed=seed,
            max_steps=max_steps,
        )
        row["id"] = (
            f"level1-wall-{split}-state{int(record['state_index']):04d}"
        )
        row["state_prefix_actions"] = list(record["prefix"])
        row["decision_steps"] = 1
        row["state_open_actions_for_audit"] = list(record["open_actions"])
        row["state_prefix_audit"] = dict(record["prefix_audit"])
        validate_episode_row(row)
        yield row


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> str:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        validate_episode_row(row)
        normalized.append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in normalized
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_hf_dataset(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' package to write HF datasets") from exc
    materialized = [dict(row) for row in rows]
    for row in materialized:
        validate_episode_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace HF dataset: {path}")
    Dataset.from_list(materialized).save_to_disk(str(path))
