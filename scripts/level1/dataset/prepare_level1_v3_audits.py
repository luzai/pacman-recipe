"""Generate deterministic API-v3 planner and option-candidate audits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml
from pacman_recipe.level1.recipe import (
    DIRECT_ACTION_PROTOCOL,
    EDWARD_OPTION_PROTOCOL,
    load_recipe_document,
    load_planner_audit_settings,
    load_recipe_settings,
)
from pacman_env.env import (
    Position,
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
    load_bundled_level,
    nearest_reachable_distance,
)
from pacman_env.planner import EdwardPlanner

from pacman_recipe.level1.level1_dataset import (
    DATASET_CONTRACT_VERSION,
    ENV_API_VERSION,
    ENV_NAME,
    REPOSITORY_NAMES,
    SUPPORTED_MAX_STEPS,
    audit_anchor_semantics,
    repository_revisions,
)
from pacman_recipe.level1.prompts import encode_png, png_sha256
from pacman_recipe.level1.rewards import (
    REWARD_RECIPE_VERSION,
    RewardBreakdown,
    RewardConfig,
    audit_reward,
    shape_reward,
)
from pacman_recipe.level1.trajectories import audit_step_environment_evidence
from pacman_env.env.ghost_modes import validate_ghost_mode, validate_ghost_state
from pacman_env.env.pygame_environment import ruleset_revision


AUDIT_CONTRACT_VERSION = "maapacman-level1-planner-audit-v4"
AUDIT_DATASET_ROLES = (
    "deterministic_planner_baseline",
    "option_candidate_audit",
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256_text(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _planner_source_sha256() -> str:
    source = inspect.getsourcefile(EdwardPlanner)
    if source is None:
        raise RuntimeError("cannot resolve Edward planner source for provenance")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _generator_provenance(source_paths: list[Path]) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("areal-pacman git commit is invalid")
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    sources: dict[str, dict[str, str]] = {}
    for source_path in sorted({path.resolve() for path in source_paths}):
        relative = source_path.relative_to(REPO_ROOT).as_posix()
        sources[relative] = {
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()
        }
    return {
        "pacman_recipe_commit": commit,
        "pacman_recipe_dirty": dirty,
        "sources": sources,
    }


def _audit_generator_provenance() -> dict[str, Any]:
    return _generator_provenance(
        [
            Path(__file__),
            REPO_ROOT / "pacman_recipe" / "level1" / "level1_dataset.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "recipe.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "prompts.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "rewards.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "trajectories.py",
        ]
    )


def _write_text_exclusive(
    path: Path, text: str, *, encoding: str = "utf-8"
) -> None:
    with path.open("x", encoding=encoding, newline="\n") as stream:
        stream.write(text)


def _metadata(
    env: PygamePacmanEnv, *, action_protocol: str = EDWARD_OPTION_PROTOCOL
) -> dict[str, Any]:
    provenance = env.provenance
    recipe_revision = repository_revisions()["areal-pacman"]
    if (
        provenance["maapacman_commit"] != recipe_revision["commit"]
        or bool(provenance["maapacman_dirty"]) is not recipe_revision["dirty"]
    ):
        raise RuntimeError(
            "maapacman must be bundled in the active areal-pacman checkout"
        )
    return {
        "name": env.spec.env_id,
        "api_version": env.spec.api_version,
        "backend": "original-pygame",
        "ghost_mode": env.config.ghost_mode,
        "pacman_python_revision": provenance["pacman_python_commit"],
        "pacman_python_source_sha256": provenance[
            "pacman_python_source_sha256"
        ],
        "pacman_python_dirty": bool(provenance["pacman_python_dirty"]),
        "maapacman_revision": recipe_revision["commit"],
        "maapacman_env_source_sha256": provenance[
            "maapacman_env_source_sha256"
        ],
        "maapacman_planner_source_sha256": (
            _planner_source_sha256() if action_protocol == EDWARD_OPTION_PROTOCOL else None
        ),
        "maapacman_dirty": recipe_revision["dirty"],
        "level": int(provenance["level"]),
        "max_steps": int(env.config.max_steps),
        "level_revision": env.spec.level_revision,
        "renderer_revision": env.spec.renderer_revision,
        "ruleset_revision": env.spec.ruleset_revision,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
    }


def _reward_config(config_path: Path) -> tuple[RewardConfig, str]:
    config_bytes = config_path.read_bytes()
    raw = yaml.safe_load(config_bytes)
    if not isinstance(raw, Mapping):
        raise ValueError("training config must be a mapping")
    if raw.get("reward_recipe_version") != REWARD_RECIPE_VERSION:
        raise ValueError("training config does not select the v3 reward recipe")
    values: dict[str, Any] = {"recipe_version": REWARD_RECIPE_VERSION}
    for field in fields(RewardConfig):
        if field.name == "recipe_version":
            continue
        if field.name not in raw:
            if field.name == "safety_refusal_penalty":
                values[field.name] = field.default
                continue
            raise ValueError(f"training config is missing {field.name}")
        values[field.name] = raw[field.name]
    return RewardConfig(**values), hashlib.sha256(config_bytes).hexdigest()


def _nearest_distance(level: Any, start: Position, targets: set[Position]) -> int:
    if not targets:
        return 0
    return int(nearest_reachable_distance(level, start, targets))


def _selected_candidate(decision: Any) -> dict[str, Any]:
    selected = [
        candidate
        for candidate in decision.candidates
        if candidate.option_id == decision.option_id
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "planner decision must select exactly one advertised candidate"
        )
    if selected[0].first_action != decision.action:
        raise RuntimeError("planner decision action disagrees with selected option")
    return selected[0].as_dict()


def audit_planner_record(record: Mapping[str, Any]) -> None:
    """Fail closed when a serialized planner audit loses source evidence."""

    required = {
        "audit_contract_version",
        "dataset_contract_version",
        "seed",
        "step",
        "env",
        "source_revisions",
        "observation_png_sha256",
        "structured_state_sha256",
        "structured_state",
        "legal_actions",
        "planner_candidates",
        "selected_option",
        "executed_primitive_action",
        "next_observation_png_sha256",
        "next_structured_state_sha256",
        "next_structured_state",
        "environment_score_delta",
        "score_before",
        "score",
        "logic_frame_before",
        "logic_frames",
        "atomic_substeps",
        "logic_frame_events",
        "events",
        "score_components",
        "ghosts",
        "edible_ticks",
        "pygame_mode",
        "pellets_remaining",
        "normal_pellets_remaining",
        "power_pellets_remaining",
        "reward_breakdown",
        "terminated",
        "truncated",
        "terminal_reason",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"planner audit is missing fields: {sorted(missing)}")
    if record["audit_contract_version"] != AUDIT_CONTRACT_VERSION:
        raise ValueError("planner audit contract version mismatch")
    if record["dataset_contract_version"] != DATASET_CONTRACT_VERSION:
        raise ValueError("planner audit dataset contract mismatch")
    action_protocol = record.get("action_protocol", EDWARD_OPTION_PROTOCOL)
    audit_anchor_semantics(action_protocol)
    revisions = record["source_revisions"]
    if not isinstance(revisions, Mapping) or set(revisions) != set(
        REPOSITORY_NAMES
    ):
        raise ValueError("planner audit source revisions are incomplete")
    for repository_name, revision in revisions.items():
        if not isinstance(revision, Mapping):
            raise ValueError(
                f"planner audit {repository_name} revision must be an object"
            )
        commit = revision.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not isinstance(revision.get("dirty"), bool)
        ):
            raise ValueError(
                f"planner audit {repository_name} revision is invalid"
            )
    environment = record["env"]
    if not isinstance(environment, Mapping):
        raise ValueError("planner audit env must be an object")
    ghost_mode = validate_ghost_mode(environment.get("ghost_mode"))
    if environment.get("ruleset_revision") != ruleset_revision(ghost_mode):
        raise ValueError("planner audit ruleset_revision does not match ghost_mode")
    if (
        environment.get("api_version") != ENV_API_VERSION
        or environment.get("name") != ENV_NAME
        or environment.get("backend") != "original-pygame"
        or environment.get("level") != 1
        or environment.get("max_steps") not in SUPPORTED_MAX_STEPS
        or isinstance(environment.get("max_steps"), bool)
        or environment.get("dataset_contract_version")
        != DATASET_CONTRACT_VERSION
    ):
        raise ValueError("planner audit environment identity mismatch")
    for digest_field in (
        "ruleset_revision",
        "pacman_python_source_sha256",
        "maapacman_env_source_sha256",
        "level_revision",
    ):
        digest = environment.get(digest_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                f"planner audit environment has invalid {digest_field}"
            )
    planner_digest = environment.get("maapacman_planner_source_sha256")
    if action_protocol == DIRECT_ACTION_PROTOCOL:
        if planner_digest is not None:
            raise ValueError("direct audit must not claim Edward planner provenance")
    elif not isinstance(planner_digest, str) or len(planner_digest) != 64:
        raise ValueError("planner audit environment has invalid maapacman_planner_source_sha256")
    renderer_revision = environment.get("renderer_revision")
    if not isinstance(renderer_revision, str) or not renderer_revision:
        raise ValueError("planner audit environment has invalid renderer_revision")
    for revision_field in ("pacman_python_revision", "maapacman_revision"):
        revision = environment.get(revision_field)
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(
                f"planner audit environment has invalid {revision_field}"
            )
    for dirty_field in ("pacman_python_dirty", "maapacman_dirty"):
        if not isinstance(environment.get(dirty_field), bool):
            raise ValueError(
                f"planner audit environment has invalid {dirty_field}"
            )
    recipe_revision = revisions["areal-pacman"]
    if (
        environment["maapacman_revision"] != recipe_revision["commit"]
        or environment["maapacman_dirty"] is not recipe_revision["dirty"]
    ):
        raise ValueError(
            "planner audit maapacman provenance must match bundled "
            "areal-pacman"
        )
    pacman_revision = revisions["pacman-python"]
    if (
        environment["pacman_python_revision"] != pacman_revision["commit"]
        or environment["pacman_python_dirty"] is not pacman_revision["dirty"]
    ):
        raise ValueError(
            "planner audit pacman-python provenance must match source revisions"
        )
    if environment["renderer_revision"] != (
        f"pacman-python:{pacman_revision['commit']}"
    ):
        raise ValueError(
            "planner audit renderer revision must match pacman-python"
        )
    for digest_field in (
        "observation_png_sha256",
        "structured_state_sha256",
        "next_observation_png_sha256",
        "next_structured_state_sha256",
    ):
        digest = record[digest_field]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"planner audit has invalid {digest_field}")
    if record["structured_state_sha256"] != _canonical_sha256(
        record["structured_state"]
    ):
        raise ValueError("planner audit structured-state hash mismatch")
    if record["next_structured_state_sha256"] != _canonical_sha256(
        record["next_structured_state"]
    ):
        raise ValueError("planner audit next-state hash mismatch")
    validate_ghost_state(record["structured_state"], ghost_mode)
    validate_ghost_state(record["next_structured_state"], ghost_mode)

    legal_actions = record["legal_actions"]
    if (
        not isinstance(legal_actions, list)
        or len(legal_actions) != len(set(legal_actions))
        or any(action not in {"U", "D", "L", "R", "S"} for action in legal_actions)
    ):
        raise ValueError("planner audit legal_actions are invalid")
    action = record["executed_primitive_action"]
    if action not in legal_actions:
        raise ValueError("planner audit executed an action outside legal_actions")
    structured_state = record["structured_state"]
    if not isinstance(structured_state, Mapping) or not isinstance(
        structured_state.get("open"), list
    ):
        raise ValueError("planner audit structured state has no open-action mask")
    if set(legal_actions) != set(structured_state["open"]) | {"S"}:
        raise ValueError("planner audit legal_actions disagree with structured state")
    candidates = record["planner_candidates"]
    selected = record["selected_option"]
    if action_protocol == DIRECT_ACTION_PROTOCOL:
        if candidates != [] or selected is not None:
            raise ValueError("direct audit cannot contain Edward options")
        directions = [token for token in ("U", "D", "L", "R") if token in legal_actions]
        if not directions or action != directions[0]:
            raise ValueError("direct anchor must execute the first legal U/D/L/R")
    else:
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("planner audit requires advertised candidates")
        matching = [item for item in candidates if item == selected]
        if len(matching) != 1:
            raise ValueError("selected option is not unique in planner candidates")
        if not isinstance(selected, Mapping) or selected.get("first_action") != action:
            raise ValueError("selected option disagrees with executed action")

    breakdown = record["reward_breakdown"]
    if not isinstance(breakdown, Mapping):
        raise ValueError("planner audit reward_breakdown must be an object")
    required_reward_fields = {field.name for field in fields(RewardBreakdown)}
    if set(breakdown) != required_reward_fields:
        missing_reward_fields = sorted(required_reward_fields - set(breakdown))
        extra_reward_fields = sorted(set(breakdown) - required_reward_fields)
        raise ValueError(
            "planner audit reward_breakdown schema mismatch: "
            f"missing={missing_reward_fields}, extra={extra_reward_fields}"
        )
    reward_record = {
        **breakdown,
        "logic_frame_events": record["logic_frame_events"],
    }
    audit_reward(reward_record)
    if int(float(record["environment_score_delta"])) != int(
        float(breakdown["base_reward"])
    ):
        raise ValueError("planner audit environment reward mismatch")
    audit_step_environment_evidence(
        {
            **breakdown,
            "logic_frames": record["logic_frames"],
            "atomic_substeps": record["atomic_substeps"],
            "logic_frame_events": record["logic_frame_events"],
            "events": record["events"],
            "score_components": record["score_components"],
            "score": record["score"],
            "ghosts": record["ghosts"],
            "edible_ticks": record["edible_ticks"],
            "pygame_mode": record["pygame_mode"],
            "pellets_remaining": record["pellets_remaining"],
            "normal_pellets_remaining": record[
                "normal_pellets_remaining"
            ],
            "power_pellets_remaining": record[
                "power_pellets_remaining"
            ],
            "death": breakdown["death"],
            "level_completed": breakdown["level_completed"],
            "terminated": record["terminated"],
            "truncated": record["truncated"],
            "terminal_reason": record["terminal_reason"],
        },
        previous_score=int(record["score_before"]),
        previous_logic_frame=int(record["logic_frame_before"]),
        ghost_mode=ghost_mode,
    )


def audit_planner_sequence(records: list[Mapping[str, Any]]) -> None:
    """Reconcile observation/state/score continuity across a full replay."""

    if not records:
        raise ValueError("planner baseline sequence must not be empty")
    seed = records[0].get("seed")
    environment = records[0].get("env")
    for index, record in enumerate(records, start=1):
        audit_planner_record(record)
        if int(record["step"]) != index:
            raise ValueError("planner baseline steps are not contiguous")
        if record.get("seed") != seed or record.get("env") != environment:
            raise ValueError("planner baseline provenance changed mid-sequence")
        if index > 1:
            previous = records[index - 2]
            if (
                previous["next_observation_png_sha256"]
                != record["observation_png_sha256"]
                or previous["next_structured_state_sha256"]
                != record["structured_state_sha256"]
                or previous["next_structured_state"]
                != record["structured_state"]
            ):
                raise ValueError("planner baseline observation/state continuity failed")
            if int(previous["score"]) != int(record["score_before"]):
                raise ValueError("planner baseline score continuity failed")
            previous_frame = int(previous["atomic_substeps"][-1]["frame"])
            if previous_frame != int(record["logic_frame_before"]):
                raise ValueError("planner baseline logic-frame continuity failed")
        if index < len(records) and (
            record["terminated"] or record["truncated"]
        ):
            raise ValueError("planner baseline contains post-terminal records")
    final = records[-1]
    if (
        not final["terminated"]
        or final["truncated"]
        or final["terminal_reason"] != "all_normal_pellets"
        or final["reward_breakdown"]["death"]
    ):
        raise ValueError("planner baseline did not end in a death-free clear")


def _collect_seed(
    *,
    seed: int,
    env: PygamePacmanEnv,
    planner: EdwardPlanner | None,
    reward_config: RewardConfig,
    max_records: int | None = None,
    require_successful_clear: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive when provided")
    level = load_bundled_level(1)
    remaining_normal_pellets = set(level.pellets)
    initial_normal_pellets = len(remaining_normal_pellets)
    image, info = env.reset(seed=seed)
    action_protocol = EDWARD_OPTION_PROTOCOL if planner is not None else DIRECT_ACTION_PROTOCOL
    if planner is None and (max_records != 1 or require_successful_clear):
        raise ValueError("direct audit collection is limited to one initial transition")
    metadata = _metadata(env, action_protocol=action_protocol)
    records: list[dict[str, Any]] = []
    terminated = truncated = False
    previous_score = 0
    previous_logic_frame = 0
    while not (terminated or truncated):
        snapshot = env.snapshot()
        legal_actions = list(info["legal_actions"])
        if planner is None:
            directions = [token for token in ("U", "D", "L", "R") if token in legal_actions]
            if not directions:
                raise RuntimeError("initial direct state has no legal direction")
            action = directions[0]
            selected = None
            candidates = []
        else:
            decision = planner.decide(snapshot)
            action = decision.action
            selected = _selected_candidate(decision)
            candidates = [candidate.as_dict() for candidate in decision.candidates]
        if action not in legal_actions:
            raise RuntimeError("planner selected an illegal primitive action")
        before_ratio = len(remaining_normal_pellets) / initial_normal_pellets
        distance_before = None
        earliest_ratio_after = (
            len(remaining_normal_pellets)
            if reward_config.nearest_pellet_skip_on_eat
            else max(0, len(remaining_normal_pellets) - 1)
        ) / initial_normal_pellets
        if (
            reward_config.nearest_pellet_alpha > 0
            and earliest_ratio_after
            <= reward_config.nearest_pellet_remaining_ratio_threshold
        ):
            row, col = info["pacman_position"]
            distance_before = _nearest_distance(
                level,
                Position(int(row), int(col)),
                remaining_normal_pellets,
            )

        next_image, base_reward, terminated, truncated, next_info = env.step(
            action
        )
        next_position = Position(
            int(next_info["pacman_position"][0]),
            int(next_info["pacman_position"][1]),
        )
        if bool(next_info["pellet_eaten"]):
            if next_position not in remaining_normal_pellets:
                raise RuntimeError("planner audit pellet tracker diverged")
            remaining_normal_pellets.remove(next_position)
        if len(remaining_normal_pellets) != int(
            next_info["normal_pellets_remaining"]
        ):
            raise RuntimeError("planner audit normal-pellet count diverged")
        after_ratio = len(remaining_normal_pellets) / initial_normal_pellets
        distance_after = None
        if (
            distance_before is not None
            and after_ratio
            <= reward_config.nearest_pellet_remaining_ratio_threshold
            and not (
                reward_config.nearest_pellet_skip_on_eat
                and bool(next_info["pellet_eaten"])
            )
        ):
            distance_after = _nearest_distance(
                level, next_position, remaining_normal_pellets
            )
        reward = shape_reward(
            base_reward,
            info,
            next_info,
            reward_config,
            normal_pellet_remaining_ratio=after_ratio,
            normal_pellet_remaining_ratio_before=before_ratio,
            nearest_pellet_distance_before=distance_before,
            nearest_pellet_distance_after=distance_after,
        )
        evidence = {
            **reward.as_dict(),
            "logic_frames": int(next_info["logic_frames"]),
            "atomic_substeps": list(next_info["atomic_substeps"]),
            "logic_frame_events": list(next_info["logic_frame_events"]),
            "events": list(next_info["events"]),
            "score_components": dict(next_info["score_components"]),
            "score": int(next_info["score"]),
            "ghosts": list(next_info["ghosts"]),
            "edible_ticks": int(next_info["edible_ticks"]),
            "pygame_mode": int(next_info["pygame_mode"]),
            "pellets_remaining": int(next_info["pellets_remaining"]),
            "normal_pellets_remaining": int(
                next_info["normal_pellets_remaining"]
            ),
            "power_pellets_remaining": int(
                next_info["power_pellets_remaining"]
            ),
            "death": bool(reward.death),
            "level_completed": bool(reward.level_completed),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_reason": next_info["terminal_reason"],
        }
        score_before = previous_score
        logic_frame_before = previous_logic_frame
        previous_score, previous_logic_frame = audit_step_environment_evidence(
            evidence,
            previous_score=previous_score,
            previous_logic_frame=previous_logic_frame,
            ghost_mode=env.config.ghost_mode,
        )
        audit_reward(evidence)
        next_snapshot = env.snapshot()
        record = {
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "dataset_contract_version": DATASET_CONTRACT_VERSION,
            "action_protocol": action_protocol,
            "seed": seed,
            "step": int(next_info["step"]),
            "env": metadata,
            "source_revisions": repository_revisions(),
            "observation_png_sha256": png_sha256(encode_png(image)),
            "structured_state_sha256": _canonical_sha256(snapshot),
            "structured_state": snapshot,
            "legal_actions": legal_actions,
            "planner_candidates": candidates,
            "selected_option": selected,
            "executed_primitive_action": action,
            "next_observation_png_sha256": png_sha256(
                encode_png(next_image)
            ),
            "next_structured_state_sha256": _canonical_sha256(next_snapshot),
            "next_structured_state": next_snapshot,
            "environment_score_delta": int(base_reward),
            "score_before": score_before,
            "score": int(next_info["score"]),
            "logic_frame_before": logic_frame_before,
            "logic_frames": int(next_info["logic_frames"]),
            "atomic_substeps": list(next_info["atomic_substeps"]),
            "logic_frame_events": list(next_info["logic_frame_events"]),
            "events": list(next_info["events"]),
            "score_components": dict(next_info["score_components"]),
            "ghosts": list(next_info["ghosts"]),
            "edible_ticks": int(next_info["edible_ticks"]),
            "pygame_mode": int(next_info["pygame_mode"]),
            "pellets_remaining": int(next_info["pellets_remaining"]),
            "normal_pellets_remaining": int(
                next_info["normal_pellets_remaining"]
            ),
            "power_pellets_remaining": int(
                next_info["power_pellets_remaining"]
            ),
            "reward_breakdown": asdict(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_reason": next_info["terminal_reason"],
        }
        audit_planner_record(record)
        records.append(record)
        image, info = next_image, next_info
        if max_records is not None and len(records) >= max_records:
            break
    if require_successful_clear and (
        not terminated
        or truncated
        or info.get("terminal_reason") != "all_normal_pellets"
    ):
        raise RuntimeError(
            "Edward planner baseline did not clear level 1; "
            f"seed={seed} terminal_reason={info.get('terminal_reason')!r}"
        )
    if require_successful_clear:
        audit_planner_sequence(records)
    elif max_records is None or len(records) != max_records:
        raise RuntimeError("partial planner audit did not produce its requested rows")
    return records, metadata


def collect_initial_audit_anchor(
    *,
    seed: int,
    max_steps: int,
    reward_config: RewardConfig,
    pacman_python_root: str | None = None,
    ghost_mode: str = "normal",
    action_protocol: str = EDWARD_OPTION_PROTOCOL,
) -> dict[str, Any]:
    """Capture one real stage-specific transition, never a model rollout."""

    semantics = audit_anchor_semantics(action_protocol)

    env = PygamePacmanEnv(
        PygamePacmanEnvConfig(
            pacman_python_root=pacman_python_root,
            max_steps=max_steps,
            ghost_mode=ghost_mode,
        )
    )
    try:
        records, _ = _collect_seed(
            seed=seed,
            env=env,
            planner=EdwardPlanner() if action_protocol == EDWARD_OPTION_PROTOCOL else None,
            reward_config=reward_config,
            max_records=1,
            require_successful_clear=False,
        )
    finally:
        env.close()
    anchor = {
        **records[0],
        "audit_role": "episode_spec_audit_anchor",
        "audit_anchor_semantics": semantics,
    }
    audit_planner_record(anchor)
    return anchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pacman-python-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment, _ = load_recipe_settings(args.config)
    planner_audit = load_planner_audit_settings(args.config)
    if args.max_steps is None:
        args.max_steps = planner_audit.max_steps
    if args.max_steps != planner_audit.max_steps:
        raise ValueError("--max-steps must match config planner_audit.max_steps")
    if load_recipe_document(args.config).get("action_protocol") != EDWARD_OPTION_PROTOCOL:
        raise ValueError("Edward baseline audit requires an Edward recipe; C1 uses direct anchors")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("audit seeds must be unique")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reward_config, config_sha256 = _reward_config(args.config)
    baseline: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    per_seed_rows: dict[str, int] = {}
    for seed in args.seeds:
        env = PygamePacmanEnv(
            PygamePacmanEnvConfig(
                pacman_python_root=args.pacman_python_root,
                max_steps=args.max_steps,
                ghost_mode=environment.ghost_mode,
            )
        )
        try:
            records, seed_metadata = _collect_seed(
                seed=seed,
                env=env,
                planner=EdwardPlanner(),
                reward_config=reward_config,
            )
        finally:
            env.close()
        if metadata is None:
            metadata = seed_metadata
        elif metadata != seed_metadata:
            raise RuntimeError("environment provenance changed across audit seeds")
        baseline.extend(records)
        per_seed_rows[str(seed)] = len(records)
    if metadata is None:
        raise RuntimeError("planner audit produced no environment metadata")

    candidates = [
        {
            **record,
            "audit_role": "option_candidate_audit",
            "audit_dataset_semantics": {
                "scope": "all_advertised_candidates_at_each_baseline_decision",
                "source": "deterministic_planner_baseline",
                "independent_rollout": False,
            },
        }
        for record in baseline
    ]
    baseline_rows = [
        {
            **record,
            "audit_role": "deterministic_planner_baseline",
            "audit_dataset_semantics": {
                "scope": "complete_deterministic_successful_replay",
                "source": "EdwardPlanner",
                "independent_rollout": True,
            },
        }
        for record in baseline
    ]
    files = {}
    for name, rows in (
        ("planner_baseline", baseline_rows),
        ("option_candidate_audit", candidates),
    ):
        path = args.output_root / f"{name}.jsonl"
        files[name] = {
            "path": path.relative_to(args.output_root).as_posix(),
            "rows": len(rows),
            "sha256": _sha256_text(path, rows),
            "per_row_provenance": True,
            "per_row_reward_breakdown": "complete_and_audited",
        }
    manifest = {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "dataset_roles": list(AUDIT_DATASET_ROLES),
        "dataset_role_semantics": {
            "deterministic_planner_baseline": (
                "complete deterministic successful replay"
            ),
            "option_candidate_audit": (
                "candidate-set audit derived from each baseline decision; "
                "not an independent rollout"
            ),
        },
        "baseline_acceptance": {
            "death_accepted": False,
            "required_terminal_reason": "all_normal_pellets",
        },
        "environment": metadata,
        "source_revisions": repository_revisions(),
        "generator_provenance": _audit_generator_provenance(),
        "seeds": args.seeds,
        "per_seed_rows": per_seed_rows,
        "audit_max_steps": args.max_steps,
        "training_environment_max_steps": environment.max_steps,
        "reward_recipe_version": REWARD_RECIPE_VERSION,
        "reward_config": asdict(reward_config),
        "training_config_sha256": config_sha256,
        "files": files,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path = args.output_root / "manifest.json"
    _write_text_exclusive(manifest_path, manifest_text)
    digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    _write_text_exclusive(
        args.output_root / "manifest.sha256",
        f"{digest}  manifest.json\n",
        encoding="ascii",
    )
    print(manifest_text, end="")


if __name__ == "__main__":
    main()
