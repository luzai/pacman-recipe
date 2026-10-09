"""Framework-independent Pacman episode execution and audit."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pacman_env.env import (
    Action,
    PacmanEnvSpec,
    Position,
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
    load_bundled_level,
    nearest_reachable_distance,
)
from pacman_env.planner import (
    EdwardPlanner,
    EdwardSafetyRefusal,
    PlannerCandidate,
    nearest_lethal_ghost_distance,
    validate_fallback_mode,
)

from ..actions import ActionParseError, parse_action
from .level1_dataset import SUPPORTED_MAX_STEPS, validate_episode_row
from .prompts import (
    build_image_messages,
    layout_image_user_content,
    edward_system_prompt,
    render_edward_decision_prompt,
    render_vision_edward_decision_prompt,
    encode_png,
    image_count,
    png_sha256,
    prompt_text,
    prompt_contract_metadata,
    prompt_user_template,
    sent_prompt_sha256,
    text_sha256,
)
from .rewards import RewardConfig, shape_reward
from .trajectories import TrajectoryAuditError, audit_trajectory, write_trajectory
from .token_constraints import (
    EDWARD_OPTION_CONSTRAINT,
    ObjectiveParseError,
    ObjectiveTokenConstraint,
)
from .vision_prompt import vision_model_image


EXPECTED_ACTIONS = ("U", "D", "L", "R", "S")
MOVEMENT_ACTIONS = ("U", "D", "L", "R")
ACTION_MASK_BIT = {
    action: 1 << index for index, action in enumerate(MOVEMENT_ACTIONS)
}
OPPOSITE_ACTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
class _NamedLogger:
    """Resolve the logger by name on every call.

    AReaL's logging setup can replace ``logging.Logger.manager``; a logger object captured at import
    time then is no longer the one registered under this name, so handlers attached by name (for
    example ``assertLogs``) never see its records.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str):
        return getattr(logging.getLogger(self._name), attribute)


LOGGER = _NamedLogger(__name__)

# vLLM (observed on 0.22.1) occasionally drops the `allowed_token_ids`
# sampling mask for a single request under high concurrency + multimodal
# input, most often when the allowed set contains a byte-fallback BPE token
# that only decodes to a valid character together with a sibling token
# (decodes alone to U+FFFD). The request then samples unconstrained and
# returns a token outside the allowed set, which previously always failed
# the whole episode as a contract violation. Retrying the same decision with
# a fresh request id almost always recovers a compliant sample, since the
# underlying failure is a transient serving-layer glitch, not a policy or
# prompt problem. 3 = 1 initial attempt + 2 retries.
_MASK_LEAK_RETRY_ATTEMPTS = 3


DECISION_BOUNDARY_SCHEMA = "pacman-decision-boundary-v1"


def _boundary_verify(png, candidates, constraint, system_prompt, user_prompt):
    """What a restored boundary must reproduce exactly before the model is asked."""
    return json.loads(json.dumps({
        "png_sha256": png_sha256(png) if png else None,
        **({"observation_mode": "ascii", "observation_text_sha256": text_sha256(user_prompt)} if not png else {}),
        "candidates": [candidate.as_dict() for candidate in candidates],
        "allowed_token_ids": [int(token) for token in constraint.allowed_token_ids],
        "system_prompt_sha256": text_sha256(system_prompt),
        "user_prompt_sha256": text_sha256(user_prompt),
    }))


def _decision_boundary(
    env, planner, cell_exit_history, recent_actions, recent_positions, no_progress_steps, verify,
    live_snapshot, previous_info,
):
    """Full continuation point before an Edward option choice (env + planner + runner)."""
    return {
        "schema": DECISION_BOUNDARY_SCHEMA,
        "env_state": env.save_state(),
        "context": {
            "planner": {
                "remaining": sorted([p.row, p.col] for p in planner.remaining),
                "last_action": planner.last_action,
                "fallback_mode": planner.fallback_mode,
            },
            "runner": {
                "cell_exit_history": sorted(
                    [row, col, list(actions)] for (row, col), actions in cell_exit_history.items()
                ),
                "recent_actions": list(recent_actions),
                "recent_positions": [list(position) for position in recent_positions],
                "no_progress_steps": int(no_progress_steps),
            },
            "verify": verify,
        },
        # Bank diversity features (not used for restoration).
        "features": {
            "pacman_position": [int(v) for v in previous_info["pacman_position"]],
            "normal_pellets_remaining": int(previous_info["normal_pellets_remaining"]),
            "nearest_lethal_ghost_distance": nearest_lethal_ghost_distance(live_snapshot, planner.level),
            "env_step": int(previous_info["step"]),
        },
    }


def _compact_edward_decision_prompt(
    state_context: Mapping[str, Any],
    candidates: tuple[PlannerCandidate, ...],
    constraint: ObjectiveTokenConstraint,
) -> str:
    """Render the exemplar's decision facts without its free-form reason."""
    return render_vision_edward_decision_prompt(
        state_context, candidates, constraint,
        fallback_mode=state_context.get("edward_fallback_mode", "refuse"),
    )


def _nearest_reachable_distance_with_diagnostics(
    level: Any,
    start: Position,
    targets: set[Position],
    *,
    phase: str,
    previous_position: tuple[int, int] | None = None,
    action: str | None = None,
    live_legal_actions: list[str] | None = None,
    allow_unreachable: bool = False,
) -> int | None:
    if not targets:
        return 0
    try:
        return nearest_reachable_distance(level, start, targets)
    except ValueError as exc:
        if str(exc) != "no target is reachable from the requested position":
            raise
        diagnostic = {
            "phase": phase,
            "start_position": [start.row, start.col],
            "start_in_bounds": bool(level.in_bounds(start)),
            "start_tile": (
                int(level.tile_at(start)) if level.in_bounds(start) else None
            ),
            "previous_position": (
                list(previous_position)
                if previous_position is not None
                else None
            ),
            "action": action,
            "live_legal_actions": list(live_legal_actions or []),
            "remaining_normal_pellet_count": len(targets),
            "remaining_normal_pellet_positions": [
                [position.row, position.col]
                for position in sorted(
                    targets,
                    key=lambda position: (position.row, position.col),
                )
            ],
            "level_height": int(level.height),
            "level_width": int(level.width),
            "level_revision": str(level.revision),
        }
        encoded = json.dumps(diagnostic, sort_keys=True)
        if allow_unreachable:
            LOGGER.warning(
                "nearest-pellet BFS unreachable; truncating trapped rollout: %s",
                encoded,
            )
            return None
        LOGGER.error("nearest-pellet BFS unreachable: %s", encoded)
        raise RuntimeError(
            f"nearest-pellet BFS unreachable: {encoded}"
        ) from exc


def _normal_pellet_event_position(
    info: Mapping[str, Any],
) -> Position | None:
    events = [
        event
        for event in info.get("logic_frame_events", [])
        if event.get("event_type") == "normal_pellet_eaten"
    ]
    if not bool(info.get("pellet_eaten")):
        if events:
            raise RuntimeError(
                "normal-pellet event ledger disagrees with pellet_eaten"
            )
        return None
    if len(events) != 1:
        raise RuntimeError(
            "pellet_eaten requires exactly one normal-pellet source event"
        )
    position = events[0].get("pacman_position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or any(not isinstance(value, int) for value in position)
    ):
        raise RuntimeError("normal-pellet source event has invalid position")
    return Position(position[0], position[1])


def preferred_open_actions(
    open_actions: list[str],
    tried_actions: list[str],
    last_action: str | None,
) -> list[str]:
    """Match the live demo's untried/least-used and anti-reverse preference."""
    counts = {action: 0 for action in open_actions}
    for action in tried_actions:
        if action in counts:
            counts[action] += 1
    untried = [action for action in open_actions if counts[action] == 0]
    if untried:
        preferred = untried
    elif open_actions:
        least = min(counts.values())
        preferred = [
            action for action in open_actions if counts[action] == least
        ]
    else:
        preferred = []
    reverse = OPPOSITE_ACTION.get(last_action or "")
    without_reverse = [action for action in preferred if action != reverse]
    return without_reverse or preferred


@dataclass(frozen=True)
class ModelTurn:
    completion: str
    completion_id: str | None
    messages: list[dict[str, Any]]
    reasoning_content: str | None = None
    raw_response: dict[str, Any] | None = None
    request_extra_body: dict[str, Any] | None = None


def validate_env_spec(
    spec: PacmanEnvSpec,
    requested: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
) -> None:
    if spec.api_version != "3.0" or requested.get("api_version") != spec.api_version:
        raise RuntimeError("unsupported Pacman environment API")
    if spec.env_id != "pacman-python-level1-ghostdoor-v3":
        raise RuntimeError("unsupported Pacman environment ID")
    if requested.get("name") != spec.env_id:
        raise RuntimeError("dataset environment ID does not match Pacman")
    if requested.get("backend") != "original-pygame":
        raise RuntimeError("production recipe requires original-pygame backend")
    if requested.get("ghost_mode") != provenance.get("ghost_mode"):
        raise RuntimeError("ghost_mode does not match dataset row")
    if requested.get("pacman_python_revision") != provenance.get(
        "pacman_python_commit"
    ):
        raise RuntimeError("pacman-python revision does not match dataset row")
    if requested.get("pacman_python_source_sha256") != provenance.get(
        "pacman_python_source_sha256"
    ):
        raise RuntimeError("pacman-python source hash does not match dataset row")
    if requested.get("maapacman_revision") != provenance.get(
        "maapacman_commit"
    ):
        raise RuntimeError("Pacman revision does not match dataset row")
    if requested.get("maapacman_env_source_sha256") != provenance.get(
        "maapacman_env_source_sha256"
    ):
        raise RuntimeError("Pacman source hash does not match dataset row")
    if spec.action_tokens != EXPECTED_ACTIONS:
        raise RuntimeError("incompatible Pacman action contract")
    if requested.get("level_revision") != spec.level_revision:
        raise RuntimeError("Pacman level revision does not match dataset row")
    if requested.get("ruleset_revision") != spec.ruleset_revision:
        raise RuntimeError("Pacman ruleset revision does not match dataset row")
    if spec.observation_dtype != "uint8" or len(spec.observation_shape) != 3:
        raise RuntimeError("incompatible Pacman RGB observation contract")


class PacmanEpisodeRunner:
    """One dataset row to one complete level-1 image-only rollout."""

    def __init__(
        self,
        *,
        env_factory: Callable[[PygamePacmanEnvConfig], PygamePacmanEnv] = (
            PygamePacmanEnv
        ),
        tokenizer: Any | None = None,
        row_validator: Callable[[Mapping[str, Any]], None] = validate_episode_row,
        **workflow_kwargs: Any,
    ) -> None:
        self.env_factory = env_factory
        self.row_validator = row_validator
        self.workflow_kwargs = workflow_kwargs
        self.last_episode: dict[str, Any] | None = None
        self._episode_payload: ContextVar[dict[str, Any] | None] = ContextVar(
            "pacman_episode_payload", default=None
        )
        self.action_token_id_by_action: dict[str, int] = {}
        self.objective_tokenizer: Any | None = None
        # Optional: called with a full decision boundary before each Edward option choice.
        self.decision_boundary_sink: Callable[[dict[str, Any]], None] | None = None
        if workflow_kwargs.get("open_action_mask") or workflow_kwargs.get(
            "edward_options"
        ):
            if tokenizer is None:
                tokenizer_path = workflow_kwargs.get("tokenizer_path")
                if not tokenizer_path:
                    raise ValueError("token constraints require tokenizer_path")
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.objective_tokenizer = tokenizer

        if workflow_kwargs.get("open_action_mask"):
            for token in MOVEMENT_ACTIONS:
                token_ids = tokenizer.encode(
                    token, add_special_tokens=False
                )
                if len(token_ids) != 1:
                    raise ValueError(
                        f"action {token!r} must map to one tokenizer ID"
                    )
                self.action_token_id_by_action[token] = int(token_ids[0])

    async def run(self, data: Mapping[str, Any], **extra_kwargs: Any) -> Any:
        # Prevent a refused episode from exposing the previous call's payload
        # through the debugging attribute on a reused workflow instance.
        self.last_episode = None
        options = {**self.workflow_kwargs, **extra_kwargs}
        if options.get("enable_thinking") is None:
            options["enable_thinking"] = False
        if options["enable_thinking"] is not False:
            raise ValueError(
                "production image-only Pacman requires enable_thinking=false"
            )
        self.row_validator(data)
        requested = data["env"]
        if options.get("ghost_mode", requested["ghost_mode"]) != requested["ghost_mode"]:
            raise ValueError("training ghost_mode does not match dataset row")
        if (
            options.get("environment_max_steps", requested["max_steps"])
            != requested["max_steps"]
        ):
            raise ValueError("training environment.max_steps does not match dataset row")
        seed = int(requested["seed"])
        state_prefix_actions = list(data.get("state_prefix_actions") or [])
        restart_fields = ("restart_state_path", "restart_state_sha256", "restart_state_id")
        restart_saved = None
        restart_metadata = None
        if any(data.get(name) is not None for name in restart_fields):
            if not all(isinstance(data.get(name), str) and data[name] for name in restart_fields):
                raise ValueError("restart state requires path, SHA-256 and ID")
            if state_prefix_actions:
                raise ValueError("restart state and state_prefix_actions are mutually exclusive")
            raw_restart = Path(data["restart_state_path"]).read_bytes()
            if hashlib.sha256(raw_restart).hexdigest() != data["restart_state_sha256"]:
                raise ValueError("restart state file SHA-256 mismatch")
            restart_saved = json.loads(raw_restart)
            restart_episode = restart_saved["payload"]["episode"]
            if restart_episode["finished"]:
                raise ValueError("restart state must not be finished")
            if restart_episode["seed"] != seed:
                raise ValueError("restart state seed differs from dataset row")
        single_step = data.get("decision_steps") == 1
        options["single_step"] = single_step
        trajectory_sample_id = str(
            options.get("trajectory_sample_id") or uuid.uuid4().hex
        )
        config = PygamePacmanEnvConfig(
            pacman_python_root=options.get("pacman_python_root"),
            level=int(requested["level"]),
            ghost_mode=requested["ghost_mode"],
            max_steps=int(requested["max_steps"]),
            episode_life_mode=str(
                options.get("episode_life_mode", "single_death")
            ),
            video_driver=options.get("video_driver", "dummy"),
            audio_driver=options.get("audio_driver", "dummy"),
            worker_base_dir=options.get("worker_base_dir"),
        )
        reward_config = RewardConfig(
            recipe_version=str(
                options.get(
                    "reward_recipe_version",
                    "maapacman-level1-event-reward-v3",
                )
            ),
            step_penalty=float(options.get("step_penalty", 1.0)),
            step_penalty_cleared_ratio_scale=float(
                options.get("step_penalty_cleared_ratio_scale", 0.0)
            ),
            wall_penalty=float(options.get("wall_penalty", 1.0)),
            use_base_reward=bool(options.get("use_base_reward", True)),
            normal_pellet_reward=float(
                options.get("normal_pellet_reward", 0.0)
            ),
            power_pellet_reward=float(
                options.get("power_pellet_reward", 0.0)
            ),
            ghost_reward=float(options.get("ghost_reward", 0.0)),
            fruit_reward=float(options.get("fruit_reward", 0.0)),
            death_penalty=float(options.get("death_penalty", 0.0)),
            completion_reward=float(options.get("completion_reward", 0.0)),
            safety_refusal_penalty=float(
                options.get("safety_refusal_penalty", 0.0)
            ),
            nearest_pellet_alpha=float(
                options.get("nearest_pellet_alpha", 0.0)
            ),
            nearest_pellet_remaining_ratio_threshold=float(
                options.get(
                    "nearest_pellet_remaining_ratio_threshold",
                    1.0,
                )
            ),
            nearest_pellet_scale_by_cleared_ratio=bool(
                options.get(
                    "nearest_pellet_scale_by_cleared_ratio", False
                )
            ),
            nearest_pellet_skip_on_eat=bool(
                options.get("nearest_pellet_skip_on_eat", False)
            ),
        )
        contract_violation_return = float(
            options.get("contract_violation_return", -1.0)
        )
        if (
            not math.isfinite(contract_violation_return)
            or abs(contract_violation_return + 1.0) > 1e-9
        ):
            raise ValueError(
                "contract_violation_return must be exactly -1.0"
            )
        image_prompt_style = str(
            options.get("image_prompt_style", "minimal_v1")
        )
        from .text_observation import text_sent_prompt_sha256
        from .prompts import ASCII_STYLES
        ascii_observation = image_prompt_style in ASCII_STYLES
        system_prompt, user_instruction = prompt_text(image_prompt_style)
        scripted = list(options.get("scripted_actions") or [])
        scripted_objectives = list(options.get("scripted_objectives") or [])
        scripted_ids = list(options.get("scripted_completion_ids") or [])
        edward_options = bool(options.get("edward_options", False))
        if ascii_observation and not edward_options:
            raise ValueError('ASCII experiment requires Edward actions')
        fallback_mode = validate_fallback_mode(
            options.get("edward_fallback_mode", "refuse"), edward_options=edward_options
        )
        if edward_options:
            if (
                options.get("objective_encoding", EDWARD_OPTION_CONSTRAINT)
                != EDWARD_OPTION_CONSTRAINT
            ):
                raise ValueError(
                    "Edward options require objective_encoding="
                    f"{EDWARD_OPTION_CONSTRAINT}"
                )
            if ascii_observation:
                from .prompts import ascii_system_prompt_for
                system_prompt = ascii_system_prompt_for(image_prompt_style, fallback_mode)
            else:
                system_prompt = edward_system_prompt(fallback_mode)
        if edward_options and scripted:
            raise ValueError(
                "edward_options uses scripted_objectives, not scripted_actions"
            )
        if edward_options and options.get("open_action_mask"):
            raise ValueError(
                "edward objective constraints replace the legacy action mask"
            )
        prompt_metadata = prompt_contract_metadata(
            image_prompt_style, edward_options=edward_options, fallback_mode=fallback_mode
        )
        if not edward_options and not options.get("open_action_mask"):
            prompt_metadata["action_protocol"] = "direct-action-token-v1"
        for field in ("action_protocol", "prompt_version"):
            configured = options.get(field)
            if configured not in (None, "legacy"):
                if configured != prompt_metadata[field]:
                    raise ValueError(f"configured {field} does not match actual harness")
                if data.get(field) != configured:
                    raise ValueError(f"dataset {field} does not match actual harness")
        recipe_contract = options.get("recipe_contract")
        if recipe_contract is not None:
            if (recipe_contract.get("harness") or {}).get(
                "edward_fallback_mode", "refuse"
            ) != fallback_mode:
                raise ValueError("recipe_contract edward_fallback_mode differs from runtime")
            # Config bounds are strings (e.g. "inf"); rewards remain finite.
            json.dumps(recipe_contract, allow_nan=False)
            if recipe_contract.get("ghost_mode", config.ghost_mode) != config.ghost_mode:
                raise ValueError("recipe_contract ghost_mode differs from runtime")
            coefficients = (recipe_contract.get("reward") or {}).get("coefficients") or {}
            if any(value != getattr(reward_config, name, None) for name, value in coefficients.items()):
                raise ValueError("recipe_contract reward coefficients differ from runtime")
        user_instruction = prompt_user_template(
            image_prompt_style, edward_options=edward_options, fallback_mode=fallback_mode
        )
        trajectory: list[dict[str, Any]] = []
        rewards_by_completion: dict[str, float] = {}
        parse_failures = 0
        canonical_violations = 0
        cell_exit_history: dict[tuple[int, int], list[str]] = {}
        recent_actions: list[str] = []
        recent_positions: list[tuple[int, int]] = []
        wall_clock_limit_seconds = options.get("wall_clock_limit_seconds")
        if wall_clock_limit_seconds is not None:
            wall_clock_limit_seconds = float(wall_clock_limit_seconds)
            if wall_clock_limit_seconds <= 0:
                raise ValueError("wall_clock_limit_seconds must be positive")
        stuck_no_progress_steps = options.get("stuck_no_progress_steps")
        if stuck_no_progress_steps is not None:
            stuck_no_progress_steps = int(stuck_no_progress_steps)
            if stuck_no_progress_steps <= 0:
                raise ValueError("stuck_no_progress_steps must be positive")
        no_progress_steps = 0
        episode_started_at = time.monotonic()
        level = (
            load_bundled_level(config.level)
            if reward_config.nearest_pellet_alpha > 0
            else None
        )
        remaining_normal_pellets = (
            set(level.pellets) if level is not None else None
        )
        planner = (
            (EdwardPlanner(fallback_mode=fallback_mode) if fallback_mode != "refuse"
             else EdwardPlanner())
            if edward_options else None
        )
        active_option: PlannerCandidate | None = None
        active_objective_constraint: ObjectiveTokenConstraint | None = None
        active_turn: ModelTurn | None = None
        active_action: str | None = None
        active_remaining = 0
        active_option_step = 0
        active_reward = 0.0

        with self.env_factory(config) as env:
            validate_env_spec(
                env.spec,
                requested,
                provenance=env.provenance,
            )
            if config.max_steps not in SUPPORTED_MAX_STEPS:
                supported = ", ".join(
                    str(value) for value in sorted(SUPPORTED_MAX_STEPS)
                )
                raise RuntimeError(
                    f"level-1 recipe requires max_steps in: {supported}"
            )
            if restart_saved is None:
                image, previous_info = env.reset(seed=seed)
            else:
                image, previous_info = env.restore_state(restart_saved)
                if previous_info["terminated"] or previous_info["truncated"]:
                    raise ValueError("restart state must not be terminal")
                restart_metadata = {
                    "id": data["restart_state_id"],
                    "path": data["restart_state_path"],
                    "sha256": data["restart_state_sha256"],
                    "identity": restart_saved["payload"]["identity"],
                    "source_step": int(previous_info["step"]),
                    "score": int(previous_info["score"]),
                    "logic_frame": int(previous_info["logic_frame"]),
                    "death_count": int(previous_info.get("death_count", 0)),
                    "normal_pellets": int(previous_info["normal_pellets_remaining"]),
                    "power_pellets": int(previous_info["power_pellets_remaining"]),
                    "remaining_budget": config.max_steps - int(previous_info["step"]),
                }
            if planner is not None:
                planner.observe(env.snapshot())
            state_prefix_evidence: list[dict[str, Any]] = []
            for prefix_token in state_prefix_actions:
                image, _, terminated, truncated, previous_info = env.step(
                    Action(prefix_token)
                )
                state_prefix_evidence.append(
                    {
                        "action": prefix_token,
                        "env_step": int(previous_info["step"]),
                        "score": int(previous_info["score"]),
                        "score_delta": int(previous_info["score_delta"]),
                        "logic_frame": int(previous_info["logic_frame"]),
                        "logic_frames": int(previous_info["logic_frames"]),
                        "wall_collision": bool(previous_info["wall_collision"]),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if previous_info["wall_collision"]:
                    raise RuntimeError(
                        "state_prefix_actions must be collision-free"
                    )
                if terminated or truncated:
                    raise RuntimeError(
                        "state_prefix_actions reached a terminal state"
                    )
                if planner is not None:
                    planner.observe(env.snapshot())
            prefix_end_score = int(previous_info["score"])
            prefix_end_logic_frame = int(previous_info["logic_frame"])
            initial_normal_pellets = int(
                previous_info["normal_pellets_remaining"]
            )
            if initial_normal_pellets <= 0:
                raise RuntimeError("level-1 must start with normal pellets")
            # Reward coefficients retain their true-start interpretation even
            # when episode metrics measure only the learner's suffix.
            reward_normal_pellets_initial = (
                len(load_bundled_level(config.level).pellets)
                if restart_saved is not None else initial_normal_pellets
            )
            if restart_saved is not None and remaining_normal_pellets is not None:
                remaining_normal_pellets = {
                    Position(int(row), int(col))
                    for row, col in env.snapshot()["normal_pellet_positions"]
                }
            recent_positions.append(
                (
                    int(previous_info["pacman_position"][0]),
                    int(previous_info["pacman_position"][1]),
                )
            )
            boundary_expected = None
            boundary_context = (restart_saved or {}).get("boundary_context")
            if boundary_context is not None:
                # Dynamic-bank decision boundary: resume planner and runner history
                # exactly; the first decision must reproduce the recorded one.
                if planner is None or boundary_context["planner"]["fallback_mode"] != fallback_mode:
                    raise ValueError("decision boundary requires the same Edward fallback mode")
                planner.remaining = {
                    Position(int(row), int(col))
                    for row, col in boundary_context["planner"]["remaining"]
                }
                planner.last_action = boundary_context["planner"]["last_action"]
                runner_state = boundary_context["runner"]
                cell_exit_history = {
                    (int(row), int(col)): list(actions)
                    for row, col, actions in runner_state["cell_exit_history"]
                }
                recent_actions = list(runner_state["recent_actions"])
                recent_positions = [
                    (int(row), int(col)) for row, col in runner_state["recent_positions"]
                ]
                no_progress_steps = int(runner_state["no_progress_steps"])
                boundary_expected = boundary_context["verify"]
            if level is not None:
                live_state = env.snapshot()
                if (
                    level.width != int(live_state["width"])
                    or level.height != int(live_state["height"])
                ):
                    raise RuntimeError(
                        "nearest-pellet topology dimensions do not match "
                        "the live environment"
                    )
            if (
                remaining_normal_pellets is not None
                and len(remaining_normal_pellets)
                != int(previous_info["normal_pellets_remaining"])
            ):
                raise RuntimeError(
                    "nearest-pellet tracker does not match reset state"
                )
            final_info = previous_info
            while True:
                turn: ModelTurn | None = None
                model_image = image if ascii_observation else vision_model_image(image)
                png = b'' if ascii_observation else encode_png(model_image)
                live_snapshot = env.snapshot()
                option_candidates: tuple[PlannerCandidate, ...] = ()
                objective_constraint: ObjectiveTokenConstraint | None = None
                if edward_options and active_option is None:
                    if planner is None or self.objective_tokenizer is None:
                        raise RuntimeError(
                            "Edward options require a planner and tokenizer"
                        )
                    try:
                        option_candidates = planner.advertised_candidates(
                            live_snapshot
                        )
                    except EdwardSafetyRefusal:
                        if not trajectory:
                            # There is no model completion or executed option to
                            # train on. Returning None lets AReaL reject and
                            # replace this rollout without inventing evidence.
                            return None
                        final_record = trajectory[-1]
                        if (
                            final_record.get("option_status") == "active"
                            or not final_record.get("option_end")
                        ):
                            raise RuntimeError(
                                "Edward safety refusal occurred with an "
                                "unfinished option"
                            )
                        completion_id = final_record.get("completion_id")
                        option_return = final_record.get("option_return")
                        if (
                            completion_id is None
                            or completion_id not in rewards_by_completion
                            or option_return is None
                        ):
                            raise RuntimeError(
                                "Edward safety refusal has no completed option "
                                "reward to penalize"
                            )
                        safety_penalty = reward_config.safety_refusal_penalty
                        final_record["safety_refusal"] = True
                        final_record["safety_refusal_penalty"] = safety_penalty
                        final_record["shaped_reward"] = (
                            float(final_record["shaped_reward"])
                            - safety_penalty
                        )
                        final_record["option_return"] = (
                            float(option_return) - safety_penalty
                        )
                        rewards_by_completion[completion_id] = (
                            float(rewards_by_completion[completion_id])
                            - safety_penalty
                        )
                        final_record["terminated"] = False
                        final_record["truncated"] = True
                        final_record["terminal_reason"] = "safety_refusal"
                        final_info = dict(final_info)
                        final_info["terminated"] = False
                        final_info["truncated"] = True
                        final_info["terminal_reason"] = "safety_refusal"
                        break
                    objective_constraint = ObjectiveTokenConstraint.build(
                        self.objective_tokenizer,
                        (candidate.option_id for candidate in option_candidates),
                    )
                current_open_actions = [
                    action
                    for action in MOVEMENT_ACTIONS
                    if action in set(live_snapshot.get("open") or [])
                ]
                if not current_open_actions:
                    raise RuntimeError(
                        "live environment reported no open movement actions"
                    )
                state_context = None
                if image_prompt_style in ("live_state_v3", "ascii_edward_v1", "ascii_edward_spaced_v1"):
                    position = (
                        int(previous_info["pacman_position"][0]),
                        int(previous_info["pacman_position"][1]),
                    )
                    open_actions = current_open_actions
                    blocked_actions = [
                        action
                        for action in MOVEMENT_ACTIONS
                        if action in set(live_snapshot.get("blocked") or [])
                    ]
                    if set(open_actions) | set(blocked_actions) != set(
                        MOVEMENT_ACTIONS
                    ):
                        blocked_actions = [
                            action
                            for action in MOVEMENT_ACTIONS
                            if action not in set(open_actions)
                        ]
                    current_history_full = list(
                        cell_exit_history.get(position, [])
                    )
                    current_history = list(
                        dict.fromkeys(current_history_full)
                    )
                    current_counts = {
                        action: current_history_full.count(action)
                        for action in MOVEMENT_ACTIONS
                        if action in current_history_full
                    }
                    last_action = (
                        recent_actions[-1] if recent_actions else None
                    )
                    state_context = {
                        "episode_life_mode": env.config.episode_life_mode,
                        "ghosts": list(live_snapshot.get("ghosts") or []),
                        "edible_ticks": int(live_snapshot.get("edible_ticks", 0)),
                        "maze_size": [int(live_snapshot["height"]), int(live_snapshot["width"])],
                        "pacman_position": list(position),
                        "facing": str(live_snapshot.get("facing") or "S"),
                        "pellets_remaining": int(
                            previous_info["pellets_remaining"]
                        ),
                        "open_actions": open_actions,
                        "legal_actions": [
                            action.value for action in env.legal_actions()
                        ],
                        "blocked_actions": blocked_actions,
                        "current_cell_exit_history": current_history,
                        "current_cell_exit_counts": current_counts,
                        "last_action": last_action,
                        "immediate_reverse_action": OPPOSITE_ACTION.get(
                            last_action or ""
                        ),
                        "recent_actions": recent_actions[-8:],
                        "recent_positions": [
                            list(item) for item in recent_positions[-8:]
                        ],
                        "preferred_open_actions": preferred_open_actions(
                            open_actions,
                            current_history_full,
                            last_action,
                        ),
                    }
                if edward_options:
                    if state_context is None:
                        state_context = {}
                    if fallback_mode != "refuse":
                        state_context["edward_fallback_mode"] = fallback_mode
                    state_context.update(
                        {
                            "episode_life_mode": config.episode_life_mode,
                            "ghosts": list(live_snapshot.get("ghosts") or []),
                            "maze_size": [
                                int(live_snapshot["height"]),
                                int(live_snapshot["width"]),
                            ],
                            "edible_ticks": int(
                                live_snapshot.get("edible_ticks", 0)
                            ),
                            "planner_candidates": [
                                candidate.as_dict()
                                for candidate in option_candidates
                            ],
                            "option_code_map": (
                                {
                                    objective_constraint.code_for_option(
                                        candidate.option_id
                                    ): candidate.option_id
                                    for candidate in option_candidates
                                }
                                if objective_constraint is not None
                                else {}
                            ),
                            "active_option": (
                                active_option.as_dict()
                                if active_option is not None
                                else None
                            ),
                        }
                    )
                if ascii_observation:
                    from .prompts import render_ascii_map_for
                    state_context['ascii_map'] = render_ascii_map_for(image_prompt_style, planner.level, live_snapshot)
                    messages = [dict(role='system', content=system_prompt),
                                dict(role='user', content=[dict(type='text', text='')])]
                else:
                    messages = build_image_messages(png, prompt_style=image_prompt_style, state_context=state_context)
                if edward_options:
                    messages[0]["content"] = system_prompt
                model_user_instruction = "".join(
                    item["text"] for item in messages[1]["content"]
                    if item.get("type") == "text")
                if objective_constraint is not None:
                    model_user_instruction = _compact_edward_decision_prompt(
                        state_context, option_candidates, objective_constraint)
                    if ascii_observation:
                        from .prompts import ascii_decision_prompt_for, render_ascii_map_for
                        model_user_instruction = ascii_decision_prompt_for(
                            image_prompt_style, state_context, option_candidates, objective_constraint,
                            render_ascii_map_for(image_prompt_style, planner.level, live_snapshot), fallback_mode=fallback_mode)
                        state_context["ascii_map"] = render_ascii_map_for(image_prompt_style, planner.level, live_snapshot)
                if ascii_observation:
                    messages[1]["content"][0]["text"] = model_user_instruction
                else:
                    messages[1]["content"] = layout_image_user_content(
                        png, model_user_instruction, prompt_style=image_prompt_style,
                        edward_options=objective_constraint is not None)
                model_called = not (
                    edward_options and active_option is not None
                )
                if not model_called:
                    model_user_instruction = None
                sent_prompt = {
                    "requested_model_id": options.get("model"),
                    "checkpoint_manifest_sha256": options.get("checkpoint_manifest_sha256"),
                    "model_system_prompt": system_prompt if model_called else None,
                    "model_user_prompt_sha256": (
                        text_sha256(model_user_instruction) if model_called else None
                    ),
                    "sent_prompt_sha256": (
                        (text_sent_prompt_sha256(system_prompt, model_user_instruction) if ascii_observation else sent_prompt_sha256(system_prompt, model_user_instruction, png_sha256(png)))
                        if model_called else None
                    ),
                }
                if image_count(messages) != (0 if ascii_observation else 1):
                    raise RuntimeError("model request observation modality differs")
                try:
                    if edward_options and active_option is not None:
                        if active_turn is None or active_action is None:
                            raise RuntimeError(
                                "active Edward option lost its model turn or action"
                            )
                        turn = active_turn
                        action = Action(active_action)
                        option_step = active_option_step + 1
                    elif edward_options:
                        if objective_constraint is None:
                            raise RuntimeError(
                                "new Edward decision has no token constraint"
                            )
                        if boundary_expected is not None or self.decision_boundary_sink is not None:
                            verify = _boundary_verify(
                                png, option_candidates, objective_constraint,
                                system_prompt, model_user_instruction,
                            )
                            if boundary_expected is not None:
                                differing = sorted(
                                    key for key in verify if verify[key] != boundary_expected.get(key)
                                )
                                if differing or set(boundary_expected) != set(verify):
                                    raise ValueError(
                                        f"restored decision boundary differs: {differing}"
                                    )
                                boundary_expected = None
                            if self.decision_boundary_sink is not None:
                                self.decision_boundary_sink(_decision_boundary(
                                    env, planner, cell_exit_history, recent_actions,
                                    recent_positions, no_progress_steps, verify,
                                    live_snapshot, previous_info,
                                ))
                        if scripted_objectives:
                            scripted_option = str(scripted_objectives.pop(0))
                            completion = objective_constraint.code_for_option(
                                scripted_option
                            )
                            completion_id = (
                                str(scripted_ids.pop(0))
                                if scripted_ids
                                else None
                            )
                            turn = ModelTurn(
                                completion, completion_id, messages
                            )
                        elif options.get("scripted_objectives") is not None:
                            raise RuntimeError(
                                "scripted objective sequence ended before the episode"
                            )
                        else:
                            turn = await self._call_model(
                                messages,
                                **options,
                                current_open_actions=current_open_actions,
                                objective_constraint=objective_constraint,
                            )
                        option_id = objective_constraint.option_for_completion(
                            turn.completion
                        )
                        active_option = next(
                            candidate
                            for candidate in option_candidates
                            if candidate.option_id == option_id
                        )
                        active_objective_constraint = objective_constraint
                        active_turn = turn
                        active_action = active_option.first_action
                        active_remaining = active_option.commit_moves
                        active_option_step = 0
                        active_reward = 0.0
                        action = Action(active_action)
                        option_step = 1
                    else:
                        if scripted:
                            completion = str(scripted.pop(0))
                            completion_id = (
                                str(scripted_ids.pop(0))
                                if scripted_ids
                                else None
                            )
                            turn = ModelTurn(
                                completion, completion_id, messages
                            )
                        else:
                            turn = await self._call_model(
                                messages,
                                **options,
                                current_open_actions=current_open_actions,
                            )
                        action = parse_action(turn.completion)
                        if options.get("open_action_mask") and action.value not in current_open_actions:
                            raise ActionParseError("direction is not in the advertised open-action mask")
                        option_step = None
                except (ActionParseError, ObjectiveParseError) as exc:
                    if turn is None:
                        raise RuntimeError(
                            "model generation violated the objective token contract"
                        ) from exc
                    parse_failures += 1
                    canonical_violations += 1
                    reward_accumulated_before_violation = sum(
                        float(step["shaped_reward"]) for step in trajectory
                    )
                    contract_violation_adjustment = (
                        contract_violation_return
                        - reward_accumulated_before_violation
                    )
                    record = {
                        "step": len(trajectory) + 1,
                        "env_step": int(previous_info["step"]),
                        "completion_id": turn.completion_id,
                        "completion": turn.completion,
                        "reasoning_content": turn.reasoning_content,
                        "raw_model_response": turn.raw_response,
                        "request_extra_body": turn.request_extra_body,
                        "model_called": model_called,
                        "model_user_instruction": model_user_instruction,
                        **sent_prompt,
                        "observation_context": state_context,
                        "open_action_mask": current_open_actions,
                        "option_id": None,
                        "option_code": None,
                        "option_code_map": {},
                        "option_strategy": None,
                        "option_target": None,
                        "option_step": None,
                        "option_end": True,
                        "option_status": "parse_failed",
                        "option_invalidated": False,
                        "option_return": contract_violation_adjustment,
                        "action": None,
                        "parse_failed": True,
                        "contract_violation": True,
                        "contract_violation_type": "parse_failure",
                        "contract_violation_target_return": (
                            contract_violation_return
                        ),
                        "reward_accumulated_before_violation": (
                            reward_accumulated_before_violation
                        ),
                        "contract_violation_adjustment": (
                            contract_violation_adjustment
                        ),
                        "base_reward": 0.0,
                        "base_reward_contribution": 0.0,
                        "reward_recipe_version": reward_config.recipe_version,
                        "event_game_score_delta": 0,
                        "event_count": 0,
                        "normal_pellet_eaten": False,
                        "normal_pellet_reward": 0.0,
                        "power_pellet_eaten": False,
                        "power_pellet_reward": 0.0,
                        "ghost_eaten": False,
                        "ghost_reward": 0.0,
                        "fruit_eaten": False,
                        "fruit_reward": 0.0,
                        "death": False,
                        "death_penalty": 0.0,
                        "death_count": int(previous_info.get("death_count", 0)),
                        # No env step happened: this record starts and ends with the lives the
                        # previous step ended with (its "lives" is the count before that step).
                        "lives": int(previous_info.get("lives_after_step", 0)),
                        "lives_after_step": int(
                            previous_info.get("lives_after_step", 0)
                        ),
                        "respawned": False,
                        "level_completed": False,
                        "completion_reward": 0.0,
                        "safety_refusal": False,
                        "safety_refusal_penalty": 0.0,
                        "step_penalty": 0.0,
                        "wall_penalty": 0.0,
                        "normal_pellet_remaining_ratio": (
                            int(previous_info["normal_pellets_remaining"])
                            / reward_normal_pellets_initial
                        ),
                        "nearest_pellet_shaping_active": False,
                        "nearest_pellet_distance_before": None,
                        "nearest_pellet_distance_after": None,
                        "nearest_pellet_progress_weight": 0.0,
                        "nearest_pellet_progress_reward": 0.0,
                        "shaped_reward": contract_violation_adjustment,
                        "pellet_clear_rate": float(previous_info["pellet_clear_rate"]),
                        "pellets_remaining": int(previous_info["pellets_remaining"]),
                        "normal_pellets_remaining": int(
                            previous_info["normal_pellets_remaining"]
                        ),
                        "normal_pellets_eaten": (
                            initial_normal_pellets
                            - int(previous_info["normal_pellets_remaining"])
                        ),
                        "normal_pellet_clear_rate": (
                            initial_normal_pellets
                            - int(previous_info["normal_pellets_remaining"])
                        )
                        / initial_normal_pellets,
                        "power_pellets_remaining": int(
                            previous_info["power_pellets_remaining"]
                        ),
                        "ghosts": list(previous_info.get("ghosts", [])),
                        "edible_ticks": int(
                            previous_info.get("edible_ticks", 0)
                        ),
                        "events": [],
                        "logic_frame_events": [],
                        "score_components": {
                            "normal_pellet": 0,
                            "power_pellet": 0,
                            "ghost": 0,
                            "fruit": 0,
                            "total": 0,
                        },
                        "logic_frames": 0,
                        "atomic_substeps": [],
                        "score": int(previous_info["score"]),
                        "pygame_mode": int(previous_info["pygame_mode"]),
                        "wall_collision": False,
                        "oscillation_return": False,
                        "terminated": True,
                        "truncated": False,
                        "terminal_reason": "parse_failed",
                        "observation_png_sha256": None if ascii_observation else png_sha256(png),
                        **({"observation_mode": "ascii", "observation_text_sha256": text_sha256(state_context["ascii_map"]), "observation_ascii_map": state_context["ascii_map"]} if ascii_observation else {}),
                    }
                    trajectory.append(record)
                    if turn.completion_id:
                        rewards_by_completion[turn.completion_id] = (
                            contract_violation_adjustment
                        )
                    break

                distance_before = None
                if level is not None and remaining_normal_pellets is not None:
                    # A step can consume at most one normal pellet.  Avoid
                    # consulting the hidden BFS topology while its reward term
                    # cannot activate; in skip-on-eat mode even a threshold-
                    # crossing pellet step does not need distances.
                    earliest_ratio_after_step = (
                        len(remaining_normal_pellets)
                        if reward_config.nearest_pellet_skip_on_eat
                        else max(0, len(remaining_normal_pellets) - 1)
                    ) / reward_normal_pellets_initial
                    if (
                        earliest_ratio_after_step
                        <= reward_config.nearest_pellet_remaining_ratio_threshold
                    ):
                        before_row, before_col = previous_info[
                            "pacman_position"
                        ]
                        distance_before = (
                            _nearest_reachable_distance_with_diagnostics(
                                level,
                                Position(int(before_row), int(before_col)),
                                remaining_normal_pellets,
                                phase="before_step",
                                action=action.value,
                                live_legal_actions=current_open_actions,
                                allow_unreachable=True,
                            )
                        )
                source_position = (
                    int(previous_info["pacman_position"][0]),
                    int(previous_info["pacman_position"][1]),
                )
                if (
                    options.get("open_action_mask")
                    and action.value not in current_open_actions
                ):
                    raise RuntimeError(
                        "model backend violated open action mask: "
                        f"{action.value} not in {current_open_actions}"
                    )
                previous_action = (
                    recent_actions[-1] if recent_actions else None
                )
                next_image, base_reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                if planner is not None:
                    planner.record_action(action.value)
                score_progress = int(info["score"]) != int(previous_info["score"])
                pellet_progress = int(info["normal_pellets_remaining"]) != int(
                    previous_info["normal_pellets_remaining"]
                )
                no_progress_steps = (
                    0 if score_progress or pellet_progress else no_progress_steps + 1
                )
                safety_reason = None
                if not (terminated or truncated):
                    if (
                        wall_clock_limit_seconds is not None
                        and time.monotonic() - episode_started_at
                        >= wall_clock_limit_seconds
                    ):
                        safety_reason = "safety_timeout"
                    elif (
                        stuck_no_progress_steps is not None
                        and no_progress_steps >= stuck_no_progress_steps
                    ):
                        safety_reason = "stuck"
                elif (
                    truncated
                    and int(info["step"]) >= config.max_steps
                    and config.max_steps == 2000
                ):
                    safety_reason = "safety_step_limit"
                if safety_reason is not None:
                    info["terminal_reason"] = safety_reason
                    terminated = False
                    truncated = True
                if single_step and not (terminated or truncated):
                    info["terminal_reason"] = "single_step_complete"
                    terminated = True
                next_position = (
                    int(info["pacman_position"][0]),
                    int(info["pacman_position"][1]),
                )
                oscillation_return = (
                    len(recent_positions) >= 2
                    and next_position == recent_positions[-2]
                    and action.value
                    == OPPOSITE_ACTION.get(previous_action or "")
                )
                normal_pellet_remaining_ratio = (
                    int(info["normal_pellets_remaining"])
                    / reward_normal_pellets_initial
                )
                normal_pellet_remaining_ratio_before = (
                    int(previous_info["normal_pellets_remaining"])
                    / reward_normal_pellets_initial
                )
                distance_after = None
                if level is not None and remaining_normal_pellets is not None:
                    after_row, after_col = info["pacman_position"]
                    after_position = Position(int(after_row), int(after_col))
                    eaten_position = _normal_pellet_event_position(info)
                    if eaten_position is not None:
                        if eaten_position not in remaining_normal_pellets:
                            raise RuntimeError(
                                "live game ate a normal pellet outside the tracker"
                            )
                        remaining_normal_pellets.remove(eaten_position)
                    if len(remaining_normal_pellets) != int(
                        info["normal_pellets_remaining"]
                    ):
                        raise RuntimeError(
                            "nearest-pellet tracker diverged from the live game"
                        )
                    if (
                        distance_before is not None
                        and normal_pellet_remaining_ratio
                        <= reward_config.nearest_pellet_remaining_ratio_threshold
                        and not (
                            reward_config.nearest_pellet_skip_on_eat
                            and bool(info["pellet_eaten"])
                        )
                    ):
                        distance_after = (
                            _nearest_reachable_distance_with_diagnostics(
                                level,
                                after_position,
                                remaining_normal_pellets,
                                phase="after_step",
                                previous_position=source_position,
                                action=action.value,
                                live_legal_actions=list(info["legal_actions"]),
                                allow_unreachable=True,
                            )
                        )
                        if distance_after is None:
                            info["terminal_reason"] = "unreachable_normal_pellets"
                            terminated = False
                            truncated = True
                reward = shape_reward(
                    base_reward,
                    previous_info,
                    info,
                    reward_config,
                    normal_pellet_remaining_ratio=(
                        normal_pellet_remaining_ratio
                    ),
                    normal_pellet_remaining_ratio_before=(
                        normal_pellet_remaining_ratio_before
                    ),
                    nearest_pellet_distance_before=distance_before,
                    nearest_pellet_distance_after=distance_after,
                )
                record_option = active_option if edward_options else None
                option_end = False
                option_status: str | None = None
                option_invalidated = False
                completed_option_reward: float | None = None
                if edward_options:
                    if record_option is None or planner is None:
                        raise RuntimeError(
                            "Edward action has no active planner option"
                        )
                    active_reward += reward.shaped_reward
                    active_remaining -= 1
                    if terminated or truncated:
                        option_status = "terminal"
                    elif record_option.strategy == "RISK_FALLBACK":
                        option_status = "max_commit"
                    else:
                        next_action, option_status = planner.continue_option(
                            record_option,
                            env.snapshot(),
                        )
                        if option_status == "active" and active_remaining <= 0:
                            option_status = "max_commit"
                        elif option_status == "active":
                            if next_action is None:
                                raise RuntimeError(
                                    "active Edward option has no next action"
                                )
                            active_action = next_action
                    option_end = option_status != "active"
                    option_invalidated = option_status == "invalidated"
                    info["option_invalidated"] = option_invalidated
                    if option_end:
                        completed_option_reward = active_reward
                record = {
                    "step": len(trajectory) + 1,
                    "env_step": int(info["step"]),
                    "completion_id": turn.completion_id,
                    "completion": turn.completion,
                    "reasoning_content": turn.reasoning_content,
                    "raw_model_response": turn.raw_response,
                    "request_extra_body": turn.request_extra_body,
                    "model_called": model_called,
                    "model_user_instruction": model_user_instruction,
                    **sent_prompt,
                    "observation_context": state_context,
                    "open_action_mask": current_open_actions,
                    "option_id": (
                        record_option.option_id
                        if record_option is not None
                        else None
                    ),
                    "option_code": (
                        active_objective_constraint.code_for_option(
                            record_option.option_id
                        )
                        if active_objective_constraint is not None
                        and record_option is not None
                        else None
                    ),
                    "option_code_map": (
                        {
                            code: option_id
                            for option_id, code in zip(
                                active_objective_constraint.option_ids,
                                active_objective_constraint.rendered_choices,
                                strict=True,
                            )
                        }
                        if active_objective_constraint is not None
                        else {}
                    ),
                    "option_strategy": (
                        record_option.strategy
                        if record_option is not None
                        else None
                    ),
                    "option_target": (
                        list(record_option.target)
                        if record_option is not None
                        else None
                    ),
                    "option_step": option_step,
                    "option_end": option_end,
                    "option_status": option_status,
                    "option_invalidated": option_invalidated,
                    "option_return": completed_option_reward,
                    "action": action.value,
                    "parse_failed": False,
                    **reward.as_dict(),
                    "pellet_clear_rate": float(info["pellet_clear_rate"]),
                    "pellets_remaining": int(info["pellets_remaining"]),
                    "normal_pellets_remaining": int(
                        info["normal_pellets_remaining"]
                    ),
                    "normal_pellets_eaten": (
                        initial_normal_pellets
                        - int(info["normal_pellets_remaining"])
                    ),
                    "normal_pellet_clear_rate": (
                        initial_normal_pellets
                        - int(info["normal_pellets_remaining"])
                    )
                    / initial_normal_pellets,
                    "power_pellets_remaining": int(info["power_pellets_remaining"]),
                    "ghosts": list(info.get("ghosts", [])),
                    "edible_ticks": int(info.get("edible_ticks", 0)),
                    "events": list(info.get("events", [])),
                    "logic_frame_events": list(
                        info.get("logic_frame_events", [])
                    ),
                    "score_components": dict(info.get("score_components", {})),
                    "logic_frames": int(info.get("logic_frames", 0)),
                    "atomic_substeps": list(info.get("atomic_substeps", [])),
                    "score": int(info["score"]),
                    "death_count": int(info.get("death_count", 0)),
                    "lives": int(info.get("lives", 0)),
                    "lives_after_step": int(info.get("lives_after_step", 0)),
                    "respawned": bool(info.get("respawned", False)),
                    "pygame_mode": int(info["pygame_mode"]),
                    "wall_collision": bool(info["wall_collision"]),
                    "oscillation_return": oscillation_return,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_reason": info["terminal_reason"],
                    "observation_png_sha256": None if ascii_observation else png_sha256(png),
                        **({"observation_mode": "ascii", "observation_text_sha256": text_sha256(state_context["ascii_map"]), "observation_ascii_map": state_context["ascii_map"]} if ascii_observation else {}),
                }
                trajectory.append(record)
                if edward_options:
                    if option_end:
                        if turn.completion_id:
                            if completed_option_reward is None:
                                raise RuntimeError(
                                    "completed Edward option lost its reward"
                                )
                            rewards_by_completion[
                                turn.completion_id
                            ] = completed_option_reward
                        active_option = None
                        active_objective_constraint = None
                        active_turn = None
                        active_action = None
                        active_remaining = 0
                        active_option_step = 0
                        active_reward = 0.0
                    else:
                        active_option_step = int(option_step or 0)
                elif turn.completion_id:
                    rewards_by_completion[
                        turn.completion_id
                    ] = reward.shaped_reward
                if action.value in MOVEMENT_ACTIONS:
                    cell_exit_history.setdefault(source_position, []).append(
                        action.value
                    )
                recent_actions.append(action.value)
                recent_positions.append(next_position)
                image, previous_info, final_info = next_image, info, info
                if terminated or truncated:
                    break
                if (
                    not edward_options
                    and not scripted
                    and options.get("scripted_actions") is not None
                ):
                    raise RuntimeError("scripted action sequence ended before the episode")

            total_base = sum(float(step["base_reward"]) for step in trajectory)
            total_shaped = sum(float(step["shaped_reward"]) for step in trajectory)
            terminal_reason = str(trajectory[-1]["terminal_reason"])
            if terminal_reason == "parse_failed":
                if abs(total_shaped - contract_violation_return) > 1e-9:
                    raise RuntimeError(
                        "parse-failure adjustment did not produce the "
                        "fail-closed episode return"
                    )
                total_shaped = contract_violation_return
            payload = {
                "id": str(data["id"]),
                "trajectory_sample_id": trajectory_sample_id,
                "model": options.get("model"),
                "checkpoint_manifest_sha256": options.get("checkpoint_manifest_sha256"),
                "split": str(data["split"]),
                "env_api_version": env.spec.api_version,
                "env_id": env.spec.env_id,
                "backend": "original-pygame",
                "ruleset_revision": env.spec.ruleset_revision,
                "pacman_python_revision": env.pacman_python_revision,
                "pacman_python_source_sha256": env.provenance[
                    "pacman_python_source_sha256"
                ],
                "maapacman_revision": env.provenance["maapacman_commit"],
                "maapacman_env_source_sha256": env.provenance[
                    "maapacman_env_source_sha256"
                ],
                "dataset_contract_version": data["dataset_contract_version"],
                **({"training_backend": data["training_backend"]} if "training_backend" in data else {}),
                "source_revisions": data["source_revisions"],
                "reward_recipe_version": reward_config.recipe_version,
                "level_revision": env.spec.level_revision,
                "renderer_revision": env.spec.renderer_revision,
                "level": config.level,
                "seed": seed,
                "max_steps": config.max_steps,
                "ghost_mode": config.ghost_mode,
                "episode_life_mode": config.episode_life_mode,
                "state_prefix_actions": state_prefix_actions,
                "state_prefix_actions_executed": len(state_prefix_evidence),
                "state_prefix_evidence": state_prefix_evidence,
                "prefix_end_score": prefix_end_score,
                "prefix_end_logic_frame": prefix_end_logic_frame,
                "restart_state": restart_metadata,
                "reward_normal_pellets_initial": reward_normal_pellets_initial,
                "decision_steps": 1 if single_step else None,
                "image_prompt_style": image_prompt_style,
                **prompt_metadata,
                "recipe_contract": recipe_contract,
                "system_prompt": system_prompt,
                "user_instruction": user_instruction,
                "observation_contract": (
                    "ascii_plus_live_state_and_navigation_history" if ascii_observation else
                    "screenshot_plus_live_state_and_navigation_history"
                    if image_prompt_style == "live_state_v3"
                    else "screenshot_only"
                ),
                "action_constraint": (
                    EDWARD_OPTION_CONSTRAINT
                    if edward_options
                    else (
                        "dynamic_open_movement_actions"
                        if options.get("open_action_mask")
                        else list(EXPECTED_ACTIONS)
                    )
                ),
                "decoding": {
                    "temperature": float(options.get("temperature", 0.0)),
                    "top_p": float(options.get("top_p", 1.0)),
                    "max_completion_tokens": (
                        1
                        if edward_options or options.get("open_action_mask")
                        else int(options.get("max_completion_tokens", 3))
                    ),
                    "enable_thinking": False,
                    "generation_seed": options.get("generation_seed"),
                    "open_action_mask": bool(
                        options.get("open_action_mask", False)
                    ),
                    "edward_options": edward_options,
                    **({"edward_fallback_mode": fallback_mode}
                       if fallback_mode != "refuse" else {}),
                },
                "reward_objective_contract": str(
                    options.get("reward_objective_contract", "legacy")
                ),
                "step_penalty_coefficient": reward_config.step_penalty,
                "step_penalty_cleared_ratio_scale": (
                    reward_config.step_penalty_cleared_ratio_scale
                ),
                "wall_penalty_coefficient": reward_config.wall_penalty,
                "use_base_reward": reward_config.use_base_reward,
                "normal_pellet_reward_coefficient": (
                    reward_config.normal_pellet_reward
                ),
                "power_pellet_reward_coefficient": (
                    reward_config.power_pellet_reward
                ),
                "ghost_reward_coefficient": reward_config.ghost_reward,
                "fruit_reward_coefficient": reward_config.fruit_reward,
                "death_penalty_coefficient": reward_config.death_penalty,
                "completion_reward_coefficient": (
                    reward_config.completion_reward
                ),
                "safety_refusal_penalty_coefficient": (
                    reward_config.safety_refusal_penalty
                ),
                "contract_violation_return": contract_violation_return,
                "nearest_pellet_alpha": reward_config.nearest_pellet_alpha,
                "nearest_pellet_remaining_ratio_threshold": (
                    reward_config.nearest_pellet_remaining_ratio_threshold
                ),
                "nearest_pellet_scale_by_cleared_ratio": (
                    reward_config.nearest_pellet_scale_by_cleared_ratio
                ),
                "nearest_pellet_skip_on_eat": (
                    reward_config.nearest_pellet_skip_on_eat
                ),
                "nearest_pellet_topology_revision": (
                    level.revision if level is not None else None
                ),
                "total_base_reward": total_base,
                "suffix_score_delta": int(final_info["score"]) - prefix_end_score,
                "suffix_death_count": int(final_info.get("death_count", 0)) - (
                    restart_metadata["death_count"] if restart_metadata is not None else 0
                ),
                "suffix_normal_pellets_eaten": initial_normal_pellets - int(
                    final_info["normal_pellets_remaining"]
                ),
                "total_shaped_reward": total_shaped,
                "steps": len(trajectory),
                "pellet_clear_rate": float(final_info["pellet_clear_rate"]),
                "normal_pellets_initial": initial_normal_pellets,
                "normal_pellets_eaten": (
                    initial_normal_pellets
                    - int(final_info["normal_pellets_remaining"])
                ),
                "normal_pellet_clear_rate": (
                    initial_normal_pellets
                    - int(final_info["normal_pellets_remaining"])
                )
                / initial_normal_pellets,
                "normal_pellets_remaining": int(
                    final_info["normal_pellets_remaining"]
                ),
                "power_pellets_remaining": int(
                    final_info["power_pellets_remaining"]
                ),
                "final_score": int(final_info["score"]),
                # From the final record: a parse failure adds a record without an env step.
                "death_count": int((trajectory[-1] if trajectory else final_info).get("death_count", 0)),
                "lives": int((trajectory[-1] if trajectory else final_info).get("lives", 0)),
                "lives_after_step": int(
                    (trajectory[-1] if trajectory else final_info).get("lives_after_step", 0)
                ),
                "pygame_mode": int(final_info["pygame_mode"]),
                "wall_collisions": sum(
                    bool(step["wall_collision"]) for step in trajectory
                ),
                "oscillation_returns": sum(
                    bool(step["oscillation_return"]) for step in trajectory
                ),
                "parse_failures": parse_failures,
                "canonical_action_violations": canonical_violations,
                "won": (
                    terminal_reason == "all_normal_pellets"
                    and int(final_info["normal_pellets_remaining"]) == 0
                ),
                "terminal_reason": terminal_reason,
                "terminated": bool(trajectory[-1]["terminated"]),
                "truncated": bool(trajectory[-1]["truncated"]),
                "trajectory": trajectory,
            }
            try:
                audit_trajectory(payload)
            except TrajectoryAuditError:
                raise
            except ValueError as exc:
                raise TrajectoryAuditError(str(exc), payload) from exc
            self._episode_payload.set(payload)
            self.last_episode = payload
            self._record_metrics(payload)
            trajectory_dir = options.get("trajectory_dir") or os.getenv(
                "PACMAN_TRAJECTORY_DIR"
            )
            if trajectory_dir:
                write_trajectory(payload, Path(trajectory_dir))

        if rewards_by_completion:
            return rewards_by_completion
        return float(payload["total_shaped_reward"])

    def _record_metrics(self, payload: Mapping[str, Any]) -> None:
        """Adapters may consume completed, audited episode metrics."""

    async def _call_model(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> ModelTurn:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "install the openai extra or provide scripted_actions"
            ) from exc
        http_client = options.get("http_client")
        client = AsyncOpenAI(
            base_url=options.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=options.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            http_client=http_client,
            max_retries=0,
        )
        objective_constraint = options.get("objective_constraint")
        if objective_constraint is not None and not isinstance(
            objective_constraint, ObjectiveTokenConstraint
        ):
            raise TypeError("objective_constraint has the wrong type")
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if objective_constraint is not None:
            extra_body["allowed_token_ids"] = (
                objective_constraint.allowed_token_ids
            )
        else:
            constrained_choices = (
                list(options["current_open_actions"])
                if options.get("open_action_mask")
                else list(EXPECTED_ACTIONS)
            )
            extra_body["structured_outputs"] = {"choice": constrained_choices}
        if objective_constraint is None and options.get("open_action_mask"):
            current_open_actions = list(options["current_open_actions"])
            extra_body["allowed_token_ids"] = [
                self.action_token_id_by_action[action]
                for action in current_open_actions
            ]
        request = {
            "model": options.get("model", "default"),
            "messages": messages,
            "temperature": float(options.get("temperature", 0.0)),
            "top_p": float(options.get("top_p", 1.0)),
            "max_tokens": (
                objective_constraint.max_new_tokens
                if objective_constraint is not None
                else (
                    1
                    if options.get("open_action_mask")
                    else int(options.get("max_completion_tokens", 3))
                )
            ),
            "extra_body": extra_body,
        }
        if options.get("generation_seed") is not None:
            request["seed"] = int(options["generation_seed"])
        try:
            response = await client.chat.completions.create(**request)
        finally:
            # AsyncOpenAI creates an httpx connection pool when no client is
            # supplied.  A Pacman episode makes hundreds of calls, so leaving
            # each short-lived pool open eventually exhausts local TCP
            # connections.  Externally supplied clients remain caller-owned.
            if http_client is None:
                await client.close()
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        raw_response = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else None
        )
        return ModelTurn(
            completion=message.content or "",
            completion_id=getattr(response, "id", None),
            messages=messages,
            reasoning_content=reasoning_content,
            raw_response=raw_response,
            request_extra_body=extra_body,
        )
