"""Auditable trajectory validation, persistence, and summary metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from pacman_env.env.ghost_modes import validate_ghost_mode, validate_ghost_state
from pacman_env.env.pygame_environment import ruleset_revision
from pacman_env.planner import validate_fallback_mode

from .level1_dataset import (
    DATASET_CONTRACT_VERSION,
    ENV_API_VERSION,
    ENV_NAME,
    REPOSITORY_NAMES,
    SUPPORTED_MAX_STEPS,
)
from .rewards import REWARD_RECIPE_VERSION, audit_reward
from .prompts import (
    live_state_instruction, prompt_contract_metadata, prompt_text,
    sent_prompt_sha256, text_sha256,
    edward_system_prompt, render_edward_decision_prompt,
)
from .token_constraints import EDWARD_OPTION_CONSTRAINT, OPTION_CODE_BY_ID


REQUIRED_ENV_FIELDS = {
    "ghost_mode",
    "env_api_version",
    "env_id",
    "backend",
    "ruleset_revision",
    "pacman_python_revision",
    "pacman_python_source_sha256",
    "maapacman_revision",
    "maapacman_env_source_sha256",
    "dataset_contract_version",
    "source_revisions",
    "reward_recipe_version",
    "level_revision",
    "renderer_revision",
    "seed",
    "max_steps",
    "state_prefix_actions",
    "state_prefix_actions_executed",
    "state_prefix_evidence",
    "prefix_end_score",
    "prefix_end_logic_frame",
    "normal_pellets_initial",
    "normal_pellets_eaten",
    "normal_pellet_clear_rate",
    "final_score",
    "pygame_mode",
    "terminated",
    "truncated",
    "terminal_reason",
    "action_constraint",
}
REQUIRED_STEP_FIELDS = {
    "step",
    "completion",
    "action",
    "parse_failed",
    "base_reward",
    "base_reward_contribution",
    "reward_recipe_version",
    "event_game_score_delta",
    "event_count",
    "normal_pellet_eaten",
    "normal_pellet_reward",
    "power_pellet_eaten",
    "power_pellet_reward",
    "ghost_eaten",
    "ghost_reward",
    "fruit_eaten",
    "fruit_reward",
    "death",
    "death_penalty",
    "level_completed",
    "completion_reward",
    "step_penalty",
    "wall_penalty",
    "normal_pellet_remaining_ratio",
    "nearest_pellet_shaping_active",
    "nearest_pellet_distance_before",
    "nearest_pellet_distance_after",
    "nearest_pellet_progress_weight",
    "nearest_pellet_progress_reward",
    "shaped_reward",
    "pellet_clear_rate",
    "pellets_remaining",
    "normal_pellets_remaining",
    "normal_pellets_eaten",
    "normal_pellet_clear_rate",
    "power_pellets_remaining",
    "ghosts",
    "edible_ticks",
    "events",
    "logic_frame_events",
    "score_components",
    "logic_frames",
    "atomic_substeps",
    "score",
    "pygame_mode",
    "oscillation_return",
    "terminated",
    "truncated",
    "terminal_reason",
    "observation_png_sha256",
}

_SCORE_COMPONENT_NAMES = (
    "normal_pellet",
    "power_pellet",
    "ghost",
    "fruit",
    "other",
    "total",
)
_REQUIRED_ATOMIC_FIELDS = {
    "frame",
    "logic_frame_index",
    "pacman_position",
    "pacman_facing",
    "level",
    "mode",
    "mode_name",
    "score",
    "score_delta",
    "score_components",
    "lives",
    "width",
    "height",
    "normal_pellets_remaining",
    "power_pellets_remaining",
    "collectibles_remaining",
    "edible_ticks",
    "ghosts",
    "fruit",
    "ghost_door",
    "blocked",
    "open",
    "events",
}
_REQUIRED_EVENT_FIELDS = {
    "logic_frame_index",
    "logic_frame",
    "event_type",
    "pacman_position",
    "ghost_id",
    "score_delta",
    "post_ghost_state",
    "edible_ticks",
}


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _audit_prefix_evidence(payload: Mapping[str, Any]) -> tuple[int, int]:
    actions = payload.get("state_prefix_actions")
    evidence = payload.get("state_prefix_evidence")
    if not isinstance(actions, list) or any(
        action not in {"U", "D", "L", "R"} for action in actions
    ):
        raise ValueError("state_prefix_actions must contain only movement actions")
    if not isinstance(evidence, list):
        raise ValueError("state_prefix_evidence must be a list")
    executed = _integer(
        payload.get("state_prefix_actions_executed"),
        "state_prefix_actions_executed",
    )
    if executed != len(actions) or executed != len(evidence):
        raise ValueError("prefix action count does not match its evidence")

    restart = payload.get("restart_state")
    if restart is not None:
        if not isinstance(restart, Mapping) or actions or evidence or executed:
            raise ValueError("restart baseline cannot contain prefix actions")
        required_restart = {"id", "path", "sha256", "identity", "source_step", "score",
                            "logic_frame", "death_count", "normal_pellets", "power_pellets",
                            "remaining_budget"}
        if not required_restart.issubset(restart):
            raise ValueError("restart baseline is incomplete")
        if any(not isinstance(restart[name], str) or not restart[name] for name in ("id", "path")):
            raise ValueError("restart identity/path is invalid")
        digest = restart["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("restart SHA-256 is invalid")
        for name in required_restart - {"id", "path", "sha256", "identity"}:
            if _integer(restart[name], f"restart {name}") < 0:
                raise ValueError("restart counters must be nonnegative")
        if not isinstance(restart["identity"], Mapping):
            raise ValueError("restart source identity is invalid")
        if restart["source_step"] + restart["remaining_budget"] != int(payload["max_steps"]):
            raise ValueError("restart horizon does not reconcile")
        if restart["remaining_budget"] <= 0 or restart["normal_pellets"] <= 0:
            raise ValueError("restart baseline is finished")
        if restart["normal_pellets"] != int(payload["normal_pellets_initial"]):
            raise ValueError("restart pellet baseline does not reconcile")
        if (restart["score"] != payload["prefix_end_score"]
                or restart["logic_frame"] != payload["prefix_end_logic_frame"]):
            raise ValueError("restart score/frame baseline does not reconcile")
        return restart["score"], restart["logic_frame"]

    score = 0
    logic_frame = 0
    required = {
        "action",
        "env_step",
        "score",
        "score_delta",
        "logic_frame",
        "logic_frames",
        "wall_collision",
        "terminated",
        "truncated",
    }
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise ValueError("state prefix evidence is incomplete")
        if item["action"] != actions[index - 1]:
            raise ValueError("state prefix evidence action order mismatch")
        if _integer(item["env_step"], "prefix env_step") != index:
            raise ValueError("state prefix env steps are not contiguous")
        logic_frames = _integer(item["logic_frames"], "prefix logic_frames")
        if not 1 <= logic_frames <= 16:
            raise ValueError("prefix action must contain 1 to 16 logic frames")
        next_logic_frame = _integer(item["logic_frame"], "prefix logic_frame")
        if next_logic_frame != logic_frame + logic_frames:
            raise ValueError("state prefix logic frames are not contiguous")
        logic_frame = next_logic_frame
        score_delta = _integer(item["score_delta"], "prefix score_delta")
        next_score = _integer(item["score"], "prefix score")
        if next_score != score + score_delta:
            raise ValueError("state prefix score delta does not reconcile")
        score = next_score
        if item["wall_collision"] is not False:
            raise ValueError("state prefix evidence contains a wall collision")
        if item["terminated"] is not False or item["truncated"] is not False:
            raise ValueError("state prefix evidence reached a terminal state")

    prefix_end_score = _integer(payload.get("prefix_end_score"), "prefix_end_score")
    prefix_end_logic_frame = _integer(
        payload.get("prefix_end_logic_frame"),
        "prefix_end_logic_frame",
    )
    if prefix_end_score != score:
        raise ValueError("prefix_end_score does not match prefix evidence")
    if prefix_end_logic_frame != logic_frame:
        raise ValueError("prefix_end_logic_frame does not match prefix evidence")
    return prefix_end_score, prefix_end_logic_frame


def audit_step_environment_evidence(
    step: Mapping[str, Any],
    *,
    previous_score: int,
    previous_logic_frame: int,
    ghost_mode: str = "normal",
    episode_life_mode: str = "single_death",
) -> tuple[int, int]:
    """Reconcile one recorded action with every API-v3 atomic frame."""

    logic_frames = _integer(step.get("logic_frames"), "logic_frames")
    atomic_substeps = step.get("atomic_substeps")
    if not isinstance(atomic_substeps, list):
        raise ValueError("atomic_substeps must be a list")
    if logic_frames != len(atomic_substeps):
        raise ValueError("atomic_substeps length does not match logic_frames")
    max_logic_frames = (
        256
        if episode_life_mode == "original_three_lives"
        and bool(step.get("death"))
        else 16
    )
    if not 1 <= logic_frames <= max_logic_frames:
        raise ValueError(
            f"executed action must contain 1 to {max_logic_frames} logic frames"
        )

    component_totals = {name: 0 for name in _SCORE_COMPONENT_NAMES}
    flattened_events: list[dict[str, Any]] = []
    score = previous_score
    logic_frame = previous_logic_frame
    death_source_frame: int | None = None
    for index, raw_substep in enumerate(atomic_substeps, start=1):
        if not isinstance(raw_substep, Mapping):
            raise ValueError("atomic substep must be an object")
        missing = _REQUIRED_ATOMIC_FIELDS - raw_substep.keys()
        if missing:
            raise ValueError(
                f"atomic substep is missing fields: {sorted(missing)}"
            )
        if _integer(
            raw_substep["logic_frame_index"], "logic_frame_index"
        ) != index:
            raise ValueError("logic-frame indexes are not contiguous")
        current_logic_frame = _integer(raw_substep["frame"], "frame")
        if death_source_frame is not None:
            # pacman-python advances GAME_LOGIC_FRAME only while mode 1 runs.
            # The original death and READY animations render inside the same
            # env.step without executing gameplay, so every post-death atomic
            # presentation frame must retain the death event's source frame.
            if current_logic_frame != death_source_frame:
                raise ValueError(
                    "source logic frame must remain frozen after death event"
                )
        elif current_logic_frame != logic_frame + 1:
            raise ValueError("source logic frames are not globally contiguous")
        logic_frame = current_logic_frame
        validate_ghost_state(raw_substep, ghost_mode)

        frame_score = _integer(raw_substep["score"], "atomic score")
        frame_delta = _integer(
            raw_substep["score_delta"], "atomic score_delta"
        )
        if frame_score - score != frame_delta:
            raise ValueError("atomic frame score delta does not reconcile")
        score = frame_score
        components = raw_substep["score_components"]
        if not isinstance(components, Mapping) or set(components) != set(
            _SCORE_COMPONENT_NAMES
        ):
            raise ValueError("atomic score_components schema mismatch")
        parsed_components = {
            name: _integer(components[name], f"score_components.{name}")
            for name in _SCORE_COMPONENT_NAMES
        }
        if parsed_components["total"] != frame_delta or parsed_components[
            "total"
        ] != sum(
            parsed_components[name] for name in _SCORE_COMPONENT_NAMES[:-1]
        ):
            raise ValueError("atomic score_components do not add up")
        for name, value in parsed_components.items():
            component_totals[name] += value

        frame_events = raw_substep["events"]
        if not isinstance(frame_events, list):
            raise ValueError("atomic events must be a list")
        event_score = 0
        event_components = {
            "normal_pellet": 0,
            "power_pellet": 0,
            "ghost": 0,
            "fruit": 0,
            "other": 0,
        }
        event_component_by_type = {
            "normal_pellet_eaten": "normal_pellet",
            "power_pellet_eaten": "power_pellet",
            "ghost_eaten": "ghost",
            "fruit_eaten": "fruit",
            "death": "other",
            "level_cleared": "other",
        }
        for raw_event in frame_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("logic-frame event must be an object")
            event_missing = _REQUIRED_EVENT_FIELDS - raw_event.keys()
            if event_missing:
                raise ValueError(
                    f"logic-frame event is missing fields: {sorted(event_missing)}"
                )
            if _integer(
                raw_event["logic_frame_index"], "event logic_frame_index"
            ) != index:
                raise ValueError("event logic-frame index mismatch")
            if _integer(raw_event["logic_frame"], "event logic_frame") != logic_frame:
                raise ValueError("event source logic frame mismatch")
            source_position = raw_event["pacman_position"]
            if not isinstance(source_position, list) or len(source_position) != 2:
                raise ValueError("event source Pacman position is invalid")
            source_row = _integer(
                source_position[0], "event source Pacman row"
            )
            source_col = _integer(
                source_position[1], "event source Pacman column"
            )
            if not (
                0 <= source_row < _integer(raw_substep["height"], "atomic height")
                and 0
                <= source_col
                < _integer(raw_substep["width"], "atomic width")
            ):
                raise ValueError("event source Pacman position is outside the level")
            atomic_position = list(raw_substep["pacman_position"])
            if source_position != atomic_position:
                raise ValueError(
                    "event Pacman position mismatch: "
                    f"event_type={raw_event['event_type']!r}, "
                    f"logic_frame={raw_event['logic_frame']}, "
                    f"event_position={source_position!r}, "
                    f"atomic_position={atomic_position!r}"
                )
            if _integer(raw_event["edible_ticks"], "event edible_ticks") < 0:
                raise ValueError("event edible timer is invalid")
            event_delta = _integer(raw_event["score_delta"], "event score_delta")
            event_type = raw_event.get("event_type")
            if event_type not in event_component_by_type:
                raise ValueError("logic-frame event type is not part of API v3")
            event_score += event_delta
            event_components[event_component_by_type[str(event_type)]] += event_delta
            flattened_events.append(dict(raw_event))
            if event_type == "death":
                if death_source_frame is not None:
                    raise ValueError("step contains more than one death event")
                death_source_frame = logic_frame
        if event_score != frame_delta:
            raise ValueError("atomic events do not reconcile frame score")
        if any(
            event_components[name] != parsed_components[name]
            for name in event_components
        ):
            raise ValueError("atomic events do not reconcile score_components")

    if score != _integer(step.get("score"), "step score"):
        raise ValueError("step score does not match final atomic frame")
    recorded_components = step.get("score_components")
    if not isinstance(recorded_components, Mapping) or set(
        recorded_components
    ) != set(_SCORE_COMPONENT_NAMES):
        raise ValueError("step score_components schema mismatch")
    if {
        name: _integer(recorded_components[name], f"score_components.{name}")
        for name in _SCORE_COMPONENT_NAMES
    } != component_totals:
        raise ValueError("step score_components do not match atomic frames")
    base_reward = float(step["base_reward"])
    if not math.isfinite(base_reward) or not base_reward.is_integer():
        raise ValueError("step base_reward must be a finite integer")
    if component_totals["total"] != base_reward:
        raise ValueError("step score_components do not reconcile base_reward")
    if step.get("logic_frame_events") != flattened_events:
        raise ValueError("logic_frame_events do not match atomic event order")
    ordered_event_types = list(
        dict.fromkeys(event["event_type"] for event in flattened_events)
    )
    if step.get("events") != ordered_event_types:
        raise ValueError("step events do not match the source event ledger")
    final = atomic_substeps[-1]
    if list(step.get("ghosts", [])) != list(final["ghosts"]):
        raise ValueError("step ghosts do not match final atomic frame")
    if _integer(step.get("edible_ticks"), "step edible_ticks") != _integer(
        final["edible_ticks"], "atomic edible_ticks"
    ):
        raise ValueError("step edible_ticks does not match final atomic frame")
    if _integer(step.get("pygame_mode"), "pygame_mode") != _integer(
        final["mode"], "atomic mode"
    ):
        raise ValueError("step pygame_mode does not match final atomic frame")
    remaining_fields = {
        "pellets_remaining": "collectibles_remaining",
        "normal_pellets_remaining": "normal_pellets_remaining",
        "power_pellets_remaining": "power_pellets_remaining",
    }
    for step_field, atomic_field in remaining_fields.items():
        if _integer(step.get(step_field), step_field) != _integer(
            final[atomic_field], atomic_field
        ):
            raise ValueError(f"step {step_field} does not match final atomic frame")

    death = bool(step.get("death"))
    respawned = bool(step.get("respawned"))
    terminal_reason = step.get("terminal_reason")
    if respawned:
        if (
            not death
            or step.get("terminated")
            or step.get("truncated")
            or terminal_reason is not None
            or int(step["pygame_mode"]) != 1
        ):
            raise ValueError("death respawn state is inconsistent")
    elif death:
        if (
            not step.get("terminated")
            or step.get("truncated")
            or terminal_reason not in {"death", "game_over"}
            or int(step["pygame_mode"]) not in {2, 3}
        ):
            raise ValueError("death event and terminal state are inconsistent")
    elif terminal_reason in {"death", "game_over"}:
        raise ValueError("death terminal reason requires a death event")
    completed = bool(step.get("level_completed"))
    if completed:
        if (
            not step.get("terminated")
            or step.get("truncated")
            or step.get("terminal_reason") != "all_normal_pellets"
            or int(step["pygame_mode"]) not in {6, 9}
        ):
            raise ValueError("level-cleared event and terminal state are inconsistent")
    return score, logic_frame


def _audit_parse_failure_evidence(
    step: Mapping[str, Any],
    *,
    previous_score: int,
    accumulated_shaped_reward: float,
    expected_target_return: float,
    ghost_mode: str,
) -> None:
    validate_ghost_state(
        {"ghost_mode": ghost_mode, "ghosts": step.get("ghosts"),
         "edible_ticks": step.get("edible_ticks"), "events": step.get("events")},
        ghost_mode,
    )
    score_components = step.get("score_components")
    zero_fields = (
        "base_reward",
        "base_reward_contribution",
        "event_game_score_delta",
        "event_count",
        "normal_pellet_reward",
        "power_pellet_reward",
        "ghost_reward",
        "fruit_reward",
        "death_penalty",
        "completion_reward",
        "step_penalty",
        "wall_penalty",
        "nearest_pellet_progress_weight",
        "nearest_pellet_progress_reward",
    )
    false_fields = (
        "normal_pellet_eaten",
        "power_pellet_eaten",
        "ghost_eaten",
        "fruit_eaten",
        "death",
        "level_completed",
        "nearest_pellet_shaping_active",
    )
    target_return = float(step.get("contract_violation_target_return", math.nan))
    recorded_accumulated = float(
        step.get("reward_accumulated_before_violation", math.nan)
    )
    adjustment = float(step.get("contract_violation_adjustment", math.nan))
    shaped_reward = float(step["shaped_reward"])
    option_return = step.get("option_return")
    expected_adjustment = expected_target_return - accumulated_shaped_reward
    if (
        step["action"] is not None
        or not step["terminated"]
        or step["truncated"]
        or step["terminal_reason"] != "parse_failed"
        or step["reward_recipe_version"] != REWARD_RECIPE_VERSION
        or step["logic_frame_events"]
        or step["events"]
        or int(step["logic_frames"]) != 0
        or step["atomic_substeps"]
        or not isinstance(score_components, Mapping)
        or any(int(value) != 0 for value in score_components.values())
        or any(float(step[field]) != 0.0 for field in zero_fields)
        or any(bool(step[field]) for field in false_fields)
        or step["nearest_pellet_distance_before"] is not None
        or step["nearest_pellet_distance_after"] is not None
        or int(step["score"]) != previous_score
        or step.get("contract_violation") is not True
        or step.get("contract_violation_type") != "parse_failure"
        or not math.isfinite(target_return)
        or abs(target_return - expected_target_return) > 1e-9
        or not math.isfinite(recorded_accumulated)
        or abs(recorded_accumulated - accumulated_shaped_reward) > 1e-9
        or not math.isfinite(adjustment)
        or abs(adjustment - expected_adjustment) > 1e-9
        or not math.isfinite(shaped_reward)
        or abs(shaped_reward - adjustment) > 1e-9
        or (
            option_return is not None
            and (
                not math.isfinite(float(option_return))
                or abs(float(option_return) - adjustment) > 1e-9
            )
        )
    ):
        raise ValueError("parse-failure step has invalid fail-closed contract evidence")


def _audit_risk_fallback_step(payload: Mapping[str, Any], step: Mapping[str, Any]) -> None:
    context = step.get("observation_context") or {}
    candidates = context.get("planner_candidates") or []
    risky = [candidate for candidate in candidates if candidate.get("strategy") == "RISK_FALLBACK"]
    selected_risk = step.get("option_strategy") == "RISK_FALLBACK"
    if not risky and not selected_risk:
        return
    mode = (payload.get("decoding") or {}).get("edward_fallback_mode", "refuse")
    if mode != "risk_ranked" or not payload.get("prompt_template_sha256"):
        raise ValueError("risk fallback requires opt-in mode and prompt evidence")
    if (
        len(risky) != len(candidates) or not 1 <= len(risky) <= 4
        or not selected_risk or not step.get("model_called")
        or step.get("option_step") != 1 or not step.get("option_end")
        or step.get("option_status") not in {"max_commit", "terminal"}
    ):
        raise ValueError("risk fallback must be an unmixed single-step model decision")
    actions = [candidate.get("first_action") for candidate in risky]
    if len(set(actions)) != len(actions) or set(actions) != set(step.get("open_action_mask") or []):
        raise ValueError("risk fallback must advertise every open direction exactly once")
    for rank, candidate in enumerate(risky, 1):
        risk = candidate.get("risk")
        if (
            candidate.get("option_id") != f"A{rank - 1}"
            or candidate.get("commit_moves") != 1 or candidate.get("route_distance") != 1
            or candidate.get("safety_margin") is not None
            or candidate.get("future_safe_exits") is not None
            or not isinstance(risk, Mapping) or risk.get("rank") != rank
            or not {"rank", "motion", "ghost_clearance", "route_margin",
                    "safe_next_cells", "dead_end", "reverse"}.issubset(risk)
            or risk.get("motion") not in {"clear_estimate", "unknown", "collision_predicted"}
            or type(risk.get("safe_next_cells")) is not int
            or not 0 <= risk["safe_next_cells"] <= 4
            or type(risk.get("dead_end")) is not bool
            or type(risk.get("reverse")) is not bool
            or any(risk.get(key) is not None and type(risk[key]) is not int
                   for key in ("ghost_clearance", "route_margin"))
        ):
            raise ValueError("invalid risk fallback candidate evidence")
        if candidate["option_id"] == step.get("option_id") and (
            candidate["first_action"] != step.get("action")
            or list(candidate["target"]) != list(step.get("option_target") or [])
        ):
            raise ValueError("risk fallback executed action/target differs from candidate")


def _audit_prompt_evidence(payload: Mapping[str, Any]) -> None:
    # Older archived payloads did not record prompt fingerprints.
    if "prompt_template_sha256" not in payload:
        return
    decoding = payload.get("decoding") or {}
    edward = bool(decoding.get("edward_options"))
    style = payload["image_prompt_style"]
    fallback_mode = validate_fallback_mode(
        decoding.get("edward_fallback_mode", "refuse"), edward_options=edward
    )
    # Archived v1 evidence must be checked against its original renderer/hash.
    from . import legacy_prompts_v1, prompts

    prompt_module = (
        legacy_prompts_v1
        if edward and payload.get("prompt_version") == "edward-option-code-v1"
        else prompts
    )
    expected = prompt_module.prompt_contract_metadata(
        style, edward_options=edward, fallback_mode=fallback_mode
    )
    if not edward and not decoding.get("open_action_mask"):
        expected["action_protocol"] = "direct-action-token-v1"
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("trajectory prompt template does not match actual harness")
    system = prompt_module.edward_system_prompt(fallback_mode) if edward else prompt_text(style)[0]
    if payload.get("system_prompt") != system:
        raise ValueError("trajectory system prompt does not match harness")
    for step in payload.get("trajectory", []):
        fields = (
            "model_system_prompt",
            "model_user_prompt_sha256",
            "sent_prompt_sha256",
        )
        if not step.get("model_called"):
            if (
                any(step.get(field) is not None for field in fields)
                or step.get("model_user_instruction") is not None
            ):
                raise ValueError(
                    "non-model environment step cannot claim a sent prompt"
                )
            continue
        user = step.get("model_user_instruction")
        if not isinstance(user, str) or step.get("model_system_prompt") != system:
            raise ValueError("trajectory missing actual model prompt text")
        if step.get("model_user_prompt_sha256") != text_sha256(user) or step.get(
            "sent_prompt_sha256"
        ) != sent_prompt_sha256(system, user, step["observation_png_sha256"]):
            raise ValueError("trajectory actual model prompt hash mismatch")
        context = step.get("observation_context")
        if style == "live_state_v3":
            validate_ghost_state(
                {**context, "ghost_mode": payload["ghost_mode"], "events": []},
                payload["ghost_mode"],
            )
        if edward:
            if context.get("edward_fallback_mode", "refuse") != fallback_mode:
                raise ValueError("trajectory fallback mode disagrees with observation context")
            code_map = context.get("option_code_map") or {}
            inverse = {option: code for code, option in code_map.items()}
            candidates = [
                SimpleNamespace(**candidate)
                for candidate in context["planner_candidates"]
            ]
            constraint = SimpleNamespace(
                code_for_option=inverse.__getitem__,
                rendered_choices=tuple(
                    inverse[candidate.option_id] for candidate in candidates
                ),
            )
            actual = prompt_module.render_edward_decision_prompt(
                context, candidates, constraint, fallback_mode=fallback_mode
            )
        else:
            actual = (
                live_state_instruction(context)
                if style == "live_state_v3"
                else prompt_text(style)[1]
            )
        if user != actual:
            raise ValueError("actual model prompt disagrees with observation context")
        if (
            not edward
            and decoding.get("open_action_mask")
            and not step.get("parse_failed")
        ):
            if step.get("action") not in step.get("open_action_mask", []):
                raise ValueError("direct action violates the open-action mask")


class TrajectoryAuditError(ValueError):
    """An episode failed its trajectory audit; ``payload`` is the rejected episode for diagnosis."""

    def __init__(self, message: str, payload: dict):
        super().__init__(message)
        self.payload = payload


def audit_trajectory(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_ENV_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"trajectory missing environment fields: {sorted(missing)}")
    if payload["env_api_version"] != ENV_API_VERSION:
        raise ValueError(f"trajectory env_api_version must be {ENV_API_VERSION!r}")
    if payload["env_id"] != ENV_NAME:
        raise ValueError("trajectory has the wrong environment ID")
    if payload["backend"] != "original-pygame":
        raise ValueError("trajectory backend must be original-pygame")
    neutral = payload["dataset_contract_version"] == "pacman-episode-v1"
    if not neutral and payload["dataset_contract_version"] != DATASET_CONTRACT_VERSION:
        raise ValueError("trajectory has the wrong dataset contract")
    source_revisions = payload["source_revisions"]
    expected_repositories = set(REPOSITORY_NAMES)
    if neutral:
        training_backend = payload.get("training_backend")
        if training_backend not in {"slime", "areal"}:
            raise ValueError("neutral trajectory requires its actual training backend")
        expected_repositories = {"pacman-recipe", "pacman-python", training_backend}
    if not isinstance(source_revisions, Mapping) or set(source_revisions) != (
        expected_repositories
    ):
        raise ValueError("trajectory source revisions are incomplete")
    for name, revision in source_revisions.items():
        if not isinstance(revision, Mapping):
            raise ValueError(f"trajectory {name} revision must be an object")
        commit = revision.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not isinstance(revision.get("dirty"), bool)
        ):
            raise ValueError(f"trajectory {name} revision is invalid")
    if (
        payload["maapacman_revision"]
        != source_revisions["pacman-recipe" if neutral else "areal-pacman"]["commit"]
    ):
        raise ValueError(
            "trajectory maapacman revision must match bundled areal-pacman"
        )
    pacman_revision = source_revisions["pacman-python"]["commit"]
    if payload["pacman_python_revision"] != pacman_revision:
        raise ValueError("trajectory pacman-python revision must match source revisions")
    if payload["renderer_revision"] != f"pacman-python:{pacman_revision}":
        raise ValueError("trajectory renderer revision must match pacman-python")
    if payload["reward_recipe_version"] != REWARD_RECIPE_VERSION:
        raise ValueError("trajectory has the wrong reward recipe")
    has_safety_refusal_contract = (
        "safety_refusal_penalty_coefficient" in payload
    )
    safety_refusal_coefficient = float(
        payload.get("safety_refusal_penalty_coefficient", 0.0)
    )
    if (
        not math.isfinite(safety_refusal_coefficient)
        or safety_refusal_coefficient < 0
    ):
        raise ValueError(
            "trajectory safety_refusal_penalty_coefficient is invalid"
        )
    contract_violation_return = float(
        payload.get("contract_violation_return", -1.0)
    )
    if (
        not math.isfinite(contract_violation_return)
        or abs(contract_violation_return + 1.0) > 1e-9
    ):
        raise ValueError(
            "trajectory contract_violation_return must be exactly -1.0"
        )
    for digest_field in (
        "ruleset_revision",
        "pacman_python_source_sha256",
        "maapacman_env_source_sha256",
        "level_revision",
    ):
        digest = payload.get(digest_field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"trajectory has invalid {digest_field}")
    mode = validate_ghost_mode(payload["ghost_mode"])
    has_life_contract = "episode_life_mode" in payload
    life_mode = payload.get("episode_life_mode", "single_death")
    if life_mode not in {"single_death", "original_three_lives"}:
        raise ValueError("trajectory episode_life_mode is invalid")
    if payload["ruleset_revision"] != ruleset_revision(mode):
        raise ValueError("trajectory ruleset_revision does not match ghost_mode")
    if payload["max_steps"] not in SUPPORTED_MAX_STEPS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_MAX_STEPS))
        raise ValueError(f"trajectory max_steps must be one of: {supported}")
    if payload.get("action_constraint") == EDWARD_OPTION_CONSTRAINT:
        decoding = payload.get("decoding")
        max_completion_tokens = (
            decoding.get("max_completion_tokens")
            if isinstance(decoding, Mapping)
            else None
        )
        if (
            not isinstance(max_completion_tokens, int)
            or isinstance(max_completion_tokens, bool)
            or max_completion_tokens != 1
        ):
            raise ValueError(
                "Edward trajectory requires decoding.max_completion_tokens=1"
            )
    steps = payload.get("trajectory")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain at least one step")
    shaped_total = 0.0
    base_total = 0.0
    previous_score, previous_logic_frame = _audit_prefix_evidence(payload)
    active_option_key: tuple[Any, ...] | None = None
    active_option_step = 0
    audited_parse_failures = 0
    restart = payload.get("restart_state")
    audited_deaths = int(restart["death_count"]) if restart is not None else 0
    previous_env_step = int(restart["source_step"]) if restart is not None else 0
    for index, step in enumerate(steps, 1):
        if not isinstance(step, Mapping):
            raise ValueError(f"trajectory step {index} must be an object")
        step_missing = REQUIRED_STEP_FIELDS - step.keys()
        if step_missing:
            raise ValueError(
                f"trajectory step {index} missing fields: {sorted(step_missing)}"
            )
        if has_life_contract:
            life_missing = {
                "death_count", "lives", "lives_after_step", "respawned"
            } - step.keys()
            if life_missing:
                raise ValueError(
                    f"trajectory step {index} missing life fields: "
                    f"{sorted(life_missing)}"
                )
        if has_safety_refusal_contract:
            safety_fields = {
                "safety_refusal",
                "safety_refusal_penalty",
            }
            safety_missing = safety_fields - step.keys()
            if safety_missing:
                raise ValueError(
                    f"trajectory step {index} missing safety-refusal fields: "
                    f"{sorted(safety_missing)}"
                )
            safety_refusal = bool(step["safety_refusal"])
            expected_safety_refusal = (
                step.get("terminal_reason") == "safety_refusal"
            )
            safety_penalty = float(step["safety_refusal_penalty"])
            expected_safety_penalty = (
                safety_refusal_coefficient if expected_safety_refusal else 0.0
            )
            if (
                safety_refusal != expected_safety_refusal
                or not math.isfinite(safety_penalty)
                or abs(safety_penalty - expected_safety_penalty) > 1e-9
            ):
                raise ValueError(
                    f"trajectory step {index} has inconsistent "
                    "safety-refusal reward evidence"
                )
        parse_failed = bool(step.get("parse_failed"))
        if restart is not None:
            expected_env_step = previous_env_step + int(not parse_failed)
            if int(step.get("env_step", -1)) != expected_env_step:
                raise ValueError("restart suffix env steps are not contiguous")
            previous_env_step = expected_env_step
        if parse_failed:
            audited_parse_failures += 1
            _audit_parse_failure_evidence(
                step,
                previous_score=previous_score,
                accumulated_shaped_reward=shaped_total,
                expected_target_return=contract_violation_return,
                ghost_mode=mode,
            )
        else:
            if step["action"] not in {"U", "D", "L", "R", "S"}:
                raise ValueError(f"trajectory step {index} has invalid action")
            audit_reward(step)
            previous_score, previous_logic_frame = audit_step_environment_evidence(
                step,
                previous_score=previous_score,
                previous_logic_frame=previous_logic_frame,
                ghost_mode=mode,
                episode_life_mode=life_mode,
            )
        death = bool(step.get("death"))
        audited_deaths += int(death)
        if has_life_contract:
            if int(step["death_count"]) != audited_deaths:
                raise ValueError("trajectory death_count does not reconcile")
            lives = int(step["lives"])
            lives_after = int(step["lives_after_step"])
            if lives < 0 or lives_after < 0:
                raise ValueError("trajectory lives cannot be negative")
            expected_lives_after = max(0, lives - int(death))
            if lives_after != expected_lives_after:
                raise ValueError(
                    "trajectory lives do not reconcile with death event "
                    f"(step {index}: lives={lives} lives_after_step={lives_after} death={death} "
                    f"events={step.get('events')} mode_terminal={step.get('terminal_reason')})"
                )
        if payload.get("action_constraint") == EDWARD_OPTION_CONSTRAINT:
            option_missing = {
                "model_called",
                "option_id",
                "option_code",
                "option_code_map",
                "option_strategy",
                "option_target",
                "option_step",
                "option_end",
                "option_status",
                "option_invalidated",
                "option_return",
            } - step.keys()
            if option_missing:
                raise ValueError(
                    f"trajectory step {index} missing Edward fields: "
                    f"{sorted(option_missing)}"
                )
            option_id = step.get("option_id")
            if not parse_failed and (
                not isinstance(option_id, str)
                or len(option_id) < 2
                or option_id[0] not in "CAE"
                or not option_id[1:].isdigit()
            ):
                raise ValueError(
                    f"trajectory step {index} has invalid Edward option"
                )
            if not parse_failed and int(step.get("option_step", 0)) < 1:
                raise ValueError(
                    f"trajectory step {index} has invalid Edward option step"
                )
            if not parse_failed:
                option_code = step.get("option_code")
                option_code_map = step.get("option_code_map")
                normalized_code_map = (
                    dict(option_code_map)
                    if isinstance(option_code_map, Mapping)
                    else {}
                )
                code_map_types_valid = all(
                    isinstance(code, str)
                    and len(code) == 1
                    and isinstance(option, str)
                    for code, option in normalized_code_map.items()
                )
                canonical_code_map = (
                    {
                        OPTION_CODE_BY_ID.get(option): option
                        for option in normalized_code_map.values()
                    }
                    if code_map_types_valid
                    else {}
                )
                observation_context = step.get("observation_context")
                structured_code_map = (
                    observation_context.get("option_code_map")
                    if isinstance(observation_context, Mapping)
                    else None
                )
                if (
                    not isinstance(option_code, str)
                    or len(option_code) != 1
                    or not isinstance(option_code_map, Mapping)
                    or not normalized_code_map
                    or not code_map_types_valid
                    or normalized_code_map.get(option_code) != option_id
                    or step.get("completion") != option_code
                    or len(set(normalized_code_map.values()))
                    != len(normalized_code_map)
                    or None in canonical_code_map
                    or normalized_code_map != canonical_code_map
                    or (
                        step.get("model_called")
                        and structured_code_map != normalized_code_map
                    )
                ):
                    raise ValueError(
                        f"trajectory step {index} has invalid Edward option code"
                    )
                if step.get("model_called"):
                    planner_candidates = observation_context.get(
                        "planner_candidates"
                    )
                    if (
                        not isinstance(planner_candidates, list)
                        or any(
                            not isinstance(candidate, Mapping)
                            or not isinstance(candidate.get("option_id"), str)
                            for candidate in planner_candidates
                        )
                    ):
                        raise ValueError(
                            f"trajectory step {index} has invalid planner candidates"
                        )
                    candidate_option_ids = [
                        candidate["option_id"] for candidate in planner_candidates
                    ]
                    if (
                        len(candidate_option_ids)
                        != len(set(candidate_option_ids))
                        or set(candidate_option_ids)
                        != set(normalized_code_map.values())
                        or len(candidate_option_ids) != len(normalized_code_map)
                    ):
                        raise ValueError(
                            f"trajectory step {index} option-code map does not "
                            "match planner candidates"
                        )
            if not parse_failed:
                _audit_risk_fallback_step(payload, step)
            if bool(step.get("option_invalidated")) != (
                step.get("option_status") == "invalidated"
            ):
                raise ValueError(
                    f"trajectory step {index} has inconsistent option invalidation"
                )
            status = step.get("option_status")
            if status not in {
                "active",
                "completed",
                "invalidated",
                "max_commit",
                "terminal",
                "parse_failed",
            }:
                raise ValueError(
                    f"trajectory step {index} has invalid option status"
                )
            option_end = bool(step.get("option_end"))
            if option_end != (status != "active"):
                raise ValueError(
                    f"trajectory step {index} has inconsistent option end"
                )
            option_return = step.get("option_return")
            if option_end != (option_return is not None):
                raise ValueError(
                    f"trajectory step {index} has inconsistent option return"
                )
            if parse_failed:
                if status != "parse_failed" or not step.get("model_called"):
                    raise ValueError(
                        f"trajectory step {index} has invalid parse-failure option"
                    )
                active_option_key = None
                active_option_step = 0
            else:
                option_key = (
                    step.get("completion_id"),
                    step.get("completion"),
                    option_id,
                    option_code,
                    tuple(sorted(normalized_code_map.items())),
                    step.get("option_strategy"),
                    tuple(step.get("option_target") or ()),
                )
                option_step = int(step["option_step"])
                if option_step == 1:
                    if active_option_key is not None or not step.get(
                        "model_called"
                    ):
                        raise ValueError(
                            f"trajectory step {index} has invalid option start"
                        )
                elif (
                    step.get("model_called")
                    or option_key != active_option_key
                    or option_step != active_option_step + 1
                ):
                    raise ValueError(
                        f"trajectory step {index} breaks option continuity"
                    )
                if option_end:
                    active_option_key = None
                    active_option_step = 0
                else:
                    active_option_key = option_key
                    active_option_step = option_step
        digest = step["observation_png_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"trajectory step {index} has invalid PNG hash")
        if int(step["step"]) != index:
            raise ValueError("trajectory steps must be contiguous and one-based")
        if index < len(steps) and (step["terminated"] or step["truncated"]):
            raise ValueError("trajectory contains steps after a terminal state")
        shaped_total += float(step["shaped_reward"])
        base_total += float(step["base_reward"])
    if abs(shaped_total - float(payload["total_shaped_reward"])) > 1e-9:
        raise ValueError("total_shaped_reward does not match trajectory steps")
    if abs(base_total - float(payload["total_base_reward"])) > 1e-9:
        raise ValueError("total_base_reward does not match trajectory steps")
    if audited_parse_failures != int(payload.get("parse_failures", 0)):
        raise ValueError("trajectory parse-failure count does not reconcile")
    if has_life_contract and audited_deaths != int(payload["death_count"]):
        raise ValueError("trajectory payload death_count does not reconcile")
    final = steps[-1]
    if restart is not None:
        suffix_score = int(final["score"]) - int(restart["score"])
        suffix_deaths = audited_deaths - int(restart["death_count"])
        suffix_pellets = int(restart["normal_pellets"]) - int(final["normal_pellets_remaining"])
        if (payload.get("suffix_score_delta") != suffix_score
                or payload.get("suffix_death_count") != suffix_deaths
                or payload.get("suffix_normal_pellets_eaten") != suffix_pellets
                or abs(base_total - suffix_score) > 1e-9):
            raise ValueError("restart suffix metrics do not reconcile")
    if int(payload.get("steps", -1)) != len(steps):
        raise ValueError("trajectory payload steps does not match trajectory length")
    if not (final["terminated"] or final["truncated"]):
        raise ValueError("trajectory final step is not terminal")
    if active_option_key is not None:
        raise ValueError("trajectory ended with an unfinished Edward option")
    if final.get("terminal_reason") == "parse_failed":
        if (
            not final.get("parse_failed")
            or int(payload.get("parse_failures", 0)) != 1
            or "contract_violation_return" not in payload
            or abs(shaped_total - contract_violation_return) > 1e-9
        ):
            raise ValueError(
                "parse-failure trajectory must terminate at total return -1.0"
            )
    if final.get("terminal_reason") == "safety_refusal":
        if (
            payload.get("action_constraint") != EDWARD_OPTION_CONSTRAINT
            or final.get("terminated")
            or not final.get("truncated")
            or final.get("death")
            or final.get("level_completed")
            or (
                has_safety_refusal_contract
                and not final.get("safety_refusal")
            )
            or not final.get("option_end")
            or final.get("option_status")
            not in {"completed", "invalidated", "max_commit"}
        ):
            raise ValueError("trajectory has an invalid Edward safety refusal")
    final_fields = ["terminated", "truncated", "terminal_reason", "pygame_mode"]
    if has_life_contract:
        final_fields.extend(["death_count", "lives", "lives_after_step"])
    for field in final_fields:
        if payload[field] != final[field]:
            raise ValueError(f"trajectory payload {field} does not match final step")
    if int(payload["final_score"]) != int(final["score"]):
        raise ValueError("trajectory final_score does not match final step")
    if int(payload["normal_pellets_eaten"]) != int(
        final["normal_pellets_eaten"]
    ):
        raise ValueError("trajectory normal-pellet total does not match final step")
    expected_normal_eaten = int(payload["normal_pellets_initial"]) - int(
        final["normal_pellets_remaining"]
    )
    if expected_normal_eaten != int(final["normal_pellets_eaten"]):
        raise ValueError("trajectory normal-pellet counts do not reconcile")
    _audit_prompt_evidence(payload)


def write_trajectory(payload: Mapping[str, Any], directory: Path) -> Path:
    audit_trajectory(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(payload["id"])
    )
    sample_id = payload.get("trajectory_sample_id")
    if not sample_id:
        raise ValueError("trajectory_sample_id is required for collision-free persistence")
    safe_sample_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(sample_id)
    )
    output = directory / f"{safe_id}--sample-{safe_sample_id}.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return output


def _max_no_progress_streak(episode: Mapping[str, Any]) -> int:
    longest = 0
    current = 0
    for step in episode.get("trajectory", []):
        if float(step.get("base_reward", 0.0)) > 0.0:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def summarize_episodes(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")
    parse_failures = sum(int(item.get("parse_failures", 0)) for item in episodes)
    canonical_violations = sum(
        int(item.get("canonical_action_violations", 0)) for item in episodes
    )
    clear_rates = [float(item["pellet_clear_rate"]) for item in episodes]
    normal_clear_rates = [
        float(item["normal_pellet_clear_rate"]) for item in episodes
    ]
    action_counts = {action: 0 for action in ("U", "D", "L", "R", "S")}
    reasoning_turns = 0
    no_progress_streaks = []
    for item in episodes:
        no_progress_streaks.append(_max_no_progress_streak(item))
        for step in item.get("trajectory", []):
            action = step.get("action")
            if action in action_counts:
                action_counts[str(action)] += 1
            if str(step.get("reasoning_content") or "").strip():
                reasoning_turns += 1
    return {
        "episodes": len(episodes),
        "average_pellet_clear_rate": sum(clear_rates) / len(clear_rates),
        "average_normal_pellet_clear_rate": (
            sum(normal_clear_rates) / len(normal_clear_rates)
        ),
        "average_normal_pellets_eaten": sum(
            int(item["normal_pellets_eaten"]) for item in episodes
        )
        / len(episodes),
        "full_completions": sum(bool(item.get("won")) for item in episodes),
        "parse_failures": parse_failures,
        "canonical_action_violations": canonical_violations,
        "average_episode_length": sum(int(item["steps"]) for item in episodes)
        / len(episodes),
        "average_base_reward": sum(
            float(item["total_base_reward"]) for item in episodes
        )
        / len(episodes),
        "average_shaped_reward": sum(
            float(item["total_shaped_reward"]) for item in episodes
        )
        / len(episodes),
        "average_final_score": sum(int(item["final_score"]) for item in episodes)
        / len(episodes),
        "average_wall_collisions": sum(
            int(item.get("wall_collisions", 0)) for item in episodes
        )
        / len(episodes),
        "average_wall_hit_rate": sum(
            int(item.get("wall_collisions", 0)) / max(int(item["steps"]), 1)
            for item in episodes
        )
        / len(episodes),
        "average_oscillation_returns": sum(
            int(item.get("oscillation_returns", 0)) for item in episodes
        )
        / len(episodes),
        "average_oscillation_rate": sum(
            int(item.get("oscillation_returns", 0)) / max(int(item["steps"]), 1)
            for item in episodes
        )
        / len(episodes),
        "average_normal_pellets_remaining": sum(
            int(item["normal_pellets_remaining"]) for item in episodes
        )
        / len(episodes),
        "average_power_pellets_remaining": sum(
            int(item["power_pellets_remaining"]) for item in episodes
        )
        / len(episodes),
        "average_death_count": sum(
            int(item.get("death_count", 0)) for item in episodes
        )
        / len(episodes),
        "max_step_rate": sum(
            item.get("terminal_reason") == "max_steps" for item in episodes
        )
        / len(episodes),
        "action_counts": action_counts,
        "reasoning_turns": reasoning_turns,
        "max_no_progress_streak": max(no_progress_streaks),
        "average_max_no_progress_streak": sum(no_progress_streaks)
        / len(no_progress_streaks),
    }
