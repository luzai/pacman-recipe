"""Auditable Level-1 reward shaping over explicit task events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

REWARD_RECIPE_VERSION = "maapacman-level1-event-reward-v3"
_SCORE_BY_EVENT = {
    "normal_pellet_eaten": {10},
    "power_pellet_eaten": {100},
    "ghost_eaten": {200, 400, 800, 1600},
    "fruit_eaten": {2500},
    "death": {0},
    "level_cleared": {0},
}


@dataclass(frozen=True)
class RewardConfig:
    recipe_version: str = REWARD_RECIPE_VERSION
    step_penalty: float = 1.0
    step_penalty_cleared_ratio_scale: float = 0.0
    wall_penalty: float = 1.0
    use_base_reward: bool = True
    normal_pellet_reward: float = 0.0
    power_pellet_reward: float = 0.0
    ghost_reward: float = 0.0
    fruit_reward: float = 0.0
    death_penalty: float = 0.0
    completion_reward: float = 0.0
    safety_refusal_penalty: float = 0.0
    nearest_pellet_alpha: float = 0.0
    nearest_pellet_remaining_ratio_threshold: float = 1.0
    nearest_pellet_scale_by_cleared_ratio: bool = False
    nearest_pellet_skip_on_eat: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite; infinity is only a reward-clip bound")
        if self.recipe_version != REWARD_RECIPE_VERSION:
            raise ValueError(
                f"reward recipe must be {REWARD_RECIPE_VERSION!r}"
            )
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.step_penalty_cleared_ratio_scale < 0:
            raise ValueError(
                "step_penalty_cleared_ratio_scale must be non-negative"
            )
        if self.wall_penalty < 0:
            raise ValueError("wall_penalty must be non-negative")
        if self.normal_pellet_reward < 0:
            raise ValueError("normal_pellet_reward must be non-negative")
        if self.power_pellet_reward < 0:
            raise ValueError("power_pellet_reward must be non-negative")
        if self.ghost_reward < 0:
            raise ValueError("ghost_reward must be non-negative")
        if self.fruit_reward < 0:
            raise ValueError("fruit_reward must be non-negative")
        if self.death_penalty < 0:
            raise ValueError("death_penalty must be non-negative")
        if self.completion_reward < 0:
            raise ValueError("completion_reward must be non-negative")
        if self.safety_refusal_penalty < 0:
            raise ValueError("safety_refusal_penalty must be non-negative")
        if self.nearest_pellet_alpha < 0:
            raise ValueError("nearest_pellet_alpha must be non-negative")
        if not 0 <= self.nearest_pellet_remaining_ratio_threshold <= 1:
            raise ValueError(
                "nearest_pellet_remaining_ratio_threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class RewardBreakdown:
    reward_recipe_version: str
    event_game_score_delta: int
    event_count: int
    base_reward: float
    base_reward_contribution: float
    normal_pellet_eaten: bool
    normal_pellet_reward: float
    power_pellet_eaten: bool
    power_pellet_reward: float
    ghost_eaten: bool
    ghost_reward: float
    fruit_eaten: bool
    fruit_reward: float
    death: bool
    death_penalty: float
    level_completed: bool
    completion_reward: float
    safety_refusal: bool
    safety_refusal_penalty: float
    step_penalty: float
    wall_penalty: float
    normal_pellet_remaining_ratio: float
    nearest_pellet_shaping_active: bool
    nearest_pellet_distance_before: int | None
    nearest_pellet_distance_after: int | None
    nearest_pellet_progress_weight: float
    nearest_pellet_progress_reward: float
    shaped_reward: float

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def shape_reward(
    base_reward: float,
    previous_info: Mapping[str, Any],
    info: Mapping[str, Any],
    config: RewardConfig,
    *,
    normal_pellet_remaining_ratio: float = 1.0,
    normal_pellet_remaining_ratio_before: float | None = None,
    nearest_pellet_distance_before: int | None = None,
    nearest_pellet_distance_after: int | None = None,
) -> RewardBreakdown:
    del previous_info
    raw_events = info.get("logic_frame_events")
    if not isinstance(raw_events, list):
        raise ValueError("API v3 reward requires logic_frame_events")
    events: list[Mapping[str, Any]] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise ValueError(f"logic_frame_events[{index}] must be an object")
        event_type = raw_event.get("event_type")
        if event_type not in _SCORE_BY_EVENT:
            raise ValueError(f"unsupported API v3 event type: {event_type!r}")
        event_score = raw_event.get("score_delta")
        if not isinstance(event_score, int) or isinstance(event_score, bool):
            raise ValueError("event score_delta must be an integer")
        if event_score not in _SCORE_BY_EVENT[str(event_type)]:
            raise ValueError(
                f"invalid {event_type} score delta: {event_score}"
            )
        events.append(raw_event)
    base_reward_value = float(base_reward)
    if not math.isfinite(base_reward_value) or not base_reward_value.is_integer():
        raise ValueError("API v3 environment score delta must be a finite integer")
    event_game_score_delta = sum(int(event["score_delta"]) for event in events)
    if event_game_score_delta != base_reward_value:
        raise ValueError(
            "event score sum does not match environment score delta: "
            f"{event_game_score_delta} != {base_reward}"
        )
    event_types = [str(event["event_type"]) for event in events]
    base_reward_contribution = (
        base_reward_value if config.use_base_reward else 0.0
    )
    normal_pellet_count = event_types.count("normal_pellet_eaten")
    normal_pellet_eaten = normal_pellet_count > 0
    normal_pellet_reward = (
        config.normal_pellet_reward * normal_pellet_count
    )
    power_pellet_count = event_types.count("power_pellet_eaten")
    power_pellet_eaten = power_pellet_count > 0
    power_pellet_reward = (
        config.power_pellet_reward * power_pellet_count
    )
    ghost_count = event_types.count("ghost_eaten")
    ghost_eaten = ghost_count > 0
    ghost_reward = config.ghost_reward * ghost_count
    fruit_count = event_types.count("fruit_eaten")
    fruit_eaten = fruit_count > 0
    fruit_reward = config.fruit_reward * fruit_count
    death = "death" in event_types
    death_penalty = config.death_penalty if death else 0.0
    level_completed = "level_cleared" in event_types
    completion_reward = (
        config.completion_reward if level_completed else 0.0
    )
    if not 0 <= normal_pellet_remaining_ratio <= 1:
        raise ValueError("normal_pellet_remaining_ratio must be in [0, 1]")
    if normal_pellet_remaining_ratio_before is None:
        normal_pellet_remaining_ratio_before = normal_pellet_remaining_ratio
    if not 0 <= normal_pellet_remaining_ratio_before <= 1:
        raise ValueError(
            "normal_pellet_remaining_ratio_before must be in [0, 1]"
        )
    cleared_ratio_before = 1.0 - normal_pellet_remaining_ratio_before
    step_penalty = (
        config.step_penalty
        + config.step_penalty_cleared_ratio_scale * cleared_ratio_before
    )
    wall_penalty = config.wall_penalty if bool(info["wall_collision"]) else 0.0
    distances_available = (
        nearest_pellet_distance_before is not None
        and nearest_pellet_distance_after is not None
    )
    shaping_active = (
        config.nearest_pellet_alpha > 0
        and normal_pellet_remaining_ratio
        <= config.nearest_pellet_remaining_ratio_threshold
        and distances_available
        and not (
            config.nearest_pellet_skip_on_eat
            and normal_pellet_eaten
        )
    )
    progress_weight = 0.0
    progress_reward = 0.0
    if shaping_active and (
        nearest_pellet_distance_before is not None
        and nearest_pellet_distance_after is not None
    ):
        progress_weight = config.nearest_pellet_alpha
        if config.nearest_pellet_scale_by_cleared_ratio:
            progress_weight *= 1.0 - normal_pellet_remaining_ratio
        progress_reward = progress_weight * (
            nearest_pellet_distance_before - nearest_pellet_distance_after
        )
    shaped = (
        base_reward_contribution
        + normal_pellet_reward
        + power_pellet_reward
        + ghost_reward
        + fruit_reward
        + completion_reward
        - step_penalty
        - wall_penalty
        - death_penalty
        + progress_reward
    )
    return RewardBreakdown(
        reward_recipe_version=config.recipe_version,
        event_game_score_delta=event_game_score_delta,
        event_count=len(events),
        base_reward=base_reward_value,
        base_reward_contribution=base_reward_contribution,
        normal_pellet_eaten=normal_pellet_eaten,
        normal_pellet_reward=normal_pellet_reward,
        power_pellet_eaten=power_pellet_eaten,
        power_pellet_reward=power_pellet_reward,
        ghost_eaten=ghost_eaten,
        ghost_reward=ghost_reward,
        fruit_eaten=fruit_eaten,
        fruit_reward=fruit_reward,
        death=death,
        death_penalty=death_penalty,
        level_completed=level_completed,
        completion_reward=completion_reward,
        safety_refusal=False,
        safety_refusal_penalty=0.0,
        step_penalty=step_penalty,
        wall_penalty=wall_penalty,
        normal_pellet_remaining_ratio=float(
            normal_pellet_remaining_ratio
        ),
        nearest_pellet_shaping_active=shaping_active,
        nearest_pellet_distance_before=nearest_pellet_distance_before,
        nearest_pellet_distance_after=nearest_pellet_distance_after,
        nearest_pellet_progress_weight=progress_weight,
        nearest_pellet_progress_reward=progress_reward,
        shaped_reward=shaped,
    )


def audit_reward(record: Mapping[str, Any], *, tolerance: float = 1e-9) -> None:
    if record.get("reward_recipe_version") != REWARD_RECIPE_VERSION:
        raise ValueError("reward audit requires the API v3 reward recipe")
    events = record.get("logic_frame_events")
    if not isinstance(events, list):
        raise ValueError("reward audit requires logic_frame_events")
    event_count = record.get("event_count")
    if (
        not isinstance(event_count, int)
        or isinstance(event_count, bool)
        or event_count != len(events)
    ):
        raise ValueError("reward event_count does not match logic_frame_events")
    event_types: list[str] = []
    event_score = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"logic_frame_events[{index}] must be an object")
        event_type = event.get("event_type")
        if event_type not in _SCORE_BY_EVENT:
            raise ValueError(f"unsupported API v3 event type: {event_type!r}")
        score_delta = event.get("score_delta")
        if not isinstance(score_delta, int) or isinstance(score_delta, bool):
            raise ValueError("event score_delta must be an integer")
        if score_delta not in _SCORE_BY_EVENT[str(event_type)]:
            raise ValueError(
                f"invalid {event_type} score delta: {score_delta}"
            )
        event_types.append(str(event_type))
        event_score += score_delta
    recorded_event_score = record.get("event_game_score_delta")
    if (
        not isinstance(recorded_event_score, int)
        or isinstance(recorded_event_score, bool)
        or event_score != recorded_event_score
    ):
        raise ValueError("reward event score audit failed")
    base_reward = float(record["base_reward"])
    if not math.isfinite(base_reward) or not base_reward.is_integer():
        raise ValueError("reward base_reward must be a finite integer")
    if event_score != base_reward:
        raise ValueError("reward events do not reconcile base_reward")
    event_flags = {
        "normal_pellet_eaten": "normal_pellet_eaten",
        "power_pellet_eaten": "power_pellet_eaten",
        "ghost_eaten": "ghost_eaten",
        "fruit_eaten": "fruit_eaten",
        "death": "death",
        "level_completed": "level_cleared",
    }
    reward_fields = {
        "normal_pellet_eaten": "normal_pellet_reward",
        "power_pellet_eaten": "power_pellet_reward",
        "ghost_eaten": "ghost_reward",
        "fruit_eaten": "fruit_reward",
        "death": "death_penalty",
        "level_completed": "completion_reward",
    }
    for flag, event_type in event_flags.items():
        expected_flag = event_type in event_types
        if bool(record.get(flag, False)) != expected_flag:
            raise ValueError(f"reward {flag} does not match event ledger")
        reward_value = float(record.get(reward_fields[flag], 0.0))
        if not math.isfinite(reward_value) or reward_value < 0:
            raise ValueError(f"reward {reward_fields[flag]} is invalid")
        if not expected_flag and abs(reward_value) > tolerance:
            raise ValueError(
                f"reward {reward_fields[flag]} requires its source event"
            )
    has_safety_refusal = "safety_refusal" in record
    has_safety_penalty = "safety_refusal_penalty" in record
    if has_safety_refusal != has_safety_penalty:
        raise ValueError("safety-refusal reward fields must be recorded together")
    expected_safety_refusal = (
        record.get("terminal_reason") == "safety_refusal"
    )
    if has_safety_refusal:
        if bool(record["safety_refusal"]) != expected_safety_refusal:
            raise ValueError(
                "reward safety_refusal does not match terminal_reason"
            )
        safety_refusal_penalty = float(record["safety_refusal_penalty"])
        if (
            not math.isfinite(safety_refusal_penalty)
            or safety_refusal_penalty < 0
        ):
            raise ValueError("reward safety_refusal_penalty is invalid")
        if (
            not expected_safety_refusal
            and abs(safety_refusal_penalty) > tolerance
        ):
            raise ValueError(
                "reward safety_refusal_penalty requires safety_refusal"
            )
    else:
        # Additive v3 schema extension: archived records without these two
        # fields retain their original zero-penalty interpretation.
        safety_refusal_penalty = 0.0
    base_contribution = float(record.get("base_reward_contribution", base_reward))
    if not math.isfinite(base_contribution) or not (
        abs(base_contribution) <= tolerance
        or abs(base_contribution - base_reward) <= tolerance
    ):
        raise ValueError(
            "base_reward_contribution must be zero or the audited base_reward"
        )
    progress_weight = float(record.get("nearest_pellet_progress_weight", 0.0))
    progress_reward = float(record.get("nearest_pellet_progress_reward", 0.0))
    if (
        not math.isfinite(progress_weight)
        or progress_weight < 0
        or not math.isfinite(progress_reward)
    ):
        raise ValueError("nearest-pellet reward terms must be finite")
    distance_before = record.get("nearest_pellet_distance_before")
    distance_after = record.get("nearest_pellet_distance_after")
    if bool(record.get("nearest_pellet_shaping_active", False)):
        if distance_before is None or distance_after is None:
            raise ValueError("active nearest-pellet shaping requires both distances")
        for label, distance in (
            ("before", distance_before),
            ("after", distance_after),
        ):
            if (
                not isinstance(distance, int)
                or isinstance(distance, bool)
                or distance < 0
            ):
                raise ValueError(
                    f"nearest-pellet distance {label} must be a non-negative integer"
                )
        expected_progress = progress_weight * (
            int(distance_before) - int(distance_after)
        )
        if abs(expected_progress - progress_reward) > tolerance:
            raise ValueError(
                "nearest-pellet reward audit failed: "
                f"expected {expected_progress}, got {progress_reward}"
            )
    elif abs(progress_reward) > tolerance or abs(progress_weight) > tolerance:
        raise ValueError("inactive nearest-pellet shaping must have zero reward and weight")
    for penalty_field in ("step_penalty", "wall_penalty"):
        penalty = float(record[penalty_field])
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError(f"reward {penalty_field} is invalid")
    remaining_ratio = float(record["normal_pellet_remaining_ratio"])
    if not math.isfinite(remaining_ratio) or not 0 <= remaining_ratio <= 1:
        raise ValueError("normal-pellet remaining ratio must be in [0, 1]")
    expected = (
        base_contribution
        + float(record.get("normal_pellet_reward", 0.0))
        + float(record.get("power_pellet_reward", 0.0))
        + float(record.get("ghost_reward", 0.0))
        + float(record.get("fruit_reward", 0.0))
        + float(record.get("completion_reward", 0.0))
        - float(record["step_penalty"])
        - float(record["wall_penalty"])
        - float(record.get("death_penalty", 0.0))
        - safety_refusal_penalty
        + progress_reward
    )
    shaped_reward = float(record["shaped_reward"])
    if not math.isfinite(shaped_reward):
        raise ValueError("reward shaped_reward must be finite")
    if abs(expected - shaped_reward) > tolerance:
        raise ValueError(
            f"reward audit failed: expected {expected}, got {record['shaped_reward']}"
        )
