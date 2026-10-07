"""Center complete episode rewards within each state before decision expansion."""

from collections import defaultdict
import math



def binary_reward(payload):
    if payload.get("parse_failures", 0) or payload.get(
        "canonical_action_violations", 0
    ):
        raise ValueError("generation failure cannot become an ordinary losing episode")
    if payload.get("episode_life_mode") != "single_death":
        raise ValueError("binary C2 requires single_death")
    reason = payload.get("terminal_reason")
    if reason not in {
        "all_normal_pellets",
        "death",
        "max_steps",
        "safety_refusal",
        "unreachable_normal_pellets",
    }:
        raise ValueError(f"unrecognized game terminal: {reason}")
    won = reason == "all_normal_pellets"
    if bool(payload.get("won")) != won or (
        won and payload.get("normal_pellets_remaining") != 0
    ):
        raise ValueError("inconsistent win evidence")
    return float(won)


def success_speed_reward(payload, coefficient=0.0):
    """Terminal success-only bonus; steps are environment actions after restore."""
    won = binary_reward(payload)
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 0.1:
        raise ValueError("invalid success speed coefficient")
    if not coefficient or not won:
        return won
    restart = payload.get("restart_state")
    horizon = restart["remaining_budget"] if restart else payload["max_steps"]
    steps = payload["steps"]
    if type(horizon) is not int or horizon <= 0 or type(steps) is not int or not 0 <= steps <= horizon:
        raise ValueError("invalid remaining horizon or executed steps")
    return 1.0 + coefficient * (1.0 - steps / horizon)


def group_advantages(rewards, *, success_speed_bonus=0.0):
    if not math.isfinite(success_speed_bonus) or not 0 <= success_speed_bonus <= 0.1:
        raise ValueError("invalid success speed coefficient")
    if len(rewards) != 12 or any(
        not math.isfinite(r) or not (r == 0.0 or 1.0 <= r <= 1.0 + success_speed_bonus) for r in rewards
    ):
        raise ValueError("expected 12 episode rewards within the declared success bonus range")
    # Constant shaped rewards must be exactly zero, including non-binary values.
    if all(r == rewards[0] for r in rewards):
        return [0.0] * 12
    # Stable host reduction; no std scaling or cross-state normalization.
    mean = math.fsum(rewards) / len(rewards)
    return [float(r - mean) for r in rewards]


def post_process_rewards(args, samples):
    if args.n_samples_per_prompt != 12:
        raise ValueError("Pacman group size must be 12")
    groups = defaultdict(dict)
    episode_groups = {}
    for sample in samples:
        meta = sample.train_metadata
        if not isinstance(meta, dict) or sample.rollout_id != meta["episode_id"]:
            raise ValueError("episode identity missing or changed")
        if (
            episode_groups.setdefault(meta["episode_id"], meta["group_id"])
            != meta["group_id"]
        ):
            raise ValueError("episode appears in multiple groups")
        episodes = groups[meta["group_id"]]
        episode = episodes.setdefault(
            meta["episode_id"],
            {
                "reward": sample.reward,
                "success_speed_bonus": meta.get("success_speed_bonus", 0.0),
                "steps": [],
                "count": meta["decision_count"],
                "version": meta["weight_version"],
                "initial_state": meta["initial_state_id"],
            },
        )
        if (
            episode["reward"],
            episode["count"],
            episode["version"],
            episode["initial_state"],
        ) != (
            sample.reward,
            meta["decision_count"],
            meta["weight_version"],
            meta["initial_state_id"],
        ):
            raise ValueError("inconsistent episode metadata")
        if episode["success_speed_bonus"] != meta.get("success_speed_bonus", 0.0):
            raise ValueError("inconsistent episode reward configuration")
        episode["steps"].append(meta["decision_index"])
    advantages = {}
    all_versions = set()
    coefficients = {e["success_speed_bonus"] for group in groups.values() for e in group.values()}
    if len(coefficients) != 1:
        raise ValueError("rollout batch mixes reward configurations")
    coefficient = next(iter(coefficients))
    for episodes in groups.values():
        if (
            len(episodes) != 12
            or len({e["initial_state"] for e in episodes.values()}) != 1
        ):
            raise ValueError("incomplete group or mixed initial states")
        for episode in episodes.values():
            expected = list(range(episode["count"])) if episode["count"] else [-1]
            if sorted(episode["steps"]) != expected:
                raise ValueError("missing or duplicated episode decisions")
            if episode["count"]:
                if not episode["version"]:
                    raise ValueError("decision has no weight version")
                all_versions.add(episode["version"])
            elif episode["reward"] != 0:
                raise ValueError("zero-decision refusal cannot be a win")
        advantages.update(
            zip(
                episodes,
                group_advantages([e["reward"] for e in episodes.values()], success_speed_bonus=coefficient),
                strict=True,
            )
        )
    if not groups or len(all_versions) > 1:
        raise ValueError("rollout batch mixes or omits weight versions")
    return [float(s.reward) for s in samples], [
        advantages[s.rollout_id] for s in samples
    ]
