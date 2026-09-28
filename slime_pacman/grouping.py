"""Normalize complete episode rewards before expanding their decisions."""

from collections import defaultdict
import math

import torch


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


def group_advantages(rewards):
    if len(rewards) != 12 or any(
        r not in (0.0, 1.0) or not math.isfinite(r) for r in rewards
    ):
        raise ValueError("expected 12 binary episode rewards")
    values = torch.tensor(rewards, dtype=torch.float32)
    # Pinned slime: sample standard deviation (correction=1), epsilon 1e-6.
    return ((values - values.mean()) / (values.std(correction=1) + 1e-6)).tolist()


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
        episode["steps"].append(meta["decision_index"])
    advantages = {}
    all_versions = set()
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
                group_advantages([e["reward"] for e in episodes.values()]),
                strict=True,
            )
        )
    if not groups or len(all_versions) > 1:
        raise ValueError("rollout batch mixes or omits weight versions")
    return [float(s.reward) for s in samples], [
        advantages[s.rollout_id] for s in samples
    ]
