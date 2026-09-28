"""Group-level rollout filters for AReaL's ``should_accept_fn`` hook.

These must be plain module-level functions referenced by import path, never
closures. ``PPOTrainer.train(dynamic_filter_fn=...)`` forwards the value to the
rollout workers through JSON RPC (``RemoteInfEngine.prepare_batch`` ->
``scheduler.async_call_engine('submit', ...)``), and a function object is not
JSON serializable: every submit fails with ``Object of type function is not
JSON serializable`` and the run retries forever without completing a single
rollout. AReaL resolves the string form on the worker with
``import_from_string``, which is the only RPC-safe way to pass a filter.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

# Episode returns are deterministic given a trajectory, so a genuinely diverse
# group never lands inside this window; it only absorbs float round-trip noise.
DEGENERATE_GROUP_REWARD_RANGE_EPS = 1e-4


def accept_non_degenerate_reward_group(trajectory: dict[str, Any]) -> bool:
    """Reject a rollout group whose episodes all return the same reward.

    Under ``reward_objective_contract=episode_return_group_v1`` the actor's
    ``reward_norm`` (mean_level=group, std_level=group) already zeroes the
    advantage of such a group, so training on it costs a full rollout plus a
    logprob-recompute pass for exactly zero gradient signal. Entropy-collapsed
    policies that sample the same trajectory for all 12 group members are the
    common cause. Rejecting the group makes ``prepare_batch`` pull a
    replacement dataset row instead.
    """

    import torch

    rewards = trajectory.get("rewards")
    if not isinstance(rewards, torch.Tensor) or rewards.numel() < 2:
        return True

    # max-min rather than std(): torch.std() on an exactly-constant float32
    # vector at this reward scale returns ~3e-5 from catastrophic
    # cancellation, while the range is exactly 0.0.
    reward_range = float((rewards.max() - rewards.min()).item())
    if reward_range > DEGENERATE_GROUP_REWARD_RANGE_EPS:
        return True

    LOGGER.warning(
        "DEGENERATE_GROUP_REJECTED rows=%d total_shaped_reward=%.6f "
        "range=%.3g (zero group-relative advantage; resampling a "
        "replacement dataset row)",
        int(rewards.numel()),
        float(rewards[0].item()),
        reward_range,
    )
    return False
