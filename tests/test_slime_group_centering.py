"""Mean-only advantages: independent reward oracle and episode expansion."""
from types import SimpleNamespace

import pytest

from slime_pacman.grouping import group_advantages, post_process_rewards


@pytest.mark.parametrize("wins", range(13))
def test_binary_group_centering_without_std(wins):
    rewards = [1.0] * wins + [0.0] * (12 - wins)
    actual = group_advantages(rewards)
    assert actual == pytest.approx([r - wins / 12 for r in rewards])
    assert sum(actual) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("reward", [0.0, 1.0, 1.00000001, 1.027272727, 1.05])
def test_identical_shaped_rewards_are_exactly_zero(reward):
    assert group_advantages([reward] * 12, success_speed_bonus=.05) == [0.0] * 12


def test_small_speed_difference_keeps_small_scale():
    actual = group_advantages([1.01] * 6 + [1.01000002] * 6, success_speed_bonus=.05)
    assert actual == pytest.approx([-1e-8] * 6 + [1e-8] * 6, abs=1e-15)


def test_two_states_center_separately_before_unequal_decision_expansion():
    samples = []
    for group, wins in [(0, 1), (1, 11)]:
        for episode in range(12):
            count = episode % 3 + 1
            for decision in range(count):
                samples.append(SimpleNamespace(
                    rollout_id=(group, episode), reward=float(episode < wins),
                    train_metadata=dict(episode_id=(group, episode), group_id=group,
                        initial_state_id=f"state-{group}", decision_count=count,
                        decision_index=decision, weight_version="v1")))
    raw, actual = post_process_rewards(SimpleNamespace(n_samples_per_prompt=12), samples)
    expected = [s.reward - (1 if s.train_metadata["group_id"] == 0 else 11) / 12
                for s in samples]
    assert raw == [s.reward for s in samples]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("rewards", [[0.] * 11, [0.] * 13, [float("nan")] * 12,
                                     [float("inf")] * 12, [-.1] * 12, [1.06] * 12])
def test_invalid_rewards_still_rejected(rewards):
    with pytest.raises(ValueError):
        group_advantages(rewards, success_speed_bonus=.05)
