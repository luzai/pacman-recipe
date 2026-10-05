"""Success-only route efficiency and informative all-win groups."""
from types import SimpleNamespace

import pytest

from slime_pacman.config import PacmanConfig
from slime_pacman.grouping import success_speed_reward, group_advantages, post_process_rewards
from slime_pacman.sampling import is_zero_variance


def episode(steps=80, won=True, restart=True):
    return dict(episode_life_mode="single_death", won=won,
                terminal_reason="all_normal_pellets" if won else "death",
                normal_pellets_remaining=0 if won else 12,
                steps=steps, max_steps=512,
                restart_state={"remaining_budget": 176} if restart else None)


def test_success_only_and_suffix_horizon():
    assert success_speed_reward(episode(), .05) == pytest.approx(1.027272727)
    assert success_speed_reward(episode(150), .05) == pytest.approx(1.007386364)
    assert success_speed_reward(episode(won=False), .05) == 0
    assert success_speed_reward(episode(176), .05) == 1
    assert success_speed_reward(episode(restart=False), .05) == pytest.approx(1 + .05 * (1 - 80 / 512))
    assert success_speed_reward(episode()) == 1


@pytest.mark.parametrize("coefficient", [-.01, .11, float("nan"), float("inf")])
def test_invalid_coefficients(coefficient):
    with pytest.raises(ValueError):
        PacmanConfig(success_speed_bonus=coefficient)
    with pytest.raises(ValueError):
        success_speed_reward(episode(), coefficient)


def test_invalid_suffix_length():
    with pytest.raises(ValueError):
        success_speed_reward(episode(177), .05)


def test_all_winners_can_learn_shorter_routes():
    rewards = [success_speed_reward(episode(n), .05) for n in [80] * 6 + [150] * 6]
    advantage = group_advantages(rewards, success_speed_bonus=.05)
    assert all(a > 0 for a in advantage[:6])
    assert all(a < 0 for a in advantage[6:])
    assert not is_zero_variance([SimpleNamespace(reward=r) for r in rewards])
    assert is_zero_variance([SimpleNamespace(reward=rewards[0])] * 12)
    assert group_advantages([0.] * 12, success_speed_bonus=.05) == [0.] * 12
    with pytest.raises(ValueError):
        group_advantages(rewards)


def test_postprocessing_propagates_bonus_and_checks_mode():
    samples = [SimpleNamespace(rollout_id=i, reward=success_speed_reward(episode(80 if i < 6 else 150), .05),
        train_metadata=dict(episode_id=i, group_id=0, decision_count=1,
                            decision_index=0, weight_version="1", initial_state_id="save",
                            success_speed_bonus=.05)) for i in range(12)]
    raw, advantages = post_process_rewards(SimpleNamespace(n_samples_per_prompt=12), samples)
    assert raw[0] > raw[-1] >= 1
    assert advantages[0] > 0 > advantages[-1]
    samples[0].train_metadata["success_speed_bonus"] = 0
    with pytest.raises(ValueError, match="mixes reward"):
        post_process_rewards(SimpleNamespace(n_samples_per_prompt=12), samples)


def test_evaluation_counts_wins_without_bonus():
    from slime_pacman.eval_stats import summarize
    rows = [dict(group="true_start", seed=14, reward=1.05, terminal_reason="all_normal_pellets")] * 24
    rows += [dict(group="true_start", seed=14, reward=0., terminal_reason="death")] * 24
    result = summarize(rows, "true_start")
    assert result["wins"] == 24
    assert result["pass_rate"] == .5


def test_runner_applies_bonus_after_binary_environment_audit(monkeypatch):
    import asyncio
    from slime_pacman import rollout
    payload = episode()
    payload["total_shaped_reward"] = 1.
    runner = object.__new__(rollout.EpisodeRunner)
    runner.record = {}
    runner.success_speed_bonus = .05
    runner.decisions = [SimpleNamespace(weight_version="v1")]
    runner.last_episode = payload
    async def run(row):
        pass
    runner.run = run
    monkeypatch.setattr(rollout, "runner_row", lambda record: {})
    monkeypatch.setattr(rollout, "neutral_trajectory", lambda value: value)
    monkeypatch.setattr(rollout, "audit_neutral_trajectory", lambda value: None)
    result = asyncio.run(runner.collect(empty_weight_version="v1"))
    assert result.reward == pytest.approx(1.027272727)
    assert result.success_speed_bonus == .05
    assert result.environment_steps == 80
    payload["total_shaped_reward"] = 1.05
    with pytest.raises(ValueError, match="binary contract"):
        asyncio.run(runner.collect(empty_weight_version="v1"))


def test_live_metrics_separate_win_rate_and_speed(monkeypatch):
    Sample = pytest.importorskip("slime.utils.types").Sample
    logging_utils = pytest.importorskip("slime.observability.logging_utils")
    from slime_pacman.rollout import log_eval, log_rollout
    captured = []
    monkeypatch.delenv("PACMAN_RUN_DIR", raising=False)
    monkeypatch.setattr(logging_utils, "log", lambda args, metrics, **kw: captured.append(metrics))
    samples = []
    for i, (reward, won, steps) in enumerate([(1.027, True, 80), (1.007, True, 150), (0., False, 1)]):
        sample = Sample(reward=reward, train_metadata=dict(episode_id=i, group_id=0,
                       won=won, environment_steps=steps))
        samples.extend([sample] * (i + 1))  # unequal decision counts must not weight episodes
    log_eval(0, SimpleNamespace(), {"pacman": {"samples": samples}})
    log_rollout(0, SimpleNamespace(), samples, {}, 1.)
    for prefix, metrics in zip(["eval/pacman", "rollout"], captured):
        assert metrics[f"{prefix}/win_rate"] == pytest.approx(2 / 3)
        assert metrics[f"{prefix}/mean_reward"] == pytest.approx(2.034 / 3)
        assert metrics[f"{prefix}/success_steps_median"] == 115
