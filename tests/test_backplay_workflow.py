"""Real simulator restart integration, suffix rewards and baseline audits."""

import asyncio
import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig, load_bundled_level
from pacman_env.planner import EdwardPlanner
from pacman_recipe.level1.workflow import PacmanImageOnlyWorkflow
from pacman_recipe.level1.trajectories import audit_trajectory
from test_level1_recipe import FakeObjectiveTokenizer, OneStepEnv, make_episode_row


def make_workflow():
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeObjectiveTokenizer()):
        return PacmanImageOnlyWorkflow(
            env_factory=OneStepEnv, open_action_mask=True,
            tokenizer_path="test-tokenizer", image_prompt_style="live_state_v3",
            action_protocol="direct-open-action-token-v1",
            prompt_version="live-state-direct-action-v3",
        )


@pytest.fixture
def restart_row(tmp_path):
    row = make_episode_row(1, split="train", seed=7,
                           action_protocol="direct-open-action-token-v1")
    config = PygamePacmanEnvConfig(max_steps=row["env"]["max_steps"])
    with PygamePacmanEnv(config) as env:
        env.reset(seed=7)
        planner = EdwardPlanner()
        for _ in range(8):
            _, _, terminated, truncated, info = env.step(planner.decide(env.snapshot()).action)
            assert not (terminated or truncated)
        saved = env.save_state()
        action = env.snapshot()["open"][0]
        assert len(env.snapshot()["normal_pellet_positions"]) == info["normal_pellets_remaining"]
    raw = json.dumps(saved).encode()
    path = tmp_path / "restart.json"
    path.write_bytes(raw)
    row.update(restart_state_path=str(path), restart_state_sha256=hashlib.sha256(raw).hexdigest(),
               restart_state_id="teacher-7-step-8")
    return row, info, action


def test_restart_suffix_preserves_reward_denominator_and_audits(restart_row):
    row, initial, action = restart_row
    workflow = make_workflow()
    asyncio.run(workflow.run(row, scripted_actions=[action], nearest_pellet_alpha=1.0,
                             step_penalty_cleared_ratio_scale=2.0))
    payload = workflow.last_episode
    audit_trajectory(payload)
    step = payload["trajectory"][0]
    assert step["env_step"] == initial["step"] + 1
    assert payload["restart_state"]["remaining_budget"] == row["env"]["max_steps"] - initial["step"]
    assert payload["total_base_reward"] == step["score"] - initial["score"]
    assert payload["suffix_score_delta"] == payload["total_base_reward"]
    assert payload["normal_pellets_initial"] == initial["normal_pellets_remaining"]
    denominator = len(load_bundled_level(1).pellets)
    assert payload["reward_normal_pellets_initial"] == denominator
    assert step["normal_pellet_remaining_ratio"] == step["normal_pellets_remaining"] / denominator
    assert step["step_penalty"] == 1.0 + 2.0 * (1.0 - initial["normal_pellets_remaining"] / denominator)
    assert payload["state_prefix_actions_executed"] == 0
    assert payload["prefix_end_score"] == initial["score"] > 0
    for field in ("score", "logic_frame", "death_count", "source_step", "normal_pellets"):
        corrupt = copy.deepcopy(payload)
        corrupt["restart_state"][field] += 1
        with pytest.raises(ValueError):
            audit_trajectory(corrupt)


@pytest.mark.parametrize("damage", ["hash", "prefix", "finished"])
def test_restart_rejections_before_rollout(restart_row, damage):
    row, _, action = restart_row
    if damage == "hash":
        row["restart_state_sha256"] = "0" * 64
    elif damage == "prefix":
        row["state_prefix_actions"] = ["L"]
    else:
        from pathlib import Path
        path = Path(row["restart_state_path"])
        saved = json.loads(path.read_bytes())
        saved["payload"]["episode"]["finished"] = True
        raw = json.dumps(saved).encode()
        path.write_bytes(raw)
        row["restart_state_sha256"] = hashlib.sha256(raw).hexdigest()
    workflow = make_workflow()
    with pytest.raises(ValueError):
        asyncio.run(workflow.run(row, scripted_actions=[action]))


def test_parse_failure_after_a_death_records_the_post_death_lives():
    # Seed 7, alternating L/R: pacman is caught at step 25 (3 -> 2 lives) with only L/R open.
    row = make_episode_row(1, split="train", seed=7, action_protocol="direct-open-action-token-v1")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeObjectiveTokenizer()):
        workflow = PacmanImageOnlyWorkflow(
            env_factory=PygamePacmanEnv, open_action_mask=True,
            tokenizer_path="test-tokenizer", image_prompt_style="live_state_v3",
            action_protocol="direct-open-action-token-v1",
            prompt_version="live-state-direct-action-v3",
            episode_life_mode="original_three_lives",
        )
    asyncio.run(workflow.run(row, scripted_actions=["L", "R"] * 12 + ["L", "U"]))
    payload = workflow.last_episode
    death, failure = payload["trajectory"][-2:]
    assert death["death"] and (death["lives"], death["lives_after_step"]) == (3, 2)
    assert failure["contract_violation_type"] == "parse_failure"
    assert (failure["lives"], failure["lives_after_step"], failure["death"]) == (2, 2, False)
    audit_trajectory(payload)
