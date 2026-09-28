"""Release prompt/reward checks; workflow cases require real AReaL imports."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from pacman_recipe.level1.prompts import (
    compact_edward_decision_prompt,
    live_state_instruction,
    prompt_contract_metadata,
    sent_prompt_sha256,
)
from pacman_recipe.level1.rewards import RewardConfig


def _context():
    return {
        "pacman_position": [23, 10],
        "facing": "L",
        "pellets_remaining": 194,
        "open_actions": ["L", "R"],
        "blocked_actions": ["U", "D"],
        "ghosts": [],
        "edible_ticks": 0,
        "maze_size": [31, 28],
    }


def test_native_request_rejects_historical_image_accumulation():
    from pacman_recipe.level1.workflow import PacmanNativeVisionWorkflow
    from pacman_recipe.level1.prompts import build_image_messages, encode_png
    import numpy as np

    messages = build_image_messages(
        encode_png(np.zeros((16, 16, 3), dtype=np.uint8)),
        prompt_style="live_state_v3", state_context=_context(),
    )
    messages[1]["content"].append(copy.deepcopy(messages[1]["content"][-1]))
    with pytest.raises(ValueError, match="exactly one current image"):
        PacmanNativeVisionWorkflow._pil_and_chat_messages(messages)


def test_stage_prompt_templates_identify_actual_distinct_protocols():
    c1 = prompt_contract_metadata("live_state_v3", edward_options=False)
    c2 = prompt_contract_metadata("live_state_v3", edward_options=True)
    assert c1["action_protocol"] == "direct-open-action-token-v1"
    assert c1["prompt_version"] == "live-state-direct-action-v3"
    assert c2["action_protocol"] == "edward-option-code-v1"
    assert c2["prompt_version"] == "edward-option-code-v2"
    assert c1["prompt_template_sha256"] != c2["prompt_template_sha256"]
    assert c1 == prompt_contract_metadata("live_state_v3", edward_options=False)


def test_c1_actual_prompt_has_four_directions_empty_ghosts_and_no_options():
    user = live_state_instruction(_context())
    assert "one of U/D/L/R" in user
    assert "Ghosts [id,state,position]: []" in user
    assert "edible_ticks=0" in user
    assert "COLLECT" not in user and "option code" not in user


def test_c2_actual_prompt_advertises_codes_not_direction_or_json_completion():
    candidate = SimpleNamespace(
        option_id="C0",
        strategy="COLLECT",
        target=[1, 2],
        first_action="L",
        route_distance=2,
        commit_moves=2,
        safety_margin=None,
        future_safe_exits=None,
        entity_id=None,
    )
    constraint = SimpleNamespace(code_for_option=lambda _: "B", rendered_choices=("B",))
    user = compact_edward_decision_prompt(_context(), [candidate], constraint)
    assert '["B","C0","COLLECT",[1,2],"L",2,2,null,null,null]' in user
    assert "Return exactly one code from [B]; nothing else." in user
    assert '"objective_id"' not in user


@pytest.mark.parametrize("change", ["system", "user", "image"])
def test_actual_sent_prompt_hash_binds_text_and_image(change):
    values = {"system": "s", "user": "u", "image": "0" * 64}
    before = sent_prompt_sha256(values["system"], values["user"], values["image"])
    values[change] += "changed"
    assert before != sent_prompt_sha256(
        values["system"], values["user"], values["image"]
    )


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize(
    "field", ["normal_pellet_reward", "death_penalty", "nearest_pellet_alpha"]
)
def test_reward_coefficients_must_remain_finite_even_with_infinite_clip(field, value):
    with pytest.raises(ValueError, match="finite"):
        RewardConfig(**{field: value})


def test_c1_runtime_does_not_construct_edward_and_emits_step_rewards(monkeypatch):
    from pacman_env.env import PygamePacmanEnv
    from pacman_recipe.level1.level1_dataset import make_episode_row
    from pacman_recipe.level1.workflow import ModelTurn, PacmanImageOnlyWorkflow
    from pacman_recipe.level1.trajectories import audit_trajectory

    class FourStepEnv(PygamePacmanEnv):
        def step(self, action):
            image, reward, _, _, info = super().step(action)
            done = int(info["step"]) == 4
            info = {
                **info,
                "terminated": done,
                "truncated": False,
                "terminal_reason": "test_complete" if done else None,
            }
            return image, reward, done, False, info

    class Policy(PacmanImageOnlyWorkflow):
        calls = 0

        async def _call_model(self, messages, **options):
            self.calls += 1
            # The spawn corridor has two empty tiles before its first pellet.
            # Hold one legal direction so this tests a real eat event, rather
            # than alternating whichever action appears first in the mask.
            assert "L" in options["current_open_actions"]
            return ModelTurn("L", str(self.calls), messages)

    def forbidden():
        pytest.fail("C1 constructed EdwardPlanner")

    monkeypatch.setattr("pacman_recipe.level1.episode.EdwardPlanner", forbidden)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda _: SimpleNamespace(encode=lambda action, **_: ["UDLR".index(action)]),
    )
    workflow = Policy(
        env_factory=FourStepEnv,
        tokenizer_path="fake",
        image_prompt_style="live_state_v3",
        edward_options=False,
        open_action_mask=True,
        action_token_choice=True,
        action_protocol="direct-open-action-token-v1",
        prompt_version="live-state-direct-action-v3",
        reward_objective_contract="step_local_raw_v1",
        use_base_reward=False,
        normal_pellet_reward=51.0,
        step_penalty=0.05,
        wall_penalty=0.5,
        recipe_contract={"reward": {"clip": "inf", "normalization": None}},
    )
    row = make_episode_row(
        28,
        split="train",
        max_steps=512,
        ghost_mode="disabled",
        action_protocol="direct-open-action-token-v1",
    )
    rewards = asyncio.run(workflow.run(row))
    payload = workflow.last_episode
    assert workflow.calls == len(rewards) == 4
    assert all(step["option_id"] is None for step in payload["trajectory"])
    assert payload["recipe_contract"]["reward"]["clip"] == "inf"
    assert any(
        event["event_type"] == "normal_pellet_eaten"
        for step in payload["trajectory"]
        for event in step["logic_frame_events"]
    )
    assert max(rewards.values()) > 20
    for step in payload["trajectory"]:
        assert rewards[step["completion_id"]] == step["shaped_reward"]
        assert step["action"] in step["open_action_mask"]
        assert step["observation_context"]["ghosts"] == []
    json.dumps(payload, allow_nan=False)
    audit_trajectory(payload)
    audit_trajectory(json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)))
    tampered = copy.deepcopy(payload)
    tampered["trajectory"][0]["model_user_instruction"] += " altered"
    with pytest.raises(ValueError, match="prompt hash"):
        audit_trajectory(tampered)


def test_runtime_rejects_mismatched_harness_row_before_environment(monkeypatch):
    from pacman_recipe.level1.level1_dataset import make_episode_row
    from pacman_recipe.level1.workflow import PacmanImageOnlyWorkflow

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda _: SimpleNamespace(encode=lambda action, **_: ["UDLR".index(action)]),
    )
    workflow = PacmanImageOnlyWorkflow(
        tokenizer_path="fake",
        image_prompt_style="live_state_v3",
        open_action_mask=True,
        action_protocol="direct-open-action-token-v1",
        prompt_version="live-state-direct-action-v3",
    )
    row = make_episode_row(28, split="train", action_protocol="edward-option-code-v1")
    with pytest.raises(ValueError, match="dataset action_protocol"):
        asyncio.run(workflow.run(row))


@pytest.mark.parametrize("contract", ["step_local_raw_v1", "episode_return_group_v1"])
def test_native_runtime_assigns_step_or_whole_episode_task_signal(
    contract, monkeypatch
):
    from contextvars import ContextVar
    import torch
    from pacman_recipe.level1.workflow import (
        PacmanImageOnlyWorkflow,
        PacmanNativeVisionWorkflow,
    )

    workflow = object.__new__(PacmanNativeVisionWorkflow)
    workflow.workflow_kwargs = {
        "reward_objective_contract": contract,
        "edward_options": contract == "episode_return_group_v1",
    }
    workflow.gconfig = SimpleNamespace(n_samples=12)
    workflow._native_engine = ContextVar("test-engine", default=None)
    workflow._native_turns = ContextVar("test-turns", default=None)
    workflow._episode_payload = ContextVar("test-payload", default=None)
    response = SimpleNamespace(
        input_tokens=[10, 11],
        output_tokens=[20],
        output_logprobs=[-0.5],
        output_versions=[3],
    )
    processed = {
        "mm_token_type_ids": torch.tensor([[0, 1]]),
        "pixel_values": torch.tensor([[1.0]]),
    }
    is_edward = contract == "episode_return_group_v1"
    turn = (
        processed,
        response,
        [] if is_edward else ["L", "R"],
        [[20, 30]] if is_edward else [],
    )

    async def run(self, data):
        self._native_turns.set({"first": turn, "second": turn})
        self._episode_payload.set(
            {"trajectory_sample_id": "test-episode", "total_shaped_reward": 50.0}
        )
        return {"first": 50.95, "second": -0.95}

    monkeypatch.setattr(PacmanImageOnlyWorkflow, "run", run)
    samples = asyncio.run(workflow.arun_episode(SimpleNamespace(), {}))
    if is_edward:
        assert samples["rewards"].tolist() == [50.0, 50.0]
        assert samples["rollout_episode_group_sizes"].tolist() == [12, 12]
        assert len(set(samples["rollout_episode_ids"].tolist())) == 1
        assert "pacman_allowed_token_ids" in samples
    else:
        assert samples["rewards"].tolist() == pytest.approx([50.95, -0.95])
        assert not any(key.startswith("rollout_episode_") for key in samples)
        assert "pacman_action_mask_bits" in samples


@pytest.mark.parametrize("reward", [float("inf"), float("-inf"), float("nan")])
def test_native_tensor_rejects_nonfinite_task_rewards(reward):
    from pacman_recipe.level1.workflow import PacmanNativeVisionWorkflow

    with pytest.raises(ValueError, match="training reward must be finite"):
        PacmanNativeVisionWorkflow._tensor_sample({}, None, reward, [])


@pytest.mark.parametrize("protocol", ["direct-open-action-token-v1", "edward-option-code-v1"])
def test_formal_native_request_never_silently_truncates(protocol):
    from pacman_recipe.level1.workflow import PacmanNativeVisionWorkflow

    workflow = object.__new__(PacmanNativeVisionWorkflow)
    workflow.workflow_kwargs = {"action_protocol": protocol}
    budget = SimpleNamespace(max_tokens=1024, max_new_tokens=1)
    workflow._validate_native_token_budget([10] * 1023, budget)
    with pytest.raises(ValueError, match="no truncation"):
        workflow._validate_native_token_budget([10] * 1024, budget)
    with pytest.raises(ValueError, match="explicit token budget"):
        workflow._validate_native_token_budget([10], SimpleNamespace())
