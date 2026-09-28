import asyncio
import copy
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import test_level1_recipe as legacy

from pacman_env.planner import EdwardPlanner
from pacman_recipe.level1.prompts import prompt_contract_metadata
from pacman_recipe.level1.recipe import recipe_contract_metadata
from pacman_recipe.level1.token_constraints import ObjectiveTokenConstraint
from pacman_recipe.level1.trajectories import audit_trajectory
from pacman_recipe.level1.workflow import ModelTurn, PacmanImageOnlyWorkflow, PacmanNativeVisionWorkflow
from test_level1_recipe import (
    FakeObjectiveTokenizer, OneStepEnv, TwoStepEnv, make_episode_row,
)


class FallbackFirstPlanner(EdwardPlanner):
    """Force a normal-option miss once; use real risk ranking and topology."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decisions = 0

    def candidates(self, state):
        return () if self.decisions == 0 else super().candidates(state)

    def advertised_candidates(self, state):
        result = super().advertised_candidates(state)
        self.decisions += 1
        return result


class ChoosingWorkflow(PacmanImageOnlyWorkflow):
    async def _call_model(self, messages, **options):
        constraint = options["objective_constraint"]
        # The policy can choose a non-first ranked direction.
        code = constraint.rendered_choices[-1]
        self.supports.append((constraint.allowed_token_ids, ord(code)))
        return ModelTurn(code, f"decision-{len(self.supports)}", messages)


@pytest.fixture
def episode():
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeObjectiveTokenizer()), patch(
        "pacman_recipe.level1.episode.EdwardPlanner", FallbackFirstPlanner
    ):
        workflow = ChoosingWorkflow(
            env_factory=TwoStepEnv, tokenizer_path="test-tokenizer",
            edward_options=True, edward_fallback_mode="risk_ranked",
            image_prompt_style="live_state_v3",
        )
        workflow.supports = []
        returns = asyncio.run(workflow.run(make_episode_row(1, split="test")))
    return workflow.last_episode, returns, workflow.supports


def test_fallback_one_step_then_normal_replan_with_real_env_and_reward(episode):
    payload, returns, supports = episode
    audit_trajectory(payload)
    first, second = payload["trajectory"]
    assert first["option_strategy"] == "RISK_FALLBACK"
    assert first["option_status"] == "max_commit" and first["option_step"] == 1
    assert not first["terminated"] and not first["truncated"]
    assert second["model_called"] and second["option_strategy"] != "RISK_FALLBACK"
    assert "not safety-approved" in first["model_user_instruction"]
    assert "Fallback risk by code:" in first["model_user_instruction"]
    assert '"K":[2,' in first["model_user_instruction"]
    assert payload["decoding"]["edward_fallback_mode"] == "risk_ranked"
    assert set(returns) == {"decision-1", "decision-2"}
    for step in (first, second):
        assert returns[step["completion_id"]] == step["shaped_reward"]
        assert step.get("safety_refusal_penalty", 0) == 0
    assert len(supports) == 2 and all(selected in allowed for allowed, selected in supports)


@pytest.mark.parametrize("mutation", ["commit", "rank", "motion", "mix", "directions", "action", "continuation", "mode", "evidence", "missing_nullable"])
def test_auditor_rejects_corrupt_fallback(episode, mutation):
    payload = copy.deepcopy(episode[0])
    step = payload["trajectory"][0]
    candidates = step["observation_context"]["planner_candidates"]
    if mutation == "commit":
        candidates[0]["commit_moves"] = 2
    elif mutation == "rank":
        candidates[0]["risk"]["rank"] = 9
    elif mutation == "motion":
        candidates[0]["risk"]["motion"] = "safe"
    elif mutation == "mix":
        candidates[0]["strategy"] = "AVOID"
    elif mutation == "directions":
        candidates[1]["first_action"] = candidates[0]["first_action"]
    elif mutation == "action":
        step["action"] = "U"
    elif mutation == "continuation":
        step["option_step"] = 2
    elif mutation == "mode":
        payload["decoding"].pop("edward_fallback_mode")
    elif mutation == "missing_nullable":
        del candidates[0]["risk"]["ghost_clearance"]
    else:
        del candidates[0]["risk"]
    with pytest.raises(ValueError):
        audit_trajectory(payload)


def test_existing_option_codes_feed_native_ppo_support_ledger():
    tokenizer = FakeObjectiveTokenizer()
    constraint = ObjectiveTokenConstraint.build(tokenizer, ["A0", "A1", "A2", "A3"])
    assert constraint.rendered_choices == ("J", "K", "M", "N")
    for code in constraint.rendered_choices:
        token = ord(code)
        response = SimpleNamespace(input_tokens=[10, 11], output_tokens=[token],
                                   output_logprobs=[-0.25], output_versions=[1])
        support = constraint.support_ledger(response.output_tokens)
        sample = PacmanNativeVisionWorkflow._tensor_sample(
            {"mm_token_type_ids": torch.tensor([[0, 1]]), "pixel_values": torch.tensor([[1.0]])},
            response, 1.0, [], support,
            rollout_episode_id=1,
        )
        assert sample["loss_mask"].tolist() == [[0, 0, 1]]
        assert sample["pacman_allowed_token_ids"].tolist()[0][-1] == [value + 1 for value in constraint.allowed_token_ids]
        assert sample["logprobs"].tolist() == [[0.0, 0.0, -0.25]]


def test_legacy_contract_unchanged_and_opt_in_fingerprinted():
    raw = {"edward_options": True, "image_prompt_style": "live_state_v3"}
    legacy = recipe_contract_metadata(raw)
    assert legacy == recipe_contract_metadata({**raw, "edward_fallback_mode": "refuse"})
    opted = recipe_contract_metadata({**raw, "edward_fallback_mode": "risk_ranked"})
    assert opted["harness"]["edward_fallback_mode"] == "risk_ranked"
    assert opted["prompt"]["template_sha256"] != legacy["prompt"]["template_sha256"]
    assert opted["prompt"]["action_protocol"] == legacy["prompt"]["action_protocol"]
    json.dumps(opted, allow_nan=False)


@pytest.mark.parametrize("edward,mode", [(True, "typo"), (False, "risk_ranked")])
def test_invalid_mode_rejected_before_rollout(edward, mode):
    with pytest.raises(ValueError, match="edward_fallback_mode"):
        prompt_contract_metadata("live_state_v3", edward_options=edward, fallback_mode=mode)


@pytest.mark.parametrize("edward,expected", [
    (False, "d0b0be31ccd26d7f73342f0a4b820ae18c92f7e124b135bf158403e08bf8cd90"),
    (True, "48b7b0dbf1e2639cd3943e7a81dcc19830314970587f66a44bd5732d6dd78f5c"),
])
def test_archived_prompt_fingerprint_matches_da1639a(edward, expected):
    from pacman_recipe.level1 import legacy_prompts_v1

    assert legacy_prompts_v1.prompt_contract_metadata("live_state_v3", edward_options=edward)["prompt_template_sha256"] == expected
    current = prompt_contract_metadata("live_state_v3", edward_options=edward)
    if edward:
        assert current["prompt_version"] == "edward-option-code-v2"
        assert current["prompt_template_sha256"] != expected
    else:
        # Current primitive actions share the explicit single-death game rules.
        assert current["prompt_template_sha256"] == "932fb0c84a8b4bd637f4386ca1eec0fd164dec9b4411bde5740ad0d63fc68cfe"


def test_native_rollout_records_actual_fallback_sampling_support():
    _, _, GConfig, Request = legacy.WorkflowContractTests._fake_objective_call_model_fixture()

    class Engine:
        async def agenerate(self, request):
            self.request = request
            selected = request.metadata["allowed_token_ids"][-1]
            return SimpleNamespace(
                input_tokens=[10, 11], output_tokens=[selected],
                output_logprobs=[-0.25], output_versions=[1],
                input_len=2, output_len=1, stop_reason="stop",
            )

    engine = Engine()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeObjectiveTokenizer()):
        workflow = PacmanNativeVisionWorkflow(
            gconfig=GConfig(), tokenizer=FakeObjectiveTokenizer(), tokenizer_path="test-tokenizer",
            processor=legacy.WorkflowContractTests._fake_native_processor(), env_factory=OneStepEnv,
            edward_options=True, edward_fallback_mode="risk_ranked",
            image_prompt_style="live_state_v3", reward_objective_contract="option_return_raw_v1",
        )
    with patch("pacman_recipe.level1.episode.EdwardPlanner", FallbackFirstPlanner), patch.dict(
        sys.modules, {
            "areal.api": SimpleNamespace(ModelRequest=Request),
            "areal.utils.data": SimpleNamespace(concat_padded_tensors=lambda samples: samples[0]),
        },
    ):
        sample = asyncio.run(workflow.arun_episode(engine, make_episode_row(1, split="test")))
    allowed = engine.request.metadata["allowed_token_ids"]
    assert allowed == [ord("J"), ord("K")]
    assert sample["input_ids"].tolist() == [[10, 11, ord("K")]]
    assert sample["pacman_allowed_token_ids"].tolist() == [[[0, 0], [0, 0], [ord("J") + 1, ord("K") + 1]]]
    assert sample["loss_mask"].tolist() == [[0, 0, 1]]
    assert sample["logprobs"].tolist() == [[0.0, 0.0, -0.25]]


def test_repeated_fallback_runs_until_real_episode_terminal():
    class NoNormalOptions(EdwardPlanner):
        def candidates(self, state):
            return ()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeObjectiveTokenizer()), patch(
        "pacman_recipe.level1.episode.EdwardPlanner", NoNormalOptions
    ):
        workflow = ChoosingWorkflow(
            tokenizer_path="test-tokenizer", edward_options=True,
            edward_fallback_mode="risk_ranked", image_prompt_style="live_state_v3",
        )
        workflow.supports = []
        asyncio.run(workflow.run(make_episode_row(1, split="test", max_steps=32)))
    payload = workflow.last_episode
    audit_trajectory(payload)
    assert payload["terminal_reason"] in {"death", "max_steps", "all_normal_pellets"}
    assert len(payload["trajectory"]) > 1
    assert all(step["model_called"] and step["option_step"] == 1 for step in payload["trajectory"])
    assert all(step["option_strategy"] == "RISK_FALLBACK" for step in payload["trajectory"])
    assert all(step["option_status"] == "max_commit" for step in payload["trajectory"][:-1])
    assert payload["trajectory"][-1]["option_status"] == "terminal"


def test_train_and_eval_kwargs_forward_mode():
    from train_areal import _build_workflow_kwargs
    config = legacy.TrainerGenerationContractTests._config()
    config.edward_options = True
    config.open_action_mask = False
    config.edward_fallback_mode = "risk_ranked"
    generation = SimpleNamespace(temperature=1, top_p=1, max_tokens=4096, max_new_tokens=1)
    for training in (True, False):
        kwargs = _build_workflow_kwargs(config, generation, training=training)
        assert kwargs["edward_fallback_mode"] == "risk_ranked"
        assert kwargs["recipe_contract"]["harness"]["edward_fallback_mode"] == "risk_ranked"


@pytest.mark.skipif(not os.getenv("PACMAN_TEST_PROCESSOR_PATH"), reason="requires local Qwen processor")
@pytest.mark.parametrize("strategy", ["normal_max", "RISK_FALLBACK"])
def test_real_processor_prompt_budget(strategy):
    from transformers import AutoProcessor
    from pacman_env.env import PygamePacmanEnv
    from pacman_env.planner import PlannerCandidate
    from pacman_recipe.level1.prompts import (
        build_image_messages, encode_png, edward_system_prompt, render_edward_decision_prompt,
    )

    processor = AutoProcessor.from_pretrained(os.environ["PACMAN_TEST_PROCESSOR_PATH"], local_files_only=True)
    env = PygamePacmanEnv()
    try:
        observation, _ = env.reset(seed=1)
        snapshot = env.snapshot()
    finally:
        env.close()
    risk_mode = strategy == "RISK_FALLBACK"
    identifiers = [f"A{i}" for i in range(4)] if risk_mode else [
        *[f"C{i}" for i in range(4)], *[f"A{i}" for i in range(4)], "E0", "E1",
    ]
    candidates = [PlannerCandidate(
        option_id=identifier,
        strategy=strategy if risk_mode else {"C": "COLLECT", "A": "AVOID", "E": "ELIMINATE"}[identifier[0]],
        target=(12, 13), first_action="UDLR"[index % 4], route_distance=1 if risk_mode else 12,
        commit_moves=1 if risk_mode else 6,
        safety_margin=None if risk_mode else 12, future_safe_exits=None if risk_mode else 3,
        entity_id=None if risk_mode else 3,
        risk={"rank": index + 1, "motion": "collision_predicted", "ghost_clearance": 12,
              "route_margin": -12, "safe_next_cells": 0, "dead_end": True, "reverse": True} if risk_mode else None,
    ) for index, identifier in enumerate(identifiers)]
    constraint = ObjectiveTokenConstraint.build(processor.tokenizer, identifiers)
    context = dict(pacman_position=[23, 13], facing="L", pellets_remaining=244,
                   maze_size=[snapshot["height"], snapshot["width"]], ghosts=snapshot["ghosts"],
                   edible_ticks=192, last_action="R")
    messages = build_image_messages(encode_png(observation))
    messages[0]["content"] = edward_system_prompt("risk_ranked")
    messages[1]["content"][0]["text"] = render_edward_decision_prompt(
        context, candidates, constraint, fallback_mode="risk_ranked",
    )
    workflow = PacmanNativeVisionWorkflow.__new__(PacmanNativeVisionWorkflow)
    workflow.processor = processor
    _, _, _, ids = workflow._process_messages(messages)
    print(f"{strategy}: {len(ids)} input tokens (limit=1024, completion=1)")
    assert len(ids) < 1024
    from scripts.level1.train.check_edward_prompt_budget import measure
    worst = measure(processor, fallback_mode="risk_ranked", risk_fallback=risk_mode)
    print(f"{strategy} conservative bound: {worst['input_tokens']} input tokens")
    assert worst["input_tokens"] < 1024
