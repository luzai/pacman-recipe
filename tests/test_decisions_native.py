import asyncio
import base64
import copy
import json
from unittest.mock import patch

import pytest

from baselines.decisions_vision.client import DecisionError, MODEL, parse_answer
from baselines.decisions_vision.native import (
    NativeDecisionRunner, SymbolCodec, metrics, native_payload, validate_eval_row,
)
from pacman_recipe.level1.episode import ModelTurn, PacmanEpisodeRunner
from pacman_recipe.level1.level1_dataset import make_episode_row
from pacman_recipe.level1.token_constraints import ObjectiveTokenConstraint
from pacman_recipe.level1.trajectories import audit_trajectory


def smoke_row():
    row = make_episode_row(1, split="test", seed=0, max_steps=32)
    row.update(decision_steps=1, baseline_smoke_only=True)
    return row


def test_native_transport_preserves_roles_text_images_order():
    messages = [{"role": "system", "content": "exact original rules"},
        {"role": "user", "content": [
            {"type": "text", "text": "fixed prefix"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
            {"type": "text", "text": "dynamic state"}]}]
    original = copy.deepcopy(messages)
    body = native_payload(messages, ["B", "J"])
    assert messages == original
    assert [m["role"] for m in body["input"]] == ["system", "user"]
    assert body["input"][0]["content"][0]["text"] == "exact original rules"
    parts = body["input"][1]["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image", "input_text"]
    assert parts[0]["text"] == "fixed prefix" and parts[2]["text"] == "dynamic state"
    assert base64.b64decode(parts[1]["image_url"].split(",")[1]) == b"abc"
    assert [c["value"] for c in body["questions"][0]["choices"]] == ["B", "J"]


def test_edward_code_mapping_is_native():
    constraint = ObjectiveTokenConstraint.build(SymbolCodec(), ["C0", "A0", "E0"])
    assert constraint.rendered_choices == ("B", "J", "P")
    assert constraint.option_for_completion("J") == "A0"


@pytest.mark.parametrize("mode", ["primitive", "edward"])
def test_singleton_never_adds_choice_or_calls_api(mode):
    runner = NativeDecisionRunner(mode=mode, live=True, sender=lambda *a, **k: pytest.fail("API called"))
    constraint = ObjectiveTokenConstraint.build(SymbolCodec(), ["C0"]) if mode == "edward" else None
    turn = asyncio.run(runner._call_model([{"role": "user", "content": "exact"}],
        objective_constraint=constraint, current_open_actions=["L"]))
    assert turn.completion == ("B" if mode == "edward" else "L")
    assert runner.decisions[0]["source"] == "forced_singleton"


def test_invalid_transport_stops_and_records_failure(tmp_path):
    def failed(*a, **k):
        raise DecisionError("HTTP status: 401; no retry")
    runner = NativeDecisionRunner(mode="primitive", live=True, output=tmp_path, sender=failed)
    with pytest.raises(DecisionError):
        asyncio.run(runner._call_model([{"role": "user", "content": "exact"}], current_open_actions=["L", "R"]))
    assert runner.decisions[0]["status"] == "request_failed"
    assert json.loads((tmp_path / "decisions.json").read_text())[0]["status"] == "request_failed"


@pytest.mark.parametrize("choice", [[], {}, 2, True, "S"])
def test_malformed_or_out_of_support_choice_rejected(choice):
    result = {"model": MODEL, "answers": [{"name": "action", "type": "choice",
        "choice": choice, "confidence": 0.5}]}
    with pytest.raises(DecisionError):
        parse_answer(result, ["L", "R"])


def test_smoke_row_keeps_production_validation():
    row = smoke_row()
    validate_eval_row(row)
    row["env"]["max_steps"] = 1
    with pytest.raises(ValueError):
        validate_eval_row(row)


@pytest.mark.parametrize("mode", ["primitive", "edward"])
@pytest.mark.parametrize("single_step", [True, False])
def test_native_prompt_and_step_match_reference_runner(mode, single_step, tmp_path):
    adapter = NativeDecisionRunner(mode=mode, output=tmp_path, row_validator=validate_eval_row)
    row = smoke_row() if single_step else make_episode_row(1, split="test", seed=0, max_steps=32)
    asyncio.run(adapter.run(row))
    reference_messages = []
    async def choose_reference(self, messages, **options):
        reference_messages.append(messages)
        constraint = options.get("objective_constraint")
        code = constraint.rendered_choices[0] if constraint else options["current_open_actions"][0]
        return ModelTurn(code, "reference", messages)
    reference = PacmanEpisodeRunner(tokenizer=SymbolCodec(), row_validator=validate_eval_row,
                                   **adapter.workflow_kwargs)
    with patch.object(PacmanEpisodeRunner, "_call_model", choose_reference):
        asyncio.run(reference.run(row))
    assert len(reference_messages) == len(adapter.decisions)
    for i, expected_messages in enumerate(reference_messages):
        actual_messages = json.loads((tmp_path / f"decision{i:04d}_native_messages.json").read_text())
        assert actual_messages == expected_messages
    assert len(adapter.last_episode["trajectory"]) == len(reference.last_episode["trajectory"])
    for actual, expected in zip(adapter.last_episode["trajectory"], reference.last_episode["trajectory"]):
        for field in ("action", "score", "observation_context", "option_code_map",
                      "model_user_instruction", "option_id", "option_status", "wall_collision"):
            assert actual[field] == expected[field], field
    audit_trajectory(adapter.last_episode)
    assert metrics(adapter, mode, 0)["api_calls"] == 0


def test_edward_multi_step_native_continuation_and_replanning():
    runner = NativeDecisionRunner(mode="edward")
    asyncio.run(runner.run(make_episode_row(1, split="test", seed=0, max_steps=32)))
    episode = runner.last_episode
    audit_trajectory(episode)
    steps = episode["trajectory"]
    assert len(steps) > 1
    assert any(not s["model_called"] for s in steps)
    assert sum(s["model_called"] for s in steps) == len(runner.decisions)
    assert sum(s["model_called"] for s in steps) > 1
    for prev, current in zip(steps, steps[1:]):
        if not current["model_called"]:
            assert not prev["option_end"]
            assert current["completion_id"] == prev["completion_id"]
            assert current["option_step"] == prev["option_step"] + 1
            assert current["option_id"] == prev["option_id"]
        else:
            assert prev["option_end"]
