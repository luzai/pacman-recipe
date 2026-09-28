"""Structural fake-processor tests; not a real model-processor budget verdict."""

from __future__ import annotations

import ast
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pacman_recipe.level1.prompts import build_image_messages, encode_png
from scripts.level1.train import check_direct_prompt_budget as budget


class FakeProcessor:
    """Only exercise token-count plumbing, never estimate real Qwen tokens."""

    def __init__(self, lengths=(321,)):
        self.lengths = lengths
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        assert messages[0]["role"] == "system"
        assert "one action letter" in messages[0]["content"]
        assert [part["type"] for part in messages[1]["content"]] == ["text", "image"]
        assert messages[1]["content"][1]["image"].size == (336, 400)
        return messages[0]["content"] + messages[1]["content"][0]["text"]

    def __call__(self, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        assert kwargs["return_tensors"] == "pt"
        assert "max_length" not in kwargs
        assert len(kwargs["images"]) == len(kwargs["text"]) == 1
        count = self.lengths[len(self.calls) % len(self.lengths)]
        self.calls.append(kwargs)
        ids = np.zeros((1, count), dtype=np.int64)
        return {
            "input_ids": ids,
            "pixel_values": np.zeros((1, 3)),
            "mm_token_type_ids": ids,
        }


def fixture_image():
    return np.zeros(budget.OBSERVATION_SHAPE, dtype=np.uint8)


def test_conservative_contexts_cover_bounded_direct_fields():
    cases = budget.conservative_contexts()
    assert len({label for label, _ in cases}) == len(cases)
    contexts = [context for _, context in cases]
    assert len({tuple(c["open_actions"]) for c in contexts}) == 15
    assert (
        len(
            {
                tuple(c["current_cell_exit_history"])
                for c in contexts
                if len(c["current_cell_exit_history"]) == 4
            }
        )
        == 24
    )
    assert {len(c["current_cell_exit_history"]) for c in contexts} == {0, 1, 2, 3, 4}
    assert {c["last_action"] for c in contexts} == {None, "U", "D", "L", "R"}
    assert {c["facing"] for c in contexts} == {"S", "U", "D", "L", "R"}
    assert {c["pellets_remaining"] for c in contexts} == {0, 1, 9, 10, 99, 100, 196}
    for context in contexts:
        assert context["ghosts"] == [] and context["edible_ticks"] == 0
        assert set(context["open_actions"]) | set(context["blocked_actions"]) == set(
            budget.MOVEMENT_ACTIONS
        )
        assert not set(context["open_actions"]) & set(context["blocked_actions"])
        assert context["preferred_open_actions"] == context["open_actions"]
        assert "planner_candidates" not in context


def test_conversion_matches_actual_native_workflow_without_importing_runtime():
    source = Path(__file__).resolve().parents[1] / "pacman_recipe/level1/workflow.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PacmanNativeVisionWorkflow"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pil_and_chat_messages"
    )
    method.decorator_list = []
    namespace = {
        "Any": Any, "base64": base64, "BytesIO": BytesIO,
        "__name__": "pacman_recipe.level1.workflow",
        "__package__": "pacman_recipe.level1",
    }
    exec(  # noqa: S102 - execute only the checked-in pure conversion method, not runtime imports.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    png = encode_png(fixture_image())
    context = budget.conservative_contexts()[0][1]
    expected_image, expected = namespace["_pil_and_chat_messages"](
        build_image_messages(png, prompt_style="live_state_v3", state_context=context)
    )
    image, actual, text = budget.native_messages(png, context)
    assert actual[0] == expected[0]
    assert actual[1]["content"][0] == expected[1]["content"][0]
    assert text == actual[1]["content"][0]["text"]
    assert np.array_equal(np.asarray(image), np.asarray(expected_image))
    assert np.array_equal(
        np.asarray(actual[1]["content"][1]["image"]),
        np.asarray(expected[1]["content"][1]["image"]),
    )


@pytest.mark.parametrize("count,passed", [(1023, True), (1024, False), (1500, False)])
def test_actual_returned_ids_not_character_estimates_control_budget(count, passed):
    processor = FakeProcessor((17, count))
    result = budget.measure(processor, fixture_image())
    assert result["input_tokens"] == count
    assert result["passed"] is passed
    assert result["headroom_tokens"] == 1024 - count
    assert len(processor.calls) == len(budget.conservative_contexts())
    assert result["prompt_contract"]["action_protocol"] == "direct-open-action-token-v1"
    assert result["truncation"] is False
    assert "not exhaustive" in result["coverage"]
    assert "no model inference" in result["scope"]


@pytest.mark.parametrize("missing", ["input_ids", "pixel_values", "mm_token_type_ids"])
def test_missing_native_vlm_outputs_fail_closed(missing):
    class IncompleteProcessor(FakeProcessor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.pop(missing)
            return result

    with pytest.raises(ValueError, match="processor omitted"):
        budget.measure(IncompleteProcessor(), fixture_image())


@pytest.mark.parametrize("shape", [(5,), (2, 5), (1, 0)])
def test_invalid_single_request_ids_fail_closed(shape):
    class InvalidProcessor(FakeProcessor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result["input_ids"] = np.zeros(shape)
            return result

    with pytest.raises(ValueError, match="input_ids shape"):
        budget.measure(InvalidProcessor(), fixture_image())


def test_nonproduction_image_or_invalid_budget_fails():
    with pytest.raises(ValueError, match="RGB observation"):
        budget.measure(FakeProcessor(), np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="positive"):
        budget.measure(FakeProcessor(), fixture_image(), max_input_tokens=0)


def test_real_headless_capture_never_constructs_edward(monkeypatch):
    from pacman_env import planner

    def forbidden(*args, **kwargs):
        raise AssertionError("C1 preflight must never construct or call Edward")

    for method in (
        "__init__",
        "candidates",
        "advertised_candidates",
        "decide",
        "continue_option",
    ):
        monkeypatch.setattr(planner.EdwardPlanner, method, forbidden)
    image, source = budget.capture_c1_image()
    assert image.shape == budget.OBSERVATION_SHAPE
    assert image.dtype == np.uint8 and np.any(image)
    assert source["ghost_mode"] == "disabled"
    assert source["source"] == "headless-pacman-python-renderer"
    # This remains a fake-processor structural check, even with a real image.
    assert budget.measure(FakeProcessor(), image)["passed"] is True


@pytest.mark.parametrize("tokens", [1023, 1024, 1500])
def test_cli_loads_local_processor_and_exits_nonzero_on_overflow(
    tmp_path, monkeypatch, capsys, tokens
):
    loaded = []

    def from_pretrained(path, **kwargs):
        loaded.append((path, kwargs))
        return FakeProcessor((tokens,))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    monkeypatch.setattr(
        budget,
        "capture_c1_image",
        lambda root: (fixture_image(), {"source": "explicit-test-double"}),
    )
    monkeypatch.setattr(
        sys, "argv", ["check_direct_prompt_budget.py", "--model-path", str(tmp_path)]
    )
    if tokens >= 1024:
        with pytest.raises(
            SystemExit, match="no automatic truncation or recipe change"
        ):
            budget.main()
    else:
        budget.main()
    result = json.loads(capsys.readouterr().out)
    assert result["input_tokens"] == tokens
    assert result["passed"] is (tokens < 1024)
    assert result["image_source"]["source"] == "explicit-test-double"
    assert loaded == [(tmp_path, {"local_files_only": True})]


@pytest.mark.parametrize("invalid", [["--max-input-tokens", "0"], []])
def test_cli_rejects_invalid_budget_or_missing_local_processor(
    tmp_path, monkeypatch, invalid
):
    path = tmp_path if invalid else tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["budget", "--model-path", str(path), *invalid])
    with pytest.raises(SystemExit) as exc:
        budget.main()
    assert exc.value.code == 2
