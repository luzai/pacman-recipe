"""Exercise rename compatibility and the versioned, actual Edward prompt."""

from __future__ import annotations

import base64
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pacman_env.paths import configured_path, pacman_python_root
from pacman_recipe.level1 import legacy_prompts_v1, prompts
from pacman_recipe.level1.recipe import prompt_contract
from pacman_recipe.level1.trajectories import _audit_prompt_evidence
from pacman_recipe.level1.image_transport import pil_and_chat_messages


@pytest.mark.parametrize("suffix", ["actions", "env", "env.state", "env.config", "env.level", "env._pygame_worker", "planner"])
def test_old_environment_imports_preserve_type_identity(suffix):
    old = importlib.import_module("maapacman." + suffix)
    new = importlib.import_module("pacman_env." + suffix)
    if suffix == "env":
        assert old.Action is new.Action
        assert old.PygamePacmanEnv is new.PygamePacmanEnv
    else:
        assert old is new


@pytest.mark.parametrize("suffix", ["actions", "level1.prompts", "level1.recipe", "level1.rewards", "level1.token_constraints", "synthetic.env"])
def test_old_recipe_imports_forward_to_one_implementation(suffix):
    assert importlib.import_module("areal_pacman." + suffix) is importlib.import_module("pacman_recipe." + suffix)


def test_path_aliases_accept_equivalent_values_and_reject_conflicts(tmp_path, monkeypatch):
    monkeypatch.setenv("PACMAN_RECIPE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_PACMAN_ROOT", str(tmp_path / "."))
    assert configured_path("PACMAN_RECIPE_ROOT", "AREAL_PACMAN_ROOT") == str(tmp_path.resolve())
    monkeypatch.setenv("AREAL_PACMAN_ROOT", str(tmp_path / "different"))
    with pytest.raises(ValueError, match="Conflicting path variables"):
        configured_path("PACMAN_RECIPE_ROOT", "AREAL_PACMAN_ROOT")


def test_old_game_path_is_still_accepted(tmp_path, monkeypatch):
    for name in ("PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAAPACMAN_PACMAN_PYTHON_ROOT", str(tmp_path))
    assert pacman_python_root() == str(tmp_path.resolve())


def test_prompt_v2_changes_identity_without_changing_action_protocol():
    old = legacy_prompts_v1.prompt_contract_metadata("live_state_v3", edward_options=True)
    new = prompts.prompt_contract_metadata("live_state_v3", edward_options=True)
    assert old["action_protocol"] == new["action_protocol"] == "edward-option-code-v1"
    assert new["prompt_version"] == "edward-option-code-v2"
    assert old["prompt_template_sha256"] != new["prompt_template_sha256"]
    assert "MaaPacman" in prompts.EDWARD_OPTION_CODE_V1_SYSTEM_PROMPT
    system = prompts.edward_system_prompt()
    assert "MaaPacman" not in system
    assert "all normal pellets" in system.lower() and "power pellets are optional" in system
    assert "The episode ends on the first death." in system
    assert "COLLECT" in system and "AVOID" in system and "ELIMINATE" in system
    assert "uppercase option code" in system
    assert "do not infer flashing from a single screenshot" in system
    assert "its structured state is vulnerable" in system
    assert "edible_ticks allows a safe interception" in system
    assert "deep-blue or flashing" not in system
    with pytest.raises(ValueError, match="prompt_version"):
        prompt_contract({"image_prompt_style": "live_state_v3", "edward_options": True, "prompt_version": "edward-option-code-v1"})


@pytest.mark.parametrize("module", [legacy_prompts_v1, prompts])
def test_archived_and_current_prompt_evidence_use_their_own_renderer(module):
    candidate = dict(option_id="C0", strategy="COLLECT", target=[1, 2], first_action="R", route_distance=1, commit_moves=1, safety_margin=10, future_safe_exits=2, entity_id=None)
    context = {"ghosts": [], "edible_ticks": 0, "option_code_map": {"B": "C0"}, "planner_candidates": [candidate], "episode_life_mode": "single_death"}
    constraint = SimpleNamespace(code_for_option=lambda _: "B", rendered_choices=("B",))
    user = module.render_edward_decision_prompt(context, [SimpleNamespace(**candidate)], constraint)
    system = module.edward_system_prompt()
    image_hash = "0" * 64
    step = {"model_called": True, "model_system_prompt": system, "model_user_instruction": user, "model_user_prompt_sha256": module.text_sha256(user), "sent_prompt_sha256": module.sent_prompt_sha256(system, user, image_hash), "observation_png_sha256": image_hash, "observation_context": context}
    payload = {
        "decoding": {"edward_options": True}, "image_prompt_style": "live_state_v3",
        **module.prompt_contract_metadata("live_state_v3", edward_options=True),
        "system_prompt": system, "ghost_mode": "disabled", "trajectory": [step],
    }
    _audit_prompt_evidence(payload)
    # Even valid hashes cannot make a modified prompt agree with its renderer.
    step["model_user_instruction"] += "tampered"
    step["model_user_prompt_sha256"] = module.text_sha256(step["model_user_instruction"])
    step["sent_prompt_sha256"] = module.sent_prompt_sha256(system, step["model_user_instruction"], image_hash)
    with pytest.raises(ValueError, match="observation context"):
        _audit_prompt_evidence(payload)


def test_latest_image_is_the_only_image_in_each_request():
    pngs = [prompts.encode_png(np.full((16, 16, 3), value, dtype=np.uint8)) for value in (0, 255)]
    context = {"pacman_position": [1, 1], "maze_size": [25, 21], "open_actions": ["R"], "blocked_actions": ["U", "D", "L"], "pellets_remaining": 10}
    messages = [prompts.build_image_messages(png, prompt_style="live_state_v3", state_context=context) for png in pngs]
    for png, request in zip(pngs, messages, strict=True):
        assert len(request) == 2
        assert [message["role"] for message in request] == ["system", "user"]
        assert prompts.image_count(request) == 1
        images = [part for message in request if isinstance(message["content"], list) for part in message["content"] if part["type"] == "image_url"]
        raw = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
        assert hashlib.sha256(raw).digest() == hashlib.sha256(png).digest()
    assert messages[0] != messages[1]
    messages[1][1]["content"].append(messages[0][1]["content"][-1])
    with pytest.raises(ValueError, match="exactly one current image"):
        pil_and_chat_messages(messages[1])


def test_actual_option_prompt_uses_shared_single_death_rule():
    candidate = SimpleNamespace(option_id="C0", strategy="COLLECT", target=(1, 2), first_action="R", route_distance=1, commit_moves=1, safety_margin=10, future_safe_exits=2, entity_id=None)
    constraint = SimpleNamespace(code_for_option=lambda _: "B", rendered_choices=("B",))
    text = prompts.compact_edward_decision_prompt({"episode_life_mode": "single_death"}, [candidate], constraint)
    assert "The episode ends on the first death." in prompts.edward_system_prompt()
    assert "life_mode" not in text
    assert "Return exactly one code from [B]" in text
