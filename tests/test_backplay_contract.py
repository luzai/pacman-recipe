from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from train_areal import (
    _validate_backplay_contract,
    _validate_release_stage_contract,
    _validate_reward_objective_contract,
)


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


def config():
    path = Path(__file__).parents[1] / "configs/level1/experiments/backplay.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["actor"]["reward_norm"]["group_size"] = 12
    raw["tokenizer_path"] = raw["actor"]["path"]
    raw["rollout"]["tokenizer_path"] = raw["actor"]["path"]
    raw["ref"]["path"] = raw["actor"]["path"]
    return namespace(raw)


def test_primitive_episode_group_requires_explicit_backplay_contract():
    c = config()
    _validate_reward_objective_contract(c)
    _validate_release_stage_contract(c)
    c.backplay_experiment = False
    with pytest.raises(ValueError, match="edward_options"):
        _validate_reward_objective_contract(c)


@pytest.mark.parametrize("field,value", [
    ("edward_options", True), ("open_action_mask", False),
    ("recipe_version", "maapacman-level1-ghostdoor-v3"),
    ("reward_objective_contract", "step_local_raw_v1"),
])
def test_backplay_does_not_relax_unrelated_contracts(field, value):
    c = config()
    setattr(c, field, value)
    with pytest.raises(ValueError, match="Backplay requires"):
        _validate_backplay_contract(c)


def test_backplay_requires_current_sampled_policy():
    c = config()
    c.gconfig.greedy = True
    with pytest.raises(ValueError, match="sampled groups"):
        _validate_backplay_contract(c)
    c = config()
    c.rollout.max_head_offpolicyness = 1
    with pytest.raises(ValueError, match="current policy"):
        _validate_backplay_contract(c)


def test_restart_row_requires_complete_identity():
    from pacman_recipe.level1.level1_dataset import make_episode_row, validate_episode_row
    row = make_episode_row(1, split="train", action_protocol="direct-open-action-token-v1")
    row["restart_state_path"] = "/immutable/state.json"
    with pytest.raises(ValueError, match="together"):
        validate_episode_row(row)
    row.update(restart_state_sha256="a" * 64, restart_state_id="teacher-0-step-32")
    validate_episode_row(row)
    row["restart_state_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="SHA256"):
        validate_episode_row(row)
