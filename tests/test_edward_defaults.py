"""New-run defaults and explicit historical compatibility, without AReaL."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from baselines.decisions_vision.native import NativeDecisionRunner, runner_options, validate_eval_row
from pacman_env.planner import EdwardPlanner, resolve_fallback_mode
from pacman_recipe.level1.level1_dataset import make_episode_row
from pacman_recipe.level1.prompts import prompt_contract_metadata
from pacman_recipe.level1.recipe import recipe_contract_metadata
from pacman_recipe.level1.trajectories import audit_trajectory
from slime_pacman.config import PacmanConfig


def test_contextual_defaults_and_explicit_compatibility():
    assert EdwardPlanner().fallback_mode == "risk_ranked"
    assert EdwardPlanner(fallback_mode="refuse").fallback_mode == "refuse"
    assert resolve_fallback_mode(edward_options=False) == "refuse"
    with pytest.raises(ValueError):
        resolve_fallback_mode("risk_ranked", edward_options=False)
    assert runner_options("edward")["edward_fallback_mode"] == "risk_ranked"
    assert runner_options("primitive")["edward_fallback_mode"] == "refuse"
    assert PacmanConfig().edward_fallback_mode == "risk_ranked"
    assert PacmanConfig(edward_fallback_mode="refuse").edward_fallback_mode == "refuse"
    primitive = PacmanConfig(action_mode="primitive", observation_mode="ascii",
                             ascii_map_format="spaced", groups_per_update=16, clip=.05)
    assert primitive.edward_fallback_mode == "refuse"


@pytest.mark.parametrize("style", ["live_state_v3", "ascii_edward_v1", "ascii_edward_spaced_v1"])
def test_new_recipe_default_equals_explicit_risk_and_legacy_is_distinct(style):
    raw = dict(edward_options=True, image_prompt_style=style)
    default = recipe_contract_metadata(raw)
    assert default == recipe_contract_metadata({**raw, "edward_fallback_mode": "risk_ranked"})
    assert default == recipe_contract_metadata({**raw, "edward_fallback_mode": None})
    legacy = recipe_contract_metadata({**raw, "edward_fallback_mode": "refuse"})
    assert "edward_fallback_mode" not in legacy["harness"]
    assert legacy["prompt"]["template_sha256"] != default["prompt"]["template_sha256"]


@pytest.mark.parametrize("flag,expected", [(None, "risk_ranked"), ("refuse", "refuse"), ("risk_ranked", "risk_ranked")])
def test_baseline_cli_resolves_both_modes(tmp_path, monkeypatch, flag, expected):
    from baselines.decisions_vision import native
    args = ["native", "--mode", "both", "--output", str(tmp_path / "out")]
    if flag:
        args += ["--fallback", flag]
    monkeypatch.setattr(sys, "argv", args)
    calls = []

    async def run(mode, seed, max_steps, single_step, live, key, output, fallback, **kwargs):
        calls.append((mode, resolve_fallback_mode(fallback, edward_options=mode == "edward")))
        return dict(mode=mode, status="complete")

    monkeypatch.setattr(native, "run_one", run)
    assert native.main() == 0
    assert calls == [("primitive", "refuse"), ("edward", expected)]


@pytest.mark.parametrize("config_mode,expected", [(None, "risk_ranked"), ("refuse", "refuse")])
def test_budget_cli_checks_resolved_scenarios(tmp_path, monkeypatch, config_mode, expected):
    from scripts.level1.train import check_edward_prompt_budget as budget
    config = tmp_path / "config.yaml"
    config.write_text("edward_options: true\n" + (
        f"edward_fallback_mode: {config_mode}\n" if config_mode else ""))
    monkeypatch.setattr(sys, "argv", ["budget", "--model-path", str(tmp_path), "--config", str(config)])
    processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor)))
    calls = []

    def measure(processor, *, fallback_mode, risk_fallback=False):
        assert fallback_mode == expected
        calls.append(risk_fallback)
        return {"input_tokens": 100}

    monkeypatch.setattr(budget, "measure", measure)
    budget.main()
    assert calls == ([False, True] if expected == "risk_ranked" else [False])


@pytest.mark.parametrize("mode,fallback", [("edward", None), ("edward", "refuse"), ("primitive", None)])
def test_episode_record_default_matches_config(tmp_path, monkeypatch, mode, fallback):
    from pacman_recipe.level1 import contracts
    from pacman_env.paths import pacman_python_root
    identity = dict(commit="a" * 40, source_sha256="b" * 64, dirty=False)
    monkeypatch.setattr(contracts, "repository_identity", lambda _: identity.copy())
    record = contracts.make_episode_record(
        0, split="test", recipe_root=tmp_path, game_root=pacman_python_root(),
        backend_root=tmp_path, max_steps=32, observation_mode="ascii",
        ascii_map_format="spaced", action_mode=mode, edward_fallback_mode=fallback)
    contracts.validate_episode_record(record)
    config = PacmanConfig(max_steps=32, observation_mode="ascii", ascii_map_format="spaced",
                          groups_per_update=16, clip=.05, action_mode=mode,
                          edward_fallback_mode=fallback)
    from slime_pacman.config import runner_options as slime_options
    options = slime_options(config)
    assert record["prompt"] == prompt_contract_metadata(
        options["image_prompt_style"], edward_options=mode == "edward",
        fallback_mode=config.edward_fallback_mode)


class FallbackFirstPlanner(EdwardPlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = True

    def candidates(self, state):
        return () if self.first else super().candidates(state)

    def advertised_candidates(self, state):
        result = super().advertised_candidates(state)
        self.first = False
        return result


@pytest.mark.parametrize("fallback", [None, "refuse"])
def test_real_native_runner_resolves_missing_mode_and_audits(fallback):
    runner = NativeDecisionRunner(mode="edward", fallback=fallback, row_validator=validate_eval_row)
    # Exercise the common harness's omitted-option default as well as explicit refuse.
    if fallback is None:
        runner.workflow_kwargs.pop("edward_fallback_mode")
    row = make_episode_row(1, split="test", seed=0, max_steps=32)
    with patch("pacman_recipe.level1.episode.EdwardPlanner", FallbackFirstPlanner):
        asyncio.run(runner.run(row))
    payload = runner.last_episode
    audit_trajectory(payload)
    first = payload["trajectory"][0]
    if fallback is None:
        assert payload["decoding"]["edward_fallback_mode"] == "risk_ranked"
        assert first["option_strategy"] == "RISK_FALLBACK"
        assert first["option_status"] == "max_commit"
        assert first["option_step"] == 1
        assert payload["trajectory"][1]["model_called"]
        assert payload["trajectory"][1]["option_strategy"] != "RISK_FALLBACK"
    else:
        assert "edward_fallback_mode" not in payload["decoding"]
        assert first["option_strategy"] != "RISK_FALLBACK"


# Captured before this default change; explicit modes must retain their identities.
PROMPT_PINS = {'live_state_v3': {'refuse': '05bbb88f07e5e84dc5ce3e92b75ea85d1977e8a9210f49d02c81d2e0dffc3650',
                   'risk_ranked': 'ba71f73244cb8dcc2eb69ff21be6af2fa625971020907f8cdda707933e19fedd'},
 'ascii_edward_v1': {'refuse': '46d21fe542dfe878aab5f5b7d370d94679e6ce4485deab9bf7b6b99c5f2dafc1',
                     'risk_ranked': '8710d8373b61c53b3d7724e1cc869cc3faf00f60ebf4db6b6f73ef55dd3994e3'},
 'ascii_edward_spaced_v1': {'refuse': 'ab18c47c25c109cdb2a7234d8add97890306091fa27d59199d5fc0a5861366b4',
                            'risk_ranked': '6aed272667b7971e65e9f9ec0655f5358f1718c2335f8e8d1a9e85d918fe7c9d'}}

@pytest.mark.parametrize("style", PROMPT_PINS)
@pytest.mark.parametrize("mode", ["refuse", "risk_ranked"])
def test_existing_explicit_mode_prompt_fingerprints_unchanged(style, mode):
    assert prompt_contract_metadata(style, edward_options=True, fallback_mode=mode)["prompt_template_sha256"] == PROMPT_PINS[style][mode]
