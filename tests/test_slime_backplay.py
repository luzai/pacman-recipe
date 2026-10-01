from copy import deepcopy

import pytest

from pacman_recipe.level1.contracts import runner_row, validate_episode_record
from pacman_recipe.level1.prompts import prompt_contract_metadata
from slime_pacman.backplay import bind_restart_record


def record():
    return dict(schema="pacman-episode-v1", id="test", split="train",
                training_backend="slime", source_revisions={
                    name: dict(commit="a" * 40, source_sha256="b" * 64, dirty=False)
                    for name in ("pacman-recipe", "pacman-python", "slime")},
                prompt=prompt_contract_metadata("live_state_v3", edward_options=True),
                environment=dict(seed=0, max_steps=512, ghost_mode="normal",
                                 episode_life_mode="single_death"))


def test_restart_fields_reach_options_runner(tmp_path):
    source = record()
    source["restart"] = dict(path=str(tmp_path / "state.json"), sha256="c" * 64, id="restart-1")
    row = runner_row(source)
    assert row["restart_state_path"] == source["restart"]["path"]
    assert row["restart_state_sha256"] == "c" * 64
    assert row["restart_state_id"] == "restart-1"
    assert row["action_protocol"] == source["prompt"]["action_protocol"]
    assert row["state_prefix_actions"] == []


@pytest.mark.parametrize("restart", [{}, None, dict(path="relative", sha256="a" * 64, id="x"),
                                    dict(path="/state", sha256="short", id="x")])
def test_malformed_restart_fails(restart):
    source = record()
    source["restart"] = restart
    with pytest.raises(ValueError):
        validate_episode_record(source)


def test_bind_preserves_template_and_rejects_three_lives(monkeypatch, tmp_path):
    entry = dict(restart_state_id="restart-1", seed=4, state_path="state.json", state_file_sha256="c" * 64)
    saved = dict(payload=dict(identity=dict(max_steps=512, episode_life_mode="single_death")))
    monkeypatch.setattr("slime_pacman.backplay.load_restart_bank", lambda _: dict(
        restart_states=[entry], environment=dict(ghost_mode="normal", episode_life_mode="single_death", max_steps=512)))
    monkeypatch.setattr("slime_pacman.backplay.load_restart_state", lambda *_: saved)
    source = record()
    before = deepcopy(source)
    bound = bind_restart_record(source, tmp_path, "restart-1")
    assert source == before
    assert bound["environment"]["seed"] == 4
    assert bound["id"] == "restart-1"
    saved["payload"]["identity"]["episode_life_mode"] = "original_three_lives"
    with pytest.raises(ValueError, match="episode_life_mode"):
        bind_restart_record(source, tmp_path, "restart-1")


def test_real_options_restart_preserves_original_horizon(tmp_path):
    import asyncio
    import hashlib
    import json
    from pathlib import Path
    from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
    from pacman_recipe.level1.contracts import make_episode_record
    from slime_pacman.config import PacmanConfig
    from slime_pacman.generation import Decision
    from slime_pacman.rollout import EpisodeRunner
    from test_slime_pacman import Tokenizer

    root = Path(__file__).resolve().parents[1]
    source = make_episode_record(28, split="test", recipe_root=root,
                                game_root=root.parent / "pacman-python",
                                backend_root=root.parent / "slime", max_steps=32)
    with PygamePacmanEnv(PygamePacmanEnvConfig(max_steps=32)) as env:
        env.reset(seed=28)
        for _ in range(4):
            _, _, terminal, truncated, _ = env.step(env.snapshot()["open"][0])
            assert not (terminal or truncated)
        saved = env.save_state()
    path = tmp_path / "restart.json"
    raw = json.dumps(saved).encode()
    path.write_bytes(raw)
    source["restart"] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                             id="restart-" + saved["sha256"])

    async def generate(messages, constraint):
        assert len(messages) == 2
        token = constraint.allowed_token_ids[0]
        return Decision("prompt", [1, 2], token, chr(token), constraint.allowed_token_ids,
                        0., "v0", {}, "0" * 64, [])

    runner = EpisodeRunner(source, tokenizer=Tokenizer(), generate=generate,
                           config=PacmanConfig(max_steps=32))
    result = asyncio.run(runner.collect(empty_weight_version="v0"))
    episode = result.trajectory["episode"]
    assert episode["restart_state"]["source_step"] == 4
    assert episode["restart_state"]["remaining_budget"] == 28
    assert episode["trajectory"][0]["env_step"] == 5
    assert result.reward in (0., 1.)
