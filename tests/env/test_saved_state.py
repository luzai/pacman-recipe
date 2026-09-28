"""Real pygame continuation tests; no mocked transition engine."""

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from maapacman.env import (
    EpisodeFinishedError, InvalidConfigurationError,
    PygamePacmanEnv, PygamePacmanEnvConfig,
)
from maapacman.planner import EdwardPlanner

ROOT = Path(os.environ.get("MAAPACMAN_PACMAN_PYTHON_ROOT") or
            Path(__file__).resolve().parents[3] / "pacman-python")


def make_env(**kwargs):
    return PygamePacmanEnv(PygamePacmanEnvConfig(pacman_python_root=ROOT, **kwargs))


def transition_record(result):
    frame, reward, terminated, truncated, info = result
    # A fresh worker has its own temporary directory; gameplay must be identical.
    info = {k: v for k, v in info.items() if k != "worker_runtime_id"}
    return hashlib.sha256(frame.tobytes()).hexdigest(), reward, terminated, truncated, info


def saved_rng(saved):
    graph = saved["payload"]["worker"]["graph"]
    roots = dict(graph["nodes"][graph["root"]["ref"]]["items"])

    def unpack(value):
        if not isinstance(value, dict):
            return value
        return tuple(unpack(v) for v in graph["nodes"][value["ref"]]["items"])

    return unpack(roots["rng"])


@pytest.mark.parametrize("seed", [0, 7, 11])
@pytest.mark.parametrize("ghost_mode", ["normal", "disabled"])
def test_json_checkpoint_fresh_worker_and_same_worker(seed, ghost_mode, tmp_path):
    with make_env(ghost_mode=ghost_mode) as env:
        env.reset(seed=seed)
        planner = EdwardPlanner()
        for _ in range(8):
            result = env.step(planner.decide(env.snapshot()).action)
            assert not (result[2] or result[3])
        saved = env.save_state()
        frame = env.render()
        state = copy.deepcopy(env.snapshot())
        path = tmp_path / "state.json"
        path.write_text(json.dumps(saved))
        loaded = json.loads(path.read_text())
        # Saving consumes no RNG and changes no animation/logic state.
        assert env.save_state() == saved
        actions, expected = [], []
        for _ in range(24):
            action = planner.decide(env.snapshot()).action
            actions.append(action)
            result = env.step(action)
            expected.append(transition_record(result))
            if result[2] or result[3]:
                break
        end = env.save_state()
        restored_frame, _ = env.restore_state(loaded)
        assert np.array_equal(restored_frame, frame)
        assert env.snapshot() == state
        assert env.save_state() == saved
        assert [transition_record(env.step(a)) for a in actions] == expected
        assert env.save_state() == end
    # The original process is gone; continuation cannot rely on hidden state.
    with make_env(ghost_mode=ghost_mode) as fresh:
        restored_frame, _ = fresh.reset(saved_state=loaded)
        assert np.array_equal(restored_frame, frame)
        assert fresh.snapshot() == state
        assert fresh.save_state() == saved
        assert [transition_record(fresh.step(a)) for a in actions] == expected
        assert fresh.save_state() == end


def test_restore_through_respawn_preserves_rng_and_death_accounting():
    with make_env(episode_life_mode="original_three_lives") as env:
        env.reset(seed=11)
        saved = env.save_state()
        before_rng = saved_rng(saved)
        expected = []
        for _ in range(128):
            result = env.step("S")
            expected.append(transition_record(result))
            if result[4]["death"]:
                assert result[4]["respawned"]
                break
        else:
            pytest.fail("test did not exercise respawn")
        after = env.save_state()
        assert after["payload"]["episode"]["death_count"] == 1
        assert saved_rng(after) != before_rng
        env.restore_state(saved)
        assert [transition_record(env.step("S")) for _ in expected] == expected
        assert env.save_state() == after
        # Checkpoint after respawn, then exercise a second death/RNG use.
        expected = []
        for _ in range(128):
            result = env.step("S")
            expected.append(transition_record(result))
            if result[4]["death"]:
                break
        assert result[4]["death_count"] == 2
        end = env.save_state()
        env.restore_state(after)
        assert [transition_record(env.step("S")) for _ in expected] == expected
        assert env.save_state() == end


@pytest.mark.parametrize("ending", ["death", "max_steps"])
def test_finished_checkpoint_stays_finished(ending):
    config = {"max_steps": 128 if ending == "death" else 2}
    with make_env(**config) as env:
        env.reset(seed=11)
        for _ in range(config["max_steps"]):
            result = env.step("S")
            if result[2] or result[3]:
                break
        assert result[4]["terminal_reason"] == ending
        saved = env.save_state()
    with make_env(**config) as env:
        _, info = env.restore_state(saved)
        assert info["terminal_reason"] == ending
        assert env.save_state() == saved
        with pytest.raises(EpisodeFinishedError):
            env.step("S")


def test_invalid_checkpoint_rejected_without_changing_episode():
    with make_env() as env:
        env.reset(seed=0)
        saved = env.save_state()
        corrupt = copy.deepcopy(saved)
        corrupt["payload"]["episode"]["steps"] += 1
        with pytest.raises(InvalidConfigurationError, match="checksum"):
            env.restore_state(corrupt)
        assert env.save_state() == saved
        with pytest.raises(InvalidConfigurationError, match="mutually exclusive"):
            env.reset(seed=1, saved_state=saved)
        assert env.save_state() == saved
    with make_env(max_steps=42) as env:
        with pytest.raises(InvalidConfigurationError, match="source/config/schema"):
            env.restore_state(saved)
        assert env._process is None


def test_mid_power_pellet_checkpoint_continues_to_win():
    with make_env() as env:
        env.reset(seed=0)
        planner = EdwardPlanner()
        saved = None
        actions, expected = [], []
        for _ in range(512):
            action = planner.decide(env.snapshot()).action
            result = env.step(action)
            if saved is not None:
                actions.append(action)
                expected.append(transition_record(result))
            if saved is None and result[4]["power_pellet_eaten"]:
                saved = env.save_state()
                assert env.snapshot()["edible_ticks"] > 0
            if result[2] or result[3]:
                break
        assert saved is not None
        assert result[4]["terminal_reason"] == "all_normal_pellets"
        end = env.save_state()
        env.restore_state(saved)
        assert env.save_state() == saved
        for action, record in zip(actions, expected):
            assert transition_record(env.step(action)) == record
        assert env.save_state() == end


def test_game_over_replay_and_shared_highscores_are_unchanged():
    scores = ROOT / "pacman" / "res" / "hiscore.txt"
    before = scores.read_bytes() if scores.exists() else None
    with make_env(episode_life_mode="original_three_lives") as env:
        env.reset(seed=11)
        saved = env.save_state()
        expected = []
        for _ in range(512):
            result = env.step("S")
            expected.append(transition_record(result))
            if result[2] or result[3]:
                break
        assert result[4]["terminal_reason"] == "game_over"
        assert result[4]["death_count"] == 4
        end = env.save_state()
    with make_env(episode_life_mode="original_three_lives") as env:
        env.restore_state(saved)
        for record in expected:
            assert transition_record(env.step("S")) == record
        assert env.save_state() == end
    assert (scores.read_bytes() if scores.exists() else None) == before


def test_fresh_restore_preserves_remaining_horizon():
    config = {"ghost_mode": "disabled", "max_steps": 3}
    with make_env(**config) as env:
        env.reset(seed=0)
        result = env.step("S")
        assert not (result[2] or result[3])
        saved = env.save_state()
        expected = [transition_record(env.step("S")) for _ in range(2)]
        end = env.save_state()
    with make_env(**config) as env:
        _, info = env.reset(saved_state=saved)
        assert info["step"] == 1
        for step, record in enumerate(expected, start=2):
            result = env.step("S")
            assert transition_record(result) == record
            assert result[4]["step"] == step
            assert not result[2]
            assert result[3] is (step == 3)
        assert result[4]["terminal_reason"] == "max_steps"
        assert env.save_state() == end
        with pytest.raises(EpisodeFinishedError):
            env.step("S")


def test_active_fruit_checkpoint_replays_in_fresh_worker():
    with make_env(ghost_mode="disabled") as env:
        env.reset(seed=0)
        # Ghost disabling preserves fruit mechanics; staying avoids a maze win.
        for _ in range(128):
            result = env.step("S")
            assert not (result[2] or result[3])
            if env.snapshot()["fruit"]["active"]:
                break
        else:
            pytest.fail("test did not exercise active fruit")
        saved = json.loads(json.dumps(env.save_state()))
        frame = env.render()
        state = copy.deepcopy(env.snapshot())
        assert state["fruit"]["path_found"]
        assert state["fruit"]["path_remaining"]
        expected = []
        for _ in range(24):
            result = env.step("S")
            assert not (result[2] or result[3])
            expected.append(transition_record(result))
        assert env.snapshot()["fruit"]["pixel_position"] != state["fruit"]["pixel_position"]
        assert env.snapshot()["fruit"]["path_remaining"] != state["fruit"]["path_remaining"]
        end = env.save_state()
    with make_env(ghost_mode="disabled") as env:
        restored_frame, _ = env.reset(saved_state=saved)
        assert np.array_equal(restored_frame, frame)
        assert env.snapshot() == state
        assert env.save_state() == saved
        for record in expected:
            assert transition_record(env.step("S")) == record
        assert env.save_state() == end
