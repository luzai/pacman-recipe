import json
from types import SimpleNamespace
import pytest
from slime_pacman.formal_rollout import Runtime, validate_binding, write_once


def catalog():
    return [
        dict(
            task_id=f"map-{i}",
            split="train",
            tier=0,
            layout_sha256=f"{i:064x}",
            ghosts=2,
            game_root="/private/game",
            episode_record={"environment": {"level_revision": f"{i:064x}", "ghost_mode": "normal"}},
        )
        for i in range(16)
    ]


def runtime(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                catalog=catalog(),
                seed=20261009,
                max_tier=0,
                policy=dict(
                    threshold=0.8,
                    window_batches=3,
                    min_games=384,
                    consecutive_checks=2,
                    min_maps=8,
                ),
            )
        )
    )
    monkeypatch.setenv("PACMAN_FORMAL_CONFIG", str(config))
    monkeypatch.setenv("PACMAN_RUN_DIR", str(tmp_path))
    return Runtime(SimpleNamespace())


def test_collection_cannot_promote_before_optimizer_completion(tmp_path, monkeypatch):
    rt = runtime(tmp_path, monkeypatch)
    plan = rt.freeze_plan(0)
    episodes = [
        dict(group=g["group"], replica=r, task_id=g["task"]["task_id"], won=True)
        for g in plan["groups"]
        for r in range(12)
    ]
    original = rt.sampler.snapshot()
    rt.collected(0, plan, episodes)
    assert rt.sampler.snapshot() == original
    with pytest.raises(ValueError, match="previous collection"):
        rt.freeze_plan(1)
    rt.training_completed(0)
    assert rt.sampler.controller.state["last_update"] == 0
    assert rt.freeze_plan(1)["update"] == 1


def test_partial_episode_set_fails_without_changing_controller(tmp_path, monkeypatch):
    rt = runtime(tmp_path, monkeypatch)
    plan = rt.freeze_plan(0)
    before = rt.sampler.snapshot()
    with pytest.raises(ValueError, match="complete"):
        rt.collected(0, plan, [])
    assert rt.sampler.snapshot() == before


def test_initial_and_layout_guards():
    task = catalog()[0]
    validate_binding(task)
    task["episode_record"]["restart"] = True
    with pytest.raises(ValueError, match="initial"):
        validate_binding(task)
    task["episode_record"].pop("restart")
    task["layout_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_binding(task)


def test_frozen_artifact_cannot_change(tmp_path):
    path = tmp_path / "plan.json"
    write_once(path, {"tier": 0})
    write_once(path, {"tier": 0})
    with pytest.raises(ValueError, match="immutable"):
        write_once(path, {"tier": 1})
