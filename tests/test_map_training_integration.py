import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_pacman.map_training.maps import canonical, gate, generate, load_runtime_level
from slime_pacman.map_training.launch import configured_command, remaining_budget
from slime_pacman.map_training.preflight import (
    validate_training_args,
    verify_files,
    verify_runtime_receipts,
)
from slime_pacman.map_training.workers import SeededSmallMazeClient
from pacman_env.env.ghost_modes import validate_ghost_state
from pacman_env.env.pygame_environment import ruleset_revision

ROOT = Path(__file__).resolve().parents[1]


def test_connected_generation_is_frozen_and_split_disjoint():
    text = (ROOT / "configs/slime/maps/template-11.txt").read_text()
    kwargs = dict(seed=20261009, counts=dict(train=16, validation=8, test=8))
    rows = generate(text, **kwargs)
    assert rows == generate(text, **kwargs)
    assert len({r["equivalence_sha256"] for r in rows}) == 32
    assert all(gate(r["text"]) is not None for r in rows)
    assert [r["split"] for r in rows].count("train") == 16


def test_runtime_map_binding_uses_private_root(tmp_path):
    directory = tmp_path / "pacman/res/levels"
    directory.mkdir(parents=True)
    text = (ROOT / "configs/slime/maps/template-11.txt").read_text()
    (directory / "1.txt").write_text(text)
    level = load_runtime_level(1, tmp_path)
    assert (level.width, level.height) == (11, 11)
    assert level.revision == hashlib.sha256(text.encode()).hexdigest()


def test_rotation_and_reflection_equivalence():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert canonical(grid) == canonical([list(reversed(row)) for row in grid])
    assert canonical(grid) == canonical([list(row) for row in zip(*grid[::-1])])


@pytest.mark.parametrize("count", [1, 2, 4])
def test_ghost_count_and_rule_identity(monkeypatch, count):
    monkeypatch.setenv("SMALL_MAZE_GHOST_COUNT", "4")
    original = ruleset_revision()
    monkeypatch.setenv("SMALL_MAZE_GHOST_COUNT", str(count))
    validate_ghost_state(dict(ghost_mode="normal", ghosts=[{}] * count), "normal")
    assert (ruleset_revision() == original) is (count == 4)
    with pytest.raises(ValueError):
        validate_ghost_state(
            dict(ghost_mode="normal", ghosts=[{}] * (count + 1)), "normal"
        )


def test_sampling_seeds_are_paired_and_do_not_mutate_request():
    class Client:
        async def post(self, endpoint, json):
            return json

    async def exercise():
        context = dict(purpose="validation", record="map-1", replica=0)
        a, b = SeededSmallMazeClient(Client(), context), SeededSmallMazeClient(
            Client(), context
        )
        request = dict(sampling_params=dict(temperature=1))
        first = await a.post("/generate", json=request)
        second = await a.post("/generate", json=request)
        paired = await b.post("/generate", json=request)
        assert first == paired and first != second
        assert "sampling_seed" not in request["sampling_params"]

    asyncio.run(exercise())


def test_launch_preserves_budget_and_rejects_reference_kl():
    base = ["python", "train.py", "--num-rollout", "20"]
    argv = configured_command(base, 20)
    assert "--force-sync-save" in argv
    assert "--use-kl-loss" not in argv
    assert remaining_budget(dict(time_unix=10, wall_seconds=100), 100, 80) == 30
    with pytest.raises(ValueError):
        remaining_budget(dict(time_unix=10, wall_seconds=100), 100, 110)
    with pytest.raises(ValueError):
        configured_command(base + ["--use-kl-loss"], 20)
    with pytest.raises(ValueError):
        validate_training_args(SimpleNamespace(kl_coef=0.01))


def test_receipts_and_source_integrity_fail_closed(tmp_path):
    file = tmp_path / "module.py"
    file.write_text("original")
    manifest = {"module.py": hashlib.sha256(file.read_bytes()).hexdigest()}
    verify_files(tmp_path, manifest)
    file.write_text("changed")
    with pytest.raises(ValueError):
        verify_files(tmp_path, manifest)
    for name in ["IMAGE_CONTENT_VERIFIED.json", "RUNTIME_CPU_VERIFIED.json"]:
        (tmp_path / name).write_text(
            json.dumps(dict(image_id="sha256:test", passed=True))
        )
    assert verify_runtime_receipts(tmp_path, "sha256:test")
    with pytest.raises(ValueError):
        verify_runtime_receipts(tmp_path, "sha256:other")
