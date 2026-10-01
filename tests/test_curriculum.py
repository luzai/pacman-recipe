"""Curriculum orchestration with fake episodes (no game, no SGLang)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pacman_recipe.level1.contracts import make_episode_record
from slime_pacman import curriculum as cur
from slime_pacman import dynamic_bank as db

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 14, 16]


def fake_boundary(seed, step, cell):
    return {"schema": "pacman-decision-boundary-v1",
            "env_state": {"payload": {"episode": {"seed": seed, "steps": step}}, "sha256": f"env-{seed}-{step}-{cell}"},
            "context": {"planner": {"remaining": [[1, 1]], "last_action": "L", "fallback_mode": "risk_ranked"},
                        "runner": {"recent_actions": ["L"] * step}, "verify": {"png_sha256": f"{seed}-{step}"}},
            "features": {"pacman_position": [cell, cell], "nearest_lethal_ghost_distance": step % 7,
                         "normal_pellets_remaining": 50, "env_step": step}}


def write_candidate(directory, name, seed, cell):
    boundaries = [fake_boundary(seed, step, cell) for step in range(20)]
    payload = {"schema": db.CANDIDATE_SCHEMA, "source": {"artifact_id": name, "start_id": f"start-{seed}"},
               "seed": seed, "last_choice_position": [cell, cell],
               "candidates": [dict(distance=d, boundary=b) for d, b in db.death_candidates(boundaries, "death")]}
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / f"{name}.json").write_text(json.dumps(payload))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PACMAN_RUN_DIR", str(tmp_path / "run"))
    for name in ("PACMAN_TRUE_STARTS", "PACMAN_BANK_STARTS", "PACMAN_TRAIN_SEEDS"):
        monkeypatch.delenv(name, raising=False)
    template = make_episode_record(28, split="train", recipe_root=ROOT, game_root=ROOT.parent / "pacman-python",
                                   backend_root=ROOT.parent / "slime", max_steps=512)
    data_source = SimpleNamespace(get_samples=lambda n: [[[SimpleNamespace(metadata={"episode_record": template})]]] * n)
    args = SimpleNamespace(seed=3, rollout_batch_size=4, hf_checkpoint="/model")
    curriculum = cur.Curriculum(args, data_source)
    monkeypatch.setattr(cur.Curriculum, "true_start_record", lambda self, seed: dict(template, id=f"true-{seed}"))
    probes = []

    async def probe(self, record, count):
        state = self.latest_probe_entry = record["id"]
        probes.append((state, count))
        distance = next(e for e in self._entries if e["state_id"] == state)["distance"]
        wins = {4: (0, None), 8: (4, 8), 16: (2, 6)}[distance]
        return (wins[0] if count == db.FIRST_PROBE else wins[1]), ["w1"]

    original_bank_record = cur.Curriculum.bank_record

    def bank_record(self, entry):
        self._entries = getattr(self, "_entries", []) + [entry]
        return original_bank_record(self, entry)

    async def run_episodes(self, records, capture_dir=None):
        for i, record in enumerate(records):
            seed = int(record["id"].split("-")[1])
            if i % 3 == 0:  # some initialization episodes die
                write_candidate(capture_dir, f"init-{seed}-{i}", seed, cell=i)
        return [SimpleNamespace(reward=0.0, terminal_reason="death", weight_version="w0") for _ in records]

    monkeypatch.setattr(cur.Curriculum, "probe", probe)
    monkeypatch.setattr(cur.Curriculum, "bank_record", bank_record)
    monkeypatch.setattr(cur.Curriculum, "run_episodes", run_episodes)
    return curriculum, probes, tmp_path / "run" / "dynamic-bank"


def test_initialization_fills_bank_and_selects_one_true_plus_three_bank(setup):
    curriculum, probes, root = setup
    plan = asyncio.run(curriculum.plan(0))
    init = json.loads((root / "refresh" / "init.json").read_text())
    assert init["initial_rollout"]["episodes"] == 48 and len(init["chosen"]) == db.INIT_TRAJECTORIES
    assert sum(count for _, count in probes) <= db.INIT_BUDGET
    accepted = [p for p in init["probes"] if p.get("result") == "accept"]
    assert accepted and all(p["distance"] == 8 for p in accepted)  # distance 4 rejected at 0/8 first
    assert len(plan["manifest"]["states"]) == len(accepted)
    assert [s["kind"] for s in plan["starts"]] == ["true_start", "bank", "bank", "bank"]
    assert plan["starts"][0]["seed"] == 0
    assert (root / "state" / "rollout-0000.json").exists()
    assert all((root / "states" / e["file"]).exists() for e in plan["manifest"]["states"].values())


def test_later_rollouts_rotate_refresh_and_resume(setup):
    curriculum, probes, root = setup
    asyncio.run(curriculum.plan(0))
    first = asyncio.run(curriculum.plan(1))
    assert first["starts"][0]["seed"] == 1 and not (root / "refresh" / "rollout-0001.json").exists()
    count_before = len(probes)
    assert asyncio.run(curriculum.plan(1)) == first and len(probes) == count_before  # resume reuses frozen plan
    write_candidate(root / "candidates" / "rollout-0000", "train-a", 14, cell=30)
    write_candidate(root / "candidates" / "rollout-0001", "train-b", 16, cell=31)
    second = asyncio.run(curriculum.plan(2))
    refresh = json.loads((root / "refresh" / "rollout-0002.json").read_text())
    assert refresh["available_trajectories"] == 2 and len(refresh["chosen"]) == db.REFRESH_TRAJECTORIES
    assert sum(c for _, c in probes[count_before:]) <= db.REFRESH_BUDGET
    assert second["starts"][0]["seed"] == 14
    usage = {k: v["usage_count"] for k, v in second["manifest"]["states"].items()}
    assert sum(usage.values()) == 3 * 3  # three bank slots in each of rollouts 0, 1, 2


def test_resumed_run_without_state_is_refused(setup):
    curriculum, _, _ = setup
    with pytest.raises(RuntimeError, match="state missing"):
        asyncio.run(curriculum.plan(3))


def test_batch_size_must_match_start_counts(setup, monkeypatch):
    curriculum, _, _ = setup
    monkeypatch.setenv("PACMAN_BANK_STARTS", "2")
    with pytest.raises(ValueError, match="rollout-batch-size"):
        cur.Curriculum(curriculum.args, curriculum.data_source)
