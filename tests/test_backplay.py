"""Backplay selection and teacher control-flow tests, without model inference."""

import copy
import json
from types import SimpleNamespace

import pytest

from pacman_recipe.level1.backplay import (
    NoLearnableRestartState, aggregate_probe_results, build_restart_bank,
    load_restart_bank, load_restart_state, select_restart_state,
)
from pacman_env.env._saved_state import checksum
from pacman_env.planner import PlannerCandidate, PlannerDecision


class TeacherEnv:
    """Six primitive steps; seed 1 fails, all other seeds clear the test maze."""

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def info(self):
        return {"step": self.steps, "normal_pellets_remaining": 6 - self.steps,
                "terminal_reason": ("death" if self.seed == 1 else "all_normal_pellets")
                if self.steps == 6 else None}

    def reset(self, *, seed):
        self.steps, self.seed = 0, seed
        return None, self.info()

    def snapshot(self):
        return {"step": self.steps}

    def save_state(self):
        payload = {"identity": {"max_steps": self.config.max_steps, "ruleset": "test"},
                   "worker": {"runtime": {"python": [3, 11, 0]}, "graph": {"step": self.steps}},
                   "episode": {"steps": self.steps, "seed": self.seed, "finished": False}}
        return {"payload": payload, "sha256": checksum(payload)}

    def step(self, action):
        assert action == ["U", "R", "R", "D", "D", "D"][self.steps]
        self.steps += 1
        return None, 1.0, self.steps == 6, False, self.info()


class TeacherPlanner:
    def observe(self, state):
        pass

    def decide(self, state):
        step = state["step"]
        # Option 0 aborts after one move; option 1 reaches its commit bound;
        # option 2 completes; option 3 ends with the episode.
        action = {0: "U", 1: "R", 3: "D", 4: "D"}[step]
        candidate = PlannerCandidate(f"C{step}", "COLLECT", (0, step), action, 3,
                                     2 if step == 1 else 4)
        return PlannerDecision(candidate.option_id, action, (candidate,))

    def record_action(self, action):
        pass

    def continue_option(self, option, state):
        step = state["step"]
        if step == 1:
            return None, "invalidated"
        if step == 4:
            return None, "completed"
        return option.first_action, "active"


def make_bank(path, **kwargs):
    return build_restart_bank(path, seeds=kwargs.pop("seeds", [0]), stride=3, dense_tail=2,
                              config=SimpleNamespace(max_steps=8, ghost_mode="normal", episode_life_mode="single_death"),
                              env_factory=TeacherEnv, planner_factory=TeacherPlanner, **kwargs)


def test_teacher_respects_abort_commit_completion_and_retains_only_live_states(tmp_path):
    root = tmp_path / "bank"
    bank = make_bank(root)
    assert load_restart_bank(root) == bank
    assert [entry["env_step"] for entry in bank["restart_states"]] == [0, 3, 4, 5]
    trajectory = json.loads((root / bank["trajectories"][0]["path"]).read_text())
    assert [step["option_status"] for step in trajectory["actions"]] == [
        "invalidated", "active", "max_commit", "completed", "active", "terminal"]
    for entry in bank["restart_states"]:
        saved = load_restart_state(root, entry)
        assert saved["payload"]["episode"]["steps"] == entry["env_step"]
        assert entry["remaining_steps"] == 8 - entry["env_step"]
        assert not saved["payload"]["episode"]["finished"]
    with pytest.raises(FileExistsError):
        make_bank(root)


def test_failed_teacher_is_not_published_and_seed_attempts_are_bounded(tmp_path):
    with pytest.raises(RuntimeError, match="0/1 successful"):
        make_bank(tmp_path / "failed", seeds=[1])
    assert not (tmp_path / "failed").exists()
    bank = make_bank(tmp_path / "success", seeds=[1, 2, 3])
    assert len(bank["attempts"]) == 2
    assert [entry["seed"] for entry in bank["restart_states"]] == [2] * 4


def test_restart_hash_and_episode_metadata_are_verified(tmp_path):
    bank = make_bank(tmp_path / "bank")
    entry = bank["restart_states"][1]
    mismatched = {**entry, "env_step": 2}
    with pytest.raises(ValueError, match="episode metadata"):
        load_restart_state(tmp_path / "bank", mismatched)
    with pytest.raises(ValueError, match="escapes"):
        load_restart_state(tmp_path / "bank", {**entry, "state_path": "../other.json"})
    path = tmp_path / "bank" / entry["state_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file checksum"):
        load_restart_state(tmp_path / "bank", entry)


def test_bank_manifest_cannot_be_reused_after_editing(tmp_path):
    root = tmp_path / "bank"
    bank = make_bank(root)
    bank["restart_states"][0]["env_step"] = 99
    (root / "manifest.json").write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="manifest checksum"):
        load_restart_bank(root)


def records(state="r2", wins=12, n=24, policy="policy-1", with_advantage=False):
    return [{"restart_state_id": state, "policy_version": policy,
             "group_id": f"group-{i // 12}", "sample_id": str(i),
             "success": i < wins, "reward": float(i < wins), "pellets_eaten": i % 3,
             **({"advantage": 1.0 if i < wins else -1.0} if with_advantage else {})}
            for i in range(n)]


def test_probe_keeps_group_dispersion_separate_from_between_group_variation():
    # One all-win group and one all-fail group: aggregate win-rate .5 alone
    # does not imply nonzero within-group learning signal.
    summary = aggregate_probe_results(records(), policy_version="policy-1")[0]
    assert summary["success_rate"] == .5
    assert summary["samples"] == 24
    assert summary["adequately_sampled"]
    assert summary["group_reward_variance"] == 0
    assert summary["advantage_variance"] is None
    assert summary["success_rate_wilson95"][0] < .5 < summary["success_rate_wilson95"][1]
    rows = records(with_advantage=True)
    for i, row in enumerate(rows):
        row["reward"] = float(i % 2)
        row["success"] = bool(i % 2)
    summary = aggregate_probe_results(rows, policy_version="policy-1")[0]
    assert summary["group_reward_variance"] > 0
    assert summary["advantage_variance"] > 0
    assert all(group["normalized_return_advantage_variance"] > .99 for group in summary["groups"])


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate"),
    (lambda rows: rows[0].update(policy_version="stale"), "policy versions"),
    (lambda rows: rows[0].update(reward=float("nan")), "finite"),
    (lambda rows: rows[0].update(success=1), "boolean"),
    (lambda rows: rows.pop(), "exactly 12"),
])
def test_probe_rejects_invalid_or_incomplete_groups(mutation, match):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        aggregate_probe_results(rows, policy_version="policy-1")


def test_selection_is_earliest_eligible_and_requires_current_policy():
    candidates = [{"restart_state_id": name, "env_step": step}
                  for name, step in [("start", 0), ("r1", 10), ("r2", 20), ("r3", 30)]]
    rows = records("start", wins=0) + records("r1", wins=10) + records("r2", wins=15) + records("r3", wins=24)
    summaries = aggregate_probe_results(rows, policy_version="policy-1")
    assert select_restart_state(candidates, summaries, policy_version="policy-1")["restart_state_id"] == "r1"
    with pytest.raises(ValueError, match="current policy"):
        select_restart_state(candidates, summaries, policy_version="policy-2")
    with pytest.raises(NoLearnableRestartState):
        select_restart_state(candidates, summaries, policy_version="policy-1", minimum_samples=25)
    with pytest.raises(NoLearnableRestartState):
        select_restart_state(candidates, [], policy_version="policy-1")


def test_true_initial_state_can_be_selected_and_small_probe_cannot():
    candidates = [{"restart_state_id": "start", "env_step": 0}]
    small = aggregate_probe_results(records("start", wins=6, n=12), policy_version="policy-1")
    assert not small[0]["adequately_sampled"]
    with pytest.raises(NoLearnableRestartState):
        select_restart_state(candidates, small, policy_version="policy-1")
    full = aggregate_probe_results(records("start"), policy_version="policy-1")
    assert select_restart_state(candidates, full, policy_version="policy-1")["env_step"] == 0
