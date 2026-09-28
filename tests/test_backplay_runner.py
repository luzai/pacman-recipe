"""Experiment boundaries: frozen groups, measured advantages, gated progression."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from areal_pacman.level1 import backplay_runner as runner


def summary(rate, interval, reward=0.0):
    return {"samples": 24, "success_rate": rate,
            "success_rate_wilson95": interval, "mean_reward": reward,
            "mean_pellets_eaten": reward, "group_reward_variance": 1.0}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    entries = [dict(restart_state_id=f"s{step}", env_step=step, seed=7,
                    state_path=f"s{step}.json", state_file_sha256="abc")
               for step in (0, 5, 9)]
    monkeypatch.setattr(runner, "load_restart_bank", lambda path: {"bank_id": "b", "restart_states": entries})
    monkeypatch.setattr(runner, "load_restart_state", lambda path, entry: {"payload": {"identity": {"max_steps": 512}}})
    batches = []

    def rollout_batch(rows, workflow, kwargs, group_size):
        batches.append((rows, group_size, kwargs))
        return [{"group": row["id"]} for row in rows]

    trainer = SimpleNamespace(config=SimpleNamespace(actor=SimpleNamespace(path="base")),
                              rollout=SimpleNamespace(rollout_batch=rollout_batch))
    exp = runner.BackplayExperiment(trainer=trainer, template={"id": "t", "env": {"seed": 0, "max_steps": 512}},
        bank_dir=tmp_path / "bank", output_dir=tmp_path / "run", workflow="native",
        probe_kwargs={}, pilot_updates=2, adaptive_updates=3, probe_every=1)
    exp.current = entries[2]
    trainer.rollout.get_version = lambda: exp.version
    exp.pilot_state = entries[2]
    exp.before = summary(.4, [.2, .6])
    exp.advantage_history = [{"count": 48, "variance": 1.0}]
    return exp, entries, batches


def test_gate_requires_actual_learning_not_merely_dispersion():
    before = summary(.4, [.2, .6])
    after = summary(.95, [.8, 1], 2)
    assert runner.pilot_gate(before, after, [{"count": 48, "variance": 1}])["passed"]
    assert not runner.pilot_gate(before, after, [])["passed"]
    assert not runner.pilot_gate(before, before, [{"count": 48, "variance": 1}])["passed"]
    assert not runner.pilot_gate(before, summary(.6, [.4, .8], 2), [{"count": 48, "variance": 1}])["passed"]


def test_resolved_config_never_persists_authentication_values():
    raw = {"admin_api_key": "sensitive", "nested": [{"access_token": "sensitive", "password": "sensitive"}],
           "gconfig": {"max_tokens": 1024}, "tokenizer_path": "model", "action_token_choice": True}
    redacted = runner.redact_config(raw)
    assert "sensitive" not in str(redacted)
    assert redacted["gconfig"]["max_tokens"] == 1024
    assert redacted["tokenizer_path"] == "model"
    assert redacted["action_token_choice"] is True
    assert raw["admin_api_key"] == "sensitive"


def test_sync_groups_are_independent_snapshots_without_prefetch(experiment):
    exp, entries, batches = experiment
    loader = SimpleNamespace(batch_size=4)
    exp.prepare_batch(loader, "native", group_size=12)
    first = copy.deepcopy(batches[0])
    exp.current = entries[1]
    exp.version = 1
    exp.prepare_batch(loader, "native", group_size=12)
    assert len(batches) == 2
    assert batches[0] == first
    assert batches[0][1] == 12
    assert {row["restart_state_id"] for row in batches[0][0]} == {"s9"}
    assert {row["restart_state_id"] for row in batches[1][0]} == {"s5"}
    assert len({row["id"] for row in batches[0][0]}) == 4
    assert exp.template["env"]["seed"] == 0


@pytest.mark.parametrize("kwargs", [{"group_size": 1}, {"group_size": 12, "dynamic_bs": True},
                                   {"group_size": 12, "should_accept_fn": "filter"}])
def test_refuses_group_drift_and_dynamic_filtering(experiment, kwargs):
    exp, _, batches = experiment
    with pytest.raises(ValueError):
        exp.prepare_batch(SimpleNamespace(batch_size=4), "native", **kwargs)
    assert not batches


def test_pilot_keeps_state_frozen_and_inconclusive_stops(experiment, monkeypatch):
    exp, entries, _ = experiment
    monkeypatch.setattr(exp, "probe", lambda entries, label: [summary(.4, [.2, .6])])
    exp.after_update(global_step=0)
    assert exp.current is entries[2]
    with pytest.raises(runner.BackplayStopped, match="inconclusive"):
        exp.after_update(global_step=1)
    assert exp.current is entries[2]
    assert not exp.report["pilot_gate"]["passed"]


def test_adaptive_switch_only_after_pilot_gate_preserves_trainer(experiment, monkeypatch):
    exp, entries, _ = experiment
    trainer_identity = id(exp.trainer)
    monkeypatch.setattr(exp, "probe", lambda entries, label: [summary(.95, [.8, 1], 2)])
    monkeypatch.setattr(runner, "select_restart_state", lambda *a, **kw: entries[1])
    exp.after_update(global_step=1)
    assert exp.report["pilot_gate"]["passed"]
    assert exp.current is entries[1]
    assert id(exp.trainer) == trainer_identity
    exp.after_update(global_step=4)
    assert exp.report["status"] == "bounded_adaptive_run_completed"
    assert "true_start_final" in exp.report


def test_missing_frontier_evaluates_true_start_and_stops(experiment, monkeypatch):
    exp, _, _ = experiment
    labels = []

    def probe(entries, label):
        labels.append(label)
        return [summary(.1, [0, .3], 2) if label == "true-start-no-frontier" else summary(.95, [.8, 1], 2)]

    def no_frontier(*args, **kwargs):
        raise runner.NoLearnableRestartState("none")

    monkeypatch.setattr(exp, "probe", probe)
    monkeypatch.setattr(runner, "select_restart_state", no_frontier)
    with pytest.raises(runner.BackplayStopped, match="no_current"):
        exp.after_update(global_step=1)
    assert labels[-1] == "true-start-no-frontier"


def test_initial_true_start_moderate_does_not_train_direct_baseline(experiment, monkeypatch):
    exp, _, batches = experiment
    monkeypatch.setattr(exp, "probe", lambda entries, label: [summary(.4, [.2, .6])])
    with pytest.raises(runner.BackplayStopped, match="assumption_not_supported"):
        exp.initial_probe()
    assert not batches


def test_no_frontier_with_high_true_start_is_distinct(experiment, monkeypatch):
    exp, _, _ = experiment
    monkeypatch.setattr(exp, "probe", lambda entries, label: [summary(.95, [.8, 1], 2)])
    def no_frontier(*args, **kwargs):
        raise runner.NoLearnableRestartState("none")
    monkeypatch.setattr(runner, "select_restart_state", no_frontier)
    with pytest.raises(runner.BackplayStopped, match="true_start_above_frontier"):
        exp.after_update(global_step=1)


def test_bind_preserves_original_horizon(experiment, monkeypatch):
    exp, entries, _ = experiment
    template = copy.deepcopy(exp.template)
    template["env"]["max_steps"] = 16
    with pytest.raises(ValueError, match="horizon"):
        runner.bind_restart_row(template, entries[0], exp.bank_dir, "g")
    template = copy.deepcopy(exp.template)
    template["decision_steps"] = 1
    with pytest.raises(ValueError, match="complete"):
        runner.bind_restart_row(template, entries[0], exp.bank_dir, "g")


def test_pinned_trainer_hooks_still_have_required_signatures():
    path = Path(__file__).resolve().parents[2] / "AReaL" / "areal" / "trainer" / "rl_trainer.py"
    if not path.exists():
        pytest.skip("sibling AReaL source required to check integration hooks")
    functions = {node.name: node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                 if isinstance(node, ast.FunctionDef)}
    assert {"global_step", "eval_workflow", "epoch"} <= {a.arg for a in functions["_evaluate"].args.args}
    train = ast.unparse(functions["train"])
    assert "self.actor.prepare_batch" in train
    assert "self.actor.compute_advantages" in train
    assert train.index("self.actor.update_weights") < train.index("self._evaluate(")


def test_actual_advantages_preserve_remote_metadata_and_ignore_masked_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    remote = pytest.importorskip("areal.infra.rpc.rtensor")
    tensors = {"a": torch.tensor([[1., -1., 999.]]), "m": torch.tensor([[1, 1, 0]])}
    wrappers = {key: remote.RTensor(remote.TensorShardInfo(shard_id=identifier, node_addr="test"),
                                   torch.empty_like(tensors[identifier], device="meta"))
                for key, identifier in (("advantages", "a"), ("loss_mask", "m"))}
    calls = []
    class Backend:
        def fetch(self, shards):
            calls.extend(shard.shard_id for shard in shards)
            return [tensors[shard.shard_id] for shard in shards]
    monkeypatch.setattr(remote, "get_backend", lambda: Backend())
    before_cache = remote.fetch_buffer_stats()["num_entries"]
    result = runner.tensor_advantage_moments([wrappers])
    assert result["count"] == 2
    assert result["mean"] == 0
    assert result["variance"] == 1
    assert result["maximum"] == 1
    assert all(item.data.is_meta for item in wrappers.values())
    assert set(calls) == {"a", "m"}
    assert remote.fetch_buffer_stats()["num_entries"] == before_cache
