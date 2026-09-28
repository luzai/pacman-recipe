"""Continuation is a fresh optimizer segment, bounded by its parent experiment."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pacman_recipe.level1 import backplay_runner as runner


@pytest.fixture
def parent(tmp_path):
    data = {"phase": "staged", "status": "no_current_learnable_frontier",
            "pilot_updates": 10, "adaptive_updates": 40, "pilot_gate": {"passed": True},
            "true_start_before": [], "true_start_final": [{"success_rate": 0}],
            "events": [{"event": "train_batch", "policy_version": str(i)} for i in range(15)]
                      + [{"event": "probe", "label": "true-start-no-frontier", "policy_version": "15"}]}
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(data))
    return path, data


def test_explicit_reset_permission_required_and_budget_derived_from_policy(parent):
    path, data = parent
    with pytest.raises(ValueError, match="allow-optimizer-reset"):
        runner.continuation_lineage(path, source_policy_version=15, allow_optimizer_reset=False)
    lineage = runner.continuation_lineage(path, source_policy_version=15, allow_optimizer_reset=True)
    assert lineage["remaining_adaptive_updates"] == 35
    assert lineage["parent_completed_adaptive_updates"] == 5
    assert lineage["optimizer_reset"] is True and lineage["exact_resume"] is False
    assert lineage["parent_pilot_gate"] == data["pilot_gate"]


@pytest.mark.parametrize("change", ["gate", "version", "missing_batch", "not_terminal"])
def test_unproven_parent_is_rejected(parent, change):
    path, data = parent
    version = 15
    if change == "gate": data["pilot_gate"]["passed"] = False
    if change == "version": version = 14
    if change == "missing_batch": data["events"].pop(3)
    if change == "not_terminal": data["status"] = "initialized"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.continuation_lineage(path, source_policy_version=version, allow_optimizer_reset=True)


@pytest.fixture
def continuation(parent, tmp_path, monkeypatch):
    path, _ = parent
    entries = [dict(restart_state_id=f"s{i}", env_step=i, seed=0) for i in (0, 355, 366)]
    monkeypatch.setattr(runner, "load_restart_bank", lambda _: {"bank_id": "dense", "restart_states": entries})
    monkeypatch.setattr(runner, "bind_restart_row", lambda template, entry, bank, group: {"id": group})
    calls = []
    trainer = SimpleNamespace(config=SimpleNamespace(actor=SimpleNamespace(path="checkpoint-v15")),
                              rollout=SimpleNamespace(get_version=lambda: 0,
                                rollout_batch=lambda rows, *a, **kw: calls.append(rows) or rows))
    exp = runner.AdaptiveContinuationExperiment(lineage=runner.continuation_lineage(path,
        source_policy_version=15, allow_optimizer_reset=True), trainer=trainer, template={},
        bank_dir=tmp_path, output_dir=tmp_path / "segment", workflow="native", probe_kwargs={},
        candidate_ids=["s355"])
    return exp, entries, calls


def test_continuation_never_runs_pilot_and_has_independent_baseline(continuation, monkeypatch):
    exp, entries, calls = continuation
    labels = []
    monkeypatch.setattr(exp, "probe", lambda entries, label: labels.append(label) or [{"success_rate": .5}])
    monkeypatch.setattr(runner, "select_restart_state", lambda *a, **kw: entries[1])
    exp.prepare_batch(SimpleNamespace(batch_size=4), "native", group_size=12)
    assert labels == ["continuation-frontier", "continuation-independent-baseline"]
    assert len(calls) == 1
    assert exp.report["pilot_updates"] == 0
    assert exp.report["adaptive_updates"] == 35
    assert exp.report["events"][-1]["engine_policy_version"] == "0"
    assert exp.report["events"][-1]["cumulative_policy_version"] == "15"
    assert (exp.output_dir / "parent-experiment.json").exists()


def test_no_continuation_frontier_stops_before_training(continuation, monkeypatch):
    exp, _, calls = continuation
    labels = []
    monkeypatch.setattr(exp, "probe", lambda entries, label: labels.append(label) or [])
    def reject(*a, **kw): raise runner.NoLearnableRestartState("none")
    monkeypatch.setattr(runner, "select_restart_state", reject)
    with pytest.raises(runner.BackplayStopped):
        exp.prepare_batch(SimpleNamespace(batch_size=4), "native", group_size=12)
    assert not calls
    assert labels == ["continuation-frontier", "true-start-no-frontier"]


def test_remaining_budget_and_cumulative_version(continuation, monkeypatch):
    exp, _, _ = continuation
    labels = []
    monkeypatch.setattr(exp, "record_true_start", lambda key, label: labels.append(label))
    exp.after_update(global_step=34)
    assert labels == ["true-start-final"]
    assert exp.report["segment_completed_updates"] == 35
    assert exp.report["cumulative_policy_version"] == "50"
    assert exp.report["status"] == "bounded_adaptive_continuation_completed"


def test_existing_recovery_root_is_rejected_before_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.getpass, "getuser", lambda: "owner")
    section = SimpleNamespace(fileroot=str(tmp_path), experiment_name="exp", trial_name="new",
                              mode="auto", no_save_optim=False, freq_steps=1)
    config = SimpleNamespace(trial_name="new", recover=section, saver=section)
    runner.require_fresh_continuation_storage(config, {"trial_name": "old"})
    (tmp_path / "checkpoints/owner/exp/new").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        runner.require_fresh_continuation_storage(config, {"trial_name": "old"})


def test_framework_saves_recovery_before_continuation_stop_hook():
    path = Path(__file__).parents[2] / "AReaL/areal/trainer/rl_trainer.py"
    if not path.exists(): pytest.skip("pinned sibling AReaL required")
    text = path.read_text(encoding="utf-8")
    assert text.index("self._save_recover_checkpoint(") < text.index("self._evaluate(")


@pytest.fixture
def hf_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.getpass, "getuser", lambda: "owner")
    config = {"saver": {"fileroot": str(tmp_path), "experiment_name": "exp", "trial_name": "parent"}}
    path = tmp_path / "checkpoints/owner/exp/parent/default/epoch14epochstep0globalstep14"
    path.mkdir(parents=True)
    (path / "config.json").write_text('{}')
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(len(header).to_bytes(8, 'little') + header + b'1234')
    return path, config


def test_checkpoint_binding_and_content_manifest(hf_checkpoint, tmp_path):
    path, config = hf_checkpoint
    result = runner.validate_continuation_checkpoint(path, config, 15)
    assert len(result['files']['model.safetensors']['sha256']) == 64
    with pytest.raises(ValueError, match='unique parent'):
        runner.validate_continuation_checkpoint(path, config, 14)
    with pytest.raises(ValueError, match='unique parent'):
        runner.validate_continuation_checkpoint(tmp_path / 'foreign', config, 15)


def test_missing_index_shard_and_truncated_single_file_rejected(hf_checkpoint):
    path, config = hf_checkpoint
    index = path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'weight': 'missing.safetensors'}}))
    with pytest.raises(ValueError, match='shard missing'):
        runner.validate_continuation_checkpoint(path, config, 15)
    index.unlink()
    weight = path / 'model.safetensors'
    weight.write_bytes(weight.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated safetensors data'):
        runner.validate_continuation_checkpoint(path, config, 15)


@pytest.mark.parametrize('field', ['learning_rate', 'kl_ctl', 'temperature', 'reward', 'ghost_mode', 'workflow'])
def test_unapproved_hyperparameter_changes_rejected(field):
    before = {'fixed': {field: 1}, 'actor': {'path': 'base'}, 'trial_name': 'old'}
    after = copy.deepcopy(before)
    after['fixed'][field] = 2
    with pytest.raises(ValueError, match='undeclared'):
        runner.validate_continuation_config(before, after)


def test_only_declared_config_changes_allowed():
    before = {'actor': {'path': 'base', 'lr': 1}, 'trial_name': 'old', 'recover': {'mode': 'disabled'}}
    after = {'actor': {'path': 'v15', 'lr': 1}, 'trial_name': 'new', 'recover': {'mode': 'auto'}}
    runner.validate_continuation_config(before, after)
    after['new_unreviewed_setting'] = True
    with pytest.raises(ValueError): runner.validate_continuation_config(before, after)


def test_derived_trial_storage_changes_are_allowed_without_loosening_model_config():
    parent = {'actor': {'trial_name': 'old', 'lr': 1}, 'ref': {'trial_name': 'old', 'path': 'base'},
              'rollout': {'trial_name': 'old', 'fileroot': 'oldpath', 'temperature': 1},
              'cluster': {'name_resolve': {'nfs_record_root': 'oldpath'}},
              'stats_logger': {'swanlab': {'name': 'old'}}}
    current = copy.deepcopy(parent)
    for section in ('actor', 'ref', 'rollout'): current[section]['trial_name'] = 'new'
    current['rollout']['fileroot'] = 'newpath'
    current['cluster']['name_resolve']['nfs_record_root'] = 'newpath'
    current['stats_logger']['swanlab']['name'] = 'new'
    runner.validate_continuation_config(parent, current)
    current['ref']['path'] = 'newbase'
    with pytest.raises(ValueError, match='ref.path'): runner.validate_continuation_config(parent, current)
