import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from slime_pacman import backplay_dataset as dataset
from slime_pacman.config import load_config
from test_slime_backplay import record


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    bank = dict(bank_id="bank-1", restart_states=[], trajectories=[])
    for i, action in enumerate("UDLR"):
        bank["restart_states"].append(dict(restart_state_id=f"state-{i}", trajectory_id=f"teacher-{i}", is_true_initial_state=False))
        path = tmp_path / f"teacher-{i}.json"
        path.write_text(json.dumps(dict(actions=[dict(action=action)])))
        bank["trajectories"].append(dict(trajectory_id=f"teacher-{i}", path=path.name))
    monkeypatch.setattr(dataset, "load_restart_bank", lambda _: bank)
    report = dict(purpose="finalist", training_data=False, weight_version="v0", optimizer_updates=0, provenance=dict(bank_id="bank-1"),
                  summaries=[dict(state_id=f"state-{i}", samples=24, successes=12, success_rate=.5) for i in range(4)])
    return bank, report


def test_selection_rejects_easy_states_duplicates_and_foreign_policy(inputs, tmp_path):
    bank, report = inputs
    ids = [f"state-{i}" for i in range(4)]
    assert dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1")) == bank
    report["provenance"] = dict(bank_id="bank-1", prompt="old")
    with pytest.raises(ValueError, match="provenance"):
        dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1"))
    report["provenance"] = dict(bank_id="bank-1")
    with pytest.raises(ValueError, match="policy version"):
        dataset.validate_selection(tmp_path, ids, [report], "v1", dict(bank_id="bank-1"))
    for wins in (6, 20):
        report["summaries"][0].update(successes=wins, success_rate=wins / 24)
        assert dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1")) == bank
    report["summaries"][0].update(successes=1, success_rate=1 / 24)
    with pytest.raises(ValueError, match="10–90"):
        dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1"))
    report["summaries"][0].update(successes=24, success_rate=1.)
    with pytest.raises(ValueError, match="10–90"):
        dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1"))
    report["summaries"][0].update(successes=12, success_rate=.5)
    (tmp_path / "teacher-1.json").write_text((tmp_path / "teacher-0.json").read_text())
    with pytest.raises(ValueError, match="duplicated"):
        dataset.validate_selection(tmp_path, ids, [report], "v0", dict(bank_id="bank-1"))


def test_changed_selection_policy_requires_matching_frozen_probe_source(tmp_path, monkeypatch):
    current, frozen = tmp_path / "current", tmp_path / "frozen"
    for root, selector in ((current, "new"), (frozen, "old")):
        (root / "slime_pacman").mkdir(parents=True)
        (root / "slime_pacman/backplay_dataset.py").write_text(selector)
        (root / "slime_pacman/runner.py").write_text("same rollout")
    probe_source = dict(commit="base", dirty=True, source_sha256="probe")
    current_sources = {"pacman-recipe": dict(commit="base", dirty=True, source_sha256="current"),
                       "pacman-python": dict(commit="game"), "slime": dict(commit="slime")}
    probe_sources = dict(current_sources, **{"pacman-recipe": probe_source})
    monkeypatch.setattr(dataset, "repository_identity", lambda path: probe_source)
    assert dataset._check_probe_recipe(current_sources, probe_sources, current, frozen) == str(frozen)
    with pytest.raises(ValueError, match="frozen probe recipe path"):
        dataset._check_probe_recipe(current_sources, probe_sources, current, None)
    (current / "slime_pacman/preflight.py").write_text("JSON reporting only")
    assert dataset._check_probe_recipe(current_sources, probe_sources, current, frozen) == str(frozen)
    (current / "slime_pacman/runner.py").write_text("changed rollout")
    with pytest.raises(ValueError, match="outside selection policy"):
        dataset._check_probe_recipe(current_sources, probe_sources, current, frozen)


def test_smoke_dataset_roundtrip_and_evidence_tampering(inputs, tmp_path, monkeypatch):
    _, report = inputs
    config_path = tmp_path / "config.yaml"
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/slime/c2.yaml").read_text())
    raw["updates"] = 1
    config_path.write_text(yaml.safe_dump(raw))
    report_path = tmp_path / "report.json"
    server_manifest = tmp_path / "server.json"
    server_manifest.write_text("{}")
    report["provenance"] = dataset.probe_provenance(record(), load_config(config_path), "bank-1", dataset._sha(server_manifest))
    report_path.write_text(json.dumps(report))
    template = record()
    monkeypatch.setattr(dataset, "make_episode_record", lambda *a, **k: template)

    def bind(source, _, state_id):
        return dict(source, id=state_id)

    monkeypatch.setattr(dataset, "bind_restart_record", bind)
    args = SimpleNamespace(config=config_path, bank=tmp_path, output=tmp_path / "dataset",
                           recipe=tmp_path, game=tmp_path, slime=tmp_path, weight_version="v0",
                           server_manifest=server_manifest, finalist_report=[report_path], candidate_id=[f"state-{i}" for i in range(4)])
    manifest = dataset.prepare(args)
    assert manifest["files"]["train.jsonl"]["rows"] == 4
    assert manifest["files"]["validation.jsonl"]["rows"] == 1
    from slime_pacman.preflight import check_dataset
    assert check_dataset(args.output, load_config(config_path), template["source_revisions"]) == manifest
    with (args.output / "finalist-0.json").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        check_dataset(args.output, load_config(config_path), template["source_revisions"])

