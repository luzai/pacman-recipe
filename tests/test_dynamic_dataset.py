from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_pacman import dynamic_dataset as dd
from slime_pacman.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 14, 16]


@pytest.fixture
def fake_bank(monkeypatch, tmp_path):
    bank = dict(bank_id="bank-x", restart_states=[
        dict(restart_state_id=f"restart-{seed}", seed=seed, is_true_initial_state=True) for seed in SEEDS + [15]
    ] + [dict(restart_state_id="restart-mid", seed=0, is_true_initial_state=False)])
    monkeypatch.setattr(dd, "load_restart_bank", lambda _: bank)

    def bind(template, bank_dir, state_id):
        record = deepcopy(template)
        record["id"] = state_id
        record["environment"]["seed"] = int(state_id.split("-")[1])
        record["restart"] = dict(path=str((tmp_path / f"{state_id}.json").resolve()), sha256="a" * 64, id=state_id)
        return record

    monkeypatch.setattr(dd, "bind_restart_record", bind)
    return bank


def args(tmp_path):
    return SimpleNamespace(config=ROOT / "configs/slime/c2.yaml", output=tmp_path / "dataset", bank=tmp_path / "bank",
                           recipe=ROOT, game=ROOT.parent / "pacman-python", slime=ROOT.parent / "slime",
                           train_seed=SEEDS)


def test_dynamic_dataset_roundtrip_and_tamper(fake_bank, tmp_path):
    a = args(tmp_path)
    manifest = dd.prepare(a)
    assert manifest["schema"] == dd.SCHEMA and manifest["state_ids"] == [f"restart-{s}" for s in SEEDS]
    from slime_pacman.preflight import check_dataset
    config = load_config(a.config)
    assert check_dataset(a.output, config, manifest["source_revisions"]) == manifest
    with pytest.raises(ValueError, match="source changed"):
        changed = deepcopy(manifest["source_revisions"])
        changed["pacman-recipe"]["source_sha256"] = "0" * 64
        check_dataset(a.output, config, changed)
    with (a.output / "train.jsonl").open("a") as out:
        out.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        check_dataset(a.output, config, manifest["source_revisions"])


def test_missing_true_start_is_rejected(fake_bank, tmp_path):
    a = args(tmp_path)
    a.train_seed = [0, 2]
    with pytest.raises(ValueError, match="exactly one true start for seed 2"):
        dd.prepare(a)


def test_fixed_states_rebind_earlier_selection(fake_bank, tmp_path):
    import json
    source = tmp_path / "source-manifest.json"
    source.write_text(json.dumps(dict(bank_id="bank-x", state_ids=["restart-16", "restart-15"])))
    a = args(tmp_path)
    a.train_seed, a.selected_from = [], source
    manifest = dd.prepare(a)
    assert manifest["selection"] == "fixed_states" and manifest["state_ids"] == ["restart-16", "restart-15"]
    from slime_pacman.preflight import check_dataset
    config = load_config(a.config)
    assert check_dataset(a.output, config, manifest["source_revisions"]) == manifest
    (a.output / "selected-from.json").write_text(json.dumps(dict(bank_id="bank-x", state_ids=["restart-mid"])))
    with pytest.raises(ValueError, match="selected-from evidence"):
        check_dataset(a.output, config, manifest["source_revisions"])


def test_fixed_states_reject_other_bank(fake_bank, tmp_path):
    import json
    source = tmp_path / "source-manifest.json"
    source.write_text(json.dumps(dict(bank_id="bank-y", state_ids=["restart-mid"])))
    a = args(tmp_path)
    a.train_seed, a.selected_from = [], source
    with pytest.raises(ValueError, match="different teacher bank"):
        dd.prepare(a)
