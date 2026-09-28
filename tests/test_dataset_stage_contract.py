"""Stage-specific real-engine anchors and fail-closed bundle provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pacman_recipe.level1 import level1_dataset
from pacman_recipe.level1.recipe import (
    DIRECT_ACTION_PROTOCOL,
    EDWARD_OPTION_PROTOCOL,
    load_recipe_document,
    recipe_contract_metadata,
)
from scripts.level1.dataset import prepare_level1_dataset as dataset
from scripts.level1.dataset import prepare_level1_v3_audits as audits


ROOT = Path(__file__).resolve().parents[1]


def _fixture_recipe(root, stage):
    raw = load_recipe_document(ROOT / f"configs/level1/train/curriculum{stage}.yaml")
    raw["dataset_generation"].update(train_episodes=1, validation_episodes=1)
    path = root / f"curriculum{stage}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _arguments(root, config):
    return SimpleNamespace(
        config=config, output_root=root, train_episodes=None,
        validation_episodes=None, seed=None, max_steps=None,
        write_hf=False, pacman_python_root=None,
    )


@pytest.fixture(scope="module")
def stage_bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage-bundles")
    result = {}
    for stage in (1, 2):
        config = _fixture_recipe(root, stage)
        output = root / f"stage{stage}"
        dataset._prepare_dataset(_arguments(output, config))
        result[stage] = (output, config)
    return result


def _validate(root, config):
    raw = load_recipe_document(config)
    return dataset.validate_prepared_dataset_manifest(
        root / "manifest.json",
        expected_environment=level1_dataset.environment_metadata(raw["environment"]["ghost_mode"]),
        expected_source_revisions=level1_dataset.repository_revisions(),
        expected_training_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        expected_recipe_contract=recipe_contract_metadata(raw),
    )


def _rewrite_manifest(root, manifest):
    encoded = (json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n").encode()
    (root / "manifest.json").write_bytes(encoded)
    (root / "manifest.sha256").write_bytes(
        f"{hashlib.sha256(encoded).hexdigest()}  manifest.json\n".encode("ascii")
    )


def test_c1_preparation_never_constructs_or_inspects_edward(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("C1 must not construct or inspect Edward")

    monkeypatch.setattr(audits, "EdwardPlanner", forbidden)
    monkeypatch.setattr(audits, "_planner_source_sha256", forbidden)
    monkeypatch.setattr(level1_dataset, "EdwardPlanner", forbidden)
    config = _fixture_recipe(tmp_path, 1)
    output = tmp_path / "dataset"
    dataset._prepare_dataset(_arguments(output, config))
    manifest = _validate(output, config)
    assert manifest["audit_anchor_contract"]["planner"] is None
    assert manifest["recipe_contract"]["reward"]["clip"] == "inf"
    row = json.loads((output / "train.jsonl").read_text())
    anchor = row["audit_anchor"]
    assert row["action_protocol"] == anchor["action_protocol"] == DIRECT_ACTION_PROTOCOL
    assert anchor["selected_option"] is None
    assert anchor["planner_candidates"] == []
    assert anchor["ghosts"] == []
    assert anchor["env"]["maapacman_planner_source_sha256"] is None
    assert anchor["executed_primitive_action"] in {"U", "D", "L", "R"}
    assert anchor["audit_anchor_semantics"] == {
        "scope": "initial_state_one_direct_step", "model_rollout": False,
        "training_sample": False,
    }


@pytest.mark.parametrize("stage", [1, 2])
def test_real_stage_bundle_protocol_reward_horizon_and_seeds(stage_bundles, stage):
    output, config = stage_bundles[stage]
    manifest = _validate(output, config)
    assert manifest["max_steps"] == 512
    assert manifest["splits"]["train"]["seeds"] == [28]
    assert manifest["splits"]["validation"]["seeds"] == [29]
    contract = manifest["recipe_contract"]
    assert contract["data"]["training_rng_seed"] == 1
    assert contract["data"]["dataset_seed"] == 28
    assert contract["reward"]["coefficients"]["death_penalty"] == 100
    row = json.loads((output / "train.jsonl").read_text())
    assert row["recipe_contract_sha256"] == manifest["recipe_contract_sha256"]
    if stage == 2:
        anchor = row["audit_anchor"]
        assert anchor["action_protocol"] == EDWARD_OPTION_PROTOCOL
        assert anchor["selected_option"] in anchor["planner_candidates"]
        assert anchor["selected_option"]["first_action"] == anchor["executed_primitive_action"]
        assert len(anchor["ghosts"]) == 4


@pytest.mark.parametrize("tamper,message", [
    ("generator", "generator provenance"),
    ("path", "canonical name"),
    ("recipe", "recipe_contract does not match config"),
    ("legacy", "explicit recipe_contract"),
    ("null", "explicit recipe_contract"),
    ("list", "explicit recipe_contract"),
    ("string", "explicit recipe_contract"),
])
def test_recomputed_sidecar_does_not_hide_tampered_contract(
    tmp_path, stage_bundles, tamper, message
):
    original, config = stage_bundles[1]
    output = tmp_path / "dataset"
    shutil.copytree(original, output)
    manifest = json.loads((output / "manifest.json").read_text())
    if tamper == "generator":
        first = next(iter(manifest["generator_provenance"]["sources"]))
        manifest["generator_provenance"]["sources"][first]["sha256"] = "0" * 64
    elif tamper == "path":
        shutil.copyfile(output / "train.jsonl", output / "replacement.jsonl")
        manifest["splits"]["train"]["jsonl"] = "replacement.jsonl"
    elif tamper == "recipe":
        manifest["recipe_contract"]["reward"]["clip"] = 20.0
    elif tamper == "legacy":
        del manifest["recipe_contract"]
    else:
        manifest["recipe_contract"] = {"null": None, "list": [], "string": "invalid"}[tamper]
    _rewrite_manifest(output, manifest)
    with pytest.raises(ValueError, match=message):
        _validate(output, config)


def test_direct_anchor_rejects_edward_fields_even_with_recomputed_hash(stage_bundles):
    output, _ = stage_bundles[1]
    row = json.loads((output / "train.jsonl").read_text())
    corrupted = copy.deepcopy(row)
    corrupted["audit_anchor"]["selected_option"] = {"first_action": "R"}
    corrupted["audit_anchor_sha256"] = audits._canonical_sha256(corrupted["audit_anchor"])
    with pytest.raises(ValueError, match="cannot contain Edward"):
        dataset.audit_episode_spec_row(corrupted)


@pytest.mark.parametrize("field,value", [("seed", 0), ("train_episodes", 2), ("max_steps", 32)])
def test_preparation_rejects_overrides_before_creating_output(tmp_path, field, value):
    config = _fixture_recipe(tmp_path, 1)
    arguments = _arguments(tmp_path / "dataset", config)
    setattr(arguments, field, value)
    with pytest.raises(ValueError, match="must match config"):
        dataset._prepare_dataset(arguments)
    assert not arguments.output_root.exists()


def test_c1_cannot_launch_edward_baseline_audit(tmp_path, monkeypatch):
    arguments = SimpleNamespace(
        config=ROOT / "configs/level1/train/curriculum1.yaml",
        output_root=tmp_path / "audit", seeds=[28], max_steps=None,
        pacman_python_root=None,
    )
    monkeypatch.setattr(audits, "parse_args", lambda: arguments)
    with pytest.raises(ValueError, match="C1 uses direct anchors"):
        audits.main()
    assert not arguments.output_root.exists()
