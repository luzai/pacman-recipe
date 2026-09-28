from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from pacman_env.env.pygame_environment import ruleset_revision

from pacman_recipe.level1 import level1_dataset
from pacman_recipe.level1.level1_dataset import write_jsonl
from pacman_recipe.level1.rewards import (
    REWARD_RECIPE_VERSION,
    RewardConfig,
    audit_reward,
    shape_reward,
)
from pacman_recipe.level1.trajectories import (
    REQUIRED_ENV_FIELDS,
    audit_step_environment_evidence,
    audit_trajectory,
)
from scripts.level1.dataset import (
    prepare_level1_dataset,
    prepare_level1_v3_audits,
    write_level1_manifest,
)
from scripts.level1.dataset.prepare_level1_v3_audits import (
    AUDIT_CONTRACT_VERSION,
    AUDIT_DATASET_ROLES,
    _audit_generator_provenance,
    _canonical_sha256,
    _metadata,
    _reward_config,
    _selected_candidate,
    _sha256_text,
    audit_planner_record,
    parse_args as parse_audit_args,
)
from scripts.level1.dataset.prepare_level1_dataset import (
    DATASET_ROLES,
    _tree_sha256,
    _write_text_exclusive,
    audit_episode_spec_row,
    validate_prepared_dataset_manifest,
)


def _event(event_type: str = "normal_pellet_eaten", score: int = 10):
    return {
        "logic_frame_index": 1,
        "logic_frame": 1,
        "event_type": event_type,
        "pacman_position": [1, 2],
        "ghost_id": None,
        "score_delta": score,
        "post_ghost_state": None,
        "edible_ticks": 0,
    }


def _atomic_frame(event=None):
    events = [] if event is None else [event]
    score = sum(item["score_delta"] for item in events)
    return {
        "ghost_mode": "normal",
        "frame": 1,
        "logic_frame_index": 1,
        "pacman_position": [1, 2],
        "pacman_facing": "R",
        "level": 1,
        "mode": 1,
        "mode_name": "play",
        "score": score,
        "score_delta": score,
        "score_components": {
            "normal_pellet": score,
            "power_pellet": 0,
            "ghost": 0,
            "fruit": 0,
            "other": 0,
            "total": score,
        },
        "lives": 3,
        "width": 28,
        "height": 31,
        "normal_pellets_remaining": 191,
        "power_pellets_remaining": 4,
        "collectibles_remaining": 195,
        "edible_ticks": 0,
        "ghosts": [{"id": index} for index in range(4)],
        "fruit": {"active": False},
        "ghost_door": {"positions": [[14, 13]]},
        "blocked": ["U"],
        "open": ["L", "R"],
        "events": events,
    }


def _reward(event):
    result = shape_reward(
        float(event["score_delta"]),
        {},
        {
            "logic_frame_events": [event],
            "wall_collision": False,
        },
        RewardConfig(step_penalty=0.0),
    )
    return result.as_dict()


def _step_evidence():
    event = _event()
    frame = _atomic_frame(event)
    return {
        **_reward(event),
        "logic_frames": 1,
        "atomic_substeps": [frame],
        "logic_frame_events": [event],
        "events": ["normal_pellet_eaten"],
        "score_components": dict(frame["score_components"]),
        "score": 10,
        "ghosts": list(frame["ghosts"]),
        "edible_ticks": 0,
        "pygame_mode": 1,
        "pellets_remaining": 195,
        "normal_pellets_remaining": 191,
        "power_pellets_remaining": 4,
        "death": False,
        "level_completed": False,
        "terminated": False,
        "truncated": False,
        "terminal_reason": None,
    }


def _extended_death_step(logic_frames: int = 17):
    death_event = _event("death", score=0)
    step = _step_evidence()
    frames = []
    for index in range(1, logic_frames + 1):
        frame = copy.deepcopy(step["atomic_substeps"][0])
        # GAME_LOGIC_FRAME advances on the lethal gameplay frame, then stays
        # frozen throughout the automatic death and READY animations.
        frame["frame"] = 1
        frame["logic_frame_index"] = index
        frame["score"] = 0
        if index == 1:
            frame["events"] = [death_event]
            frame["score_delta"] = 0
            frame["score_components"] = {
                name: 0 for name in frame["score_components"]
            }
        else:
            frame["events"] = []
            frame["score_delta"] = 0
            frame["score_components"] = {
                name: 0 for name in frame["score_components"]
            }
        frames.append(frame)
    step.update(
        {
            **_reward(death_event),
            "logic_frames": logic_frames,
            "atomic_substeps": frames,
            "logic_frame_events": [death_event],
            "events": ["death"],
            "score_components": dict(frames[0]["score_components"]),
            "score": 0,
            "death": True,
            "respawned": True,
        }
    )
    return step


def test_extended_death_animation_requires_original_three_lives():
    step = _extended_death_step()
    with pytest.raises(ValueError, match="1 to 16 logic frames"):
        audit_step_environment_evidence(
            step,
            previous_score=0,
            previous_logic_frame=0,
        )

    score, logic_frame = audit_step_environment_evidence(
        step,
        previous_score=0,
        previous_logic_frame=0,
        episode_life_mode="original_three_lives",
    )
    assert score == 0
    assert logic_frame == 1


def test_extended_death_animation_rejects_source_frame_advance_after_death():
    step = _extended_death_step()
    step["atomic_substeps"][1]["frame"] = 2
    with pytest.raises(ValueError, match="must remain frozen after death event"):
        audit_step_environment_evidence(
            step,
            previous_score=0,
            previous_logic_frame=0,
            episode_life_mode="original_three_lives",
        )


def _planner_record():
    evidence = _step_evidence()
    ghost_state = {"ghost_mode": "normal", "ghosts": [{"id": i} for i in range(4)], "edible_ticks": 0}
    state = {**ghost_state, "row": 1, "col": 1, "open": ["R"]}
    next_state = {**ghost_state, "row": 1, "col": 2, "open": ["L", "R"]}
    candidate = {
        "option_id": "C0",
        "strategy": "COLLECT",
        "target": [1, 2],
        "first_action": "R",
        "route_distance": 1,
        "commit_moves": 1,
        "safety_margin": 2,
        "future_safe_exits": 2,
        "entity_id": None,
        "lethal_ghost_etas": [],
        "choke_points": [],
        "safe_return_distance": 1,
    }
    return {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "dataset_contract_version": level1_dataset.DATASET_CONTRACT_VERSION,
        "seed": 0,
        "step": 1,
        "source_revisions": {
            "pacman-python": {"commit": "1" * 40, "dirty": True},
            "areal-pacman": {"commit": "2" * 40, "dirty": True},
            "AReaL": {"commit": "4" * 40, "dirty": True},
        },
        "env": {
            "ghost_mode": "normal",
            "name": "pacman-python-level1-ghostdoor-v3",
            "api_version": "3.0",
            "backend": "original-pygame",
            "level": 1,
            "dataset_contract_version": (
                level1_dataset.DATASET_CONTRACT_VERSION
            ),
            "pacman_python_revision": "1" * 40,
            "pacman_python_dirty": True,
            "maapacman_revision": "2" * 40,
            "maapacman_dirty": True,
            "max_steps": 256,
            "ruleset_revision": ruleset_revision("normal"),
            "pacman_python_source_sha256": "d" * 64,
            "maapacman_env_source_sha256": "e" * 64,
            "maapacman_planner_source_sha256": "9" * 64,
            "level_revision": "f" * 64,
            "renderer_revision": f"pacman-python:{'1' * 40}",
        },
        "observation_png_sha256": "a" * 64,
        "structured_state_sha256": _canonical_sha256(state),
        "structured_state": state,
        "legal_actions": ["R", "S"],
        "planner_candidates": [candidate],
        "selected_option": candidate,
        "executed_primitive_action": "R",
        "next_observation_png_sha256": "b" * 64,
        "next_structured_state_sha256": _canonical_sha256(next_state),
        "next_structured_state": next_state,
        "environment_score_delta": 10,
        "score_before": 0,
        "score": 10,
        "logic_frame_before": 0,
        "logic_frames": evidence["logic_frames"],
        "atomic_substeps": evidence["atomic_substeps"],
        "logic_frame_events": evidence["logic_frame_events"],
        "events": evidence["events"],
        "score_components": evidence["score_components"],
        "ghosts": evidence["ghosts"],
        "edible_ticks": evidence["edible_ticks"],
        "pygame_mode": evidence["pygame_mode"],
        "pellets_remaining": evidence["pellets_remaining"],
        "normal_pellets_remaining": evidence[
            "normal_pellets_remaining"
        ],
        "power_pellets_remaining": evidence[
            "power_pellets_remaining"
        ],
        "reward_breakdown": {
            key: value
            for key, value in evidence.items()
            if key
            in {
                "reward_recipe_version",
                "event_game_score_delta",
                "event_count",
                "base_reward",
                "base_reward_contribution",
                "normal_pellet_eaten",
                "normal_pellet_reward",
                "power_pellet_eaten",
                "power_pellet_reward",
                "ghost_eaten",
                "ghost_reward",
                "fruit_eaten",
                "fruit_reward",
                "death",
                "death_penalty",
                "level_completed",
                "completion_reward",
                "safety_refusal",
                "safety_refusal_penalty",
                "step_penalty",
                "wall_penalty",
                "normal_pellet_remaining_ratio",
                "nearest_pellet_shaping_active",
                "nearest_pellet_distance_before",
                "nearest_pellet_distance_after",
                "nearest_pellet_progress_weight",
                "nearest_pellet_progress_reward",
                "shaped_reward",
            }
        },
        "terminated": False,
        "truncated": False,
        "terminal_reason": None,
    }


def test_atomic_environment_evidence_reconciles_every_layer():
    evidence = _step_evidence()
    assert audit_step_environment_evidence(
        evidence, previous_score=0, previous_logic_frame=0
    ) == (10, 1)

    corrupted = copy.deepcopy(evidence)
    corrupted["atomic_substeps"][0]["score_components"]["total"] = 9
    with pytest.raises(ValueError, match="score_components"):
        audit_step_environment_evidence(
            corrupted, previous_score=0, previous_logic_frame=0
        )

    corrupted = copy.deepcopy(evidence)
    components = corrupted["atomic_substeps"][0]["score_components"]
    components["normal_pellet"] = 0
    components["other"] = 10
    with pytest.raises(ValueError, match="events.*score_components"):
        audit_step_environment_evidence(
            corrupted, previous_score=0, previous_logic_frame=0
        )

    corrupted = copy.deepcopy(evidence)
    corrupted["logic_frame_events"] = []
    with pytest.raises(ValueError, match="atomic event order"):
        audit_step_environment_evidence(
            corrupted, previous_score=0, previous_logic_frame=0
        )

    stale_event_position = copy.deepcopy(evidence)
    stale_event_position["logic_frame_events"][0]["pacman_position"] = [
        1,
        1,
    ]
    stale_event_position["atomic_substeps"][0]["events"][0][
        "pacman_position"
    ] = [1, 1]
    with pytest.raises(
        ValueError,
        match="event Pacman position mismatch.*normal_pellet_eaten",
    ):
        audit_step_environment_evidence(
            stale_event_position, previous_score=0, previous_logic_frame=0
        )


def test_reward_audit_rejects_event_flag_invention():
    event = _event()
    record = {**_reward(event), "logic_frame_events": [event]}
    record["death"] = True
    with pytest.raises(ValueError, match="death.*event ledger"):
        audit_reward(record)


def test_serialized_planner_audit_is_self_reconciling():
    record = _planner_record()
    audit_planner_record(record)

    corrupted = copy.deepcopy(record)
    corrupted["structured_state"]["row"] = 9
    with pytest.raises(ValueError, match="structured-state hash"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["selected_option"]["first_action"] = "L"
    with pytest.raises(ValueError, match="selected option"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    del corrupted["env"]["maapacman_planner_source_sha256"]
    with pytest.raises(ValueError, match="planner_source_sha256"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["env"]["pacman_python_dirty"] = "yes"
    with pytest.raises(ValueError, match="pacman_python_dirty"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    del corrupted["reward_breakdown"]["normal_pellet_reward"]
    with pytest.raises(ValueError, match="reward_breakdown schema mismatch"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    del corrupted["source_revisions"]["AReaL"]
    with pytest.raises(ValueError, match="source revisions are incomplete"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["source_revisions"]["areal-pacman"]["commit"] = "bad"
    with pytest.raises(ValueError, match="areal-pacman revision is invalid"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["env"]["maapacman_revision"] = "3" * 40
    with pytest.raises(ValueError, match="must match bundled areal-pacman"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["env"]["pacman_python_revision"] = "3" * 40
    with pytest.raises(ValueError, match="pacman-python provenance"):
        audit_planner_record(corrupted)

    corrupted = copy.deepcopy(record)
    corrupted["env"]["renderer_revision"] = "pacman-python:" + "3" * 40
    with pytest.raises(ValueError, match="renderer revision"):
        audit_planner_record(corrupted)


def test_trajectory_rejects_legacy_or_mismatched_maapacman_repository():
    payload = {field: None for field in REQUIRED_ENV_FIELDS}
    payload.update(
        env_api_version=level1_dataset.ENV_API_VERSION,
        env_id=level1_dataset.ENV_NAME,
        backend="original-pygame",
        dataset_contract_version=level1_dataset.DATASET_CONTRACT_VERSION,
        maapacman_revision="2" * 40,
        pacman_python_revision="1" * 40,
        renderer_revision=f"pacman-python:{'1' * 40}",
        source_revisions={
            "pacman-python": {"commit": "1" * 40, "dirty": False},
            "areal-pacman": {"commit": "2" * 40, "dirty": True},
            "AReaL": {"commit": "3" * 40, "dirty": False},
        },
    )

    legacy = copy.deepcopy(payload)
    legacy["source_revisions"]["Pacman"] = {
        "commit": "4" * 40,
        "dirty": False,
    }
    with pytest.raises(ValueError, match="source revisions are incomplete"):
        audit_trajectory(legacy)

    mismatched = copy.deepcopy(payload)
    mismatched["maapacman_revision"] = "4" * 40
    with pytest.raises(ValueError, match="must match bundled areal-pacman"):
        audit_trajectory(mismatched)

    mismatched = copy.deepcopy(payload)
    mismatched["pacman_python_revision"] = "4" * 40
    with pytest.raises(ValueError, match="pacman-python revision"):
        audit_trajectory(mismatched)

    mismatched = copy.deepcopy(payload)
    mismatched["renderer_revision"] = "pacman-python:" + "4" * 40
    with pytest.raises(ValueError, match="renderer revision"):
        audit_trajectory(mismatched)


def test_selected_candidate_uses_advertised_option_not_missing_property():
    candidate = SimpleNamespace(
        option_id="C0",
        first_action="R",
        as_dict=lambda: {"option_id": "C0", "first_action": "R"},
    )
    decision = SimpleNamespace(
        option_id="C0", action="R", candidates=(candidate,)
    )
    assert _selected_candidate(decision) == {
        "option_id": "C0",
        "first_action": "R",
    }


def test_reward_config_manifest_uses_the_exact_training_recipe(tmp_path):
    path = tmp_path / "config.yaml"
    fields = {
        "reward_recipe_version": "maapacman-level1-event-reward-v3",
        "step_penalty": 0.05,
        "step_penalty_cleared_ratio_scale": 0.0,
        "wall_penalty": 0.5,
        "use_base_reward": False,
        "normal_pellet_reward": 1.0,
        "power_pellet_reward": 1.0,
        "ghost_reward": 5.0,
        "fruit_reward": 0.0,
        "death_penalty": 25.0,
        "completion_reward": 50.0,
        "safety_refusal_penalty": 25.0,
        "nearest_pellet_alpha": 0.1,
        "nearest_pellet_remaining_ratio_threshold": 1.0,
        "nearest_pellet_scale_by_cleared_ratio": True,
        "nearest_pellet_skip_on_eat": True,
    }
    path.write_text(
        "\n".join(f"{key}: {str(value).lower()}" for key, value in fields.items()),
        encoding="utf-8",
    )
    config, digest = _reward_config(path)
    assert config.death_penalty == 25.0
    assert config.safety_refusal_penalty == 25.0
    assert config.use_base_reward is False
    assert len(digest) == 64

    legacy_path = tmp_path / "legacy-config.yaml"
    legacy_fields = dict(fields)
    del legacy_fields["safety_refusal_penalty"]
    legacy_path.write_text(
        "\n".join(
            f"{key}: {str(value).lower()}"
            for key, value in legacy_fields.items()
        ),
        encoding="utf-8",
    )
    legacy_config, _ = _reward_config(legacy_path)
    assert legacy_config.safety_refusal_penalty == 0.0


def test_four_v3_dataset_roles_are_distinct_and_complete():
    assert set(DATASET_ROLES) | set(AUDIT_DATASET_ROLES) == {
        "train",
        "validation",
        "deterministic_planner_baseline",
        "option_candidate_audit",
    }
    assert set(DATASET_ROLES).isdisjoint(AUDIT_DATASET_ROLES)


def test_audit_generator_provenance_uses_relative_source_hashes():
    provenance = _audit_generator_provenance()
    assert len(provenance["pacman_recipe_commit"]) == 40
    assert isinstance(provenance["pacman_recipe_dirty"], bool)
    assert {
        "scripts/level1/dataset/prepare_level1_v3_audits.py",
        "pacman_recipe/level1/level1_dataset.py",
        "pacman_recipe/level1/recipe.py",
        "pacman_recipe/level1/prompts.py",
        "pacman_recipe/level1/rewards.py",
        "pacman_recipe/level1/trajectories.py",
    } == set(provenance["sources"])
    assert all(
        not Path(path).is_absolute() and len(record["sha256"]) == 64
        for path, record in provenance["sources"].items()
    )


def test_v3_audit_defaults_to_recipe_horizon(tmp_path):
    with patch(
        "sys.argv",
        [
            "prepare_level1_v3_audits.py",
            "--output-root",
            str(tmp_path / "output"),
            "--config",
            str(tmp_path / "config.yaml"),
        ],
    ):
        arguments = parse_audit_args()
    assert arguments.seeds == [0]
    assert arguments.max_steps is None


def test_v3_audit_rejects_horizon_that_differs_from_recipe(tmp_path):
    arguments = SimpleNamespace(
        output_root=tmp_path / "output",
        config=(
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "train"
            / "curriculum1.yaml"
        ),
        seeds=[0],
        max_steps=256,
        pacman_python_root=None,
    )
    with (
        patch.object(
            prepare_level1_v3_audits,
            "parse_args",
            return_value=arguments,
        ),
        pytest.raises(ValueError, match="planner_audit.max_steps"),
    ):
        prepare_level1_v3_audits.main()
    assert not arguments.output_root.exists()


def test_split_main_uses_explicit_non_sibling_pacman_root_without_leaking_env(
    tmp_path, monkeypatch
):
    explicit_root = tmp_path / "external" / "pacman-source"
    (explicit_root / "pacman" / "res" / "levels").mkdir(parents=True)
    (explicit_root / "pacman" / "pacman.pyw").write_text(
        "# provenance-only fixture\n", encoding="utf-8"
    )
    (explicit_root / "pacman" / "res" / "levels" / "1.txt").write_text(
        "fixture-level\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=explicit_root, check=True)
    subprocess.run(["git", "add", "."], cwd=explicit_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=explicit_root,
        check=True,
    )
    explicit_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=explicit_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    original_root = Path(
        os.environ.get(
            "MAAPACMAN_PACMAN_ROOT",
            Path(__file__).parents[2] / "pacman-python",
        )
    ).resolve()
    monkeypatch.setenv("MAAPACMAN_PACMAN_ROOT", str(original_root))
    observed: dict[str, object] = {}

    def capture_provenance(_args):
        observed["environment"] = level1_dataset.environment_metadata()
        observed["revisions"] = level1_dataset.repository_revisions()
        observed["selected_root"] = os.environ["MAAPACMAN_PACMAN_ROOT"]

    arguments = SimpleNamespace(pacman_python_root=str(explicit_root))
    with (
        patch.object(prepare_level1_dataset, "parse_args", return_value=arguments),
        patch.object(
            prepare_level1_dataset,
            "_prepare_dataset",
            side_effect=capture_provenance,
        ),
    ):
        prepare_level1_dataset.main()

    environment = observed["environment"]
    revisions = observed["revisions"]
    assert environment["pacman_python_revision"] == explicit_commit
    assert revisions["pacman-python"]["commit"] == explicit_commit
    assert Path(observed["selected_root"]) == explicit_root.resolve()
    assert os.environ["MAAPACMAN_PACMAN_ROOT"] == str(original_root)
    assert (
        level1_dataset.repository_revisions()["pacman-python"]["commit"]
        != explicit_commit
    )
    level1_dataset.repository_revisions.cache_clear()
    level1_dataset.environment_metadata.cache_clear()


def test_manifest_and_jsonl_writes_refuse_overwrite(tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_text_exclusive(manifest, "{}\n")
    with pytest.raises(FileExistsError):
        _write_text_exclusive(manifest, "replaced\n")
    assert manifest.read_text(encoding="utf-8") == "{}\n"

    rows = [{"id": "first"}]
    jsonl = tmp_path / "baseline.jsonl"
    _sha256_text(jsonl, rows)
    with pytest.raises(FileExistsError):
        _sha256_text(jsonl, [{"id": "replacement"}])
    assert json.loads(jsonl.read_text(encoding="utf-8"))["id"] == "first"

    episode_jsonl = tmp_path / "episodes.jsonl"
    with patch.object(level1_dataset, "validate_episode_row"):
        write_jsonl([{"id": "first"}], episode_jsonl)
        with pytest.raises(FileExistsError):
            write_jsonl([{"id": "replacement"}], episode_jsonl)
    assert json.loads(episode_jsonl.read_text(encoding="utf-8"))["id"] == "first"


def test_tree_digest_is_relative_to_artifact_root(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "elsewhere" / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "dataset_info.json").write_text("{}\n", encoding="utf-8")
        (root / "nested" / "data.bin").write_bytes(b"same")
    assert _tree_sha256(first) == _tree_sha256(second)


def test_split_generator_writes_relative_immutable_manifest(tmp_path):
    fixture_config = tmp_path / "fixture.yaml"
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs/level1/train/curriculum2.yaml").read_text())
    raw["dataset_generation"].update(train_episodes=2, validation_episodes=1, seed=10)
    fixture_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output_root = tmp_path / "v3-splits"
    arguments = SimpleNamespace(
        output_root=output_root,
        config=fixture_config,
        train_episodes=2,
        validation_episodes=1,
        seed=10,
        max_steps=512,
        write_hf=False,
        pacman_python_root=None,
    )
    with patch.object(prepare_level1_dataset, "parse_args", return_value=arguments):
        prepare_level1_dataset.main()
    manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_roles"] == ["train", "validation"]
    assert manifest["splits"]["train"]["jsonl"] == "train.jsonl"
    assert manifest["splits"]["validation"]["jsonl"] == "validation.jsonl"
    assert manifest["splits"]["train"]["seeds"] == [10, 11]
    assert manifest["splits"]["validation"]["seeds"] == [12]
    assert Path(manifest["splits"]["train"]["jsonl"]).is_absolute() is False
    assert len(manifest["splits"]["train"]["sha256"]) == 64
    assert manifest["splits"]["train"]["reward_breakdown"] == {
        "episode_spec": "not_applicable",
        "audit_anchor": "complete_and_audited",
        "model_rollout": "required_at_training",
    }
    assert isinstance(manifest["environment"]["pacman_python_dirty"], bool)
    assert isinstance(manifest["environment"]["maapacman_dirty"], bool)
    generator = manifest["generator_provenance"]
    assert len(generator["pacman_recipe_commit"]) == 40
    assert isinstance(generator["pacman_recipe_dirty"], bool)
    assert {
        "scripts/level1/dataset/prepare_level1_dataset.py",
        "scripts/level1/dataset/prepare_level1_v3_audits.py",
        "pacman_recipe/level1/level1_dataset.py",
        "pacman_recipe/level1/recipe.py",
        "pacman_recipe/level1/prompts.py",
        "pacman_recipe/level1/rewards.py",
        "pacman_recipe/level1/trajectories.py",
    } == set(generator["sources"])
    assert all(
        not Path(path).is_absolute() and len(record["sha256"]) == 64
        for path, record in generator["sources"].items()
    )
    train_rows = [
        json.loads(line)
        for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_rows) == 2
    assert all(row["audit_anchor"]["step"] == 1 for row in train_rows)
    assert all(len(row["audit_anchor_sha256"]) == 64 for row in train_rows)
    assert all(
        set(row["source_revisions"])
        == {"pacman-python", "areal-pacman", "AReaL"}
        and row["source_revisions"]
        == row["audit_anchor"]["source_revisions"]
        for row in train_rows
    )
    assert all(
        row["audit_anchor"]["audit_anchor_semantics"]["model_rollout"] is False
        for row in train_rows
    )
    assert all(
        len(row["audit_anchor"]["reward_breakdown"]) == 28
        for row in train_rows
    )
    for row in train_rows:
        audit_episode_spec_row(row)
    corrupted = copy.deepcopy(train_rows[0])
    executed = corrupted["audit_anchor"]["executed_primitive_action"]
    corrupted["audit_anchor"]["selected_option"]["first_action"] = (
        "U" if executed != "U" else "D"
    )
    with pytest.raises(ValueError, match="selected option"):
        audit_episode_spec_row(corrupted)
    with pytest.raises(ValueError, match="canonical hash mismatch"):
        level1_dataset.validate_episode_row(corrupted)
    with (
        patch.object(prepare_level1_dataset, "parse_args", return_value=arguments),
        pytest.raises(FileExistsError),
    ):
        prepare_level1_dataset.main()
    validated = validate_prepared_dataset_manifest(
        output_root / "manifest.json",
        expected_environment=manifest["environment"],
        expected_source_revisions=manifest["source_revisions"],
        expected_training_config_sha256=hashlib.sha256(
            arguments.config.read_bytes()
        ).hexdigest(),
    )
    assert validated == manifest
    checksum = output_root / "manifest.sha256"
    original_checksum = checksum.read_bytes()
    checksum.write_bytes(f"{'0' * 64}  manifest.json\n".encode("ascii"))
    with pytest.raises(ValueError, match="checksum sidecar"):
        validate_prepared_dataset_manifest(
            output_root / "manifest.json",
            expected_environment=manifest["environment"],
            expected_source_revisions=manifest["source_revisions"],
            expected_training_config_sha256=manifest["training_config_sha256"],
        )
    checksum.write_bytes(original_checksum)
    train_jsonl = output_root / "train.jsonl"
    train_jsonl.write_bytes(train_jsonl.read_bytes() + b" ")
    with pytest.raises(ValueError, match="JSONL digest"):
        validate_prepared_dataset_manifest(
            output_root / "manifest.json",
            expected_environment=manifest["environment"],
            expected_source_revisions=manifest["source_revisions"],
            expected_training_config_sha256=manifest["training_config_sha256"],
        )


def test_audit_metadata_carries_complete_environment_and_planner_provenance():
    env = SimpleNamespace(
        config=SimpleNamespace(ghost_mode="normal", max_steps=256),
        provenance={
            "pacman_python_commit": "1" * 40,
            "pacman_python_source_sha256": "2" * 64,
            "pacman_python_dirty": True,
            "maapacman_commit": "3" * 40,
            "maapacman_env_source_sha256": "4" * 64,
            "maapacman_dirty": False,
            "level": 1,
        },
        spec=SimpleNamespace(
            env_id="pacman-python-level1-ghostdoor-v3",
            api_version="3.0",
            level_revision="5" * 64,
            renderer_revision="6" * 64,
            ruleset_revision="7" * 64,
        ),
    )
    with (
        patch(
            "scripts.level1.dataset.prepare_level1_v3_audits."
            "_planner_source_sha256",
            return_value="8" * 64,
        ),
        patch(
            "scripts.level1.dataset.prepare_level1_v3_audits."
            "repository_revisions",
            return_value={
                "pacman-python": {"commit": "1" * 40, "dirty": True},
                "areal-pacman": {"commit": "3" * 40, "dirty": False},
                "AReaL": {"commit": "4" * 40, "dirty": False},
            },
        ),
    ):
        metadata = _metadata(env)
    assert metadata == {
        "ghost_mode": "normal",
        "name": "pacman-python-level1-ghostdoor-v3",
        "api_version": "3.0",
        "backend": "original-pygame",
        "pacman_python_revision": "1" * 40,
        "pacman_python_source_sha256": "2" * 64,
        "pacman_python_dirty": True,
        "maapacman_revision": "3" * 40,
        "maapacman_env_source_sha256": "4" * 64,
        "maapacman_planner_source_sha256": "8" * 64,
        "maapacman_dirty": False,
        "level": 1,
        "max_steps": 256,
        "level_revision": "5" * 64,
        "renderer_revision": "6" * 64,
        "ruleset_revision": "7" * 64,
        "dataset_contract_version": level1_dataset.DATASET_CONTRACT_VERSION,
    }


def test_run_manifest_records_three_repositories_and_bundled_revision(
    tmp_path,
):
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps({"splits": {"train": {"seeds": [11, 12]}}}),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "recipe_version": "maapacman-level1-ghostdoor-v3",
                "reward_recipe_version": REWARD_RECIPE_VERSION,
                "environment": {"ghost_mode": "normal", "max_steps": 256},
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "run"
    dataset_manifest.write_text(json.dumps({
        "splits": {"train": {"seeds": [11, 12]}},
        "environment": {"ghost_mode": "normal", "ruleset_revision": "5" * 64},
        "max_steps": 256,
        "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    revisions = {
        "pacman-python": {"commit": "1" * 40, "dirty": False},
        "areal-pacman": {"commit": "2" * 40, "dirty": True},
        "AReaL": {"commit": "3" * 40, "dirty": False},
    }
    env = SimpleNamespace(
        provenance={
            "maapacman_commit": "2" * 40,
            "maapacman_dirty": True,
        },
        spec=SimpleNamespace(
            api_version="3.0",
            env_id="pacman-python-level1-ghostdoor-v3",
            level_revision="4" * 64,
            renderer_revision="pacman-python:renderer-v1",
            ruleset_revision="5" * 64,
            action_tokens=("U", "D", "L", "R"),
        ),
        close=lambda: None,
    )
    argv = [
        "write_level1_manifest.py",
        "--artifact-root",
        str(artifact_root),
        "--model-revision",
        "model-revision",
        "--dataset-manifest",
        str(dataset_manifest),
        "--config",
        str(config),
    ]
    with (
        patch.object(write_level1_manifest.sys, "argv", argv),
        patch.object(
            write_level1_manifest, "model_initialization_identity",
            return_value={"path": "model-revision", "identity_sha256": "test-model-content", "fixture": True},
        ),
        patch.object(
            write_level1_manifest,
            "PygamePacmanEnv",
            return_value=env,
        ),
        patch.object(
            write_level1_manifest,
            "repository_revisions",
            return_value=revisions,
        ),
        patch.object(
            write_level1_manifest,
            "environment_metadata",
            return_value={"ghost_mode": "normal"},
        ),
        patch.object(
            write_level1_manifest,
            "validate_prepared_dataset_manifest",
            return_value=json.loads(dataset_manifest.read_text(encoding="utf-8")),
        ),
    ):
        write_level1_manifest.main()

    manifest = json.loads(
        (artifact_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["source_revisions"]) == {
        "pacman-python",
        "areal-pacman",
        "AReaL",
    }
    assert manifest["pacman_recipe_revision"] == "2" * 40
    assert manifest["maapacman_revision"] == "2" * 40
    assert manifest["environment_provenance"]["maapacman_commit"] == "2" * 40


def test_death_prefixes_are_auditable_but_never_a_successful_baseline():
    class DeadEnv:
        def reset(self, seed):
            return None, {"terminal_reason": None}

        def snapshot(self):
            return {"open": ["R"]}

        def step(self, action):
            return None, 0.0, True, False, {
                "terminal_reason": "death",
                "wall_collision": False,
            }

        def close(self):
            pass

    planner = SimpleNamespace(
        decide=lambda state: SimpleNamespace(action="R")
    )
    with (
        patch.object(level1_dataset, "PygamePacmanEnv", DeadEnv),
        patch.object(level1_dataset, "EdwardPlanner", return_value=planner),
    ):
        records = level1_dataset.planner_audit_state_records(seed=0)
        assert len(records) == 1
        assert records[0]["prefix"] == []
        audit = records[0]["prefix_audit"]
        assert audit["source_terminal_reason"] == "death"
        assert audit["prefix_verified_nonterminal"] is True
        assert audit["prefix_state_index"] == 0
        assert audit["source_episode_steps"] == 1
        assert audit["source_cleared_level"] is False
        assert audit["successful_baseline"] is False
        with pytest.raises(RuntimeError, match="did not clear level 1"):
            level1_dataset.planner_baseline_state_records(seed=0)
        with pytest.raises(RuntimeError, match="did not clear level 1"):
            level1_dataset.oracle_state_records(seed=0)


def test_planner_audit_rejects_truncated_or_unknown_outcomes():
    class TruncatedEnv:
        def reset(self, seed):
            return None, {"terminal_reason": None}

        def snapshot(self):
            return {"open": ["R"]}

        def step(self, action):
            return None, 0.0, False, True, {
                "terminal_reason": "max_steps",
                "wall_collision": False,
            }

        def close(self):
            pass

    planner = SimpleNamespace(decide=lambda state: SimpleNamespace(action="R"))
    with (
        patch.object(level1_dataset, "PygamePacmanEnv", TruncatedEnv),
        patch.object(level1_dataset, "EdwardPlanner", return_value=planner),
        pytest.raises(RuntimeError, match="did not reach an accepted terminal"),
    ):
        level1_dataset.planner_audit_state_records(seed=0)


def test_single_step_and_corridor_use_audit_prefixes_not_success_baseline():
    audit = {
        "contract_version": level1_dataset.PREFIX_AUDIT_CONTRACT_VERSION,
        "source": "edward-planner-preterminal-replay",
        "source_seed": 0,
        "prefix_state_index": 0,
        "source_episode_steps": 48,
        "source_terminal_reason": "death",
        "source_cleared_level": False,
        "successful_baseline": False,
        "prefix_verified_nonterminal": True,
        "prefix_wall_collisions": 0,
    }
    records = []
    for index in range(48):
        record_audit = {
            **audit,
            "prefix_state_index": index,
        }
        records.append(
            {
                "state_index": index,
                "prefix": ["R"] * index,
                "open_actions": ["L", "R"] if index < 24 else ["U", "D"],
                "prefix_audit": record_audit,
            }
        )
    with (
        patch.object(
            level1_dataset, "planner_audit_state_records", return_value=records
        ),
        patch.object(
            level1_dataset,
            "planner_baseline_state_records",
            side_effect=AssertionError("success baseline must not be consulted"),
        ),
    ):
        single = list(
            level1_dataset.generate_single_step_rows(
                2, split="train", seed=0
            )
        )
        corridor = list(
            level1_dataset.generate_balanced_corridor_rows(
                2, split="train", seed=0
            )
        )
    assert len(single) == len(corridor) == 2
    assert all(
        row["state_prefix_audit"]["successful_baseline"] is False
        for row in single + corridor
    )


def test_prefix_rows_fail_closed_on_missing_or_falsely_successful_audit():
    row = level1_dataset.make_episode_row(1, split="train")
    row["state_prefix_actions"] = ["R"]
    row["decision_steps"] = 1
    with pytest.raises(ValueError, match="require state_prefix_audit"):
        level1_dataset.validate_episode_row(row)

    row["state_prefix_audit"] = {
        "contract_version": level1_dataset.PREFIX_AUDIT_CONTRACT_VERSION,
        "source": "edward-planner-preterminal-replay",
        "source_seed": 0,
        "prefix_state_index": 1,
        "source_episode_steps": 2,
        "source_terminal_reason": "death",
        "source_cleared_level": False,
        "successful_baseline": True,
        "prefix_verified_nonterminal": True,
        "prefix_wall_collisions": 0,
    }
    with pytest.raises(ValueError, match="successful_baseline"):
        level1_dataset.validate_episode_row(row)
