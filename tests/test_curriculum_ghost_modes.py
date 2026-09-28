"""Real headless engine and immutable-data checks for both published stages."""

import asyncio
import copy
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
from pacman_env.env._pygame_worker import _PygameBridge
from pacman_env.env.ghost_modes import validate_ghost_state
from pacman_env.planner import EdwardPlanner
from pacman_recipe.level1.level1_dataset import make_episode_row, validate_episode_row
from pacman_recipe.level1.recipe import load_recipe_settings
from scripts.level1.dataset.prepare_level1_dataset import (
    _prepare_dataset,
    audit_episode_spec_row,
)
from scripts.level1.dataset.prepare_level1_v3_audits import audit_planner_record

ROOT = Path(__file__).resolve().parents[1]


def _small_recipe(tmp_path, stage, *, train=2, validation=1):
    raw = yaml.safe_load((ROOT / f"configs/level1/train/curriculum{stage}.yaml").read_text())
    raw["dataset_generation"].update(train_episodes=train, validation_episodes=validation)
    path = tmp_path / f"curriculum{stage}-fixture.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_ghost_mode_schema_uses_new_dataset_contract() -> None:
    from pacman_recipe.level1.level1_dataset import DATASET_CONTRACT_VERSION
    from scripts.level1.dataset.prepare_level1_dataset import (
        DATASET_PREPARATION_CONTRACT_VERSION,
    )
    from scripts.level1.dataset.prepare_level1_v3_audits import (
        AUDIT_CONTRACT_VERSION,
    )

    assert DATASET_CONTRACT_VERSION == "maapacman-level1-dataset-v4"
    assert DATASET_PREPARATION_CONTRACT_VERSION == "maapacman-level1-split-bundle-v4"
    assert AUDIT_CONTRACT_VERSION == "maapacman-level1-planner-audit-v4"


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/level1/report/export_level1_rollout_video.py",
        "scripts/level1/report/export_level1_ab_demo_video.py",
        "scripts/level1/evaluate/evaluate_single_step_wall.py",
    ],
)
def test_replay_consumers_forward_recorded_mode(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PygamePacmanEnvConfig"
    )
    for mode in ("disabled", "normal"):
        recorded = {"level": 1, "max_steps": 32, "ghost_mode": mode}
        config = eval(
            compile(ast.Expression(call), relative, "eval"),
            {
                "PygamePacmanEnvConfig": PygamePacmanEnvConfig,
                "episode": recorded,
                "requested": recorded,
            },
        )
        assert config.ghost_mode == mode


def test_edward_prompt_keeps_common_observation_fields_between_modes():
    # Exercise the real pure renderer without importing distributed AReaL.
    from pacman_recipe.level1.prompts import compact_edward_decision_prompt as render
    constraint = SimpleNamespace(rendered_choices=("0",))
    prompts = [
        render({"ghosts": ghosts}, (), constraint)
        for ghosts in ([], [{"id": 0, "state": "normal", "position": [1, 2]}])
    ]
    lines = [prompt.splitlines() for prompt in prompts]
    assert lines[0][:-2] == lines[1][:-2]
    assert lines[0][-1] == lines[1][-1]
    states = [
        json.loads(next(line for line in prompt if line.startswith("{")))
        for prompt in lines
    ]
    assert states[0].keys() == states[1].keys()
    assert states[0]["ghosts"] == []
    assert states[1]["ghosts"] == [[0, "normal", [1, 2]]]


def test_formal_stages_have_explicit_distinct_settings():
    first, second = [
        yaml.safe_load((ROOT / f"configs/level1/train/curriculum{i}.yaml").read_text())
        for i in (1, 2)
    ]
    assert first["environment"] == {
        "ghost_mode": "disabled", "max_steps": 512,
        "episode_life_mode": "single_death",
    }
    assert second["environment"] == {
        "ghost_mode": "normal", "max_steps": 512,
        "episode_life_mode": "original_three_lives",
    }
    assert first["actor"]["path"] == "Qwen/Qwen3.5-9B"
    assert second["actor"]["path"] == "Qwen/Qwen3.5-9B"
    assert first["actor"]["optimizer"]["lr"] == 5e-7
    assert second["actor"]["optimizer"]["lr"] == 5e-7
    assert first["nearest_pellet_alpha"] == second["nearest_pellet_alpha"] == 0.1
    assert first["action_protocol"] == "direct-open-action-token-v1"
    assert first["edward_options"] is False
    assert first["reward_objective_contract"] == "step_local_raw_v1"
    assert second["action_protocol"] == "edward-option-code-v1"
    assert second["edward_options"] is True
    assert second["reward_objective_contract"] == "episode_return_group_v1"
    for config, expected_train_rows, expected_updates in (
        (first, 80, 100),
        (second, 40, 50),
    ):
        assert config["recover"]["mode"] == "disabled"
        assert config["dataset_generation"] == {
            "train_episodes": expected_train_rows,
            "validation_episodes": 4,
            "seed": 28,
        }
        assert config["saver"]["freq_steps"] == 1
        if config is first:
            assert all(
                config["evaluator"][key] is None
                for key in ("freq_steps", "freq_epochs", "freq_secs")
            )
            assert config["evaluator"]["eval_before_train"] is False
        else:
            assert config["evaluator"] == {
                "experiment_name": "${experiment_name}",
                "trial_name": "${trial_name}",
                "fileroot": "${cluster.fileroot}",
                "freq_epochs": 1,
                "freq_steps": None,
                "freq_secs": None,
                "eval_before_train": True,
            }
        assert (
            config["dataset_generation"]["train_episodes"]
            // config["train_dataset"]["batch_size"]
            * config["total_train_epochs"]
            == expected_updates
        )
    for field in (
        "image_prompt_style",
        "observation_mode",
        "gconfig",
    ):
        assert first[field] == second[field]


@pytest.mark.parametrize("mode", ["disabled", "normal"])
def test_real_mode_reset_step_and_seed_determinism(mode, monkeypatch):
    # Ambient legacy curriculum must never turn fruit off or override C2.
    monkeypatch.setenv("MAAPACMAN_CURRICULUM", "1")
    traces = []
    for _ in range(2):
        with PygamePacmanEnv(PygamePacmanEnvConfig(ghost_mode=mode)) as env:
            frame, info = env.reset(seed=7)
            assert frame.shape == (400, 336, 3)
            initial = copy.deepcopy(info["ghosts"])
            trace = []
            for _ in range(4):
                frame, _, terminated, truncated, info = env.step("S")
                validate_ghost_state(env.snapshot(), mode)
                for atomic in info["atomic_substeps"]:
                    validate_ghost_state(atomic, mode)
                trace.append(
                    (hashlib.sha256(frame.tobytes()).hexdigest(), info["ghosts"])
                )
                if terminated or truncated:
                    break
            if mode == "normal":
                assert initial != info["ghosts"]
            else:
                assert initial == info["ghosts"] == []
            traces.append(trace)
    assert traces[0] == traces[1]


def test_disabled_ghosts_stay_absent_after_power_pellets_and_fruit_ticks():
    planner = EdwardPlanner()
    power_events = 0
    with PygamePacmanEnv(
        PygamePacmanEnvConfig(ghost_mode="disabled", max_steps=64)
    ) as env:
        env.reset(seed=0)
        for _ in range(64):
            _, _, terminated, truncated, info = env.step(
                planner.decide(env.snapshot()).action
            )
            assert not terminated
            assert info["ghosts"] == []
            for atomic in info["atomic_substeps"]:
                validate_ghost_state(atomic, "disabled")
                power_events += sum(
                    event["event_type"] == "power_pellet_eaten"
                    for event in atomic["events"]
                )
            if truncated:
                break
        assert power_events > 0
        assert info["step"] == 64
        assert env.snapshot()["fruit_timer"] > 0


def test_missing_native_switch_fails_closed():
    bridge = object.__new__(_PygameBridge)
    bridge._ghost_mode = "disabled"
    with pytest.raises(RuntimeError, match="explicit ghost-mode"):
        bridge._capture({})


@pytest.mark.parametrize("mode", ["disabled", "normal"])
def test_dataset_row_mode_is_part_of_identity(mode):
    row = make_episode_row(1, split="train", ghost_mode=mode)
    validate_episode_row(row)
    row["env"]["ghost_mode"] = "normal" if mode == "disabled" else "disabled"
    with pytest.raises(ValueError, match="ruleset_revision"):
        validate_episode_row(row)
    del row["env"]["ghost_mode"]
    with pytest.raises(ValueError, match="ghost_mode"):
        validate_episode_row(row)


@pytest.mark.parametrize("stage", [1, 2])
def test_real_dataset_anchors_match_stage_and_reject_tampering(tmp_path, stage):
    config = _small_recipe(tmp_path, stage)
    environment, generation = load_recipe_settings(config)
    output = tmp_path / "dataset"
    _prepare_dataset(
        SimpleNamespace(
            config=config,
            output_root=output,
            train_episodes=2,
            validation_episodes=1,
            seed=None,
            max_steps=None,
            write_hf=False,
            pacman_python_root=None,
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["environment"]["ghost_mode"] == environment.ghost_mode
    assert manifest["max_steps"] == environment.max_steps
    assert manifest["splits"]["train"]["seeds"] == [
        generation.seed,
        generation.seed + 1,
    ]
    assert manifest["splits"]["validation"]["seeds"] == [generation.seed + 2]
    for line in (output / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        audit_episode_spec_row(row)
        anchor = row["audit_anchor"]
        assert len(anchor["ghosts"]) == (0 if stage == 1 else 4)
        corrupted = copy.deepcopy(anchor)
        corrupted["atomic_substeps"][0]["ghost_mode"] = (
            "normal" if stage == 1 else "disabled"
        )
        with pytest.raises(ValueError, match="ghost_mode"):
            audit_planner_record(corrupted)


def test_disabled_mode_rejects_ghost_events():
    with pytest.raises(ValueError, match="ghost/death"):
        validate_ghost_state(
            {
                "ghost_mode": "disabled",
                "ghosts": [],
                "edible_ticks": 0,
                "events": [{"event_type": "death"}],
            },
            "disabled",
        )


@pytest.mark.parametrize("mode", ["disabled", "normal"])
def test_parse_failure_keeps_fail_closed_ghost_evidence(mode):
    from pacman_recipe.level1.trajectories import audit_trajectory
    from pacman_recipe.level1.workflow import PacmanImageOnlyWorkflow

    row = make_episode_row(
        1,
        split="train",
        ghost_mode=mode,
        max_steps=512,
    )
    workflow = PacmanImageOnlyWorkflow()
    reward = asyncio.run(
        workflow.run(row, scripted_actions=["Action: R"])
    )
    assert reward == -1.0
    payload = workflow.last_episode
    assert payload is not None
    audit_trajectory(payload)

    corrupted = copy.deepcopy(payload)
    corrupted["trajectory"][-1]["ghosts"] = (
        [{"id": index} for index in range(4)] if mode == "disabled" else []
    )
    with pytest.raises(ValueError, match="ghost count"):
        audit_trajectory(corrupted)
    if mode == "disabled":
        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][-1]["edible_ticks"] = 1
        with pytest.raises(ValueError, match="vulnerability ticks"):
            audit_trajectory(corrupted)


@pytest.mark.parametrize("stage", [1, 2])
def test_training_preflight_rejects_wrong_stage_data(tmp_path, monkeypatch, stage):
    import train_areal

    # Isolate framework discovery; exercise the real dataset/config validation.
    monkeypatch.setattr(
        train_areal,
        "_load_workflow",
        lambda _: SimpleNamespace(
            __module__="pacman_recipe.level1.workflow",
            __name__="PacmanNativeVisionWorkflow",
        ),
    )
    config = _small_recipe(tmp_path, stage, train=1, validation=1)
    environment, _ = load_recipe_settings(config)
    args = ["--config", str(config)]
    output = tmp_path / "dataset"
    _prepare_dataset(SimpleNamespace(
        config=config, output_root=output, train_episodes=None,
        validation_episodes=None, seed=None, max_steps=None,
        write_hf=True, pacman_python_root=None,
    ))
    for split, key in (("train", "train_dataset"), ("validation", "valid_dataset")):
        path = output / f"{split}_hf"
        args.append(f"{key}.path={path}")
    assert train_areal._production_dry_run(config, config_args=args)
    opposite = "normal" if stage == 1 else "disabled"
    with pytest.raises(ValueError, match="ghost_mode"):
        train_areal._production_dry_run(
            config, config_args=[*args, f"environment.ghost_mode={opposite}"]
        )
    horizon = 256 if stage == 1 else 32
    with pytest.raises(ValueError, match="max_steps"):
        train_areal._production_dry_run(
            config, config_args=[*args, f"environment.max_steps={horizon}"]
        )


@pytest.mark.parametrize("stage", [1, 2])
def test_new_config_sections_support_omegaconf_structured_loading(stage):
    from omegaconf import OmegaConf
    from pacman_recipe.level1.recipe import (
        DatasetGenerationConfig,
        EnvironmentConfig,
        PlannerAuditConfig,
    )

    raw = yaml.safe_load(
        (ROOT / f"configs/level1/train/curriculum{stage}.yaml").read_text()
    )
    for key, schema in (
        ("environment", EnvironmentConfig),
        ("dataset_generation", DatasetGenerationConfig),
        ("planner_audit", PlannerAuditConfig),
    ):
        value = OmegaConf.merge(OmegaConf.structured(schema), raw[key])
        instance = OmegaConf.to_object(value)
        assert isinstance(instance, schema)
        assert vars(instance) == vars(schema(**raw[key]))


@pytest.mark.parametrize("stage", [1, 2])
def test_hf_roundtrip_and_run_manifest_preserve_mode(tmp_path, monkeypatch, stage):
    datasets = pytest.importorskip("datasets")
    from scripts.level1.dataset import write_level1_manifest

    monkeypatch.setattr(
        write_level1_manifest, "model_initialization_identity",
        lambda path: {"path": str(path), "identity_sha256": "test-model-content", "fixture": True},
    )

    config = _small_recipe(tmp_path, stage, train=1, validation=1)
    output = tmp_path / "dataset"
    _prepare_dataset(
        SimpleNamespace(
            config=config,
            output_root=output,
            train_episodes=1,
            validation_episodes=1,
            seed=None,
            max_steps=None,
            write_hf=True,
            pacman_python_root=None,
        )
    )
    row = datasets.load_from_disk(str(output / "train_hf"))[0]
    audit_episode_spec_row(row)
    mode = "disabled" if stage == 1 else "normal"
    assert row["env"]["ghost_mode"] == mode
    assert len(row["audit_anchor"]["ghosts"]) == (0 if stage == 1 else 4)
    argv = [
        "write_level1_manifest.py",
        "--artifact-root",
        str(tmp_path / "run"),
        "--model-revision",
        "test-model",
        "--dataset-manifest",
        str(output / "manifest.json"),
        "--config",
        str(config),
        "--smoke-updates",
        "2",
    ]
    monkeypatch.setattr("sys.argv", argv)
    write_level1_manifest.main()
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["ghost_mode"] == mode
    assert manifest["environment_provenance"]["ghost_mode"] == mode
    assert manifest["total_train_steps"] == 2
    other_config = ROOT / f"configs/level1/train/curriculum{3 - stage}.yaml"
    argv[argv.index("--config") + 1] = str(other_config)
    with pytest.raises(ValueError, match="ghost_mode"):
        write_level1_manifest.main()
