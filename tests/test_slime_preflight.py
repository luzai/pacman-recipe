import hashlib
import json
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from pacman_recipe.level1.contracts import make_episode_record
from slime_pacman.command import build_command
from slime_pacman.config import PacmanConfig
from slime_pacman.preflight import check_dataset, check_model, check_patch

ROOT = Path(__file__).resolve().parents[1]


def test_model_preflight_result_is_json_serializable(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    (tmp_path / "config.json").write_text(json.dumps(dict(
        model_type="qwen3_5", text_config=dict(hidden_size=4096, num_hidden_layers=32),
        vision_config=dict(hidden_size=1))))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(dict(
        weight_map=dict(weight="model-00001.safetensors"))))
    (tmp_path / "model-00001.safetensors").write_bytes(b"fixture")
    processor = SimpleNamespace(image_processor=SimpleNamespace(size=UserDict({"height": 224, "width": 224})))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    result = check_model(tmp_path)
    assert json.loads(json.dumps(result))["image_size"] == {"height": 224, "width": 224}


def test_dataset_manifest_rejects_tampering_and_source_drift(tmp_path):
    if not (ROOT.parent / "slime/.git").exists():
        pytest.skip("pinned slime checkout unavailable")
    config = PacmanConfig(train_seed_count=1, validation_seed_count=1)
    files, sources = {}, None
    for split, seed in (("train", 28), ("validation", 68)):
        record = make_episode_record(
            seed,
            split=split,
            recipe_root=ROOT,
            game_root=ROOT.parent / "pacman-python",
            backend_root=ROOT.parent / "slime",
        )
        sources = record["source_revisions"]
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"metadata": {"episode_record": record}}) + "\n")
        files[path.name] = dict(
            rows=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    manifest = dict(
        schema="pacman-dataset-manifest-v1",
        training_backend="slime",
        config=config.as_dict(),
        source_revisions=sources,
        files=files,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert check_dataset(tmp_path, config, sources) == manifest
    with pytest.raises(ValueError, match="source changed"):
        check_dataset(tmp_path, config, {})
    with (tmp_path / "train.jsonl").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="hash"):
        check_dataset(tmp_path, config, sources)


def test_pinned_patch_and_smoke_command():
    if not (ROOT.parent / "slime/.git").exists():
        pytest.skip("pinned slime checkout unavailable")
    assert len(check_patch(ROOT.parent / "slime", ROOT)) == 64
    command = build_command(
        slime_root=ROOT.parent / "slime",
        model="/model",
        dataset="/data",
        run_dir="/run",
        config=PacmanConfig(),
        updates=1,
    )

    def value(name):
        return command[command.index("--" + name) + 1]

    assert value("spec") == "slime_plugins.models.qwen3_5_vl"
    # The pinned actor formats this template using a keyword argument.
    assert Path(value("save-hf").format(rollout_id=0)).name == "hf-0"
    assert value("global-batch-size") == "48" and value("micro-batch-size") == "1"
    assert value("n-samples-per-prompt") == "12" and value("eps-clip-high") == "0.2"
    assert (
        value("custom-convert-samples-to-train-data-path")
        == "slime_pacman.batching.convert_samples"
    )
    assert (
        "--use-rollout-logprobs" in command and "--normalize-advantages" not in command
    )
    assert {
        "--optimizer-cpu-offload",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--use-precision-aware-optimizer",
    } <= set(command)
    with pytest.raises(ValueError, match="acceptance"):
        build_command(
            slime_root=ROOT.parent / "slime",
            model="/model",
            dataset="/data",
            run_dir="/run",
            config=PacmanConfig(),
            updates=50,
        )


def test_eval_win_rate_counts_episodes_once(monkeypatch, tmp_path):
    Sample = pytest.importorskip("slime.utils.types").Sample
    logging_utils = pytest.importorskip("slime.observability.logging_utils")
    from slime_pacman.rollout import log_eval

    monkeypatch.setenv("PACMAN_RUN_DIR", str(tmp_path))

    captured = []
    monkeypatch.setattr(
        logging_utils, "log", lambda args, metrics, **kw: captured.append(metrics)
    )
    samples = [Sample(reward=1.0, train_metadata={"episode_id": 0})]
    samples += [Sample(reward=0.0, train_metadata={"episode_id": 1}) for _ in range(3)]
    assert log_eval(0, SimpleNamespace(), {"pacman": {"samples": samples}})
    assert captured[0]["eval/pacman/win_rate"] == 0.5
    assert captured[0]["eval/pacman/episodes"] == 2
    assert log_eval(0, SimpleNamespace(), {"pacman": {"samples": samples}})
    assert len(list((tmp_path / "metrics").glob("eval-0-*.json"))) == 2
    samples[0].weight_versions = ["1"]
    samples[-1].weight_versions = ["2"]
    with pytest.raises(ValueError, match="mixes weight"):
        log_eval(0, SimpleNamespace(), {"pacman": {"samples": samples}})
