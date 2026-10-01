"""Read-only source, dataset, model, and optional GPU-runtime checks."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from pacman_recipe.level1.contracts import repository_identity, validate_episode_record
from . import SLIME_REVISION
from .config import load_config


def check_dataset(directory, config, sources):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") == "pacman-backplay-smoke-dataset-v1":
        from .backplay_dataset import check_dataset as check_backplay_dataset
        return check_backplay_dataset(directory, config, sources)
    if manifest.get("schema") == "pacman-dynamic-bank-dataset-v1":
        from .dynamic_dataset import check_dataset as check_dynamic_dataset
        return check_dynamic_dataset(directory, config, sources)
    if (
        manifest.get("schema") != "pacman-dataset-manifest-v1"
        or manifest.get("training_backend") != "slime"
    ):
        raise ValueError("invalid dataset manifest")
    if (
        manifest["config"] != config.as_dict()
        or manifest["source_revisions"] != sources
    ):
        raise ValueError("dataset config/source changed; generate a fresh dataset")
    if set(manifest["files"]) != {"train.jsonl", "validation.jsonl"}:
        raise ValueError("unexpected dataset splits")
    for split, seeds in (
        ("train", config.train_seeds),
        ("validation", config.validation_seeds),
    ):
        path = directory / f"{split}.jsonl"
        entry = manifest["files"][path.name]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("dataset file hash mismatch")
        rows = [
            json.loads(line)["metadata"]["episode_record"]
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != entry["rows"] or [
            r["environment"]["seed"] for r in rows
        ] != list(seeds):
            raise ValueError("dataset seed membership changed")
        for row in rows:
            validate_episode_record(row, expected_sources=sources)
            if (
                row["split"] != split
                or row["environment"]["max_steps"] != config.max_steps
            ):
                raise ValueError("dataset split/horizon mismatch")
    return manifest


def check_patch(slime_root, recipe_root):
    revision = subprocess.check_output(
        ["git", "-C", str(slime_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SLIME_REVISION:
        raise ValueError("slime revision does not match the pinned revision")
    patch = Path(recipe_root) / "patches/slime-pacman-metadata.patch"
    subprocess.run(
        [
            "git",
            "-C",
            str(slime_root),
            "apply",
            "--reverse",
            "--check",
            str(patch.resolve()),
        ],
        check=True,
    )
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def check_model(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config", {})
    if config.get("model_type") != "qwen3_5" or (
        text.get("hidden_size"),
        text.get("num_hidden_layers"),
    ) != (4096, 32):
        raise ValueError("expected the Qwen3.5-9B vision model")
    if not config.get("vision_config"):
        raise ValueError("vision tower is missing")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        path = (directory / shard).resolve()
        if path.parent != directory.resolve() or not path.is_file():
            raise ValueError("model shard missing or outside model directory")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    return {
        "path": str(directory.resolve()),
        "shards": shards,
        "processor_class": type(processor).__name__,
        "image_size": dict(processor.image_processor.size),
    }


def check_runtime():
    from .command import PROCESSOR_ENV

    for key, value in PROCESSOR_ENV.items():
        if os.environ.get(key) != value:
            raise ValueError(f"runtime requires {key}={value}")
    importlib.import_module("slime_pacman.sglang_processors.qwen35")
    versions = {}
    for name, module in (
        ("torch", "torch"),
        ("sglang", "sglang"),
        ("megatron-core", "megatron.core"),
        ("transformer-engine", "transformer_engine.pytorch"),
        ("transformers", "transformers"),
        ("ray", "ray"),
    ):
        importlib.import_module(module)
        versions[name] = importlib.metadata.version(name)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("runtime acceptance requires eight visible CUDA GPUs")
    if versions["sglang"] != "0.5.15.post1":
        raise ValueError(
            "SGLang differs from the pinned slime Docker runtime; re-audit before changing it"
        )
    versions["cuda"] = torch.version.cuda
    versions["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(8)]
    return versions


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/slime/c2.yaml")
    parser.add_argument("--slime-root", type=Path, default=root.parent / "slime")
    parser.add_argument("--game-root", type=Path, default=root.parent / "pacman-python")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    sources = {
        "pacman-recipe": repository_identity(root),
        "pacman-python": repository_identity(args.game_root),
        "slime": repository_identity(args.slime_root),
    }
    patch_hash = check_patch(args.slime_root, root)
    manifest = check_dataset(args.dataset, load_config(args.config), sources)
    result = {
        "schema": "pacman-preflight-v1",
        "sources": sources,
        "patch_sha256": patch_hash,
        "dataset_files": manifest["files"],
        "model": check_model(args.model) if args.model else None,
        "runtime": check_runtime() if args.runtime else None,
        "gpu_acceptance": "not_evaluated",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
