from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
from pacman_recipe.level1.recipe import load_recipe_settings, recipe_contract_metadata

from pacman_recipe.level1.level1_dataset import (
    environment_metadata,
    repository_revisions,
)
from pacman_recipe.level1.rewards import REWARD_RECIPE_VERSION
from scripts.level1.dataset.prepare_level1_dataset import (
    validate_prepared_dataset_manifest,
)
from scripts.level1.train.validate_model_checkpoint import (
    WEIGHT_LAYOUTS,
    _validate_weight_layout,
    validate_model_checkpoint,
)


# Runtime model assets only: optimizer/recovery state, training logs, README,
# screenshots and unrelated files must not affect initialization identity.
MODEL_RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
)
MODEL_FILE_REFERENCE_KEYS = frozenset(
    {
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "sp_model_file",
        "sentencepiece_model_file",
        "chat_template_file",
    }
)
EXPORT_PROVENANCE_FILES = ("merge_manifest.json", "merge_manifest.sha256")


def _safe_model_file(root: Path, path: Path) -> Path:
    """Require every used file (including symlink targets) inside the bundle."""
    if not path.absolute().is_relative_to(root) or ".." in path.parts:
        raise ValueError(f"initialization file is externally linked: {path.name}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"missing initialization file: {path.name}") from exc
    if not resolved.is_relative_to(root):
        raise ValueError(f"initialization file is externally linked: {path.name}")
    if not resolved.is_file():
        raise ValueError(f"initialization asset is not a file: {path.name}")
    return path


def _metadata_file_references(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in MODEL_FILE_REFERENCE_KEYS and item is not None:
                if not isinstance(item, str) or not item:
                    raise ValueError(f"invalid model metadata file reference: {key}")
                yield item
            elif isinstance(item, (dict, list)):
                yield from _metadata_file_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _metadata_file_references(item)


def _initialization_files(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Reject outside targets before even inspecting weight headers.
    for single_name, index_name in WEIGHT_LAYOUTS:
        candidate = root / (single_name or index_name)
        if candidate.exists() or candidate.is_symlink():
            _safe_model_file(root, candidate)
            if index_name:
                index = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(index, dict) or not isinstance(
                    index.get("weight_map"), dict
                ):
                    raise ValueError("invalid initialization weight index")
                for shard in index["weight_map"].values():
                    if (
                        not isinstance(shard, str)
                        or Path(shard).name != shard
                        or "\\" in shard
                    ):
                        raise ValueError("invalid initialization shard filename")
                    _safe_model_file(root, root / shard)
    layout, weights = _validate_weight_layout(root)
    names = {layout, *weights}
    names.update(
        name
        for name in MODEL_RUNTIME_FILES
        if (root / name).exists() or (root / name).is_symlink()
    )
    for name in names:
        _safe_model_file(root, root / name)
    # Some tokenizers explicitly name additional vocabulary/model files. Bind
    # these too; stale external paths are not a self-contained release bundle.
    references = set()
    for name in sorted(names):
        if name in MODEL_RUNTIME_FILES and name.endswith(".json"):
            document = json.loads((root / name).read_text(encoding="utf-8"))
            for reference in _metadata_file_references(document):
                path = Path(reference)
                if ".." in path.parts:
                    raise ValueError(
                        "initialization metadata file reference contains parent traversal"
                    )
                path = path if path.is_absolute() else root / path
                _safe_model_file(root, path)
                references.add(path.relative_to(root).as_posix())
    names.update(references)
    templates = root / "chat_templates"
    if templates.exists() or templates.is_symlink():
        if not templates.resolve(strict=True).is_relative_to(root):
            raise ValueError("initialization chat_templates is externally linked")
        if not templates.is_dir():
            raise ValueError("initialization chat_templates is not a directory")
        names.update(
            path.relative_to(root).as_posix() for path in templates.glob("*.jinja")
        )
    provenance = tuple(
        name
        for name in EXPORT_PROVENANCE_FILES
        if (root / name).exists() or (root / name).is_symlink()
    )
    for name in (*names, *provenance):
        _safe_model_file(root, root / name)
    return tuple(sorted(names)), provenance


def _file_snapshot(root: Path, name: str) -> tuple:
    path = _safe_model_file(root, root / name)
    stat = path.stat()
    return (
        str(path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _stable_file_sha256(root: Path, name: str, before: tuple) -> str:
    path = _safe_model_file(root, root / name)
    if _file_snapshot(root, name) != before:
        raise ValueError(f"initialization file changed before hashing: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        # Windows stat and fstat can expose different ctime semantics. Compare
        # their common identity/size/mtime fields, then compare ctime within
        # each API before/after so real mutations still fail closed.
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != before[1:5]:
            raise ValueError(f"initialization file changed while opening: {name}")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise ValueError(f"initialization file changed during hashing: {name}")
    if _file_snapshot(root, name) != before:
        raise ValueError(f"initialization file changed during hashing: {name}")
    return digest.hexdigest()


def model_initialization_identity(checkpoint: Path) -> dict[str, Any]:
    """Bind actual local initialization bytes, not a path label or stage claim."""
    root = checkpoint.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model initialization must be a local checkpoint directory")
    runtime_names, provenance_names = _initialization_files(root)
    snapshots = {
        name: _file_snapshot(root, name) for name in (*runtime_names, *provenance_names)
    }
    # Default validator checks architecture keys/shapes on meta plus offline
    # tokenizer/processor metadata and any export hashes, without a forward.
    validation = validate_model_checkpoint(root)
    if not validation.get("architecture_verified") or not validation.get(
        "offline_metadata"
    ):
        raise ValueError(
            "initialization requires verified architecture and offline metadata"
        )
    hashes = {
        name: _stable_file_sha256(root, name, snapshots[name]) for name in snapshots
    }
    if _initialization_files(root) != (runtime_names, provenance_names):
        raise ValueError(
            "initialization file inventory changed during validation/hashing"
        )
    for name, before in snapshots.items():
        if _file_snapshot(root, name) != before:
            raise ValueError(
                f"initialization file changed during validation/hashing: {name}"
            )
    content = {
        "identity_contract": "checkpoint-runtime-files-sha256-v1",
        "files_sha256": {name: hashes[name] for name in runtime_names},
    }
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        **content,
        "identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "path": str(root),
        "path_is_hf_revision": False,
        "hf_revision_verified": False,
        "training_lineage_verified": False,
        "training_completion_verified": False,
        "identity_scope": "runtime file bytes; excludes location and export provenance",
        "export_provenance_sha256": {name: hashes[name] for name in provenance_names},
        "merge_manifest_sha256": hashes.get("merge_manifest.json"),
        "checkpoint_validation": validation,
        "snapshot_scope": "files stable during validation and hashing; not a later model-load attestation",
    }


def _publish_manifest(artifact_root: Path, payload: dict[str, Any]) -> None:
    """Expose only a fully serialized manifest, atomically and without overwrite."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".manifest-",
            suffix=".tmp",
            dir=artifact_root,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, artifact_root / "manifest.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _Once(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be specified only once")
        setattr(namespace, self.dest, value)


def reward_ablation_metadata(config, *, reward_ablation=None, smoke_updates=None):
    """Bind explicit ablation intent without changing default manifests."""
    direct = config.get("action_protocol") == "direct-open-action-token-v1"
    if reward_ablation is None:
        if direct and config.get("nearest_pellet_scale_by_cleared_ratio") is False:
            raise ValueError("fixed-distance C1 requires explicit --reward-ablation")
        return None
    if reward_ablation == "binary-outcome":
        # Terminal win signal only: every shaping coefficient must be zero, so a
        # manifest carrying this opt-in cannot be confused with the shaped recipe.
        zeroed = (
            "normal_pellet_reward",
            "power_pellet_reward",
            "ghost_reward",
            "fruit_reward",
            "step_penalty",
            "wall_penalty",
            "death_penalty",
            "safety_refusal_penalty",
            "nearest_pellet_alpha",
        )
        nonzero = [f for f in zeroed if float(config.get(f, 0.0)) != 0.0]
        if nonzero:
            raise ValueError(
                "binary-outcome ablation requires zero shaping coefficients, "
                f"got non-zero {nonzero}"
            )
        if float(config.get("completion_reward", 0.0)) <= 0.0:
            raise ValueError(
                "binary-outcome ablation requires completion_reward>0 as the only reward"
            )
        return {
            "experiment_role": "B",
            "opt_in": "binary-outcome",
            "completion_reward": float(config.get("completion_reward")),
            "shaping_coefficients": "zeroed",
            "episode_life_mode": config.get("environment", {}).get("episode_life_mode"),
        }
    if reward_ablation != "fixed-distance" or smoke_updates != 4:
        raise ValueError("--reward-ablation fixed-distance requires --smoke-updates 4")
    if not (
        direct
        and config.get("environment", {}).get("ghost_mode") == "disabled"
        and config.get("edward_options") is False
        and config.get("nearest_pellet_alpha") == 0.1
        and config.get("nearest_pellet_scale_by_cleared_ratio") is False
        and config.get("total_train_steps") is None
        and config.get("total_train_epochs") == 5
        and config.get("dataset_generation", {}).get("train_episodes") == 80
        and config.get("train_dataset", {}).get("batch_size") == 4
        and config.get("recover", {}).get("mode") == "disabled"
    ):
        raise ValueError("fixed-distance ablation requires fresh C1 alpha=0.1, scale=false and full 100-update recipe")
    optimizer = config.get("actor", {}).get("optimizer", {})
    if not (
        optimizer.get("lr_scheduler_type") == "constant"
        and optimizer.get("warmup_steps_proportion") == 0
        and optimizer.get("lr") == 5e-7
    ):
        raise ValueError("fixed-distance ablation requires matched constant 5e-7 learning rate and zero warmup")
    return {
        "experiment_role": "A",
        "opt_in": "fixed-distance",
        "requested_update_cap": 4,
        "full_recipe_schedule_updates": 100,
        "nearest_pellet_alpha": 0.1,
        "nearest_pellet_scale_by_cleared_ratio": False,
        "optimizer_scheduler_initialization": "fresh",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an auditable level-1 run manifest."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        "--model-revision",
        dest="model_revision",
        required=True,
        help="Complete local initialization checkpoint (legacy --model-revision is a path, not a HF revision).",
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-updates", type=int, action=_Once)
    parser.add_argument(
        "--reward-ablation",
        choices=["fixed-distance", "binary-outcome"],
        action=_Once,
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    ablation = reward_ablation_metadata(
        config, reward_ablation=args.reward_ablation, smoke_updates=args.smoke_updates
    )
    environment, _ = load_recipe_settings(args.config)
    if args.smoke_updates is not None and args.smoke_updates < 1:
        raise ValueError("smoke updates must be positive")
    env = PygamePacmanEnv(
        PygamePacmanEnvConfig(
            ghost_mode=environment.ghost_mode,
            max_steps=environment.max_steps,
            episode_life_mode=environment.episode_life_mode,
        )
    )
    try:
        spec = env.spec
        environment_provenance = env.provenance
    finally:
        env.close()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    recipe_contract = recipe_contract_metadata(config)
    source_revisions = repository_revisions()
    dataset = validate_prepared_dataset_manifest(
        args.dataset_manifest,
        expected_environment=environment_metadata(environment.ghost_mode),
        expected_source_revisions=source_revisions,
        expected_training_config_sha256=config_sha256,
        expected_recipe_contract=recipe_contract,
    )
    if dataset.get("environment", {}).get("ghost_mode") != environment.ghost_mode:
        raise ValueError("dataset manifest ghost_mode does not match config")
    if dataset.get("environment", {}).get("ruleset_revision") != spec.ruleset_revision:
        raise ValueError("dataset manifest ruleset_revision does not match config")
    if dataset.get("max_steps") != environment.max_steps:
        raise ValueError("dataset manifest max_steps does not match config")
    if dataset.get("training_config_sha256") != config_sha256:
        raise ValueError("dataset was not prepared with this training config")
    if config.get("recipe_version") != "maapacman-level1-ghostdoor-v3":
        raise ValueError("run manifest requires the ghostdoor-v3 recipe")
    if config.get("reward_recipe_version") != REWARD_RECIPE_VERSION:
        raise ValueError("run manifest requires the API-v3 reward recipe")
    split_seeds = {
        int(seed)
        for split in dataset.get("splits", {}).values()
        for seed in split.get("seeds", [])
    }
    if not split_seeds:
        raise ValueError("dataset manifest does not enumerate split seeds")
    recipe_revision = source_revisions["areal-pacman"]["commit"]
    if (
        environment_provenance["maapacman_commit"] != recipe_revision
        or bool(environment_provenance["maapacman_dirty"])
        is not source_revisions["areal-pacman"]["dirty"]
    ):
        raise RuntimeError(
            "maapacman must be bundled in the active areal-pacman checkout"
        )
    initialization = model_initialization_identity(Path(args.model_revision))
    payload = {
        "recipe_version": config["recipe_version"],
        "reward_recipe_version": config["reward_recipe_version"],
        "recipe_contract": recipe_contract,
        "action_protocol": recipe_contract["harness"]["action_protocol"],
        "prompt_version": recipe_contract["prompt"]["version"],
        "prompt_template_sha256": recipe_contract["prompt"]["template_sha256"],
        "source_revisions": source_revisions,
        "pacman_recipe_revision": recipe_revision,
        "maapacman_revision": recipe_revision,
        "pacman_python_revision": source_revisions["pacman-python"]["commit"],
        "areal_revision": source_revisions["AReaL"]["commit"],
        "env_api_version": spec.api_version,
        "env_id": spec.env_id,
        "ghost_mode": environment.ghost_mode,
        "episode_life_mode": environment.episode_life_mode,
        "max_steps": environment.max_steps,
        "total_train_steps": (
            args.smoke_updates
            or config.get("total_train_steps")
            or (recipe_contract["data"]["updates_per_epoch"] or 0)
            * (recipe_contract["data"]["epochs"] or 0)
        ),
        "level_revision": spec.level_revision,
        "renderer_revision": spec.renderer_revision,
        "ruleset_revision": spec.ruleset_revision,
        "environment_provenance": environment_provenance,
        "action_tokens": list(spec.action_tokens),
        "model_revision": args.model_revision,
        "model_revision_kind": "local_checkpoint_path_not_hf_revision",
        "model_initialization_identity": initialization,
        "actor_initialization": initialization["path"],
        "reference_initialization": initialization["path"],
        "actor_initialization_identity_sha256": initialization["identity_sha256"],
        "reference_initialization_identity_sha256": initialization["identity_sha256"],
        "optimizer_scheduler_initialization": "fresh",
        "recovery_mode": (config.get("recover") or {}).get("mode"),
        "dataset": dataset,
        "seed_set": sorted(split_seeds),
        "config_sha256": config_sha256,
        "launch_command": shlex.join(sys.argv),
    }
    if ablation is not None:
        payload["reward_ablation"] = ablation
    _publish_manifest(args.artifact_root, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
