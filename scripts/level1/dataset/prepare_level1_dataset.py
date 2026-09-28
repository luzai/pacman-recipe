from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from pacman_recipe.level1.recipe import (
    DIRECT_ACTION_PROTOCOL,
    EDWARD_OPTION_PROTOCOL,
    load_recipe_document,
    load_recipe_settings,
    recipe_contract_metadata,
)

from pacman_recipe.level1.level1_dataset import (
    DATASET_CONTRACT_VERSION,
    audit_anchor_semantics,
    environment_metadata,
    generate_episode_rows,
    repository_revisions,
    validate_episode_row,
    write_hf_dataset,
    write_jsonl,
)
try:
    from scripts.level1.dataset.prepare_level1_v3_audits import (
        AUDIT_CONTRACT_VERSION,
        REPO_ROOT,
        _canonical_sha256,
        _generator_provenance,
        _reward_config,
        audit_planner_record,
        collect_initial_audit_anchor,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.level1"}:
        raise
    from prepare_level1_v3_audits import (  # type: ignore[no-redef]
        AUDIT_CONTRACT_VERSION,
        REPO_ROOT,
        _canonical_sha256,
        _generator_provenance,
        _reward_config,
        audit_planner_record,
        collect_initial_audit_anchor,
    )


DATASET_PREPARATION_CONTRACT_VERSION = "maapacman-level1-split-bundle-v4"
DATASET_ROLES = ("train", "validation")


def _anchor_contract(action_protocol: str) -> dict[str, Any]:
    return {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        **audit_anchor_semantics(action_protocol),
        "planner": "EdwardPlanner" if action_protocol == EDWARD_OPTION_PROTOCOL else None,
        "direct_action_selection": "first_legal_U_D_L_R" if action_protocol == DIRECT_ACTION_PROTOCOL else None,
        "per_row_reward_breakdown": "complete_and_audited",
    }


def _tree_sha256(root: Path) -> str:
    """Hash a directory without embedding its machine-specific absolute path."""

    digest = hashlib.sha256()
    discovered = list(root.rglob("*"))
    if any(path.is_symlink() for path in discovered):
        raise ValueError(f"artifact directory cannot contain symlinks: {root}")
    files = sorted(path for path in discovered if path.is_file())
    if not files:
        raise ValueError(f"cannot hash empty artifact directory: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_text_exclusive(
    path: Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Create an immutable manifest/checksum file and reject replacement."""

    with path.open("x", encoding=encoding, newline="\n") as stream:
        stream.write(text)


def _split_generator_provenance() -> dict[str, Any]:
    return _generator_provenance(
        [
            Path(__file__),
            REPO_ROOT
            / "scripts"
            / "level1"
            / "dataset"
            / "prepare_level1_v3_audits.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "level1_dataset.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "recipe.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "prompts.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "rewards.py",
            REPO_ROOT / "pacman_recipe" / "level1" / "trajectories.py",
        ]
    )


def audit_episode_spec_row(row: dict[str, Any]) -> None:
    """Validate the real one-step evidence nested beside an episode spec."""

    anchor = row.get("audit_anchor")
    if not isinstance(anchor, dict):
        raise ValueError("episode specification is missing audit_anchor")
    audit_planner_record(anchor)
    if row.get("source_revisions") != anchor.get("source_revisions"):
        raise ValueError("episode source revisions differ from audit_anchor")
    if anchor.get("audit_role") != "episode_spec_audit_anchor":
        raise ValueError("episode audit_anchor has an invalid role")
    protocol = row.get("action_protocol", EDWARD_OPTION_PROTOCOL)
    if anchor.get("audit_anchor_semantics") != audit_anchor_semantics(protocol):
        raise ValueError("episode audit_anchor semantics are invalid")
    if anchor.get("seed") != row["env"]["seed"] or anchor.get("step") != 1:
        raise ValueError("episode audit_anchor does not match the initial seed/state")
    if anchor.get("terminated") or anchor.get("truncated"):
        raise ValueError("episode audit_anchor initial transition must be nonterminal")
    identity_fields = {
        "ghost_mode",
        "name",
        "api_version",
        "backend",
        "pacman_python_revision",
        "pacman_python_source_sha256",
        "pacman_python_dirty",
        "maapacman_revision",
        "maapacman_env_source_sha256",
        "maapacman_dirty",
        "level_revision",
        "renderer_revision",
        "ruleset_revision",
        "max_steps",
    }
    if any(
        anchor["env"].get(field) != row["env"].get(field)
        for field in identity_fields
    ):
        raise ValueError("episode audit_anchor provenance differs from episode spec")
    validate_episode_row(row)


def _artifact_path(root: Path, value: Any, label: str) -> Path:
    """Resolve one portable manifest path without allowing root escape."""

    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if not relative.parts or relative.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError(f"{label} must remain within the dataset root")
    resolved_root = root.resolve()
    candidate = resolved_root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{label} cannot contain symlinks")
    resolved = (resolved_root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain within the dataset root") from exc
    return resolved


def validate_prepared_dataset_manifest(
    manifest_path: Path,
    *,
    expected_environment: Mapping[str, Any],
    expected_source_revisions: Mapping[str, Any],
    expected_training_config_sha256: str,
    expected_recipe_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a prepared split bundle is complete and current."""

    manifest_path = manifest_path.resolve()
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise ValueError("dataset manifest must be an existing manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_path = manifest_path.with_name("manifest.sha256")
    expected_checksum = f"{manifest_digest}  manifest.json\n"
    if (
        not checksum_path.is_file()
        or checksum_path.read_bytes() != expected_checksum.encode("ascii")
    ):
        raise ValueError("dataset manifest checksum sidecar does not match")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("dataset manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be a JSON object")
    if (
        manifest.get("preparation_contract_version")
        != DATASET_PREPARATION_CONTRACT_VERSION
    ):
        raise ValueError("dataset preparation contract version does not match")
    if manifest.get("dataset_contract_version") != DATASET_CONTRACT_VERSION:
        raise ValueError("dataset contract version does not match")
    if manifest.get("dataset_roles") != list(DATASET_ROLES):
        raise ValueError("dataset roles must be train and validation")
    contract = manifest.get("recipe_contract")
    if not isinstance(contract, dict):
        raise ValueError("dataset requires explicit recipe_contract metadata")
    expected_seed_contract = (
        "explicit_validation_seed_range"
        if (contract.get("data") or {}).get("validation_seed_start") is not None
        else "disjoint_contiguous_seeds"
    )
    if manifest.get("split_seed_contract") != expected_seed_contract:
        raise ValueError("dataset split seed contract does not match")
    if manifest.get("source_revisions") != dict(expected_source_revisions):
        raise ValueError("dataset source revisions do not match installed repositories")
    manifest_environment = manifest.get("environment")
    if not isinstance(manifest_environment, dict):
        raise ValueError("dataset environment must be an object")
    if manifest_environment.get("ghost_mode") != expected_environment.get(
        "ghost_mode"
    ):
        raise ValueError("dataset environment ghost_mode does not match config")
    if manifest_environment != dict(expected_environment):
        raise ValueError("dataset environment does not match installed environment")
    if manifest.get("generator_provenance") != _split_generator_provenance():
        raise ValueError("dataset generator provenance does not match current source")
    if manifest.get("training_config_sha256") != expected_training_config_sha256:
        raise ValueError("dataset was not prepared with this training config")
    if expected_recipe_contract is not None and contract != dict(expected_recipe_contract):
        raise ValueError("dataset recipe_contract does not match config")
    protocol = (contract.get("harness") or {}).get("action_protocol")
    if manifest.get("audit_anchor_contract") != _anchor_contract(protocol):
        raise ValueError("dataset audit-anchor contract does not match")
    contract_digest = _canonical_sha256(contract)
    if manifest.get("recipe_contract_sha256") != contract_digest:
        raise ValueError("dataset recipe_contract hash does not match")
    if contract.get("ghost_mode") != expected_environment.get("ghost_mode"):
        raise ValueError("dataset recipe_contract ghost_mode does not match")
    reward_contract = contract.get("reward") or {}
    if manifest.get("reward_config") != {
        "recipe_version": reward_contract.get("formula_version"),
        **(reward_contract.get("coefficients") or {}),
    }:
        raise ValueError("dataset reward_config does not match recipe_contract")

    first_seed = manifest.get("seed")
    max_steps = manifest.get("max_steps")
    if not isinstance(first_seed, int) or isinstance(first_seed, bool):
        raise ValueError("dataset seed must be an integer")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool):
        raise ValueError("dataset max_steps must be an integer")
    data_contract = contract.get("data") or {}
    if first_seed != data_contract.get("dataset_seed") or max_steps != data_contract.get("environment_steps_per_episode"):
        raise ValueError("dataset seed or horizon does not match recipe_contract")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(DATASET_ROLES):
        raise ValueError("dataset manifest must contain exactly two splits")

    root = manifest_path.parent
    all_seeds: list[int] = []
    artifact_paths: set[Path] = set()
    for split in DATASET_ROLES:
        artifact = splits[split]
        if not isinstance(artifact, dict):
            raise ValueError(f"dataset split {split} must be an object")
        if artifact.get("artifact_kind") != "episode_specification":
            raise ValueError(f"dataset split {split} has the wrong artifact kind")
        if artifact.get("reward_breakdown") != {
            "episode_spec": "not_applicable",
            "audit_anchor": "complete_and_audited",
            "model_rollout": "required_at_training",
        }:
            raise ValueError(f"dataset split {split} reward contract does not match")
        row_count = artifact.get("rows")
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 1
        ):
            raise ValueError(f"dataset split {split} must be non-empty")
        if row_count != data_contract.get(f"{split}_rows"):
            raise ValueError(f"dataset split {split} row count does not match recipe_contract")
        if artifact.get("jsonl") != f"{split}.jsonl":
            raise ValueError(f"dataset split {split} JSONL path must use its canonical name")
        jsonl_path = _artifact_path(root, artifact.get("jsonl"), f"{split}.jsonl")
        if not jsonl_path.is_file() or jsonl_path in artifact_paths:
            raise ValueError(f"dataset split {split} JSONL is missing or reused")
        artifact_paths.add(jsonl_path)
        jsonl_bytes = jsonl_path.read_bytes()
        if hashlib.sha256(jsonl_bytes).hexdigest() != artifact.get("sha256"):
            raise ValueError(f"dataset split {split} JSONL digest does not match")
        try:
            text = jsonl_bytes.decode("utf-8")
            lines = text.splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError(f"dataset split {split} contains blank rows")
            rows = [json.loads(line) for line in lines]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"dataset split {split} is not valid JSONL") from exc
        if len(rows) != row_count:
            raise ValueError(f"dataset split {split} row count does not match")
        seeds = artifact.get("seeds")
        if (
            not isinstance(seeds, list)
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
            or len(seeds) != row_count
        ):
            raise ValueError(f"dataset split {split} seeds are invalid")
        row_seeds: list[int] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"dataset split {split} row must be an object")
            audit_episode_spec_row(row)
            if row.get("action_protocol") != protocol or row.get("prompt_version") != (contract.get("prompt") or {}).get("version"):
                raise ValueError(f"dataset split {split} row protocol does not match recipe_contract")
            if row.get("recipe_contract_sha256") != contract_digest:
                raise ValueError(f"dataset split {split} row recipe_contract hash does not match")
            if row.get("split") != split:
                raise ValueError(f"dataset split {split} contains a mislabeled row")
            if row.get("source_revisions") != dict(expected_source_revisions):
                raise ValueError(f"dataset split {split} source revisions do not match")
            row_environment = row.get("env")
            if not isinstance(row_environment, dict):
                raise ValueError(f"dataset split {split} row environment is invalid")
            if row_environment.get("max_steps") != max_steps:
                raise ValueError(f"dataset split {split} max_steps does not match")
            row_seeds.append(row_environment["seed"])
        if seeds != row_seeds:
            raise ValueError(f"dataset split {split} seed list does not match rows")
        if seeds != data_contract.get(f"{split}_seeds"):
            raise ValueError(f"dataset split {split} seeds do not match recipe_contract")
        all_seeds.extend(seeds)

        hf = artifact.get("hf")
        if hf is not None:
            if not isinstance(hf, dict) or set(hf) != {"path", "sha256"}:
                raise ValueError(f"dataset split {split} HF artifact is invalid")
            if hf.get("path") != f"{split}_hf":
                raise ValueError(f"dataset split {split} HF path must use its canonical name")
            hf_path = _artifact_path(root, hf.get("path"), f"{split}.hf")
            if not hf_path.is_dir() or hf_path in artifact_paths:
                raise ValueError(f"dataset split {split} HF artifact is missing or reused")
            artifact_paths.add(hf_path)
            if _tree_sha256(hf_path) != hf.get("sha256"):
                raise ValueError(f"dataset split {split} HF digest does not match")
            from datasets import load_from_disk

            if list(load_from_disk(str(hf_path))) != rows:
                raise ValueError(f"dataset split {split} HF rows do not match audited JSONL")

    if (
        manifest["split_seed_contract"] == "disjoint_contiguous_seeds"
        and all_seeds != list(range(first_seed, first_seed + len(all_seeds)))
    ):
        raise ValueError("dataset split seeds must be disjoint and contiguous")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare immutable API-v3 Pacman level-1 episode rows.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/level1_dataset"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--write-hf", action="store_true")
    parser.add_argument("--pacman-python-root")
    return parser.parse_args()


@contextmanager
def _temporary_pacman_python_root(
    pacman_python_root: str | os.PathLike[str] | None,
) -> Iterator[None]:
    """Select one checkout for all provenance calls without leaking state."""

    if pacman_python_root is None:
        yield
        return

    variables = ("PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT")
    previous = {variable: os.environ.get(variable) for variable in variables}
    for variable in variables:
        os.environ[variable] = str(Path(pacman_python_root).resolve())
    repository_revisions.cache_clear()
    environment_metadata.cache_clear()
    try:
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value
        repository_revisions.cache_clear()
        environment_metadata.cache_clear()


def _prepare_dataset(args: argparse.Namespace) -> None:
    environment, generation = load_recipe_settings(args.config)
    raw_config = load_recipe_document(args.config)
    contract = recipe_contract_metadata(raw_config)
    protocol = contract["harness"]["action_protocol"]
    anchor_contract = _anchor_contract(protocol)
    for name in ("train_episodes", "validation_episodes", "seed"):
        if getattr(args, name, None) is None:
            setattr(args, name, getattr(generation, name))
        if getattr(args, name) != getattr(generation, name):
            raise ValueError(f"--{name.replace('_', '-')} must match config dataset_generation.{name}")
    if args.max_steps is None:
        args.max_steps = environment.max_steps
    if args.max_steps != environment.max_steps:
        raise ValueError("--max-steps must match config environment.max_steps")
    if args.train_episodes < 1 or args.validation_episodes < 1:
        raise ValueError("train and validation datasets must both be non-empty")
    args.output_root.mkdir(parents=True, exist_ok=False)
    metadata = environment_metadata(environment.ghost_mode)
    reward_config, config_sha256 = _reward_config(args.config)
    manifest: dict[str, Any] = {
        "preparation_contract_version": DATASET_PREPARATION_CONTRACT_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "dataset_roles": list(DATASET_ROLES),
        "environment": metadata,
        "source_revisions": repository_revisions(),
        "generator_provenance": _split_generator_provenance(),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "split_seed_contract": (
            "explicit_validation_seed_range"
            if raw_config.get("dataset_generation", {}).get("validation_seed_start")
            is not None
            else "disjoint_contiguous_seeds"
        ),
        "training_config_sha256": config_sha256,
        "reward_config": asdict(reward_config),
        "recipe_contract": contract,
        "recipe_contract_sha256": _canonical_sha256(contract),
        "audit_anchor_contract": anchor_contract,
        "splits": {},
    }
    validation_seed_start = generation.validation_seed_start
    for split, count in (
        ("train", args.train_episodes),
        ("validation", args.validation_episodes),
    ):
        split_seed_start = (
            args.seed
            if split == "train"
            else (
                validation_seed_start
                if validation_seed_start is not None
                else args.seed + args.train_episodes
            )
        )
        rows = [
            next(
                generate_episode_rows(
                    1,
                    split=split,
                    seed=split_seed_start + index,
                    max_steps=args.max_steps,
                    ghost_mode=environment.ghost_mode,
                    action_protocol=protocol,
                )
            )
            for index in range(count)
        ]
        for row in rows:
            row["recipe_contract_sha256"] = manifest["recipe_contract_sha256"]
            row["audit_anchor"] = collect_initial_audit_anchor(
                seed=int(row["env"]["seed"]),
                max_steps=int(row["env"]["max_steps"]),
                reward_config=reward_config,
                pacman_python_root=args.pacman_python_root,
                ghost_mode=environment.ghost_mode,
                action_protocol=protocol,
            )
            row["source_revisions"] = row["audit_anchor"]["source_revisions"]
            row["audit_anchor_sha256"] = _canonical_sha256(row["audit_anchor"])
            audit_episode_spec_row(row)
        jsonl = args.output_root / f"{split}.jsonl"
        digest = write_jsonl(rows, jsonl)
        artifact: dict[str, Any] = {
            "artifact_kind": "episode_specification",
            "rows": count,
            "seeds": [row["env"]["seed"] for row in rows],
            "jsonl": jsonl.relative_to(args.output_root).as_posix(),
            "sha256": digest,
            "reward_breakdown": {
                "episode_spec": "not_applicable",
                "audit_anchor": "complete_and_audited",
                "model_rollout": "required_at_training",
            },
        }
        if args.write_hf:
            hf_path = args.output_root / f"{split}_hf"
            write_hf_dataset(rows, hf_path)
            artifact["hf"] = {
                "path": hf_path.relative_to(args.output_root).as_posix(),
                "sha256": _tree_sha256(hf_path),
            }
        manifest["splits"][split] = artifact
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_exclusive(
        args.output_root / "manifest.json",
        manifest_text,
    )
    manifest_digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    _write_text_exclusive(
        args.output_root / "manifest.sha256",
        f"{manifest_digest}  manifest.json\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    args = parse_args()
    with _temporary_pacman_python_root(args.pacman_python_root):
        _prepare_dataset(args)


if __name__ == "__main__":
    main()
