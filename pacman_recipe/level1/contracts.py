"""Versioned neutral records; legacy ledgers are converted only in memory."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
from .prompts import prompt_contract_metadata

EPISODE_SCHEMA = "pacman-episode-v1"
TRAJECTORY_SCHEMA = "pacman-trajectory-v1"
FIELD_NAMES = {
    "maapacman_revision": "recipe_revision",
    "maapacman_env_source_sha256": "environment_source_sha256",
    "maapacman_dirty": "recipe_dirty",
}


def _rename(mapping, names):
    return {names.get(key, key): deepcopy(value) for key, value in mapping.items()}


def repository_identity(path):
    path = Path(path).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    digest = hashlib.sha256()
    roots = [
        path / name
        for name in (
            "pacman_recipe",
            "pacman_env",
            "slime_pacman",
            "slime",
            "slime_plugins",
            "pacman",
            "configs",
            "scripts",
        )
    ]
    files = {
        p
        for root in roots
        if root.is_dir()
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".py", ".pyw", ".json", ".yaml", ".txt", ".sh"}
    }
    files.update(
        p for p in path.iterdir() if p.is_file() and p.suffix in {".py", ".toml"}
    )
    for file in sorted(files):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
    }


def make_episode_record(
    seed, *, split, recipe_root, game_root, backend_root, backend="slime", max_steps=512,
    observation_mode="image", edward_fallback_mode="refuse"
):
    if observation_mode not in {'image', 'ascii'}:
        raise ValueError('Unsupported observation mode')
    if (
        backend not in {"slime", "areal"}
        or split not in {"train", "validation", "test"}
        or type(seed) is not int
    ):
        raise ValueError("invalid episode backend, split or seed")
    sources = {
        "pacman-recipe": repository_identity(recipe_root),
        "pacman-python": repository_identity(game_root),
        backend: repository_identity(backend_root),
    }
    with PygamePacmanEnv(
        PygamePacmanEnvConfig(
            pacman_python_root=game_root,
            max_steps=max_steps,
            ghost_mode="normal",
            episode_life_mode="single_death",
        )
    ) as env:
        info, spec = env.provenance, env.spec
        environment = dict(
            name=spec.env_id,
            api_version=spec.api_version,
            backend="original-pygame",
            level=1,
            seed=seed,
            max_steps=max_steps,
            ghost_mode="normal",
            episode_life_mode="single_death",
            observation_mode="rgb",
            pacman_python_revision=info["pacman_python_commit"],
            pacman_python_source_sha256=info["pacman_python_source_sha256"],
            pacman_python_dirty=info["pacman_python_dirty"],
            recipe_revision=info["maapacman_commit"],
            environment_source_sha256=info["maapacman_env_source_sha256"],
            recipe_dirty=info["maapacman_dirty"],
            level_revision=spec.level_revision,
            ruleset_revision=spec.ruleset_revision,
            renderer_revision=spec.renderer_revision,
        )
    return dict(
        schema=EPISODE_SCHEMA,
        id=f"pacman-{split}-{seed}",
        split=split,
        training_backend=backend,
        environment=environment,
        source_revisions=sources,
        prompt=prompt_contract_metadata("ascii_edward_v1" if observation_mode == 'ascii' else "live_state_v3",
                                        edward_options=True, fallback_mode=edward_fallback_mode),
    )


def validate_episode_record(record, *, expected_sources=None):
    if record.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported episode schema; regenerate historical rows")
    backend = record.get("training_backend")
    if backend not in {"slime", "areal"} or set(record.get("source_revisions", {})) != {
        "pacman-recipe",
        "pacman-python",
        backend,
    }:
        raise ValueError("backend/source provenance mismatch")
    if expected_sources is not None and record["source_revisions"] != expected_sources:
        raise ValueError("source revisions or hashes differ; regenerate data")
    for source in record["source_revisions"].values():
        if type(source.get("dirty")) is not bool:
            raise ValueError("missing source dirty state")
        for key, length in (("commit", 40), ("source_sha256", 64)):
            value = source.get(key, "")
            if len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid source hash")
    known_prompts = [prompt_contract_metadata(style, edward_options=True, fallback_mode=mode)
                     for style in ("live_state_v3", "ascii_edward_v1") for mode in ("refuse", "risk_ranked")]
    if record.get("prompt") not in known_prompts:
        raise ValueError("prompt contract changed; regenerate data")
    env = record["environment"]
    if (
        env.get("ghost_mode") != "normal"
        or env.get("episode_life_mode") != "single_death"
    ):
        raise ValueError("C2 requires normal ghosts and single_death")
    if (
        type(env.get("seed")) is not int
        or type(env.get("max_steps")) is not int
        or not 0 < env["max_steps"] <= 512
    ):
        raise ValueError("invalid seed/horizon")
    if not record.get("id") or record.get("split") not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("invalid record identity")
    if "restart" in record:
        restart = record["restart"]
        if not isinstance(restart, dict) or set(restart) != {"path", "sha256", "id"}:
            raise ValueError("restart requires path, sha256 and id")
        if not all(isinstance(value, str) and value for value in restart.values()):
            raise ValueError("restart fields must be nonempty strings")
        if not Path(restart["path"]).is_absolute():
            raise ValueError("restart path must be absolute on the rollout worker")
        digest = restart["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid restart file hash")


def runner_row(record):
    validate_episode_record(record)
    row = dict(
        id=record["id"],
        split=record["split"],
        dataset_contract_version=EPISODE_SCHEMA,
        training_backend=record["training_backend"],
        source_revisions=deepcopy(record["source_revisions"]),
        env=_rename(record["environment"], {v: k for k, v in FIELD_NAMES.items()}),
        state_prefix_actions=[],
        action_protocol=record["prompt"]["action_protocol"],
        prompt_version=record["prompt"]["prompt_version"],
    )
    if "restart" in record:
        row.update({"restart_state_" + key: value for key, value in record["restart"].items()})
    return row


def neutral_trajectory(payload):
    from .trajectories import audit_trajectory

    audit_trajectory(payload)
    return {"schema": TRAJECTORY_SCHEMA, "episode": _rename(payload, FIELD_NAMES)}


def audit_neutral_trajectory(record):
    from .trajectories import audit_trajectory

    if record.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("unsupported trajectory schema")
    payload = _rename(record["episode"], {v: k for k, v in FIELD_NAMES.items()})
    if payload.get("dataset_contract_version") != EPISODE_SCHEMA:
        raise ValueError("neutral trajectory requires neutral episode provenance")
    audit_trajectory(payload)


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(
            value, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        out.write("\n")
