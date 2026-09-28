"""Create a fresh neutral dataset and its immutable manifest on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

from pacman_recipe.level1.contracts import make_episode_record, write_json_new
from . import SLIME_REVISION
from .config import load_config


def prepare(config_path, output, *, recipe_root, game_root, slime_root):
    config = load_config(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files, sources = {}, None
    for split, seeds in (
        ("train", config.train_seeds),
        ("validation", config.validation_seeds),
    ):
        rows = [
            make_episode_record(
                seed,
                split=split,
                recipe_root=recipe_root,
                game_root=game_root,
                backend_root=slime_root,
                max_steps=config.max_steps,
            )
            for seed in seeds
        ]
        for row in rows:
            if row["source_revisions"]["slime"]["commit"] != SLIME_REVISION:
                raise ValueError("slime checkout differs from the pinned revision")
            if sources is None:
                sources = row["source_revisions"]
            if sources != row["source_revisions"]:
                raise ValueError("source tree changed during data preparation")
        path = output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as out:
            for row in rows:
                out.write(
                    json.dumps(
                        {
                            "prompt": "Pacman episode",
                            "metadata": {"episode_record": row},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        files[path.name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema": "pacman-dataset-manifest-v1",
        "training_backend": "slime",
        "config": config.as_dict(),
        "source_revisions": sources,
        "files": files,
    }
    write_json_new(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs/slime/c2.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-root", type=Path, default=root.parent / "pacman-python")
    parser.add_argument("--slime-root", type=Path, default=root.parent / "slime")
    args = parser.parse_args()
    manifest = prepare(
        args.config,
        args.output,
        recipe_root=root,
        game_root=args.game_root,
        slime_root=args.slime_root,
    )
    print(
        json.dumps({"schema": manifest["schema"], "files": manifest["files"]}, indent=2)
    )


if __name__ == "__main__":
    main()
