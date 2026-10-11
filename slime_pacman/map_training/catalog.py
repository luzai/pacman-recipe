"""Prepare immutable maps and records. No model inference or training."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from .maps import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", required=True, help="tiers with template, ghosts, counts and seeds"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--game-root", required=True)
    parser.add_argument("--slime-root", required=True)
    parser.add_argument("--max-steps", type=int, default=512)
    options = parser.parse_args()
    spec = json.loads(Path(options.spec).read_text())
    output = Path(options.output).resolve()
    if output.exists():
        raise FileExistsError("refuse replacing a frozen catalog")
    output.mkdir(parents=True)
    used, catalog = set(spec.get("excluded_equivalence_sha256", [])), []
    original_env = dict(os.environ)
    try:
        from pacman_recipe.level1.contracts import make_episode_record

        recipe = Path(__file__).resolve().parents[2]
        for tier, settings in enumerate(spec["tiers"]):
            if settings["ghosts"] not in (0, 1, 2, 4):
                raise ValueError("unsupported ghost count")
            rows = generate(
                Path(settings["template"]).read_text(),
                seed=settings["seed"],
                counts=settings["counts"],
                excluded=used,
                swaps=settings.get("swaps", 4),
                small_floor_pellets=settings.get("small_floor_pellets", False),
            )
            for index, row in enumerate(rows):
                used.add(row["equivalence_sha256"])
                map_id = f"t{tier}-map-{index:04d}"
                game = output / "games" / map_id
                shutil.copytree(
                    options.game_root,
                    game,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
                )
                (game / "pacman/res/levels/1.txt").write_text(
                    row.pop("text"), encoding="utf-8", newline="\n"
                )
                for args in [
                    ("init",),
                    ("add", "."),
                    (
                        "-c",
                        "user.name=Map Catalog",
                        "-c",
                        "user.email=map-catalog@localhost",
                        "commit",
                        "-m",
                        "Frozen map source",
                    ),
                ]:
                    subprocess.run(
                        ["git", "-C", str(game), *args], check=True, capture_output=True
                    )
                os.environ.update(
                    PACMAN_PYTHON_ROOT=str(game),
                    SMALL_MAZE_GHOST_COUNT=str(settings["ghosts"] or 1),
                    PACMAN_CUSTOM_WALL_RENDER="1",
                )
                for game_seed in settings.get("game_seeds", [0]):
                    task_id = f"{map_id}-seed-{game_seed}"
                    record = make_episode_record(
                        game_seed,
                        split=row["split"],
                        recipe_root=recipe,
                        game_root=game,
                        backend_root=Path(options.slime_root),
                        max_steps=options.max_steps,
                        observation_mode="ascii",
                        ascii_map_format="spaced",
                        action_mode=spec.get("action_mode", "edward"),
                        ghost_mode="disabled" if settings["ghosts"] == 0 else "normal",
                        episode_life_mode=spec.get("episode_life_mode", "single_death"),
                        edward_fallback_mode="refuse" if spec.get("action_mode", "edward") == "primitive" else "risk_ranked",
                    )
                    record["id"] = task_id
                    if record["environment"]["level_revision"] != row["layout_sha256"]:
                        raise ValueError("game/record level mismatch")
                    catalog.append(
                        dict(
                            row,
                            tier=tier,
                            ghosts=settings["ghosts"],
                            map_id=map_id,
                            task_id=task_id,
                            game_seed=game_seed,
                            game_root=str(game),
                            panel=settings.get("panel", "heldout_layouts"),
                            episode_record=record,
                        )
                    )
        (output / "catalog.json").write_text(json.dumps(catalog, indent=2))
    finally:
        os.environ.clear()
        os.environ.update(original_env)


if __name__ == "__main__":
    main()
