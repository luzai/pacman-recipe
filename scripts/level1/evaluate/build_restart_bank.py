"""Generate an immutable bank from successful Edward option trajectories."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pacman_recipe.level1.backplay import build_restart_bank
from pacman_env.env import PygamePacmanEnvConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--dense-tail", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--ghost-mode", choices=["normal", "disabled"], default="normal")
    parser.add_argument("--episode-life-mode", choices=["single_death", "original_three_lives"], default="original_three_lives")
    args = parser.parse_args()
    manifest = build_restart_bank(args.output_dir, seeds=args.seeds,
        trajectories=args.trajectories, stride=args.stride, dense_tail=args.dense_tail,
        config=PygamePacmanEnvConfig(max_steps=args.max_steps, ghost_mode=args.ghost_mode,
                                    episode_life_mode=args.episode_life_mode))
    print(json.dumps({"bank_id": manifest["bank_id"], "output_dir": str(args.output_dir.resolve()),
                      "trajectories": len(manifest["trajectories"]),
                      "restart_states": len(manifest["restart_states"])}, indent=2))


if __name__ == "__main__":
    main()
