from __future__ import annotations

import argparse
import json
from pathlib import Path

from pacman_recipe.level1.trajectories import audit_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute and audit level-1 trajectory rewards.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    checked = 0
    for path in args.paths:
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            audit_trajectory(json.loads(candidate.read_text(encoding="utf-8")))
            checked += 1
    if checked == 0:
        raise SystemExit("no trajectory JSON files found")
    print(f"reward_audit=ok files={checked}")


if __name__ == "__main__":
    main()
