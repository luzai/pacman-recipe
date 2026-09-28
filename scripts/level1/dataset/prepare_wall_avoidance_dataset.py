from __future__ import annotations

import argparse
import json
from pathlib import Path

from pacman_recipe.level1.level1_dataset import (
    generate_balanced_corridor_rows,
    write_hf_dataset,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare varied single-step Pacman wall-avoidance states."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/datasets/wall_avoidance_dataset"),
    )
    parser.add_argument("--train-states", type=int, default=32)
    parser.add_argument("--validation-states", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-hf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = (
        ("train", args.train_states),
        ("validation", args.validation_states),
    )
    manifest: dict[str, object] = {"seed": args.seed, "splits": {}}
    for split, count in splits:
        rows = list(
            generate_balanced_corridor_rows(
                count,
                split=split,
                seed=args.seed,
            )
        )
        jsonl = args.output_root / f"{split}.jsonl"
        digest = write_jsonl(rows, jsonl)
        if args.write_hf:
            write_hf_dataset(rows, args.output_root / f"{split}_hf")
        manifest["splits"][split] = {
            "rows": count,
            "jsonl": str(jsonl),
            "sha256": digest,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
