from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic.dataset import LAYOUTS, SPLIT_SIZES, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("run_artifacts/pacman_prompts.jsonl"))
    parser.add_argument("--hf-dir", type=Path, default=Path("run_artifacts/pacman_hf_dataset"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--mode", choices=("step", "episode"), default="step")
    parser.add_argument(
        "--prompt-style",
        choices=(
            "default",
            "legal",
            "ghost_legal",
            "ghost_legal_strict",
            "ghost_legal_reason",
            "ghost_legal_json",
            "ghost_legal_json_concise",
            "ghost_legal_json_fast_think",
            "ghost_legal_json_first",
            "ghost_legal_route_json",
            "ghost_legal_distance_json",
            "teacher_hint",
            "teacher_hint_pure",
        ),
        default="default",
    )
    parser.add_argument("--illegal-action-penalty", type=int, default=0)
    parser.add_argument("--layout-name", choices=tuple(sorted(LAYOUTS)), default="default")
    parser.add_argument("--maze-split", choices=tuple(SPLIT_SIZES))
    parser.add_argument("--episodes-per-layout", type=int, default=1)
    parser.add_argument(
        "--reward-mode",
        choices=("sparse", "route_prefix", "route_prefix_stay_penalty", "route_prefix_progress_penalty"),
        default="sparse",
    )
    args = parser.parse_args()

    count = write_jsonl(
        args.jsonl,
        args.episodes,
        args.max_steps,
        args.mode,
        prompt_style=args.prompt_style,
        illegal_action_penalty=args.illegal_action_penalty,
        layout_name=args.layout_name,
        reward_mode=args.reward_mode,
        maze_split=args.maze_split,
        episodes_per_layout=args.episodes_per_layout,
    )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to export Hugging Face dataset format") from exc

    dataset = load_dataset("json", data_files=str(args.jsonl), split="train")
    args.hf_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.hf_dir))
    print(f"wrote_jsonl={count}")
    print(f"hf_dataset={args.hf_dir}")
    print(f"hf_rows={len(dataset)}")


if __name__ == "__main__":
    main()
