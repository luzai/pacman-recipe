from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .areal_workflow import build_observation_turn
from .synthetic.dataset import system_prompt
from .synthetic.env import LAYOUTS, PacmanEnv
from .synthetic.vision import env_png_bytes


def shortest_route_action(env: PacmanEnv, avoid_stay: bool = True) -> str:
    route_actions = env.shortest_route_actions()
    if route_actions:
        return route_actions[0]
    legal_actions = env.legal_actions()
    if avoid_stay:
        moving_actions = [action for action in legal_actions if action != "stay"]
        if moving_actions:
            return moving_actions[0]
    return legal_actions[0] if legal_actions else "stay"


def build_sft_example(
    env: PacmanEnv,
    action: str,
    image_path: Path | None,
    prompt_style: str,
    vision_tile_size: int,
    include_data_url: bool,
) -> dict[str, object]:
    obs = build_observation_turn(
        env,
        prompt_style=prompt_style,
        observation_mode="image_text",
        vision_tile_size=vision_tile_size,
        include_image_data_url=include_data_url,
    )
    user_content = list(obs.model_observation) if isinstance(obs.model_observation, list) else [{"type": "text", "text": obs.text}]
    if image_path is not None:
        user_content.append({"type": "image_path", "image_path": str(image_path)})
    return {
        "messages": [
            {"role": "system", "content": system_prompt(prompt_style)},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": action},
        ],
        "answer": action,
        "obs_text": obs.text,
        "obs_image_sha256": obs.image_sha256,
        "image_path": str(image_path) if image_path is not None else None,
    }


def generate_shortest_route_sft(
    output_jsonl: Path,
    images_dir: Path,
    episodes: int = 1,
    max_steps: int = 30,
    layout_name: str = "medium_default",
    prompt_style: str = "ghost_legal_strict",
    reward_mode: str = "sparse",
    vision_tile_size: int = 32,
    include_data_url: bool = True,
    save_images: bool = True,
) -> tuple[int, dict[str, object]]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    wins = 0
    rewards: list[float] = []
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for seed in range(episodes):
            env = PacmanEnv(
                seed=seed,
                max_steps=max_steps,
                layout_name=layout_name,
                reward_mode=reward_mode,
            )
            env.reset()
            total_reward = 0.0
            while not env.state.done:
                step = env.state.steps
                action = shortest_route_action(env)
                image_path = None
                if save_images:
                    image_path = images_dir / f"episode{seed:04d}_step{step:03d}_{uuid4().hex[:8]}.png"
                    image_path.write_bytes(env_png_bytes(env, tile_size=vision_tile_size))
                example = build_sft_example(
                    env=env,
                    action=action,
                    image_path=image_path,
                    prompt_style=prompt_style,
                    vision_tile_size=vision_tile_size,
                    include_data_url=include_data_url,
                )
                state, reward, done, info = env.step(action)
                total_reward += float(reward)
                example.update(
                    {
                        "id": f"vision-sft-{seed}-{step}",
                        "seed": seed,
                        "step": step,
                        "layout_name": layout_name,
                        "max_steps": max_steps,
                        "prompt_style": prompt_style,
                        "reward_mode": reward_mode,
                        "vision_tile_size": vision_tile_size,
                        "policy": "shortest_route",
                        "legal_action": bool(info["legal_action"]),
                        "route_action": bool(info.get("route_action", False)),
                        "reward": reward,
                        "done": done,
                        "reason": info["reason"],
                        "score": state.score,
                    }
                )
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
            wins += bool(env.state.won)
            rewards.append(total_reward)

    summary = {
        "examples": count,
        "episodes": episodes,
        "wins": wins,
        "layout_name": layout_name,
        "max_steps": max_steps,
        "prompt_style": prompt_style,
        "reward_mode": reward_mode,
        "vision_tile_size": vision_tile_size,
        "include_data_url": include_data_url,
        "save_images": save_images,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
    }
    return count, summary


def maybe_save_hf_dataset(jsonl_path: Path, hf_dir: Path | None) -> int | None:
    if hf_dir is None:
        return None
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to export Hugging Face dataset format") from exc
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            rows.append(
                {
                    "id": item["id"],
                    "messages_json": json.dumps(item["messages"], ensure_ascii=False),
                    "answer": item["answer"],
                    "image_path": item["image_path"],
                    "obs_image_sha256": item["obs_image_sha256"],
                    "seed": item["seed"],
                    "step": item["step"],
                    "layout_name": item["layout_name"],
                    "prompt_style": item["prompt_style"],
                    "reward_mode": item["reward_mode"],
                    "reward": item["reward"],
                    "done": item["done"],
                    "reason": item["reason"],
                    "score": item["score"],
                }
            )
    dataset = Dataset.from_list(rows)
    hf_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(hf_dir))
    return len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shortest-route image_text SFT warm-start data.")
    parser.add_argument("--jsonl", type=Path, default=Path("run_artifacts/vision_sft_shortest_route/vision_sft.jsonl"))
    parser.add_argument("--images-dir", type=Path, default=Path("run_artifacts/vision_sft_shortest_route/images"))
    parser.add_argument("--hf-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--layout-name", choices=tuple(sorted(LAYOUTS)), default="medium_default")
    parser.add_argument("--prompt-style", default="ghost_legal_strict")
    parser.add_argument(
        "--reward-mode",
        choices=("sparse", "route_prefix", "route_prefix_stay_penalty", "route_prefix_progress_penalty"),
        default="sparse",
    )
    parser.add_argument("--vision-tile-size", type=int, default=32)
    parser.add_argument("--no-data-url", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    args = parser.parse_args()

    count, summary = generate_shortest_route_sft(
        output_jsonl=args.jsonl,
        images_dir=args.images_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        layout_name=args.layout_name,
        prompt_style=args.prompt_style,
        reward_mode=args.reward_mode,
        vision_tile_size=args.vision_tile_size,
        include_data_url=not args.no_data_url,
        save_images=not args.no_save_images,
    )
    hf_rows = maybe_save_hf_dataset(args.jsonl, args.hf_dir)
    print(f"wrote_jsonl={count}")
    print(f"jsonl={args.jsonl}")
    print(f"images_dir={args.images_dir if not args.no_save_images else 'disabled'}")
    if args.hf_dir is not None:
        print(f"hf_dataset={args.hf_dir}")
        print(f"hf_rows={hf_rows}")
    print("summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
