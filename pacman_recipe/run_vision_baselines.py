from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from .synthetic.baselines import GreedyPelletAgent, RandomAgent
from .synthetic.env import ACTIONS, LAYOUTS, PacmanEnv
from .synthetic.vision import env_png_bytes, image_sha256, render_env_image


@dataclass
class ShortestRouteAgent:
    def act(self, env: PacmanEnv) -> str:
        route_actions = env.shortest_route_actions()
        if route_actions:
            return route_actions[0]
        moving_actions = [action for action in env.legal_actions() if action != "stay"]
        if moving_actions:
            return moving_actions[0]
        return "stay"


def make_agent(name: str, seed: int = 0) -> Any:
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyPelletAgent()
    if name == "shortest_route":
        return ShortestRouteAgent()
    raise ValueError(f"unknown agent {name!r}")


def _obs_text(env: PacmanEnv) -> str:
    return (
        f"Step {env.state.steps}/{env.max_steps}\n"
        f"Score: {env.state.score}\n"
        f"Pellets left: {len(env.state.pellets)}\n"
        f"Legal actions: {', '.join(env.legal_actions())}\n"
        f"Grid:\n{env.render()}\n"
    )


def run_vision_episode(
    agent_name: str,
    seed: int = 0,
    max_steps: int = 30,
    layout_name: str = "medium_default",
    reward_mode: str = "sparse",
    tile_size: int = 32,
) -> tuple[dict[str, Any], list[Image.Image]]:
    env = PacmanEnv(
        seed=seed,
        max_steps=max_steps,
        layout_name=layout_name,
        reward_mode=reward_mode,
    )
    env.reset()
    agent = make_agent(agent_name, seed=seed)
    total_reward = 0.0
    trajectory: list[dict[str, Any]] = []
    frames: list[Image.Image] = []

    while not env.state.done:
        image = render_env_image(env, tile_size=tile_size)
        png_bytes = env_png_bytes(env, tile_size=tile_size)
        obs = _obs_text(env)
        action = agent.act(env)
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        frames.append(image)
        trajectory.append(
            {
                "step": state.steps,
                "obs": obs,
                "obs_text": obs,
                "observation_mode": "image",
                "obs_image_sha256": image_sha256(png_bytes),
                "policy": agent_name,
                "model_output": f"{agent_name}:{action}",
                "action": action,
                "raw_action": action,
                "action_masked": False,
                "parse_failed": False,
                "parse_failure_penalty": 0,
                "exact_action": True,
                "reward": reward,
                "legal_action": info["legal_action"],
                "done": done,
                "reason": info["reason"],
                "score": state.score,
            }
        )

    frames.append(render_env_image(env, tile_size=tile_size))
    payload = {
        "id": f"vision-baseline-{agent_name}-{seed}",
        "seed": seed,
        "agent": agent_name,
        "layout_name": layout_name,
        "max_steps": max_steps,
        "reward_mode": reward_mode,
        "observation_mode": "image",
        "vision_tile_size": tile_size,
        "total_reward": total_reward,
        "final_score": env.state.score,
        "steps": env.state.steps,
        "won": env.state.won,
        "trajectory": trajectory,
    }
    return payload, frames


def _frame_with_caption(image: Image.Image, step: dict[str, Any] | None, payload: dict[str, Any]) -> Image.Image:
    font = ImageFont.load_default()
    footer_h = 54
    out = Image.new("RGB", (image.width, image.height + footer_h), (18, 22, 30))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    if step is None:
        text = (
            f"{payload['agent']} final | "
            f"{'WIN' if payload['won'] else 'FAIL'} | "
            f"score {payload['final_score']} | steps {payload['steps']}"
        )
    else:
        text = (
            f"{payload['agent']} step {step['step']} | action {step['action']} | "
            f"reward {step['reward']:g} | score {step['score']} | {step['reason']}"
        )
    draw.text((8, image.height + 8), text[:92], fill=(236, 241, 247), font=font)
    draw.text((8, image.height + 28), f"layout {payload['layout_name']} | mode image", fill=(160, 171, 190), font=font)
    return out


def compose_vision_video_frames(payload: dict[str, Any], frames: list[Image.Image], fps: int) -> list[Image.Image]:
    trajectory = payload.get("trajectory", [])
    repeats = max(1, math.ceil(fps * 0.7))
    out: list[Image.Image] = []
    for idx, frame in enumerate(frames):
        step = trajectory[idx] if idx < len(trajectory) else None
        out.extend([_frame_with_caption(frame, step, payload)] * repeats)
    return out


def write_video(frames: list[Image.Image], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            frame.save(tmp_path / f"frame_{idx:05d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def run_and_write(
    agent_name: str,
    output_dir: Path,
    seed: int = 0,
    max_steps: int = 30,
    layout_name: str = "medium_default",
    reward_mode: str = "sparse",
    tile_size: int = 32,
    fps: int = 6,
    video_suffix: str = ".gif",
) -> tuple[Path, Path, dict[str, Any]]:
    payload, frames = run_vision_episode(
        agent_name=agent_name,
        seed=seed,
        max_steps=max_steps,
        layout_name=layout_name,
        reward_mode=reward_mode,
        tile_size=tile_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"vision_{agent_name}_{layout_name}_seed{seed}_{uuid4().hex[:8]}"
    json_path = output_dir / f"{stem}.json"
    video_path = output_dir / f"{stem}{video_suffix}"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_video(compose_vision_video_frames(payload, frames, fps=fps), video_path, fps=fps)
    return json_path, video_path, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vision-observation PacMan baseline rollouts.")
    parser.add_argument("--agents", nargs="+", choices=("random", "greedy", "shortest_route"), default=["random", "greedy", "shortest_route"])
    parser.add_argument("--output-dir", type=Path, default=Path("run_artifacts/vision_baselines"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--layout-name", choices=tuple(sorted(LAYOUTS)), default="medium_default")
    parser.add_argument("--reward-mode", choices=("sparse", "route_prefix", "route_prefix_stay_penalty", "route_prefix_progress_penalty"), default="sparse")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--video-suffix", choices=(".gif", ".mp4"), default=".gif")
    args = parser.parse_args()

    for agent_name in args.agents:
        json_path, video_path, payload = run_and_write(
            agent_name=agent_name,
            output_dir=args.output_dir,
            seed=args.seed,
            max_steps=args.max_steps,
            layout_name=args.layout_name,
            reward_mode=args.reward_mode,
            tile_size=args.tile_size,
            fps=args.fps,
            video_suffix=args.video_suffix,
        )
        print(
            f"agent={agent_name} won={payload['won']} steps={payload['steps']} "
            f"total_reward={payload['total_reward']} json={json_path} video={video_path}"
        )


if __name__ == "__main__":
    main()
