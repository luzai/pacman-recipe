from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pacman_env.env import Action, PygamePacmanEnv, PygamePacmanEnvConfig


def compose_frame(
    image: object,
    *,
    title: str,
    step: int,
    total_steps: int,
    action: str,
    score: int,
    open_actions: list[str],
) -> Image.Image:
    game = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    scale = min(960 / game.width, 720 / game.height)
    game = game.resize(
        (round(game.width * scale), round(game.height * scale)),
        Image.Resampling.NEAREST,
    )
    canvas_width = game.width + (game.width % 2)
    canvas_height = game.height + 92
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (15, 18, 25))
    canvas.paste(game, (0, 92))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 14), title, fill=(244, 246, 250), font=font)
    draw.text(
        (18, 40),
        f"step {step}/{total_steps}   action {action or '-'}   "
        f"score {score}   OPEN {','.join(open_actions)}",
        fill=(185, 198, 220),
        font=font,
    )
    draw.text(
        (18, 65),
        "Qwen3.5-9B initial rollout | no-thinking | dynamic action mask",
        fill=(90, 214, 154),
        font=font,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    args = parser.parse_args()

    episode = json.loads(args.trajectory.read_text(encoding="utf-8"))
    trajectory = episode["trajectory"]
    config = PygamePacmanEnvConfig(
        level=int(episode["level"]),
        ghost_mode=episode["ghost_mode"],
        max_steps=int(episode["max_steps"]),
        video_driver="dummy",
        audio_driver="dummy",
    )

    title = (
        f"{episode['id']} | {episode['terminal_reason']} | "
        f"score {episode['final_score']} | "
        f"pellets left {episode['normal_pellets_remaining']}"
    )
    frames: list[Image.Image] = []
    with PygamePacmanEnv(config) as env:
        image, info = env.reset(seed=int(episode["seed"]))
        frames.append(
            compose_frame(
                image,
                title=title,
                step=0,
                total_steps=len(trajectory),
                action="",
                score=int(info["score"]),
                open_actions=list(env.snapshot().get("open") or []),
            )
        )
        for index, recorded in enumerate(trajectory, start=1):
            image, _, terminated, truncated, info = env.step(
                Action(recorded["action"])
            )
            frames.append(
                compose_frame(
                    image,
                    title=title,
                    step=index,
                    total_steps=len(trajectory),
                    action=recorded["action"],
                    score=int(info["score"]),
                    open_actions=list(env.snapshot().get("open") or []),
                )
            )
            # A safety refusal ends the episode above the environment: the
            # planner declines to act while the env itself is still live, so
            # the recorded final step carries truncated=True and the replayed
            # env reports neither flag. Every other step stays strict.
            terminal_matches = bool(terminated or truncated) == bool(
                recorded["terminated"] or recorded["truncated"]
            )
            if recorded.get("safety_refusal"):
                terminal_matches = (
                    index == len(trajectory)
                    and episode["terminal_reason"] == "safety_refusal"
                    and recorded.get("terminal_reason") == "safety_refusal"
                    and not recorded["terminated"]
                    and bool(recorded["truncated"])
                    and not (terminated or truncated)
                )
            if not terminal_matches:
                raise RuntimeError(f"replay terminal-state mismatch at step {index}")

    hold_frames = max(1, round(args.fps * args.hold_seconds))
    frames = [frames[0]] * hold_frames + frames + [frames[-1]] * hold_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "output": str(args.output),
                "frames": len(frames),
                "fps": args.fps,
                "duration_seconds": len(frames) / args.fps,
                "final_score": episode["final_score"],
            }
        )
    )


if __name__ == "__main__":
    main()
