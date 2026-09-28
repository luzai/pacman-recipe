from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pacman_env.env import Action, PygamePacmanEnv, PygamePacmanEnvConfig


OUTCOME_LABELS = {
    "all_normal_pellets": "LEVEL CLEARED",
    "caught": "DEAD",
    "stuck": "STUCK",
    "safety_timeout": "SAFETY TIMEOUT",
    "safety_step_limit": "SAFETY STEP LIMIT",
    "parse_failed": "MODEL OUTPUT ERROR",
}


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def compose_frame(
    image: object,
    *,
    model_label: str,
    checkpoint_label: str,
    step: int,
    action: str,
    reward: float,
    score: int,
    pellet_clear_rate: float,
    terminal_reason: str | None = None,
) -> Image.Image:
    game = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )
    scale = min(960 / game.width, 680 / game.height)
    game = game.resize(
        (round(game.width * scale), round(game.height * scale)),
        Image.Resampling.NEAREST,
    )
    width = game.width + (game.width % 2)
    header = 132
    height = game.height + header
    height += height % 2
    canvas = Image.new("RGB", (width, height), (13, 17, 24))
    canvas.paste(game, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), model_label, fill=(245, 247, 250), font=_font(24))
    draw.text(
        (20, 47),
        checkpoint_label,
        fill=(92, 214, 157),
        font=_font(16),
    )
    draw.text(
        (20, 78),
        (
            f"step {step}   action {action or '-'}   reward {reward:+.1f}   "
            f"score {score}   pellets cleared {100 * pellet_clear_rate:.1f}%"
        ),
        fill=(190, 202, 221),
        font=_font(16),
    )
    if terminal_reason:
        outcome = OUTCOME_LABELS.get(terminal_reason, terminal_reason.upper())
        box = (width - 310, 14, width - 20, 66)
        draw.rounded_rectangle(box, radius=10, fill=(148, 38, 52))
        draw.text(
            (width - 290, 27),
            outcome,
            fill=(255, 255, 255),
            font=_font(20),
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
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
    frames: list[Image.Image] = []
    with PygamePacmanEnv(config) as env:
        image, info = env.reset(seed=int(episode["seed"]))
        frames.append(
            compose_frame(
                image,
                model_label=args.model_label,
                checkpoint_label=args.checkpoint_label,
                step=0,
                action="",
                reward=0.0,
                score=int(info["score"]),
                pellet_clear_rate=0.0,
            )
        )
        for index, recorded in enumerate(trajectory, start=1):
            image, _, env_terminated, env_truncated, info = env.step(
                Action(recorded["action"])
            )
            is_last = index == len(trajectory)
            frames.append(
                compose_frame(
                    image,
                    model_label=args.model_label,
                    checkpoint_label=args.checkpoint_label,
                    step=index,
                    action=recorded["action"],
                    reward=float(recorded["shaped_reward"]),
                    score=int(info["score"]),
                    pellet_clear_rate=float(
                        recorded["normal_pellet_clear_rate"]
                    ),
                    terminal_reason=(
                        str(episode["terminal_reason"]) if is_last else None
                    ),
                )
            )
            recorded_terminal = bool(
                recorded["terminated"] or recorded["truncated"]
            )
            if not is_last and bool(env_terminated or env_truncated) != recorded_terminal:
                raise RuntimeError(
                    f"replay terminal-state mismatch at step {index}"
                )

    hold = max(1, round(args.fps * args.hold_seconds))
    frames = [frames[0]] * hold + frames + [frames[-1]] * hold
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
                "steps": episode["steps"],
                "final_score": episode["final_score"],
                "normal_pellet_clear_rate": episode[
                    "normal_pellet_clear_rate"
                ],
                "terminal_reason": episode["terminal_reason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
