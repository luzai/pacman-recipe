from __future__ import annotations

import base64
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pacman_recipe.synthetic.env import PacmanEnv
from pacman_recipe.synthetic.vision import render_env_image


RUN_DIR = ROOT / "run_artifacts" / "areal_vlm_imageonly_rules_activationgate_notrain16_20260713"
TRAJECTORY_DIR = (
    RUN_DIR
    / "trajectories"
    / "trajectories_vision-qwen3p5-vlm-imageonly-rules-activationgate-notrain16-20260713"
)
SUCCESS_SOURCE = TRAJECTORY_DIR / "pacman-episode-3-3319735-e1284cec.json"
FAIL_SOURCE = TRAJECTORY_DIR / "pacman-episode-11-3316568-f7c7c580.json"

BG = (18, 22, 30)
TEXT = (238, 242, 247)
MUTED = (166, 176, 194)
GOOD = (73, 210, 138)
BAD = (255, 113, 113)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def request_parts(step: dict[str, Any]) -> tuple[str, str, str]:
    system_prompt = ""
    user_text = ""
    image_url = ""
    for message in step.get("request_messages", []):
        if message.get("role") == "system":
            system_prompt = str(message.get("content", ""))
        if message.get("role") != "user":
            continue
        for item in message.get("content", []):
            if item.get("type") == "text":
                user_text = str(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_url = str(item.get("image_url", {}).get("url", ""))
    if not image_url.startswith("data:image/png;base64,"):
        raise ValueError("trajectory step has no PNG image data URL")
    return system_prompt, user_text, image_url


def decode_image(image_url: str) -> tuple[Image.Image, bytes]:
    png_bytes = base64.b64decode(image_url.split(",", 1)[1])
    return Image.open(BytesIO(png_bytes)).convert("RGB"), png_bytes


def render_demo_frame(
    episode: dict[str, Any],
    step: dict[str, Any],
    label: str,
    accent: tuple[int, int, int],
) -> Image.Image:
    _, _, image_url = request_parts(step)
    observation, _ = decode_image(image_url)
    observation = observation.resize((576, 384), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (720, 560), BG)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((28, 20), label, fill=accent, font=font)
    draw.text(
        (28, 42),
        f"step {step['step']}/{episode['steps']} | model -> {step['model_output']!r} | action -> {step['action']}",
        fill=TEXT,
        font=font,
    )
    canvas.paste(observation, (72, 74))
    draw.text(
        (28, 476),
        f"env -> reward {step['reward']:g} | score {step['score']} | reason {step['reason']}",
        fill=TEXT,
        font=font,
    )
    draw.text(
        (28, 502),
        "Model input at this turn: system rules + fixed action text + this PNG frame",
        fill=MUTED,
        font=font,
    )
    if step.get("done"):
        outcome = "WIN: all pellets cleared" if episode.get("won") else "FAIL: caught by ghost"
        draw.text((28, 528), outcome, fill=accent, font=font)
    return canvas


def render_terminal_frame(
    episode: dict[str, Any],
    label: str,
    accent: tuple[int, int, int],
) -> Image.Image:
    env = PacmanEnv(
        layout_name=episode["layout_name"],
        max_steps=episode["max_steps"],
        seed=episode["seed"],
        illegal_action_penalty=episode["illegal_action_penalty"],
        reward_mode=episode["reward_mode"],
        route_shaping_scale=episode["route_shaping_scale"],
    )
    final_info: dict[str, Any] = {}
    for step in episode["trajectory"]:
        _, _, _, final_info = env.step(step["action"])

    if not env.state.done:
        raise ValueError("replayed trajectory does not reach a terminal state")
    if env.state.won != episode["won"] or final_info.get("reason") != episode["final_reason"]:
        raise ValueError("replayed terminal state does not match saved trajectory")

    terminal = render_env_image(env, tile_size=32).resize((576, 384), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (720, 560), BG)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    outcome = "WIN: all pellets cleared" if env.state.won else "FAIL: PacMan caught by ghost"

    draw.text((28, 20), label, fill=accent, font=font)
    draw.text((28, 42), "POST-ACTION TERMINAL STATE", fill=accent, font=font)
    canvas.paste(terminal, (72, 74))
    draw.text(
        (28, 476),
        f"env -> score {env.state.score} | reason {final_info['reason']} | done true",
        fill=TEXT,
        font=font,
    )
    draw.text((28, 502), outcome, fill=accent, font=font)
    draw.text(
        (28, 528),
        "No model call: this is the env state after applying the final action",
        fill=MUTED,
        font=font,
    )
    return canvas


def compact_episode(
    episode: dict[str, Any], source: Path, no_train: bool = True
) -> dict[str, Any]:
    compact_steps = []
    system_prompt = ""
    user_text = ""
    for step in episode["trajectory"]:
        current_system, current_user, image_url = request_parts(step)
        image, png_bytes = decode_image(image_url)
        system_prompt = current_system
        user_text = current_user
        compact_steps.append(
            {
                "step": step["step"],
                "request": {
                    "system_prompt": current_system,
                    "user_text": current_user,
                    "image": {
                        "media_type": "image/png",
                        "width": image.width,
                        "height": image.height,
                        "bytes": len(png_bytes),
                        "sha256": hashlib.sha256(png_bytes).hexdigest(),
                        "source": "request_messages user image_url data URL",
                    },
                    "structured_choice": step.get("request_extra_body", {})
                    .get("structured_outputs", {})
                    .get("choice", []),
                },
                "response": {
                    "raw_model_output": step.get("model_output"),
                    "parsed_action": step.get("action"),
                    "exact_action": step.get("exact_action"),
                    "parse_failed": step.get("parse_failed"),
                    "illegal_action": step.get("illegal_action"),
                    "action_masked": step.get("action_masked"),
                },
                "env_result": {
                    "reward": step.get("reward"),
                    "score": step.get("score"),
                    "done": step.get("done"),
                    "reason": step.get("reason"),
                },
            }
        )
    return {
        "source_trajectory": str(source),
        "episode_id": episode["id"],
        "no_train": no_train,
        "layout_name": episode["layout_name"],
        "won": episode["won"],
        "steps": episode["steps"],
        "total_reward": episode["total_reward"],
        "final_reason": episode["final_reason"],
        "prompt_contract": {
            "system_prompt": system_prompt,
            "per_turn_user_text": user_text,
            "per_turn_user_image": "one rendered PNG frame in image_url",
            "forbidden_user_metadata": "no ASCII grid, coordinates, pellet count, score, or text state",
            "decode_constraint": "structured choice over legal non-backtracking actions",
        },
        "action_sequence": [step["action"] for step in episode["trajectory"]],
        "trajectory": compact_steps,
    }


def write_contact_sheet(frames: list[Image.Image], output: Path) -> None:
    thumb_size = (360, 280)
    columns = 3
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), BG)
    for index, frame in enumerate(frames):
        sheet.paste(frame.resize(thumb_size, Image.Resampling.LANCZOS), ((index % columns) * 360, (index // columns) * 280))
    sheet.save(output)


def export(
    source: Path,
    stem: str,
    label: str,
    accent: tuple[int, int, int],
    output_dir: Path = RUN_DIR,
    no_train: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = load_json(source)
    frames = [render_demo_frame(episode, step, label, accent) for step in episode["trajectory"]]
    terminal_frame = render_terminal_frame(episode, label, accent)
    frames.append(terminal_frame)
    terminal_frame.save(output_dir / f"{stem}_final_frame.png")
    terminal_frame.save(output_dir / f"{stem}_final_frame.jpg", quality=95, subsampling=0)
    write_contact_sheet(frames, output_dir / f"{stem}_contact_sheet.png")

    gif_frames = [frame.copy() for frame in frames]
    gif_frames[0].save(
        output_dir / f"{stem}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=[900] * (len(gif_frames) - 1) + [2600],
        loop=0,
        disposal=2,
        optimize=False,
    )
    with (output_dir / f"{stem}_prompt_response.json").open("w", encoding="utf-8") as handle:
        json.dump(compact_episode(episode, source, no_train=no_train), handle, indent=2)
        handle.write("\n")


def main() -> None:
    export(SUCCESS_SOURCE, "no_train_success_demo", "NO-TRAIN SUCCESS", GOOD)
    export(FAIL_SOURCE, "no_train_failure_demo", "NO-TRAIN FAILURE", BAD)
    print(RUN_DIR)


if __name__ == "__main__":
    main()
