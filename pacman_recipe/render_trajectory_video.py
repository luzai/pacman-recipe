from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CELL = 56
PANEL_PAD = 24
HEADER_H = 112
FOOTER_H = 72
GAP = 24
BG = (18, 22, 30)
PANEL_BG = (28, 34, 46)
WALL = (36, 83, 196)
PATH = (10, 12, 18)
PELLET = (245, 214, 112)
PACMAN = (255, 214, 50)
GHOST = (235, 76, 100)
TEXT = (236, 241, 247)
MUTED = (160, 171, 190)
GOOD = (73, 210, 138)
BAD = (255, 113, 113)


def load_trajectory(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def grid_from_obs(obs: str) -> list[str]:
    lines = obs.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "Grid:":
            rows: list[str] = []
            for row in lines[idx + 1 :]:
                if not row:
                    break
                if set(row) <= {"#", "P", "G", "X", ".", " "}:
                    rows.append(row)
                else:
                    break
            if rows:
                return rows
    raise ValueError("trajectory step does not contain a parsable Grid block")


def final_grid_after_action(step: dict[str, Any]) -> list[str] | None:
    if not step.get("done"):
        return None
    action = str(step.get("action", "")).strip().lower()
    delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
        "stay": (0, 0),
    }.get(action)
    if delta is None:
        return None
    grid = [list(row) for row in grid_from_obs(str(step.get("obs", "")))]
    pacman_pos = None
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == "P":
                pacman_pos = (row_idx, col_idx)
                break
        if pacman_pos is not None:
            break
    if pacman_pos is None:
        return None
    row_idx, col_idx = pacman_pos
    next_row = row_idx + delta[0]
    next_col = col_idx + delta[1]
    if not (0 <= next_row < len(grid) and 0 <= next_col < len(grid[next_row])):
        return None
    if grid[next_row][next_col] == "#":
        return None
    grid[row_idx][col_idx] = " "
    grid[next_row][next_col] = "P"
    return ["".join(row) for row in grid]


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=fill, font=font)


def fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def pacman_angles(action: str) -> tuple[int, int]:
    return {
        "right": (35, 325),
        "left": (215, 145),
        "up": (305, 235),
        "down": (125, 55),
    }.get(action, (20, 340))


def draw_panel(
    episode: dict[str, Any],
    step_idx: int,
    label: str,
    panel_size: tuple[int, int],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    final_frame: bool = False,
) -> Image.Image:
    steps = episode.get("trajectory", [])
    if not steps:
        raise ValueError("episode has no trajectory steps")
    step = steps[min(step_idx, len(steps) - 1)]
    grid = final_grid_after_action(step) if final_frame else None
    if grid is None:
        grid = grid_from_obs(str(step.get("obs", "")))
    rows, cols = len(grid), max(len(row) for row in grid)
    width, height = panel_size
    image = Image.new("RGB", panel_size, PANEL_BG)
    draw = ImageDraw.Draw(image)

    won = bool(episode.get("won", False))
    accent = GOOD if won else BAD
    draw_text(draw, (PANEL_PAD, 18), label, accent, font)
    summary = f"{'WIN' if won else 'FAIL'} | total {float(episode.get('total_reward', 0.0)):.1f} | steps {episode.get('steps', 0)}"
    draw_text(draw, (PANEL_PAD, 54), summary, TEXT, small_font)

    grid_w = cols * CELL
    grid_h = rows * CELL
    grid_x = (width - grid_w) // 2
    grid_y = HEADER_H

    for r, row in enumerate(grid):
        for c, cell in enumerate(row.ljust(cols)):
            x0 = grid_x + c * CELL
            y0 = grid_y + r * CELL
            x1 = x0 + CELL
            y1 = y0 + CELL
            draw.rectangle((x0, y0, x1, y1), fill=PATH)
            if cell == "#":
                draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=8, fill=WALL)
            elif cell == ".":
                cx, cy = x0 + CELL // 2, y0 + CELL // 2
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=PELLET)
            elif cell == "P":
                start, end = pacman_angles(str(step.get("action", "")))
                draw.pieslice((x0 + 7, y0 + 7, x1 - 7, y1 - 7), start=start, end=end, fill=PACMAN)
                draw.ellipse((x0 + 32, y0 + 14, x0 + 38, y0 + 20), fill=(22, 22, 22))
            elif cell in {"G", "X"}:
                draw.rounded_rectangle((x0 + 8, y0 + 9, x1 - 8, y1 - 5), radius=16, fill=GHOST)
                draw.ellipse((x0 + 20, y0 + 21, x0 + 27, y0 + 28), fill=(255, 255, 255))
                draw.ellipse((x0 + 31, y0 + 21, x0 + 38, y0 + 28), fill=(255, 255, 255))
                draw.ellipse((x0 + 23, y0 + 23, x0 + 26, y0 + 26), fill=(20, 20, 30))
                draw.ellipse((x0 + 34, y0 + 23, x0 + 37, y0 + 26), fill=(20, 20, 30))

    footer_y = grid_y + grid_h + 18
    action = str(step.get("action", ""))
    reward = float(step.get("reward", 0.0))
    score = step.get("score", "")
    reason = str(step.get("reason", ""))
    model_output = fit_text(str(step.get("model_output", "")).replace("\n", " "), 64)
    status_prefix = "final" if final_frame else f"step {step.get('step', step_idx)}"
    status = f"{status_prefix} | action {action} | reward {reward:g} | score {score} | {reason}"
    draw_text(draw, (PANEL_PAD, footer_y), status, TEXT, small_font)
    draw_text(draw, (PANEL_PAD, footer_y + 30), f"model: {model_output!r}", MUTED, small_font)
    return image


def title_card(size: tuple[int, int], title: str, subtitle: str, frames: int) -> list[Image.Image]:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    cards: list[Image.Image] = []
    for _ in range(frames):
        image = Image.new("RGB", size, BG)
        draw = ImageDraw.Draw(image)
        draw_text(draw, (48, 48), title, TEXT, font)
        draw_text(draw, (48, 86), subtitle, MUTED, small_font)
        cards.append(image)
    return cards


def compose_frames(left: dict[str, Any], right: dict[str, Any], left_label: str, right_label: str, fps: int) -> list[Image.Image]:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    max_rows = 0
    max_cols = 0
    for episode in (left, right):
        for step in episode.get("trajectory", []):
            grid = grid_from_obs(str(step.get("obs", "")))
            max_rows = max(max_rows, len(grid))
            max_cols = max(max_cols, max(len(row) for row in grid))
    panel_size = (
        max_cols * CELL + PANEL_PAD * 2,
        HEADER_H + max_rows * CELL + FOOTER_H,
    )
    full_size = (panel_size[0] * 2 + GAP, panel_size[1])
    frames = title_card(
        full_size,
        "PacMan RL trajectory comparison",
        "Left: failure trajectory. Right: shaped/constrained success trajectory.",
        max(1, fps * 2),
    )
    left_steps = left.get("trajectory", [])
    right_steps = right.get("trajectory", [])
    left_has_final = bool(left.get("won")) and bool(left_steps and final_grid_after_action(left_steps[-1]))
    right_has_final = bool(right.get("won")) and bool(right_steps and final_grid_after_action(right_steps[-1]))
    steps = max(len(left_steps) + int(left_has_final), len(right_steps) + int(right_has_final))
    for idx in range(steps):
        left_final = left_has_final and idx >= len(left_steps)
        right_final = right_has_final and idx >= len(right_steps)
        left_idx = min(idx, max(0, len(left_steps) - 1))
        right_idx = min(idx, max(0, len(right_steps) - 1))
        canvas = Image.new("RGB", full_size, BG)
        canvas.paste(draw_panel(left, left_idx, left_label, panel_size, font, small_font, final_frame=left_final), (0, 0))
        canvas.paste(draw_panel(right, right_idx, right_label, panel_size, font, small_font, final_frame=right_final), (panel_size[0] + GAP, 0))
        frames.extend([canvas] * max(1, math.ceil(fps * 0.7)))
    caveat = title_card(
        full_size,
        "Caveat",
        "The success clip may use hidden reward shaping and decode-time action constraints.",
        max(1, fps * 2),
    )
    frames.extend(caveat)
    return frames


def compose_single_frames(episode: dict[str, Any], label: str, fps: int, title: str, subtitle: str) -> list[Image.Image]:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    max_rows = 0
    max_cols = 0
    for step in episode.get("trajectory", []):
        grid = grid_from_obs(str(step.get("obs", "")))
        max_rows = max(max_rows, len(grid))
        max_cols = max(max_cols, max(len(row) for row in grid))
    if max_rows == 0 or max_cols == 0:
        raise ValueError("episode has no parsable trajectory frames")
    panel_size = (
        max_cols * CELL + PANEL_PAD * 2,
        HEADER_H + max_rows * CELL + FOOTER_H,
    )
    frames = title_card(panel_size, title, subtitle, max(1, fps * 2))
    for idx in range(len(episode.get("trajectory", []))):
        frames.extend([draw_panel(episode, idx, label, panel_size, font, small_font)] * max(1, math.ceil(fps * 0.7)))
    steps = episode.get("trajectory", [])
    if bool(episode.get("won")) and steps and final_grid_after_action(steps[-1]):
        frames.extend(
            [
                draw_panel(episode, len(steps) - 1, label, panel_size, font, small_font, final_frame=True)
            ]
            * max(1, math.ceil(fps * 0.7))
        )
    return frames


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PacMan trajectory videos.")
    parser.add_argument("--single", type=Path, help="Render one trajectory JSON.")
    parser.add_argument("--left", type=Path, help="Failure trajectory JSON.")
    parser.add_argument("--right", type=Path, help="Success trajectory JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 or .gif path.")
    parser.add_argument("--label", default="Trajectory")
    parser.add_argument("--left-label", default="Failure")
    parser.add_argument("--right-label", default="Success")
    parser.add_argument("--title", default="PacMan RL trajectory")
    parser.add_argument("--subtitle", default="Rendered from saved trajectory JSON.")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    if args.single:
        frames = compose_single_frames(load_trajectory(args.single), args.label, args.fps, args.title, args.subtitle)
    else:
        if not args.left or not args.right:
            parser.error("--left and --right are required unless --single is provided")
        frames = compose_frames(
            load_trajectory(args.left),
            load_trajectory(args.right),
            args.left_label,
            args.right_label,
            args.fps,
        )
    write_video(frames, args.output, args.fps)
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
