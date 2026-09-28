from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Any

from .env import PacmanEnv


COLORS = {
    "background": (14, 18, 30),
    "empty": (22, 28, 43),
    "wall": (42, 88, 184),
    "grid": (10, 14, 24),
    "pellet": (255, 221, 128),
    "pacman": (255, 216, 45),
    "ghost": (238, 74, 93),
    "collision": (255, 255, 255),
}


def render_env_image(env: PacmanEnv, tile_size: int = 32) -> Any:
    """Render the current grid state as a deterministic RGB image."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install Pillow to use PacMan image observations") from exc

    if tile_size < 8:
        raise ValueError("tile_size must be at least 8")

    rows = len(env.layout)
    cols = max(len(row) for row in env.layout)
    image = Image.new("RGB", (cols * tile_size, rows * tile_size), COLORS["background"])
    draw = ImageDraw.Draw(image)

    for r in range(rows):
        for c in range(cols):
            x0 = c * tile_size
            y0 = r * tile_size
            x1 = x0 + tile_size - 1
            y1 = y0 + tile_size - 1
            fill = COLORS["wall"] if (r, c) in env.walls else COLORS["empty"]
            draw.rectangle((x0, y0, x1, y1), fill=fill)
            draw.rectangle((x0, y0, x1, y1), outline=COLORS["grid"])

    pellet_radius = max(2, tile_size // 8)
    for r, c in env.state.pellets:
        cx = c * tile_size + tile_size // 2
        cy = r * tile_size + tile_size // 2
        draw.ellipse(
            (cx - pellet_radius, cy - pellet_radius, cx + pellet_radius, cy + pellet_radius),
            fill=COLORS["pellet"],
        )

    pr, pc = env.state.pacman
    gr, gc = env.state.ghost
    margin = max(2, tile_size // 8)
    pacman_box = (
        pc * tile_size + margin,
        pr * tile_size + margin,
        (pc + 1) * tile_size - margin - 1,
        (pr + 1) * tile_size - margin - 1,
    )
    draw.ellipse(pacman_box, fill=COLORS["pacman"])

    ghost_box = (
        gc * tile_size + margin,
        gr * tile_size + margin,
        (gc + 1) * tile_size - margin - 1,
        (gr + 1) * tile_size - margin - 1,
    )
    ghost_color = COLORS["collision"] if (pr, pc) == (gr, gc) else COLORS["ghost"]
    draw.rounded_rectangle(ghost_box, radius=max(2, tile_size // 6), fill=ghost_color)
    return image


def image_png_bytes(image: Any) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def env_png_bytes(env: PacmanEnv, tile_size: int = 32) -> bytes:
    return image_png_bytes(render_env_image(env, tile_size=tile_size))


def image_sha256(png_bytes: bytes) -> str:
    return hashlib.sha256(png_bytes).hexdigest()


def image_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"
