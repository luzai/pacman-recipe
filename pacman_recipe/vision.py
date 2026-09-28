"""Backward-compatible imports for synthetic environment rendering."""

from .synthetic.vision import (
    COLORS,
    env_png_bytes,
    image_data_url,
    image_png_bytes,
    image_sha256,
    render_env_image,
)

__all__ = [
    "COLORS",
    "env_png_bytes",
    "image_data_url",
    "image_png_bytes",
    "image_sha256",
    "render_env_image",
]
