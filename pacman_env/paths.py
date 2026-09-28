"""Path configuration with explicit compatibility for historical launchers."""

from __future__ import annotations

import os
from pathlib import Path


def configured_path(name: str, *legacy_names: str) -> str | None:
    """Accept old names, but never silently choose between conflicting paths."""
    values = [(key, os.environ[key]) for key in (name, *legacy_names) if os.environ.get(key)]
    if not values:
        return None
    normalized = {os.path.normcase(str(Path(value).expanduser().resolve())) for _, value in values}
    if len(normalized) != 1:
        raise ValueError("Conflicting path variables: " + ", ".join(key for key, _ in values))
    return str(Path(values[0][1]).expanduser().resolve())


def pacman_python_root() -> str | None:
    return configured_path("PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT")


def recipe_root() -> Path:
    return Path(configured_path("PACMAN_RECIPE_ROOT", "AREAL_PACMAN_ROOT") or Path(__file__).resolve().parents[1])
