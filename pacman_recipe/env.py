"""Backward-compatible imports for the historical synthetic environment."""

from .synthetic.env import (
    ACTIONS,
    DEFAULT_LAYOUT,
    LAYOUTS,
    MEDIUM_DEFAULT_LAYOUT,
    MOVE_DELTAS,
    OPPOSITE_ACTION,
    ROUTE_REWARD_MODES,
    SAFE_PROGRESS_REWARD_MODE,
    SMALL_DEFAULT_LAYOUT,
    TINY_CORRIDOR_LAYOUT,
    PacmanEnv,
    PacmanState,
    layout_by_name,
)

__all__ = [
    "ACTIONS",
    "DEFAULT_LAYOUT",
    "LAYOUTS",
    "MEDIUM_DEFAULT_LAYOUT",
    "MOVE_DELTAS",
    "OPPOSITE_ACTION",
    "ROUTE_REWARD_MODES",
    "SAFE_PROGRESS_REWARD_MODE",
    "SMALL_DEFAULT_LAYOUT",
    "TINY_CORRIDOR_LAYOUT",
    "PacmanEnv",
    "PacmanState",
    "layout_by_name",
]
