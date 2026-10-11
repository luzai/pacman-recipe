"""Shared, fail-closed ghost-mode contract for live state and saved evidence."""

from typing import Any, Mapping
import os

def active_ghost_count():
    count = int(os.environ.get("SMALL_MAZE_GHOST_COUNT", "4"))
    if count not in (1, 2, 4):
        raise ValueError("active ghost count must be 1, 2 or 4")
    return count


GHOST_MODES = ("disabled", "normal")


def validate_ghost_mode(mode: str) -> str:
    if mode not in GHOST_MODES:
        raise ValueError("ghost_mode must be disabled or normal")
    return mode


def validate_ghost_state(state: Mapping[str, Any], expected: str) -> None:
    validate_ghost_mode(expected)
    if state.get("ghost_mode") != expected:
        raise ValueError("ghost_mode does not match environment evidence")
    ghosts = state.get("ghosts")
    if not isinstance(ghosts, list) or len(ghosts) != (
        0 if expected == "disabled" else active_ghost_count()
    ):
        raise ValueError("ghost count does not match ghost_mode")
    if expected == "disabled":
        if state.get("edible_ticks") != 0:
            raise ValueError("disabled ghosts cannot have vulnerability ticks")
        if any(
            event.get("type", event.get("event_type")) in {"ghost_eaten", "death"}
            for event in state.get("events", [])
        ):
            raise ValueError("disabled ghosts cannot produce ghost/death events")
