"""Canonical action tokens shared by Pacman environment consumers."""

from __future__ import annotations

from enum import Enum

from .errors import InvalidActionError


class Action(str, Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"
    STAY = "S"


ACTION_ORDER = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.STAY,
)

MOVE_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.STAY: (0, 0),
}


def coerce_action(value: Action | str) -> Action:
    """Accept only an Action or one exact canonical action token."""
    if isinstance(value, Action):
        return value
    if isinstance(value, str):
        try:
            return Action(value)
        except ValueError:
            pass
    raise InvalidActionError(
        f"invalid action {value!r}; expected U, D, L, R, or S"
    )
