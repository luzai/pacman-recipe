"""Read-only navigation helpers over Pacman's bundled level topology."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection

from pacman_env.actions import MOVE_DELTAS, Action

from .level import LevelDefinition
from .state import Position


CARDINAL_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def transition(
    level: LevelDefinition,
    position: Position,
    action: Action,
) -> Position | None:
    """Return the adjacent/portal destination, or ``None`` for a wall."""

    if action not in CARDINAL_ACTIONS:
        return position
    row_delta, col_delta = MOVE_DELTAS[action]
    candidate = Position(position.row + row_delta, position.col + col_delta)
    if level.is_wall(candidate):
        return None
    return level.portal_exit(candidate, action.value)


def route_to_nearest(
    level: LevelDefinition,
    start: Position,
    targets: Collection[Position],
) -> tuple[Action, ...]:
    """Return a deterministic shortest route to any reachable target."""

    target_set = frozenset(targets)
    if not target_set:
        return ()
    queue: deque[tuple[Position, tuple[Action, ...]]] = deque([(start, ())])
    visited = {start}
    while queue:
        position, route = queue.popleft()
        if position in target_set:
            return route
        for action in CARDINAL_ACTIONS:
            candidate = transition(level, position, action)
            if candidate is not None and candidate not in visited:
                visited.add(candidate)
                queue.append((candidate, (*route, action)))
    raise ValueError("no target is reachable from the requested position")


def nearest_reachable_distance(
    level: LevelDefinition,
    start: Position,
    targets: Collection[Position],
) -> int:
    """Return shortest-path distance; an empty target set has distance zero."""

    return len(route_to_nearest(level, start, targets))
