"""Pure-Python Edward tactical planner over structured environment state.

This module deliberately depends only on :mod:`maapacman.env`. It can run on
a Linux training node without MaaFramework, Win32 capture, OpenCV, or a game
window. Candidate identifiers are stable strategy options:

* ``C*``: collect a reachable pellet through a route with an escape margin.
* ``A*``: avoid a lethal ghost by moving toward a safer junction/region.
* ``E*``: eliminate a reachable vulnerable ghost before its timer expires.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

from .actions import Action
from .env.level import LevelDefinition, load_bundled_level
from .env.navigation import CARDINAL_ACTIONS
from .env.state import Position


_ACTION_BY_TOKEN = {action.value: action for action in CARDINAL_ACTIONS}
_OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_LETHAL_STATES = frozenset({"normal", 1})
_EDIBLE_STATES = frozenset({"vulnerable", 2})
_SAFETY_MARGIN = 1
_MAX_ESCAPE_SEARCH = 10


class EdwardSafetyRefusal(RuntimeError):
    """The planner cannot prove that any currently legal action is ghost-safe."""


@dataclass(frozen=True)
class GhostETA:
    """Auditable arrival time for one lethal ghost at an option target."""

    entity_id: int | str | None
    eta: int | None


@dataclass(frozen=True)
class PlannerCandidate:
    """One planner-approved high-level option."""

    option_id: str
    strategy: str
    target: tuple[int, int]
    first_action: str
    route_distance: int
    commit_moves: int
    safety_margin: int | None = None
    future_safe_exits: int | None = None
    entity_id: int | str | None = None
    lethal_ghost_etas: tuple[GhostETA, ...] = ()
    choke_points: tuple[tuple[int, int], ...] = ()
    safe_return_distance: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerDecision:
    """Deterministic option selection and its immediately executable action."""

    option_id: str
    action: str
    candidates: tuple[PlannerCandidate, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "action": self.action,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _position(value: Any) -> Position | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        result = Position(int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None
    return result if result.row >= 0 and result.col >= 0 else None


@dataclass(frozen=True)
class _GhostThreat:
    entity_id: int | str | None
    position: Position


def _is_wall(level: LevelDefinition, position: Position, *, actor: str) -> bool:
    """Use the v3 actor-aware wall contract and fail on an old topology."""

    return bool(level.is_wall(position, actor=actor))


def _transition(
    level: LevelDefinition,
    position: Position,
    action: Action,
    *,
    actor: str,
    use_portals: bool = True,
) -> Position | None:
    if action not in CARDINAL_ACTIONS:
        return position
    row_delta, col_delta = {
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
    }[action]
    candidate = Position(position.row + row_delta, position.col + col_delta)
    if _is_wall(level, candidate, actor=actor):
        return None
    if use_portals:
        return level.portal_exit(candidate, action.value)
    return candidate


def _neighbors(
    level: LevelDefinition,
    position: Position,
    *,
    actor: str = "pacman",
    use_portals: bool = True,
) -> Iterable[tuple[str, Position]]:
    for action in CARDINAL_ACTIONS:
        candidate = _transition(
            level,
            position,
            action,
            actor=actor,
            use_portals=use_portals,
        )
        if candidate is not None:
            yield action.value, candidate


@lru_cache(maxsize=None)
def _routes_from(
    level: LevelDefinition,
    start: Position,
    *,
    actor: str = "pacman",
    use_portals: bool = True,
) -> dict[Position, tuple[str, ...]]:
    """Return all shortest routes from one tile after a single BFS."""

    queue: deque[tuple[Position, tuple[str, ...]]] = deque([(start, ())])
    routes: dict[Position, tuple[str, ...]] = {start: ()}
    while queue:
        position, route = queue.popleft()
        for action, candidate in _neighbors(
            level, position, actor=actor, use_portals=use_portals
        ):
            if candidate in routes:
                continue
            next_route = (*route, action)
            routes[candidate] = next_route
            queue.append((candidate, next_route))
    return routes


def _shortest_path(
    level: LevelDefinition,
    start: Position,
    target: Position,
    *,
    actor: str = "pacman",
    use_portals: bool = True,
) -> tuple[str, ...] | None:
    return _routes_from(
        level,
        start,
        actor=actor,
        use_portals=use_portals,
    ).get(target)


def _distance(
    level: LevelDefinition,
    start: Position,
    target: Position,
    *,
    actor: str = "pacman",
    use_portals: bool = True,
) -> int | None:
    route = _shortest_path(
        level,
        start,
        target,
        actor=actor,
        use_portals=use_portals,
    )
    return None if route is None else len(route)


def _route_positions(
    level: LevelDefinition,
    start: Position,
    route: Iterable[str],
) -> tuple[Position, ...]:
    positions: list[Position] = []
    current = start
    for token in route:
        candidate = _transition(
            level,
            current,
            _ACTION_BY_TOKEN[token],
            actor="pacman",
        )
        if candidate is None:
            return ()
        current = candidate
        positions.append(current)
    return tuple(positions)


def _ghost_distance(
    level: LevelDefinition,
    ghosts: Iterable[_GhostThreat],
    target: Position,
) -> int | None:
    distances: list[int] = []
    for ghost in ghosts:
        distance = _distance(
            level,
            ghost.position,
            target,
            actor="ghost",
            # pacman-python only teleports Pacman when CheckIfHitSomething
            # handles a portal tile; ghost movement follows the ordinary grid.
            use_portals=False,
        )
        if distance is not None:
            distances.append(distance)
    return min(distances) if distances else None


def _ghost_etas(
    level: LevelDefinition,
    ghosts: Iterable[_GhostThreat],
    target: Position,
) -> tuple[GhostETA, ...]:
    arrivals = [
        GhostETA(
            ghost.entity_id,
            _distance(
                level,
                ghost.position,
                target,
                actor="ghost",
                use_portals=False,
            ),
        )
        for ghost in ghosts
    ]
    return tuple(
        sorted(
            arrivals,
            key=lambda item: (
                str(item.entity_id),
                item.eta is None,
                item.eta if item.eta is not None else 10**9,
            ),
        )
    )


def _route_margin(
    level: LevelDefinition,
    start: Position,
    route: tuple[str, ...],
    lethal: tuple[_GhostThreat, ...],
) -> int:
    """Conservative ghost-arrival margin along a proposed route."""

    if not lethal:
        return 10**6
    margins: list[int] = []
    for player_eta, tile in enumerate(
        _route_positions(level, start, route), start=1
    ):
        ghost_eta = _ghost_distance(level, lethal, tile)
        if ghost_eta is not None:
            margins.append(ghost_eta - player_eta)
    return min(margins, default=10**6)


def _one_step_pixel_safe(
    level: LevelDefinition,
    state: Mapping[str, Any],
    target: Position,
) -> bool:
    """Prove a margin-one emergency action against v3 pixel/path state.

    Pacman reaches the target tile atomically and then ghosts advance for 16
    logic frames.  A ghost two topology tiles away can still collide in that
    interval when it is already part-way through the first tile, so tile ETA
    alone is insufficient for the margin-one case.
    """

    target_y = float(target.row * 16)
    target_x = float(target.col * 16)
    deltas = {
        "U": (-1.0, 0.0),
        "D": (1.0, 0.0),
        "L": (0.0, -1.0),
        "R": (0.0, 1.0),
    }
    for ghost in state.get("ghosts") or ():
        if not isinstance(ghost, Mapping) or ghost.get("state") not in _LETHAL_STATES:
            continue
        pixel = ghost.get("pixel_position")
        direction = str(ghost.get("direction", ""))
        path_value = ghost.get("path_remaining")
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or direction not in deltas
            or not isinstance(path_value, str)
        ):
            # Margin-one approval requires the complete API-v3 evidence.
            return False
        try:
            y, x = float(pixel[0]), float(pixel[1])
            speed = float(ghost.get("speed", 1.0))
        except (TypeError, ValueError):
            return False
        if speed <= 0:
            speed = 0.0
        path = path_value
        for _ in range(16):
            dy, dx = deltas[direction]
            y += dy * speed
            x += dx * speed
            if abs(target_y - y) < 16 and abs(target_x - x) < 16:
                return False
            if abs(y % 16) > 1e-9 or abs(x % 16) > 1e-9:
                continue
            if path:
                path = path[1:]
            if path and path[0] in deltas:
                direction = path[0]
                continue
            ghost_tile = Position(int(round(y / 16)), int(round(x / 16)))
            chase = _shortest_path(
                level,
                ghost_tile,
                target,
                actor="ghost",
                use_portals=False,
            )
            if chase:
                direction = chase[0]
    return True


def _future_safe_exits(
    level: LevelDefinition,
    target: Position,
    lethal: tuple[_GhostThreat, ...],
    player_eta: int,
) -> int:
    result = 0
    for _, candidate in _neighbors(level, target):
        ghost_eta = _ghost_distance(level, lethal, candidate)
        if ghost_eta is None or ghost_eta > player_eta + 1:
            result += 1
    return result


def _degree(level: LevelDefinition, position: Position) -> int:
    return sum(1 for _ in _neighbors(level, position, actor="pacman"))


@lru_cache(maxsize=None)
def _components_without_tile(
    level: LevelDefinition,
    blocked: Position,
) -> tuple[frozenset[Position], ...]:
    """Cache Pacman's connected components after removing one choke tile."""

    remaining = {
        Position(row, col)
        for row in range(level.height)
        for col in range(level.width)
        if Position(row, col) != blocked
        and not _is_wall(level, Position(row, col), actor="pacman")
    }
    components: list[frozenset[Position]] = []
    while remaining:
        origin = min(remaining)
        queue: deque[Position] = deque([origin])
        component = {origin}
        remaining.remove(origin)
        while queue:
            position = queue.popleft()
            for _, candidate in _neighbors(level, position, actor="pacman"):
                if candidate not in remaining:
                    continue
                remaining.remove(candidate)
                component.add(candidate)
                queue.append(candidate)
        components.append(frozenset(component))
    return tuple(components)


def _reachable_without_tile(
    level: LevelDefinition,
    start: Position,
    target: Position,
    blocked: Position,
) -> bool:
    if blocked in {start, target}:
        return False
    return any(
        start in component and target in component
        for component in _components_without_tile(level, blocked)
    )


def _safe_escape_to_junction(
    level: LevelDefinition,
    target: Position,
    lethal: tuple[_GhostThreat, ...],
    *,
    target_eta: int,
) -> int | None:
    """Return time after target to a safe junction, if one is reachable."""

    queue: deque[tuple[Position, int]] = deque([(target, target_eta)])
    best_time = {target: target_eta}
    while queue:
        tile, absolute_time = queue.popleft()
        safe_exits = _future_safe_exits(level, tile, lethal, absolute_time)
        if _degree(level, tile) >= 3 and safe_exits >= 2:
            return absolute_time - target_eta
        if absolute_time - target_eta >= _MAX_ESCAPE_SEARCH:
            continue
        for _, candidate in _neighbors(level, tile, actor="pacman"):
            next_time = absolute_time + 1
            if best_time.get(candidate, 10**9) <= next_time:
                continue
            ghost_eta = _ghost_distance(level, lethal, candidate)
            if (
                ghost_eta is not None
                and ghost_eta <= next_time + _SAFETY_MARGIN
            ):
                continue
            best_time[candidate] = next_time
            queue.append((candidate, next_time))
    return None


def _collect_route_safety(
    level: LevelDefinition,
    start: Position,
    target: Position,
    route: tuple[str, ...],
    lethal: tuple[_GhostThreat, ...],
) -> tuple[bool, tuple[tuple[int, int], ...], int | None]:
    """Audit choke points and prove a time-safe exit from a target pocket."""

    route_tiles = _route_positions(level, start, route)
    choke_positions = tuple(
        tile
        for tile in route_tiles[:-1]
        if not _reachable_without_tile(level, start, target, tile)
    )
    choke_points = tuple((tile.row, tile.col) for tile in choke_positions)
    if not lethal:
        return True, choke_points, 0

    target_eta = len(route)
    for entry_eta, choke in enumerate(route_tiles[:-1], start=1):
        if choke not in choke_positions:
            continue
        return_eta = target_eta + (target_eta - entry_eta)
        ghost_eta = _ghost_distance(level, lethal, choke)
        if (
            ghost_eta is not None
            and ghost_eta <= return_eta + _SAFETY_MARGIN
        ):
            return False, choke_points, None

    needs_escape_proof = bool(choke_positions) or _degree(level, target) <= 1
    if not needs_escape_proof:
        return True, choke_points, 0
    escape_distance = _safe_escape_to_junction(
        level,
        target,
        lethal,
        target_eta=target_eta,
    )
    return escape_distance is not None, choke_points, escape_distance


class EdwardPlanner:
    """Stateful deterministic planner for one Level-1 episode."""

    def __init__(self, level: LevelDefinition | None = None) -> None:
        self.level = level or load_bundled_level(1)
        self.reset()

    def reset(self) -> None:
        self.remaining = set(self.level.pellets | self.level.power_pellets)
        self.last_action: str | None = None

    def record_action(self, action: str) -> None:
        if action not in _ACTION_BY_TOKEN and action != "S":
            raise ValueError(f"invalid planner action: {action!r}")
        self.last_action = action

    def observe(self, state: Mapping[str, Any]) -> Position:
        position = _position(state.get("pacman_position"))
        if position is None:
            position = _position((state.get("row"), state.get("col")))
        if position is None:
            raise ValueError("planner state must contain a valid Pacman position")
        self.remaining.discard(position)
        return position

    def candidates(self, state: Mapping[str, Any]) -> tuple[PlannerCandidate, ...]:
        player = self.observe(state)
        legal = set(state.get("open") or state.get("legal_actions") or ())
        open_actions = tuple(
            token for token in ("U", "D", "L", "R") if token in legal
        )
        if not open_actions:
            return ()

        lethal: list[_GhostThreat] = []
        edible: list[tuple[int | str | None, Position]] = []
        for ghost in state.get("ghosts") or ():
            if not isinstance(ghost, Mapping):
                continue
            position = _position(ghost.get("position"))
            if position is None:
                continue
            ghost_state = ghost.get("state")
            if ghost_state in _LETHAL_STATES:
                lethal.append(_GhostThreat(ghost.get("id"), position))
            elif ghost_state in _EDIBLE_STATES:
                edible.append((ghost.get("id"), position))
        lethal_tuple = tuple(lethal)

        candidates: list[PlannerCandidate] = []
        collect: list[
            tuple[
                int,
                int,
                Position,
                tuple[str, ...],
                int,
                tuple[tuple[int, int], ...],
                int | None,
                tuple[GhostETA, ...],
            ]
        ] = []
        reachable_collect_margins: list[int] = []
        for target in self.remaining:
            route = _shortest_path(
                self.level,
                player,
                target,
                actor="pacman",
            )
            if not route or route[0] not in open_actions:
                continue
            margin = _route_margin(self.level, player, route, lethal_tuple)
            reachable_collect_margins.append(margin)
            exits = _future_safe_exits(
                self.level, target, lethal_tuple, len(route)
            )
            safe_return, choke_points, return_distance = _collect_route_safety(
                self.level,
                player,
                target,
                route,
                lethal_tuple,
            )
            if (
                margin <= _SAFETY_MARGIN
                or (exits == 0 and lethal_tuple)
                or not safe_return
            ):
                continue
            collect.append(
                (
                    len(route),
                    -margin,
                    target,
                    route,
                    exits,
                    choke_points,
                    return_distance,
                    _ghost_etas(self.level, lethal_tuple, target),
                )
            )
        collect.sort(key=lambda item: (item[0], item[1], item[2]))
        for rank, (
            distance,
            negative_margin,
            target,
            route,
            exits,
            choke_points,
            return_distance,
            target_ghost_etas,
        ) in enumerate(collect[:4]):
            candidates.append(
                PlannerCandidate(
                    option_id=f"C{rank}",
                    strategy="COLLECT",
                    target=(target.row, target.col),
                    first_action=route[0],
                    route_distance=distance,
                    commit_moves=min(8, distance),
                    safety_margin=-negative_margin,
                    future_safe_exits=exits,
                    lethal_ghost_etas=target_ghost_etas,
                    choke_points=choke_points,
                    safe_return_distance=return_distance,
                )
            )

        nearest_lethal = _ghost_distance(self.level, lethal_tuple, player)
        collect_threatened = not collect or (
            reachable_collect_margins
            and min(reachable_collect_margins) <= _SAFETY_MARGIN + 1
        )
        if nearest_lethal is not None and (
            nearest_lethal <= 6 or collect_threatened
        ):
            avoid: list[
                tuple[
                    int,
                    int,
                    int,
                    int,
                    int,
                    Position,
                    tuple[str, ...],
                    tuple[GhostETA, ...],
                    tuple[tuple[int, int], ...],
                    int | None,
                ]
            ] = []
            for row in range(self.level.height):
                for col in range(self.level.width):
                    target = Position(row, col)
                    if (
                        _is_wall(self.level, target, actor="pacman")
                        or target == player
                    ):
                        continue
                    route = _shortest_path(
                        self.level,
                        player,
                        target,
                        actor="pacman",
                    )
                    if (
                        not route
                        or route[0] not in open_actions
                        or not 1 <= len(route) <= 8
                        # An AVOID target must be an escape anchor, not merely
                        # a temporarily clear tile inside a two-way corridor.
                        or _degree(self.level, target) < 3
                    ):
                        continue
                    margin = _route_margin(
                        self.level, player, route, lethal_tuple
                    )
                    exits = _future_safe_exits(
                        self.level, target, lethal_tuple, len(route)
                    )
                    clearance = _ghost_distance(self.level, lethal_tuple, target)
                    safe_return, choke_points, return_distance = (
                        _collect_route_safety(
                            self.level,
                            player,
                            target,
                            route,
                            lethal_tuple,
                        )
                    )
                    if (
                        margin <= _SAFETY_MARGIN
                        or exits < 2
                        or clearance is None
                        or clearance <= nearest_lethal
                        or not safe_return
                    ):
                        continue
                    reversal = int(route[0] == _OPPOSITE.get(self.last_action))
                    avoid.append(
                        (
                            -margin,
                            -clearance,
                            -exits,
                            reversal,
                            len(route),
                            target,
                            route,
                            _ghost_etas(self.level, lethal_tuple, target),
                            choke_points,
                            return_distance,
                        )
                    )
            if any(item[3] == 0 for item in avoid):
                avoid = [item for item in avoid if item[3] == 0]
            avoid.sort(key=lambda item: item[:6])
            for rank, (
                negative_margin,
                _,
                negative_exits,
                _,
                distance,
                target,
                route,
                target_ghost_etas,
                choke_points,
                return_distance,
            ) in enumerate(avoid[:4]):
                candidates.append(
                    PlannerCandidate(
                        option_id=f"A{rank}",
                        strategy="AVOID",
                        target=(target.row, target.col),
                        first_action=route[0],
                        route_distance=distance,
                        commit_moves=min(3, distance),
                        safety_margin=-negative_margin,
                        future_safe_exits=-negative_exits,
                        lethal_ghost_etas=target_ghost_etas,
                        choke_points=choke_points,
                        safe_return_distance=return_distance,
                    )
                )

        safe_edible_moves = max(
            0, (int(state.get("edible_ticks", 0)) - 32) // 16
        )
        eliminate: list[
            tuple[int, str, Position, tuple[str, ...], int | str | None]
        ] = []
        for entity_id, target in edible:
            route = _shortest_path(
                self.level,
                player,
                target,
                actor="pacman",
            )
            if (
                route
                and route[0] in open_actions
                and len(route) <= safe_edible_moves
            ):
                eliminate.append(
                    (len(route), str(entity_id), target, route, entity_id)
                )
        eliminate.sort(key=lambda item: (item[0], item[1], item[2]))
        for rank, (
            distance,
            _,
            target,
            route,
            entity_id,
        ) in enumerate(eliminate[:2]):
            candidates.append(
                PlannerCandidate(
                    option_id=f"E{rank}",
                    strategy="ELIMINATE",
                    target=(target.row, target.col),
                    first_action=route[0],
                    route_distance=distance,
                    commit_moves=min(6, distance),
                    entity_id=entity_id,
                )
            )
        return tuple(candidates)

    def advertised_candidates(
        self, state: Mapping[str, Any]
    ) -> tuple[PlannerCandidate, ...]:
        """Return at least one candidate without selecting on the model's behalf."""

        candidates = self.candidates(state)
        if candidates:
            return candidates
        player = _position(state.get("pacman_position")) or _position(
            (state.get("row"), state.get("col"))
        )
        lethal = tuple(
            _GhostThreat(ghost.get("id"), position)
            for ghost in state.get("ghosts") or ()
            if isinstance(ghost, Mapping)
            and ghost.get("state") in _LETHAL_STATES
            and (position := _position(ghost.get("position"))) is not None
        )
        return (self._emergency_candidate(state, player, lethal),)

    def decide(self, state: Mapping[str, Any]) -> PlannerDecision:
        candidates = self.advertised_candidates(state)
        edible = [item for item in candidates if item.strategy == "ELIMINATE"]
        avoid = [item for item in candidates if item.strategy == "AVOID"]
        collect = [item for item in candidates if item.strategy == "COLLECT"]

        if edible:
            selected = edible[0]
        elif collect:
            # Every advertised COLLECT route has already passed the stricter
            # per-tile arrival-margin and safe-return proofs.  Do not override
            # that proof with the coarser "ghost within six tiles" heuristic;
            # it caused repeated AVOID replanning into the seed-0 bottom loop.
            selected = collect[0]
        elif avoid:
            selected = avoid[0]
        else:
            selected = candidates[0]

        self.last_action = selected.first_action
        return PlannerDecision(
            option_id=selected.option_id,
            action=selected.first_action,
            candidates=candidates or (selected,),
        )

    def continue_option(
        self,
        option: PlannerCandidate,
        state: Mapping[str, Any],
    ) -> tuple[str | None, str]:
        """Continue an option or report completion/invalidation.

        Candidate IDs are re-ranked every decision, so continuity is matched
        by immutable strategy and target rather than by the transient rank.
        """

        player = self.observe(state)
        target = Position(*option.target)
        if player == target:
            return None, "completed"
        matches = []
        for candidate in self.candidates(state):
            if candidate.strategy != option.strategy:
                continue
            if option.strategy == "ELIMINATE" and option.entity_id is not None:
                if candidate.entity_id == option.entity_id:
                    matches.append(candidate)
            elif candidate.target == option.target:
                matches.append(candidate)
        if not matches:
            return None, "invalidated"
        return matches[0].first_action, "active"

    def _emergency_candidate(
        self,
        state: Mapping[str, Any],
        player: Position | None,
        lethal: tuple[_GhostThreat, ...],
    ) -> PlannerCandidate:
        if player is None:
            raise ValueError("planner state must contain a valid Pacman position")
        legal = set(state.get("open") or state.get("legal_actions") or ())
        open_actions = [
            token for token in ("U", "D", "L", "R") if token in legal
        ]
        if not open_actions:
            raise EdwardSafetyRefusal(
                "Edward planner has no legal cardinal action; refusing to "
                "advertise an unaudited emergency option"
            )

        scored: list[tuple[int, int, int, int, str, Position]] = []
        for token in open_actions:
            target = _transition(
                self.level,
                player,
                _ACTION_BY_TOKEN[token],
                actor="pacman",
            )
            if target is None:
                continue
            ghost_clearance = _ghost_distance(self.level, lethal, target)
            clearance = ghost_clearance if ghost_clearance is not None else 10**6
            exits = _future_safe_exits(self.level, target, lethal, 1)
            margin = _route_margin(self.level, player, (token,), lethal)
            # Margin one is allowed only when the exact API-v3 pixel/path state
            # proves that no lethal ghost enters Pacman's collision box during
            # the 16 logic frames.  Formal multi-step options retain the extra
            # one-tile buffer, and margin zero remains forbidden.
            if (
                margin <= 0
                or exits < 1
                or (margin == 1 and not _one_step_pixel_safe(self.level, state, target))
            ):
                continue
            reversal = int(token == _OPPOSITE.get(self.last_action))
            scored.append((margin, clearance, exits, -reversal, token, target))
        if not scored:
            raise EdwardSafetyRefusal(
                "Edward planner has no ghost-safe action; refusing to "
                "advertise an AVOID option with a failed safety gate"
            )
        margin, _, exits, _, token, target = max(scored)
        return PlannerCandidate(
            option_id="A0",
            strategy="AVOID",
            target=(target.row, target.col),
            first_action=token,
            route_distance=1,
            commit_moves=1,
            safety_margin=margin,
            future_safe_exits=exits,
            lethal_ghost_etas=_ghost_etas(self.level, lethal, target),
        )
