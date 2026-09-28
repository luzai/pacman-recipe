from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Iterable

ACTIONS = ("up", "down", "left", "right", "stay")
ROUTE_REWARD_MODES = {
    "route_prefix",
    "route_prefix_stay_penalty",
    "route_prefix_progress_penalty",
}
SAFE_PROGRESS_REWARD_MODE = "safe_progress"
MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}
OPPOSITE_ACTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

DEFAULT_LAYOUT = (
    "#########",
    "#P..#...#",
    "#.#.#.#G#",
    "#.#...#.#",
    "#...#...#",
    "#########",
)

TINY_CORRIDOR_LAYOUT = (
    "#######",
    "#P..#G#",
    "#######",
)

SMALL_DEFAULT_LAYOUT = (
    "#########",
    "#P..#...#",
    "#.#.#.#G#",
    "#.#...#.#",
    "#########",
)

MEDIUM_DEFAULT_LAYOUT = (
    "#########",
    "#P..#...#",
    "# # # #G#",
    "# #   # #",
    "#   #   #",
    "#########",
)

LAYOUTS = {
    "default": DEFAULT_LAYOUT,
    "medium_default": MEDIUM_DEFAULT_LAYOUT,
    "small_default": SMALL_DEFAULT_LAYOUT,
    "tiny_corridor": TINY_CORRIDOR_LAYOUT,
}

from .maze_suite import load_layout_registry

LAYOUTS.update(load_layout_registry())


def layout_by_name(layout_name: str) -> tuple[str, ...]:
    try:
        return LAYOUTS[layout_name]
    except KeyError as exc:
        names = ", ".join(sorted(LAYOUTS))
        raise ValueError(f"unknown layout_name {layout_name!r}; expected one of {names}") from exc


@dataclass(frozen=True)
class PacmanState:
    pacman: tuple[int, int]
    ghost: tuple[int, int]
    pellets: frozenset[tuple[int, int]]
    steps: int
    score: float
    done: bool
    won: bool


class PacmanEnv:
    """Small deterministic gridworld with text observations."""

    def __init__(
        self,
        layout: Iterable[str] = DEFAULT_LAYOUT,
        layout_name: str | None = None,
        max_steps: int = 80,
        seed: int = 0,
        illegal_action_penalty: int = 0,
        reward_mode: str = "sparse",
        route_shaping_scale: float = 1.0,
        safe_progress_alpha: float = 1.0,
    ):
        if layout_name is not None:
            layout = layout_by_name(layout_name)
        self.layout_name = layout_name or "custom"
        self.layout = tuple(layout)
        self.max_steps = max_steps
        self.illegal_action_penalty = illegal_action_penalty
        self.reward_mode = reward_mode
        self.route_shaping_scale = route_shaping_scale
        self.safe_progress_alpha = safe_progress_alpha
        self.rng = Random(seed)
        self.walls, self.start, self.ghost_start, self.initial_pellets = self._parse_layout(self.layout)
        self.state = self.reset()

    def reset(self) -> PacmanState:
        self.state = PacmanState(
            pacman=self.start,
            ghost=self.ghost_start,
            pellets=frozenset(self.initial_pellets),
            steps=0,
            score=0,
            done=False,
            won=False,
        )
        return self.state

    def step(self, action: str) -> tuple[PacmanState, float, bool, dict[str, object]]:
        if action not in ACTIONS:
            raise ValueError(f"invalid action {action!r}; expected one of {ACTIONS}")
        if self.state.done:
            return self.state, 0, True, {"reason": "already_done"}

        legal_actions = set(self.legal_actions())
        route_actions = set(self.shortest_route_actions()) if self.reward_mode in ROUTE_REWARD_MODES else set()
        safe_distance_before = self.safe_distance() if self.reward_mode == SAFE_PROGRESS_REWARD_MODE else None
        legal_action = action in legal_actions
        reward = -1
        current_distance = self._nearest_pellet_distance(self.state.pacman)
        if not legal_action:
            reward += self.illegal_action_penalty
        if self.reward_mode == "route_prefix":
            if route_actions and action in route_actions:
                reward += 2 * self.route_shaping_scale
            elif route_actions:
                reward -= 1 * self.route_shaping_scale
        elif self.reward_mode == "route_prefix_stay_penalty":
            if route_actions and action in route_actions:
                reward += 2 * self.route_shaping_scale
            elif route_actions:
                reward -= 1 * self.route_shaping_scale
            if action == "stay" and any(a != "stay" for a in legal_actions):
                reward -= 4 * self.route_shaping_scale
        elif self.reward_mode == "route_prefix_progress_penalty":
            if route_actions and action in route_actions:
                reward += 4 * self.route_shaping_scale
            elif route_actions:
                reward -= 4 * self.route_shaping_scale
            if action == "stay" and any(a != "stay" for a in legal_actions):
                reward -= 12 * self.route_shaping_scale
        pacman = self._move(self.state.pacman, action)
        pellets = set(self.state.pellets)
        ate_pellet = pacman in pellets
        next_distance = self._nearest_pellet_distance(pacman)
        if (
            self.reward_mode == "route_prefix_progress_penalty"
            and legal_action
            and not ate_pellet
            and current_distance is not None
            and next_distance is not None
        ):
            if next_distance > current_distance:
                reward -= 4 * self.route_shaping_scale
            elif next_distance == current_distance:
                reward -= 2 * self.route_shaping_scale
        if pacman in pellets:
            pellets.remove(pacman)
            reward += 10

        ghost = self.state.ghost if self.state.steps % 2 == 0 else self._ghost_move(self.state.ghost, pacman)
        done = False
        won = False
        reason = "running"

        if pacman == ghost:
            reward -= 50
            done = True
            reason = "caught"
        elif not pellets:
            reward += 100
            done = True
            won = True
            reason = "all_pellets"
        elif self.state.steps + 1 >= self.max_steps:
            reward -= 10
            done = True
            reason = "max_steps"

        self.state = PacmanState(
            pacman=pacman,
            ghost=ghost,
            pellets=frozenset(pellets),
            steps=self.state.steps + 1,
            score=self.state.score + reward,
            done=done,
            won=won,
        )
        safe_distance_after = self.safe_distance() if self.reward_mode == SAFE_PROGRESS_REWARD_MODE else None
        safe_progress_reward = 0.0
        if safe_distance_before is not None and safe_distance_after is not None:
            safe_progress_reward = self.safe_progress_alpha * (safe_distance_before - safe_distance_after)
            reward += safe_progress_reward
            self.state = PacmanState(
                pacman=self.state.pacman,
                ghost=self.state.ghost,
                pellets=self.state.pellets,
                steps=self.state.steps,
                score=self.state.score + safe_progress_reward,
                done=self.state.done,
                won=self.state.won,
            )
        return self.state, reward, done, {
            "reason": reason,
            "legal_action": legal_action,
            "route_action": action in route_actions,
            "safe_distance_before": safe_distance_before,
            "safe_distance_after": safe_distance_after,
            "safe_progress_reward": safe_progress_reward,
        }

    def legal_actions(self) -> list[str]:
        return self._legal_actions_from(self.state.pacman)

    def safe_distance(self) -> int:
        """Shortest collision-free distance to the next pellet under real ghost dynamics."""
        if self.state.done:
            return 0 if self.state.won else self.max_steps + 1
        if not self.state.pellets or self.state.pacman in self.state.pellets:
            return 0

        phase = self.state.steps % 2
        queue = deque([(self.state.pacman, self.state.ghost, phase, 0)])
        visited = {(self.state.pacman, self.state.ghost, phase)}
        while queue:
            pacman, ghost, ghost_phase, distance = queue.popleft()
            for action in self._legal_actions_from(pacman):
                next_pacman = self._move(pacman, action)
                next_ghost = ghost if ghost_phase == 0 else self._ghost_move(ghost, next_pacman)
                if next_pacman == next_ghost:
                    continue
                if next_pacman in self.state.pellets:
                    return distance + 1
                key = (next_pacman, next_ghost, 1 - ghost_phase)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((*key, distance + 1))
        return self.max_steps + 1

    def render(self) -> str:
        rows = [[" " if cell in "PG" else cell for cell in row] for row in self.layout]
        for r, c in self.walls:
            rows[r][c] = "#"
        for r, c in self.initial_pellets:
            rows[r][c] = "."
        for r, c in set(self.initial_pellets) - set(self.state.pellets):
            rows[r][c] = " "
        pr, pc = self.state.pacman
        gr, gc = self.state.ghost
        rows[pr][pc] = "P"
        rows[gr][gc] = "X" if (pr, pc) == (gr, gc) else "G"
        return "\n".join("".join(row) for row in rows)

    def observation_text(self, prompt_style: str = "default") -> str:
        legal_actions = ", ".join(self.legal_actions())
        if prompt_style == "legal":
            instruction = (
                f"Choose exactly one LEGAL action from this list only: {legal_actions}.\n"
                "Do not choose a wall move. Answer with exactly one token and no explanation."
            )
        elif prompt_style == "ghost_legal":
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            instruction = (
                f"Choose exactly one LEGAL action from this list only: {legal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                "Prefer a legal action that eats a pellet and is not unsafe. Answer with exactly one token and no explanation."
            )
        elif prompt_style == "ghost_legal_strict":
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            illegal_actions = ", ".join(action for action in ACTIONS if action not in self.legal_actions()) or "none"
            instruction = (
                f"Allowed output tokens now: {legal_actions}.\n"
                f"Forbidden output tokens now: {illegal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                "If a token is forbidden now, it is a wall move and loses reward.\n"
                "Output exactly one allowed token from the Allowed output tokens list. No explanation."
            )
        elif prompt_style == "ghost_legal_reason":
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            illegal_actions = ", ".join(action for action in ACTIONS if action not in self.legal_actions()) or "none"
            instruction = (
                f"Allowed output tokens now: {legal_actions}.\n"
                f"Forbidden output tokens now: {illegal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                "Briefly explain which allowed move makes progress or avoids danger.\n"
                "End with a separate final line exactly like `Action: <token>`."
            )
        elif prompt_style in {
            "ghost_legal_json",
            "ghost_legal_json_concise",
            "ghost_legal_json_fast_think",
            "ghost_legal_json_first",
        }:
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            illegal_actions = ", ".join(action for action in ACTIONS if action not in self.legal_actions()) or "none"
            extra = ""
            if prompt_style == "ghost_legal_json_concise":
                extra = (
                    "Do not analyze the maze in prose. Do not write markdown or thinking text.\n"
                    "Start immediately with `{` and output one compact JSON object only.\n"
                    "Use a reason with at most 12 words.\n"
                    "Do not choose stay unless it is the only allowed token.\n"
                )
            elif prompt_style == "ghost_legal_json_fast_think":
                extra = (
                    "Think briefly and quickly before the JSON. Use at most three short internal bullets.\n"
                    "Do not explore many alternatives; focus on legal moves, immediate pellets, ghost danger, and wall-routing.\n"
                )
            elif prompt_style == "ghost_legal_json_first":
                extra = (
                    "Start your answer with `{`. Do not write any preface, markdown, or thinking text before the JSON.\n"
                    "Do not choose stay unless it is the only allowed token.\n"
                )
            instruction = (
                f"Allowed output tokens now: {legal_actions}.\n"
                f"Forbidden output tokens now: {illegal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                "If no allowed move eats a pellet immediately, choose a legal safe move that makes progress through the maze toward remaining pellets.\n"
                f"{extra}"
                'Output exactly one JSON object with this schema: {"reason":"short reason","action":"one_allowed_token"}.\n'
                "Replace one_allowed_token with one token from Allowed output tokens. No markdown, no extra text."
            )
        elif prompt_style == "ghost_legal_route_json":
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            route_actions = ", ".join(self.shortest_route_actions()) or "none"
            illegal_actions = ", ".join(action for action in ACTIONS if action not in self.legal_actions()) or "none"
            instruction = (
                f"Allowed output tokens now: {legal_actions}.\n"
                f"Forbidden output tokens now: {illegal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                f"Shortest-route pellet actions now: {route_actions}.\n"
                "If Shortest-route pellet actions is not none, choose one of those actions unless it is unsafe.\n"
                'Output exactly one JSON object with this schema: {"reason":"short reason","action":"one_allowed_token"}.\n'
                "Replace one_allowed_token with one token from Allowed output tokens. No markdown, no extra text."
            )
        elif prompt_style == "ghost_legal_distance_json":
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            distances = ", ".join(
                f"{action}={distance if distance is not None else 'unreachable'}"
                for action, distance in self.pellet_distances_after_actions().items()
            )
            illegal_actions = ", ".join(action for action in ACTIONS if action not in self.legal_actions()) or "none"
            instruction = (
                f"Allowed output tokens now: {legal_actions}.\n"
                f"Forbidden output tokens now: {illegal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                f"Pellet distance after action: {distances}.\n"
                "Prefer an allowed safe action with the smallest finite pellet distance; distance 0 means the action eats a pellet now.\n"
                'Output exactly one JSON object with this schema: {"reason":"short reason","action":"one_allowed_token"}.\n'
                "Replace one_allowed_token with one token from Allowed output tokens. No markdown, no extra text."
            )
        elif prompt_style in {"teacher_hint", "teacher_hint_pure"}:
            unsafe_actions = ", ".join(self.unsafe_actions()) or "none"
            teacher_action = self.teacher_action()
            instruction = (
                f"Choose exactly one LEGAL action from this list only: {legal_actions}.\n"
                f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
                f"Teacher action hint: {teacher_action}.\n"
                "For this overfit warm-start gate, copy the Teacher action hint exactly. Answer with exactly one token and no explanation."
            )
        else:
            instruction = "Choose exactly one action from: up, down, left, right, stay."
        return (
            f"Step {self.state.steps}/{self.max_steps}\n"
            f"Score: {self.state.score}\n"
            f"Pellets left: {len(self.state.pellets)}\n"
            f"Legal actions: {legal_actions}\n"
            f"Grid:\n{self.render()}\n"
            f"{instruction}"
        )

    def unsafe_actions(self) -> list[str]:
        unsafe = []
        for action in self.legal_actions():
            pacman = self._move(self.state.pacman, action)
            ghost = self.state.ghost if self.state.steps % 2 == 0 else self._ghost_move(self.state.ghost, pacman)
            if pacman == ghost:
                unsafe.append(action)
        return unsafe

    def teacher_action(self) -> str:
        if not self.state.pellets:
            return "stay"

        queue = deque([(self.state.pacman, None)])
        visited = {self.state.pacman}
        while queue:
            pos, first_action = queue.popleft()
            if pos in self.state.pellets and first_action is not None:
                return first_action
            actions = self.legal_actions() if pos == self.state.pacman else ("up", "down", "left", "right")
            for action in actions:
                nxt = self._move(pos, action)
                if nxt == pos or nxt in visited:
                    continue
                if abs(nxt[0] - self.state.ghost[0]) + abs(nxt[1] - self.state.ghost[1]) <= 1:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_action or action))
        return "stay"

    def shortest_route_actions(self) -> list[str]:
        if not self.state.pellets:
            return ["stay"] if "stay" in self.legal_actions() else []

        queue = deque([(self.state.pacman, None, 0)])
        visited = {self.state.pacman}
        best_distance = None
        best_actions: set[str] = set()
        legal_actions = self.legal_actions()
        while queue:
            pos, first_action, distance = queue.popleft()
            if best_distance is not None and distance > best_distance:
                break
            if pos in self.state.pellets and first_action is not None:
                best_distance = distance
                best_actions.add(first_action)
                continue
            actions = legal_actions if pos == self.state.pacman else ("up", "down", "left", "right")
            for action in actions:
                if action == "stay":
                    continue
                nxt = self._move(pos, action)
                if nxt == pos or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_action or action, distance + 1))
        return [action for action in legal_actions if action in best_actions]

    def pellet_distances_after_actions(self) -> dict[str, int | None]:
        distances: dict[str, int | None] = {}
        for action in self.legal_actions():
            if action == "stay":
                start = self.state.pacman
            else:
                start = self._move(self.state.pacman, action)
            distances[action] = self._nearest_pellet_distance(start)
        return distances

    def _nearest_pellet_distance(self, start: tuple[int, int]) -> int | None:
        if start in self.state.pellets:
            return 0
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            pos, distance = queue.popleft()
            for action in ("up", "down", "left", "right"):
                nxt = self._move(pos, action)
                if nxt == pos or nxt in visited:
                    continue
                if nxt in self.state.pellets:
                    return distance + 1
                visited.add(nxt)
                queue.append((nxt, distance + 1))
        return None

    def _move(self, pos: tuple[int, int], action: str) -> tuple[int, int]:
        dr, dc = MOVE_DELTAS[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        return pos if nxt in self.walls else nxt

    def _legal_actions_from(self, pos: tuple[int, int]) -> list[str]:
        return [action for action in ACTIONS if self._move(pos, action) != pos or action == "stay"]

    def _ghost_move(self, ghost: tuple[int, int], pacman: tuple[int, int]) -> tuple[int, int]:
        if abs(ghost[0] - pacman[0]) + abs(ghost[1] - pacman[1]) > 3:
            return ghost
        candidates = []
        for action in ("up", "down", "left", "right"):
            nxt = self._move(ghost, action)
            dist = abs(nxt[0] - pacman[0]) + abs(nxt[1] - pacman[1])
            candidates.append((dist, action, nxt))
        candidates.sort()
        return candidates[0][2]

    @staticmethod
    def _parse_layout(layout: tuple[str, ...]):
        walls: set[tuple[int, int]] = set()
        pellets: set[tuple[int, int]] = set()
        start = None
        ghost = None
        for r, row in enumerate(layout):
            for c, cell in enumerate(row):
                if cell == "#":
                    walls.add((r, c))
                elif cell == ".":
                    pellets.add((r, c))
                elif cell == "P":
                    start = (r, c)
                elif cell == "G":
                    ghost = (r, c)
        if start is None or ghost is None:
            raise ValueError("layout must include P and G")
        return walls, start, ghost, pellets
