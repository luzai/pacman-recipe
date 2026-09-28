from __future__ import annotations

from collections import deque
from random import Random

from .env import PacmanEnv


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = Random(seed)

    def act(self, env: PacmanEnv) -> str:
        return self.rng.choice(env.legal_actions())


class GreedyPelletAgent:
    def act(self, env: PacmanEnv) -> str:
        if not env.state.pellets:
            return "stay"

        queue = deque([(env.state.pacman, None)])
        visited = {env.state.pacman}
        while queue:
            pos, first_action = queue.popleft()
            if pos in env.state.pellets and first_action is not None:
                return first_action
            for action in env.legal_actions() if pos == env.state.pacman else ("up", "down", "left", "right"):
                nxt = env._move(pos, action)
                if nxt == pos or nxt in visited:
                    continue
                if abs(nxt[0] - env.state.ghost[0]) + abs(nxt[1] - env.state.ghost[1]) <= 1:
                    continue
                visited.add(nxt)
                queue.append((nxt, first_action or action))
        return "stay"
