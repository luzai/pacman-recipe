from __future__ import annotations

import argparse
from statistics import mean

from .baselines import GreedyPelletAgent, RandomAgent
from .env import PacmanEnv


def run_episode(agent, seed: int = 0, max_steps: int = 80) -> dict[str, object]:
    env = PacmanEnv(seed=seed, max_steps=max_steps)
    env.reset()
    reason = "running"
    while not env.state.done:
        _, _, _, info = env.step(agent.act(env))
        reason = str(info["reason"])
    return {
        "score": env.state.score,
        "steps": env.state.steps,
        "won": env.state.won,
        "reason": reason,
    }


def evaluate(agent_name: str, episodes: int, max_steps: int) -> dict[str, float]:
    agent = GreedyPelletAgent() if agent_name == "greedy" else RandomAgent(seed=1)
    results = [run_episode(agent, seed=i, max_steps=max_steps) for i in range(episodes)]
    return {
        "episodes": float(episodes),
        "win_rate": mean(1.0 if r["won"] else 0.0 for r in results),
        "mean_score": mean(float(r["score"]) for r in results),
        "mean_steps": mean(float(r["steps"]) for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("random", "greedy"), default="greedy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()
    metrics = evaluate(args.agent, args.episodes, args.max_steps)
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
