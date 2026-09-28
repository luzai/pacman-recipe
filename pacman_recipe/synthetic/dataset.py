from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import GreedyPelletAgent
from .env import LAYOUTS, PacmanEnv
from .maze_suite import SPLIT_SIZES, layout_hash, split_layout_names, topology_hash

SYSTEM_PROMPT = """You are a PacMan game agent.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each turn you choose exactly one action: up, down, left, right, or stay.
- A wall blocks movement. If you choose a wall move, PacMan stays in place and wastes the turn.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Strategy:
- Prefer legal moves that eat pellets.
- Avoid moves that immediately collide with the ghost or move closer to danger without a reason.
- Do not repeat a wall move after it failed.

Output format:
- Answer with exactly one action token: up, down, left, right, or stay.
- Do not explain your reasoning."""

LEGAL_SYSTEM_PROMPT = """You are a PacMan game agent.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- The current observation includes a Legal actions list. You must choose exactly one action from that list.
- A wall blocks movement. Choosing a wall move wastes the turn and receives an extra penalty.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Strategy:
- Prefer legal moves that eat pellets.
- Avoid moves that immediately collide with the ghost or move closer to danger without a reason.
- Do not repeat a wall move after it failed.

Output format:
- Answer with exactly one legal action token from the current Legal actions list.
- Do not explain your reasoning."""

GHOST_LEGAL_SYSTEM_PROMPT = """You are a PacMan game agent.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- The current observation includes a Legal actions list. You must choose exactly one action from that list.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- A wall blocks movement. Choosing a wall move wastes the turn and receives an extra penalty.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Strategy:
- Prefer legal moves that eat pellets.
- Avoid unsafe ghost actions.
- If no pellet can be eaten immediately, move toward the nearest reachable pellet while staying legal and safe.
- Do not stay unless every movement option is worse.
- Do not repeat a wall move after it failed.

Output format:
- Answer with exactly one legal action token from the current Legal actions list.
- Do not explain your reasoning."""

GHOST_LEGAL_STRICT_SYSTEM_PROMPT = """You are a PacMan game agent.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. You must choose exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Strategy:
- Prefer allowed moves that eat pellets.
- Avoid unsafe ghost actions.
- If no pellet can be eaten immediately, move toward the nearest reachable pellet while staying allowed and safe.
- Do not stay unless every movement option is worse.
- Never output a forbidden token.

Output format:
- Answer with exactly one token from Allowed output tokens.
- Do not explain your reasoning."""

GHOST_LEGAL_REASON_SYSTEM_PROMPT = """You are a PacMan game agent in a debug run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your final action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Debug format:
- First write a short reason in one or two sentences.
- End with a separate line exactly like `Action: <token>`.
- The `<token>` must be one token from Allowed output tokens."""

GHOST_LEGAL_JSON_SYSTEM_PROMPT = """You are a PacMan game agent in a debug run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.
- If no allowed move eats a pellet immediately, choose a legal safe move that opens the shortest route to remaining pellets.
- Avoid reversing into an empty corridor unless that route is needed to reach remaining pellets.

Output format:
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence.
- The `action` value must be one token from Allowed output tokens."""

GHOST_LEGAL_JSON_CONCISE_SYSTEM_PROMPT = """You are a PacMan game agent in a concise JSON-only debug run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Decision rules:
- Do not analyze the maze in prose.
- Decide quickly from allowed actions, immediate pellets, ghost danger, and simple progress.
- Prefer an allowed move that eats a pellet immediately.
- If no allowed move eats a pellet immediately, choose an allowed safe move that makes progress through the maze.
- Do not choose `stay` unless it is the only allowed token.

Output format:
- The first character of your answer must be `{`.
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence with at most 12 words.
- The `action` value must be one token from Allowed output tokens."""

GHOST_LEGAL_JSON_FAST_THINK_SYSTEM_PROMPT = """You are a PacMan game agent in a concise thinking debug run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Concise thinking rule:
- Think briefly and quickly.
- Use at most three short internal reasoning bullets.
- Do not explore many alternatives.
- Focus only on legal moves, immediate pellets, ghost danger, and whether a legal move opens a route around a wall.

Output format:
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence.
- The `action` value must be one token from Allowed output tokens."""

GHOST_LEGAL_JSON_FIRST_SYSTEM_PROMPT = """You are a PacMan game agent in a JSON-first control run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Decision rules:
- Prefer an allowed move that eats a pellet immediately.
- Avoid unsafe ghost actions.
- If no allowed move eats a pellet immediately, choose an allowed safe move that makes progress through the maze toward remaining pellets.
- Do not choose `stay` unless it is the only allowed token.

Output format:
- The first character of your answer must be `{`.
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence.
- The `action` value must be one token from Allowed output tokens."""

GHOST_LEGAL_ROUTE_JSON_SYSTEM_PROMPT = """You are a PacMan game agent in a no-training route-feature run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- The observation includes Shortest-route pellet actions. If this list is not none, choose one action from it unless it is unsafe.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Output format:
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence.
- The `action` value must be one token from Allowed output tokens."""

GHOST_LEGAL_DISTANCE_JSON_SYSTEM_PROMPT = """You are a PacMan game agent in a no-training distance-feature run.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- Each observation has an Allowed output tokens list. Your action must be exactly one token from that list.
- Tokens outside the Allowed output tokens list are forbidden for the current turn and count as wall or illegal moves.
- The observation may include Unsafe immediate ghost actions. Avoid these unless every useful move is unsafe.
- The observation includes Pellet distance after action values. Prefer allowed safe actions with smaller finite distance.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Output format:
- Output exactly one JSON object and no other text.
- The JSON object must have keys `reason` and `action`.
- The `reason` value must be one short sentence.
- The `action` value must be one token from Allowed output tokens."""

TEACHER_HINT_SYSTEM_PROMPT = """You are a PacMan game agent in an overfit warm-start test.

Game rules:
- The grid is a maze. `#` is a wall, `P` is PacMan, `G` is the ghost, `.` is an uneaten pellet, and blank spaces are empty cells.
- The current observation includes a Legal actions list. You must choose exactly one action from that list.
- The observation includes a Teacher action hint. For this overfit gate, copy that hint exactly.
- The observation may include Unsafe immediate ghost actions. Avoid these unless the Teacher action hint says otherwise.
- A wall blocks movement. Choosing a wall move wastes the turn and receives an extra penalty.
- Eating a pellet is good. Moving without eating is usually bad because time is limited.
- Touching the ghost is very bad and ends the episode.
- Clearing all pellets wins the episode.

Output format:
- Answer with exactly the Teacher action hint token.
- Do not explain your reasoning."""


def system_prompt(prompt_style: str = "default") -> str:
    if prompt_style == "legal":
        return LEGAL_SYSTEM_PROMPT
    if prompt_style == "ghost_legal":
        return GHOST_LEGAL_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_strict":
        return GHOST_LEGAL_STRICT_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_reason":
        return GHOST_LEGAL_REASON_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_json":
        return GHOST_LEGAL_JSON_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_json_concise":
        return GHOST_LEGAL_JSON_CONCISE_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_json_fast_think":
        return GHOST_LEGAL_JSON_FAST_THINK_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_json_first":
        return GHOST_LEGAL_JSON_FIRST_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_route_json":
        return GHOST_LEGAL_ROUTE_JSON_SYSTEM_PROMPT
    if prompt_style == "ghost_legal_distance_json":
        return GHOST_LEGAL_DISTANCE_JSON_SYSTEM_PROMPT
    if prompt_style in {"teacher_hint", "teacher_hint_pure"}:
        return TEACHER_HINT_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def generate_examples(episodes: int, max_steps: int):
    agent = GreedyPelletAgent()
    for seed in range(episodes):
        env = PacmanEnv(seed=seed, max_steps=max_steps)
        env.reset()
        while not env.state.done:
            observation = env.observation_text()
            action = agent.act(env)
            yield {
                "id": f"pacman-{seed}-{env.state.steps}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation},
                ],
                "answer": action,
            }
            env.step(action)


def generate_episode_specs(
    episodes: int,
    max_steps: int,
    prompt_style: str = "default",
    illegal_action_penalty: int = 0,
    layout_name: str = "default",
    reward_mode: str = "sparse",
):
    for seed in range(episodes):
        env = PacmanEnv(
            seed=seed,
            max_steps=max_steps,
            illegal_action_penalty=illegal_action_penalty,
            reward_mode=reward_mode,
            layout_name=layout_name,
        )
        env.reset()
        yield {
            "id": f"pacman-episode-{seed}",
            "seed": seed,
            "max_steps": max_steps,
            "prompt_style": prompt_style,
            "illegal_action_penalty": illegal_action_penalty,
            "reward_mode": reward_mode,
            "layout_name": layout_name,
            "messages": [
                {"role": "system", "content": system_prompt(prompt_style)},
                {"role": "user", "content": env.observation_text(prompt_style=prompt_style)},
            ],
        }


def generate_multi_maze_episode_specs(
    split: str,
    episodes_per_layout: int,
    max_steps: int,
    prompt_style: str = "default",
    illegal_action_penalty: int = 0,
    reward_mode: str = "sparse",
):
    if episodes_per_layout <= 0:
        raise ValueError("episodes_per_layout must be positive")
    for layout_index, layout_name in enumerate(split_layout_names(split)):
        layout = LAYOUTS[layout_name]
        digest = layout_hash(layout)
        topology_digest = topology_hash(layout)
        for rollout_index in range(episodes_per_layout):
            seed = layout_index * episodes_per_layout + rollout_index
            env = PacmanEnv(
                seed=seed,
                max_steps=max_steps,
                illegal_action_penalty=illegal_action_penalty,
                reward_mode=reward_mode,
                layout_name=layout_name,
            )
            env.reset()
            yield {
                "id": f"{layout_name}-rollout-{rollout_index:03d}",
                "seed": seed,
                "max_steps": max_steps,
                "prompt_style": prompt_style,
                "illegal_action_penalty": illegal_action_penalty,
                "reward_mode": reward_mode,
                "layout_name": layout_name,
                "layout_hash": digest,
                "topology_hash": topology_digest,
                "maze_suite": "multi_maze_v1",
                "maze_split": split,
                "rollout_index": rollout_index,
                "messages": [
                    {"role": "system", "content": system_prompt(prompt_style)},
                    {"role": "user", "content": env.observation_text(prompt_style=prompt_style)},
                ],
            }


def write_jsonl(
    path: Path,
    episodes: int,
    max_steps: int,
    mode: str = "step",
    prompt_style: str = "default",
    illegal_action_penalty: int = 0,
    layout_name: str = "default",
    reward_mode: str = "sparse",
    maze_split: str | None = None,
    episodes_per_layout: int = 1,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if maze_split is not None:
        if mode != "episode":
            raise ValueError("maze_split is supported only in episode mode")
        iterator = generate_multi_maze_episode_specs(
            split=maze_split,
            episodes_per_layout=episodes_per_layout,
            max_steps=max_steps,
            prompt_style=prompt_style,
            illegal_action_penalty=illegal_action_penalty,
            reward_mode=reward_mode,
        )
    elif mode == "episode":
        iterator = generate_episode_specs(
            episodes=episodes,
            max_steps=max_steps,
            prompt_style=prompt_style,
            illegal_action_penalty=illegal_action_penalty,
            layout_name=layout_name,
            reward_mode=reward_mode,
        )
    else:
        iterator = generate_examples(episodes=episodes, max_steps=max_steps)
    with path.open("w", encoding="utf-8") as fh:
        for item in iterator:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--mode", choices=("step", "episode"), default="step")
    parser.add_argument(
        "--prompt-style",
        choices=(
            "default",
            "legal",
            "ghost_legal",
            "ghost_legal_strict",
            "ghost_legal_reason",
            "ghost_legal_json",
            "ghost_legal_json_concise",
            "ghost_legal_json_fast_think",
            "ghost_legal_json_first",
            "ghost_legal_route_json",
            "ghost_legal_distance_json",
            "teacher_hint",
            "teacher_hint_pure",
        ),
        default="default",
    )
    parser.add_argument("--illegal-action-penalty", type=int, default=0)
    parser.add_argument("--layout-name", choices=tuple(sorted(LAYOUTS)), default="default")
    parser.add_argument("--maze-split", choices=tuple(SPLIT_SIZES))
    parser.add_argument("--episodes-per-layout", type=int, default=1)
    parser.add_argument("--reward-mode", choices=("sparse", "route_prefix"), default="sparse")
    args = parser.parse_args()
    count = write_jsonl(
        args.output,
        args.episodes,
        args.max_steps,
        args.mode,
        prompt_style=args.prompt_style,
        illegal_action_penalty=args.illegal_action_penalty,
        layout_name=args.layout_name,
        reward_mode=args.reward_mode,
        maze_split=args.maze_split,
        episodes_per_layout=args.episodes_per_layout,
    )
    print(f"wrote {count} examples to {args.output}")


if __name__ == "__main__":
    main()
