from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def load_trajectories(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    trajectories = []
    for file in files:
        with file.open() as handle:
            payload = json.load(handle)
        payload["_file"] = file.name
        trajectories.append(payload)
    return trajectories


def step_legal_action(step: dict[str, Any]) -> bool:
    if "legal_action" in step:
        return bool(step["legal_action"])
    obs = str(step.get("obs", ""))
    action = str(step.get("action", ""))
    for line in obs.splitlines():
        if line.startswith("Legal actions:"):
            legal_actions = {item.strip() for item in line.split(":", 1)[1].split(",")}
            return action in legal_actions
    return True


def summarize(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    if not trajectories:
        return {
            "episodes": 0,
            "wins": 0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "illegal_actions": 0,
            "parse_failures": 0,
            "terminal_reasons": {},
            "actions": {},
            "files": [],
        }

    terminal_reasons: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    illegal_actions = 0
    parse_failures = 0
    files = []

    for episode in trajectories:
        steps = episode.get("trajectory", [])
        files.append(str(episode.get("_file", "")))
        if steps:
            terminal_reasons[str(steps[-1].get("reason", "unknown"))] += 1
        else:
            terminal_reasons["empty"] += 1
        actions.update(str(step.get("action", "unknown")) for step in steps)
        illegal_actions += sum(1 for step in steps if not step_legal_action(step))
        parse_failures += sum(1 for step in steps if step.get("parse_failed"))

    return {
        "episodes": len(trajectories),
        "wins": sum(1 for episode in trajectories if episode.get("won")),
        "avg_reward": mean(float(episode.get("total_reward", 0.0)) for episode in trajectories),
        "avg_steps": mean(float(episode.get("steps", 0.0)) for episode in trajectories),
        "illegal_actions": illegal_actions,
        "parse_failures": parse_failures,
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "actions": dict(sorted(actions.items())),
        "files": files,
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"episodes: {summary['episodes']}",
        f"wins: {summary['wins']}",
        f"avg_reward: {summary['avg_reward']:.2f}",
        f"avg_steps: {summary['avg_steps']:.2f}",
        f"illegal_actions: {summary['illegal_actions']}",
        f"parse_failures: {summary['parse_failures']}",
        f"terminal_reasons: {summary['terminal_reasons']}",
        f"actions: {summary['actions']}",
    ]
    if summary["files"]:
        lines.append("files:")
        lines.extend(f"- {file}" for file in summary["files"])
    return "\n".join(lines)


def format_trajectory(episode: dict[str, Any], max_steps: int = 12) -> str:
    lines = [
        f"file: {episode.get('_file', '')}",
        f"id: {episode.get('id', '')}",
        f"seed: {episode.get('seed', '')}",
        f"prompt_style: {episode.get('prompt_style', 'unknown')}",
        f"layout_name: {episode.get('layout_name', 'unknown')}",
        f"system_prompt: {episode.get('system_prompt', 'not recorded in this trajectory file')}",
        f"illegal_action_penalty: {episode.get('illegal_action_penalty', 'unknown')}",
        f"total_reward: {episode.get('total_reward', 0.0)}",
        f"final_score: {episode.get('final_score', 0)}",
        f"steps: {episode.get('steps', 0)}",
        f"won: {episode.get('won', False)}",
        "",
    ]
    for step in episode.get("trajectory", [])[:max_steps]:
        obs = str(step.get("obs", "")).rstrip()
        lines.extend(
            [
                f"## step {step.get('step')}",
                obs,
                f"model_output: {step.get('model_output', '')!r}",
                f"parsed_action: {step.get('action', '')}",
                f"parse_failed: {step.get('parse_failed', False)}",
                f"exact_action: {step.get('exact_action', False)}",
                f"legal_action: {step_legal_action(step)}",
                f"reward: {step.get('reward', 0)}",
                f"score: {step.get('score', 0)}",
                f"done: {step.get('done', False)}",
                f"reason: {step.get('reason', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PacMan text rollout trajectory JSON files.")
    parser.add_argument("path", type=Path, help="Trajectory JSON file or directory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    parser.add_argument("--show-first", action="store_true", help="Print a readable dump of the first trajectory.")
    parser.add_argument("--steps", type=int, default=12, help="Maximum steps to print with --show-first.")
    args = parser.parse_args()

    trajectories = load_trajectories(args.path)
    if args.show_first:
        if not trajectories:
            raise SystemExit("no trajectories found")
        print(format_trajectory(trajectories[0], max_steps=args.steps))
        return

    summary = summarize(trajectories)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
