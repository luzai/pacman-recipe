"""Produce an on-disk checkpoint and deterministic fresh-worker replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig
from maapacman.actions import coerce_action
from maapacman.env._saved_state import checksum
from maapacman.planner import EdwardPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=32)
    parser.add_argument("--continuation-steps", type=int, default=64)
    args = parser.parse_args()
    if args.prefix_steps < 0 or args.continuation_steps <= 0:
        parser.error("prefix must be nonnegative and continuation positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = PygamePacmanEnvConfig(episode_life_mode="original_three_lives")

    def record(env, result):
        frame, reward, terminated, truncated, info = result
        info = {k: v for k, v in info.items() if k != "worker_runtime_id"}
        return {
            "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "reward": reward, "terminated": terminated, "truncated": truncated,
            "transition_sha256": checksum(info),
            "complete_state_sha256": env.save_state()["sha256"],
            "step": info["step"], "logic_frame": info["logic_frame"],
            "pellets_remaining": info["pellets_remaining"],
            "death_count": info["death_count"],
        }

    with PygamePacmanEnv(config) as env:
        env.reset(seed=args.seed)
        planner = EdwardPlanner()
        for _ in range(args.prefix_steps):
            result = env.step(planner.decide(env.snapshot()).action)
            if result[2] or result[3]:
                raise RuntimeError("prefix ended before checkpoint; use fewer prefix steps")
        start = time.perf_counter()
        saved = env.save_state()
        save_seconds = time.perf_counter() - start
        state_path = args.output_dir / "state.json"
        state_path.write_text(json.dumps(saved, separators=(",", ":")), encoding="utf-8")
        observation_hash = hashlib.sha256(env.render().tobytes()).hexdigest()
        actions, expected = [], []
        for _ in range(args.continuation_steps):
            action = planner.decide(env.snapshot()).action
            actions.append(coerce_action(action).value)
            result = env.step(action)
            expected.append(record(env, result))
            if result[2] or result[3]:
                break
        provenance = env.provenance

    # Read after terminating the original worker to prove process independence.
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    with PygamePacmanEnv(config) as env:
        start = time.perf_counter()
        obs, _ = env.reset(saved_state=loaded)
        restore_seconds = time.perf_counter() - start
        if hashlib.sha256(obs.tobytes()).hexdigest() != observation_hash:
            raise AssertionError("restored observation differs")
        if env.save_state() != loaded:
            raise AssertionError("restored complete state differs")
        actual = [record(env, env.step(action)) for action in actions]
        if actual != expected:
            raise AssertionError("continuation differs")

    summary = {
        "passed": True, "seed": args.seed, "prefix_steps": args.prefix_steps,
        "matched_continuation_steps": len(actions),
        "observation_exact": True, "complete_state_exact": True,
        "transition_reward_terminal_exact": True,
        "checkpoint_bytes": state_path.stat().st_size,
        "save_seconds": save_seconds,
        "fresh_worker_restore_seconds": restore_seconds,
        "runtime": saved["payload"]["worker"]["runtime"],
        "provenance": provenance, "checkpoint_sha256": saved["sha256"],
        "actions": actions, "steps": actual,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("actions", "steps", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
