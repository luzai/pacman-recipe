"""Bounded launch from an explicit argv spec; resume retains the original budget."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from .checkpoints import atomic_json, restore
from .controller import TrainingPolicy
from .sampling import FormalSampler
from .preflight import verify_files, verify_runtime_receipts
from .reward_filter import configured_policy


HOOKS = {
    "--custom-rollout-training-completed-hook-path": "optimizer_completed_hook",
    "--custom-rollout-checkpoint-saved-hook-path": "checkpoint_saved_hook",
    "--custom-rollout-checkpoint-load-hook-path": "checkpoint_load_hook",
}


def configured_command(command, budget):
    command = list(command)
    for flag in [
        "--use-kl-loss",
        "--use-kl",
        "--normalize-advantages",
        "--no-load-optim",
        "--no-load-rng",
        "--override-opt-param-scheduler",
    ]:
        if flag in command:
            raise ValueError(f"unsupported flag: {flag}")
    if "--kl-coef" in command and float(command[command.index("--kl-coef") + 1]) != 0:
        raise ValueError("reference KL is disabled")
    if int(command[command.index("--num-rollout") + 1]) != budget:
        raise ValueError("optimizer budget mismatch")
    settings = {
        **{flag: f"slime_pacman.formal_rollout.{name}" for flag, name in HOOKS.items()},
        "--rollout-function-path": "slime_pacman.formal_rollout.generate_rollout",
        "--eval-function-path": "slime_pacman.formal_rollout.generate_rollout",
    }
    for flag, value in settings.items():
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command.extend([flag, value])
    for flag in [
        "--force-sync-save",
        "--eval-before-train-on-resume",
        "--use-checkpoint-opt-param-scheduler",
    ]:
        if flag not in command:
            command.append(flag)
    if "--sglang-disable-overlap-schedule" in command:
        raise ValueError("map training requires overlap enabled")
    return command


def remaining_budget(cold, wall_seconds, now):
    if cold["wall_seconds"] != wall_seconds:
        raise ValueError("cannot change original wall-clock budget")
    remaining = cold["time_unix"] + wall_seconds - now
    if remaining <= 0:
        raise ValueError("original wall-clock budget exhausted")
    return math.ceil(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--restore", action="store_true")
    options = parser.parse_args()
    spec = json.loads(Path(options.spec).read_text())
    if not spec["opening_only"] or any(
        spec[key] for key in ["dynamic_bank", "failure_bank", "backplay"]
    ):
        raise ValueError("only initial-stage training is supported")
    verify_files(spec["source_root"], spec["source_manifest"])
    verify_runtime_receipts(spec["runtime_receipts"], spec["image_id"])
    env = dict(os.environ, **spec["environment"])
    config = json.loads(Path(env["PACMAN_FORMAL_CONFIG"]).read_text())
    command = configured_command(spec["argv"], config["optimizer_budget"])
    run = Path(env["PACMAN_RUN_DIR"])
    run.mkdir(parents=True, exist_ok=True)
    cold_path = run / "LAUNCH_COLD.json"
    if options.restore:
        sampler = FormalSampler(
            config["catalog"],
            TrainingPolicy(**config["policy"]),
            max_tier=config["max_tier"],
            min_tier=config.get("min_tier", 1),
            seed=config["seed"],
            environment_seed_mode=config.get("environment_seed_mode", "group_randomized"),
            reward_group_filter=configured_policy(config),
        )
        checkpoint = Path(command[command.index("--load") + 1])
        receipt = restore(checkpoint, identity=sampler.controller.identity)
        next_update = receipt["update"] + 1
        if next_update >= config["optimizer_budget"]:
            raise ValueError("optimizer budget already completed")
        if "--start-rollout-id" in command:
            command[command.index("--start-rollout-id") + 1] = str(next_update)
        else:
            command.extend(["--start-rollout-id", str(next_update)])
        cold = json.loads(cold_path.read_text())
        if cold["spec"] != spec:
            raise ValueError("immutable launch spec changed during recovery")
    else:
        if cold_path.exists():
            raise ValueError("existing run requires explicit restore")
        cold = dict(time_unix=time.time(), wall_seconds=spec["wall_seconds"], spec=spec)
        atomic_json(cold_path, cold)
    remaining = remaining_budget(cold, spec["wall_seconds"], time.time())
    token = uuid.uuid4().hex
    atomic_json(
        run / f"LAUNCH_{token}.json", dict(argv=command, restored=options.restore)
    )
    result = subprocess.run(
        ["timeout", "--signal=TERM", "--kill-after=60", str(remaining), *command],
        env=env,
    )
    atomic_json(
        run / f"EXIT_{token}.json",
        dict(exit_code=result.returncode, time_unix=time.time()),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
