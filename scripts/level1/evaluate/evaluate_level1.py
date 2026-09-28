"""Recipe-aligned full-episode evaluation; a successful run is not a guaranteed win."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from pacman_env.paths import pacman_python_root
from pacman_recipe.level1.level1_dataset import make_episode_row
from pacman_recipe.level1.recipe import load_recipe_document, recipe_contract_metadata
from pacman_recipe.level1.trajectories import summarize_episodes

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs/level1/train/curriculum2.yaml"
)
EVALUATION_VERSION = "pacman-release-evaluation-v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def exception_chain(error: BaseException) -> list[dict[str, Any]]:
    """The full __cause__/__context__ chain behind a caught exception.

    str() on a wrapped exception throws the diagnosis away: every
    openai.APIConnectionError stringifies to the literal "Connection error."
    whatever the underlying fault was, so a recorded error field alone cannot
    tell a connect timeout from a server disconnect from a closed socket.
    Three full evaluation runs were unexplainable for exactly that reason.

    __context__ is followed as well as __cause__ because `raise X from None`
    suppresses the latter while keeping the former.
    """
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 12:
        seen.add(id(current))
        chain.append(
            {
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "message": str(current)[:400],
                "errno": getattr(current, "errno", None),
            }
        )
        current = (
            current.__cause__ if current.__cause__ is not None else current.__context__
        )
    return chain


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model", required=True, help="Exact ID advertised by /v1/models."
    )
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of consecutive environment seeds.",
    )
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--samples-per-seed", type=int, default=3)
    parser.add_argument("--generation-seed-base", type=int, default=0)
    parser.add_argument(
        "--purpose", choices=("heldout", "validation", "diagnostic"), default="heldout"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Explicit override, e.g. 0 for separate greedy evaluation.",
    )
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-steps", type=int, help="Must match --config.")
    parser.add_argument(
        "--ghost-mode", choices=("disabled", "normal"), help="Must match --config."
    )
    parser.add_argument(
        "--max-completion-tokens", type=int, help="Must match --config."
    )
    parser.add_argument("--prompt-style", help="Must match --config.")
    parser.add_argument(
        "--open-action-mask",
        action="store_true",
        default=None,
        help="Compatibility check; recipe controls harness.",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--wall-clock-limit-seconds", type=float)
    parser.add_argument("--stuck-no-progress-steps", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    # The OpenAI SDK defaults to a 5 second connect timeout, which a local
    # server under concurrent episode load misses often enough to abort a
    # quarter of the run. Every abort is scored as an infrastructure error and
    # silently biases the result, because long episodes offer more chances to
    # hit it and winning episodes are the long ones.
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Per failed episode; all attempts remain in the report.",
    )
    parser.add_argument(
        "--pacman-python-root",
        type=Path,
        default=pacman_python_root(),
    )
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def evaluation_settings(args):
    raw = load_recipe_document(args.config)
    contract = recipe_contract_metadata(raw)
    generation = raw.get("eval_gconfig") or raw["gconfig"]
    for name, expected in {
        "max_steps": raw["environment"]["max_steps"],
        "ghost_mode": raw["environment"]["ghost_mode"],
        "max_completion_tokens": generation["max_new_tokens"],
        "prompt_style": raw["image_prompt_style"],
        "open_action_mask": raw["open_action_mask"],
    }.items():
        if getattr(args, name, None) is not None and getattr(args, name) != expected:
            raise ValueError(f"--{name.replace('_', '-')} conflicts with --config")
    if (
        raw.get("enable_thinking") is not False
        or int(generation["max_new_tokens"]) != 1
    ):
        raise ValueError(
            "release evaluation requires no thinking and exactly one output token"
        )
    if raw["action_protocol"] not in {
        "direct-open-action-token-v1",
        "edward-option-code-v1",
    }:
        raise ValueError("unsupported release action protocol")
    edward = raw["action_protocol"] == "edward-option-code-v1"
    if (
        bool(raw["edward_options"]) != edward
        or bool(raw["open_action_mask"]) == edward
        or bool(raw["action_token_choice"]) == edward
    ):
        raise ValueError("recipe action protocol and harness switches disagree")
    temperature = (
        generation["temperature"] if args.temperature is None else args.temperature
    )
    top_p = generation["top_p"] if args.top_p is None else args.top_p
    if not math.isfinite(float(temperature)) or not 0 <= float(temperature) or not 0 < float(top_p) <= 1:
        raise ValueError("invalid evaluation decoding")
    return raw, {
        **contract["reward"]["coefficients"],
        "recipe_contract": contract,
        "reward_recipe_version": contract["reward"]["formula_version"],
        "reward_objective_contract": "evaluation_only_v1",
        "contract_violation_return": raw.get("contract_violation_return", -1.0),
        "ghost_mode": contract["ghost_mode"],
        "episode_life_mode": contract["episode_life_mode"],
        "environment_max_steps": raw["environment"]["max_steps"],
        "image_prompt_style": raw["image_prompt_style"],
        "prompt_version": raw["prompt_version"],
        **contract["harness"],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_completion_tokens": int(generation["max_new_tokens"]),
        "enable_thinking": False,
    }


async def verify_served_model(args):
    from openai import AsyncOpenAI

    async with AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, max_retries=args.retries
    ) as client:
        models = await client.models.list()
    ids = [item.id for item in models.data]
    if args.model not in ids:
        raise ValueError(
            f"requested model {args.model!r} is not advertised by /v1/models: {ids}"
        )
    return {
        "requested_model_id": args.model,
        "advertised_model_ids": ids,
        "identity_limit": "API ID checked; file-to-server binding also needs the launcher log.",
    }


def checkpoint_identity(args):
    manifest = args.checkpoint_manifest
    if manifest is None and args.checkpoint_path is not None:
        candidate = args.checkpoint_path / "merge_manifest.json"
        manifest = candidate if candidate.is_file() else None
    result = {
        "path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "manifest_path": str(manifest) if manifest else None,
        "manifest_sha256": None,
        "weight_identity_verified": False,
    }
    if manifest is not None:
        contents = manifest.read_bytes()
        json.loads(contents)
        result["manifest_sha256"] = hashlib.sha256(contents).hexdigest()
    # A manifest hash does not itself prove the server loaded these files.
    return result


def is_verified_win(episode):
    return bool(
        episode.get("won")
        and episode.get("terminal_reason") == "all_normal_pellets"
        and episode.get("normal_pellets_remaining") == 0
        and any(
            event.get("event_type", event.get("type")) == "level_cleared"
            for step in episode.get("trajectory", [])
            for event in step.get("logic_frame_events", [])
        )
    )


def workflow_class():
    # Inspecting a recipe does not need AReaL's Linux-only runtime imports.
    from pacman_recipe.level1.workflow import PacmanImageOnlyWorkflow

    return PacmanImageOnlyWorkflow


async def evaluate(args):
    if (
        min(args.episodes, args.samples_per_seed, args.concurrency) < 1
        or args.retries < 0
    ):
        raise ValueError(
            "episodes, samples-per-seed and concurrency must be positive; retries nonnegative"
        )
    raw, kwargs = evaluation_settings(args)
    if args.pacman_python_root is not None:
        for variable in ("PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT"):
            os.environ[variable] = str(args.pacman_python_root.resolve())
    seeds = list(range(args.seed, args.seed + args.episodes))
    data = kwargs["recipe_contract"]["data"]
    if args.purpose == "heldout" and set(seeds) & set(
        data["train_seeds"] + data["validation_seeds"]
    ):
        raise ValueError("heldout evaluation seeds overlap training/validation")
    if args.purpose == "validation" and not set(seeds) <= set(data["validation_seeds"]):
        raise ValueError("checkpoint selection must use the recipe validation seeds")
    trajectory_dir = (
        args.trajectory_dir or args.output.parent / f"{args.output.stem}_trajectories"
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    attempts_path = args.output.with_suffix(".attempts.jsonl")
    if any(path.exists() for path in (args.output, manifest_path, attempts_path)):
        raise FileExistsError(
            "evaluation output already exists; preserve attempts and choose a new output"
        )
    row_template = make_episode_row(
        1,
        split="test",
        seed=seeds[0],
        max_steps=kwargs["environment_max_steps"],
        ghost_mode=kwargs["ghost_mode"],
        action_protocol=kwargs["action_protocol"],
    )
    protocol = {
        "version": EVALUATION_VERSION,
        "purpose": args.purpose,
        "environment": {
            key: value for key, value in row_template["env"].items() if key != "seed"
        },
        "source_revisions": row_template["source_revisions"],
        "harness": kwargs["recipe_contract"]["harness"],
        "prompt": kwargs["recipe_contract"]["prompt"],
        "reward": {
            "formula_version": kwargs["reward_recipe_version"],
            "coefficients": kwargs["recipe_contract"]["reward"]["coefficients"],
            "contract_violation_return": kwargs["contract_violation_return"],
        },
        "decoding": {
            key: kwargs[key]
            for key in (
                "temperature",
                "top_p",
                "max_completion_tokens",
                "enable_thinking",
            )
        },
        "seeds": seeds,
        "generation_seeds": list(
            range(
                args.generation_seed_base,
                args.generation_seed_base + args.samples_per_seed,
            )
        ),
        "samples_per_seed": args.samples_per_seed,
        "retries_per_trial": args.retries,
        "safety_limits": {
            "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
            "stuck_no_progress_steps": args.stuck_no_progress_steps,
        },
        "sampling_reproducibility": "generation seeds are sent to the backend; bitwise reproducibility is not guaranteed",
    }
    identity = checkpoint_identity(args)
    plan = [
        {
            "trial_id": f"seed{seed}-sample{sample}",
            "seed": seed,
            "generation_seed": args.generation_seed_base + sample,
        }
        for seed in seeds
        for sample in range(args.samples_per_seed)
    ]
    manifest = {
        "evaluation_contract": protocol,
        "evaluation_contract_sha256": canonical_sha256(protocol),
        "recipe_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": args.model,
        "checkpoint": identity,
        "planned_trials": plan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    service = await verify_served_model(args)
    policy_class = workflow_class()
    semaphore = asyncio.Semaphore(args.concurrency)
    successful, attempts = [], []

    # One client for the whole run, with timeouts sized for a busy local server
    # rather than the SDK's 5 second connect default. The workflow builds its
    # AsyncOpenAI with max_retries=0, so a single missed connection would
    # otherwise discard a whole episode.
    #
    # Connection reuse is disabled deliberately. httpcore's has_expired()
    # (_async/http11.py) consults only the clock and never checks whether a
    # connection is already assigned to a pending request, so the pool's reap
    # loop (_async/connection_pool.py) closes a connection out from under the
    # task holding it whenever the event loop stalls for about keepalive_expiry
    # seconds -- which this evaluator's pygame work does routinely. That
    # surfaces as APIConnectionError <- ReadError <- ClosedResourceError.
    # Tuning keepalive_expiry cannot fix it: too long reuses sockets the server
    # already closed, too short reaps assigned ones. Measured on H100_1_2:
    # 40/960 requests failed with keep-alive enabled, 0/480 with it disabled.
    # See reports/APICONNECTION_ROOT_CAUSE_20260919.md.
    import httpx

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            args.request_timeout_seconds, connect=args.connect_timeout_seconds
        ),
        limits=httpx.Limits(
            max_connections=max(8, 4 * args.concurrency),
            max_keepalive_connections=0,
        ),
    )

    async def run_trial(trial):
        async with semaphore:
            for retry in range(args.retries + 1):
                started = time.monotonic()
                attempt = {
                    **trial,
                    "attempt": retry + 1,
                    "model": args.model,
                    "checkpoint_manifest_sha256": identity["manifest_sha256"],
                }
                try:
                    row = deepcopy(row_template)
                    row["id"] = trial["trial_id"] + f"-attempt{retry + 1}"
                    row["env"]["seed"] = trial["seed"]
                    workflow = policy_class(
                        **kwargs,
                        tokenizer_path=args.tokenizer_path
                        or args.checkpoint_path
                        or args.model,
                    )
                    await workflow.run(
                        row,
                        model=args.model,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        generation_seed=trial["generation_seed"],
                        checkpoint_manifest_sha256=identity["manifest_sha256"],
                        pacman_python_root=args.pacman_python_root,
                        trajectory_dir=trajectory_dir,
                        wall_clock_limit_seconds=args.wall_clock_limit_seconds,
                        stuck_no_progress_steps=args.stuck_no_progress_steps,
                        http_client=http_client,
                    )
                    episode = workflow.last_episode
                    if episode is None:
                        attempt.update(
                            status="initial_safety_refusal",
                            won=False,
                            terminal_reason="initial_safety_refusal",
                        )
                    else:
                        if bool(episode.get("won")) != is_verified_win(episode):
                            raise ValueError(
                                "claimed completion lacks terminal/event evidence"
                            )
                        successful.append(episode)
                        attempt.update(
                            status="completed",
                            **{
                                key: value
                                for key, value in episode.items()
                                if key != "trajectory"
                            },
                        )
                except Exception as error:
                    attempt.update(
                        status="error",
                        won=False,
                        error_type=type(error).__name__,
                        error=str(error),
                        error_chain=exception_chain(error),
                        terminal_reason="infrastructure_or_runtime_error",
                    )
                attempt["elapsed_seconds"] = time.monotonic() - started
                attempts.append(attempt)
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(attempt, sort_keys=True, allow_nan=False) + "\n"
                    )
                if attempt["status"] != "error" or retry == args.retries:
                    break
                await asyncio.sleep(min(2**retry, 8))

    try:
        await asyncio.gather(*(run_trial(trial) for trial in plan))
    finally:
        await http_client.aclose()
    wins = sum(bool(attempt.get("won")) for attempt in attempts)
    metrics = summarize_episodes(successful) if successful else {}
    return {
        **manifest,
        **metrics,
        "model": args.model,
        "service": service,
        "episodes": len(plan),
        "planned_episodes": len(plan),
        "completed_episodes": len(successful),
        "attempts": len(attempts),
        "error_attempts": sum(item["status"] == "error" for item in attempts),
        "full_completions": wins,
        "win_rate": wins / len(attempts),
        "planned_trial_win_rate": wins / len(plan),
        "minimum_win_rate": None,
        "evaluation_completed": all(
            any(a["trial_id"] == trial["trial_id"] for a in attempts) for trial in plan
        ),
        "conclusion": "observed_completed_games"
        if wins
        else "no_completion_observed_on_this_test_set",
        "terminal_reasons": dict(Counter(item["terminal_reason"] for item in attempts)),
        "decoding": protocol["decoding"],
        "max_steps": kwargs["environment_max_steps"],
        "ghost_mode": kwargs["ghost_mode"],
        "image_prompt_style": kwargs["image_prompt_style"],
        "trajectory_dir": str(trajectory_dir),
        "episode_results": attempts,
        "performance": {
            "total_attempt_seconds": sum(item["elapsed_seconds"] for item in attempts),
            "gpu_memory": "not measured by client; retain server/GPU telemetry",
        },
    }


def main():
    args = parse_args()
    summary = asyncio.run(evaluate(args))
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "evaluation_completed",
                    "planned_episodes",
                    "attempts",
                    "error_attempts",
                    "full_completions",
                    "win_rate",
                    "conclusion",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
