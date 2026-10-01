"""Small, immutable Edward restart banks and current-policy frontier selection.

The teacher selects advertised options and executes their primitive actions.
Learner rollout/training is deliberately outside this module. Simulator time
and remaining horizon are preserved, and terminal states are never candidates.
"""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, variance
from typing import Any, Callable, Iterable, Mapping

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig
from pacman_env.env._saved_state import checksum
from pacman_env.planner import EdwardPlanner, EdwardSafetyRefusal


BANK_SCHEMA = "pacman-restart-bank-v1"


class NoLearnableRestartState(ValueError):
    """No adequately sampled current-policy candidate meets the frontier rule."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_new(path: Path, value: Any) -> str:
    raw = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _teacher_episode(env: Any, planner: Any, seed: int, stride: int, dense_tail: int):
    _, info = env.reset(seed=seed)
    planner.observe(env.snapshot())
    sparse = {}
    tail: deque = deque(maxlen=dense_tail)
    actions = []
    active = None
    remaining = 0
    action = None
    reason = "horizon_exhausted"
    for _ in range(env.config.max_steps):
        # Capture pre-action boundaries, so even the final candidate is live.
        saved = env.save_state()
        step = int(info["step"])
        candidate = (step, saved, int(info["normal_pellets_remaining"]))
        tail.append(candidate)
        if step % stride == 0:
            sparse[step] = candidate
        if active is None:
            try:
                decision = planner.decide(env.snapshot())
            except EdwardSafetyRefusal:
                reason = "safety_refusal"
                break
            active = next(c for c in decision.candidates if c.option_id == decision.option_id)
            remaining = active.commit_moves
            action = active.first_action
        if action not in {"U", "D", "L", "R"} or remaining <= 0:
            raise RuntimeError("teacher option has an invalid primitive action or commitment")
        option = active.as_dict()
        _, reward, terminated, truncated, info = env.step(action)
        planner.record_action(action)
        remaining -= 1
        next_action = None
        if terminated or truncated:
            status = "terminal"
        elif active.strategy == "RISK_FALLBACK":
            status = "max_commit"
        else:
            next_action, status = planner.continue_option(active, env.snapshot())
            if status == "active" and remaining <= 0:
                status = "max_commit"
        actions.append({"action": action, "env_step": int(info["step"]),
                        "option": option, "option_status": status,
                        "reward": float(reward), "terminated": bool(terminated),
                        "truncated": bool(truncated)})
        if status != "active":
            active = None
        else:
            action = next_action
        if terminated or truncated:
            reason = info.get("terminal_reason")
            success = (terminated and not truncated and reason == "all_normal_pellets"
                       and info["normal_pellets_remaining"] == 0)
            if success:
                sparse.update({item[0]: item for item in tail})
                return {"seed": seed, "success": True, "terminal_reason": reason,
                        "total_steps": int(info["step"]), "actions": actions,
                        "states": [sparse[key] for key in sorted(sparse)]}
            break
    return {"seed": seed, "success": False, "terminal_reason": reason,
            "total_steps": len(actions)}


def build_restart_bank(
    output_dir: str | Path, *, seeds: Iterable[int] = range(5), trajectories: int = 1,
    config: PygamePacmanEnvConfig | None = None, stride: int = 32, dense_tail: int = 16,
    env_factory: Callable = PygamePacmanEnv, planner_factory: Callable = EdwardPlanner,
) -> dict[str, Any]:
    """Publish 1--10 successful teacher trajectories into a new directory.

    Seed attempts are bounded by the supplied list. Failed attempts contribute
    diagnostics only; no restart files are written unless enough teachers win.
    """
    _positive_integer(trajectories, "trajectories")
    if trajectories > 10:
        raise ValueError("the bounded bank supports at most ten trajectories")
    _positive_integer(stride, "stride")
    _positive_integer(dense_tail, "dense_tail")
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("teacher seeds must be unique")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"restart bank already exists: {output_dir}")
    config = config or PygamePacmanEnvConfig()
    successful, attempts = [], []
    for seed in seeds:
        with env_factory(config) as env:
            episode = _teacher_episode(env, planner_factory(), seed, stride, dense_tail)
        attempts.append({k: v for k, v in episode.items() if k not in {"states", "actions"}})
        if episode["success"]:
            successful.append(episode)
        if len(successful) == trajectories:
            break
    if len(successful) != trajectories:
        raise RuntimeError(f"teacher produced {len(successful)}/{trajectories} successful trajectories: {attempts}")

    # Exclusive creation and content hashes make artifacts auditable and prevent
    # accidental reuse of a directory from an earlier policy or source version.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": BANK_SCHEMA, "teacher": "edward-advertised-options-v1",
                "environment": {"ghost_mode": config.ghost_mode,
                                "episode_life_mode": config.episode_life_mode,
                                "max_steps": config.max_steps},
                "horizon_semantics": "preserve_original_remaining_steps",
                "learner_actions": ["U", "D", "L", "R"],
                "stride": stride, "dense_tail": dense_tail,
                "attempts": attempts, "trajectories": [], "restart_states": []}
    for episode in successful:
        states = episode.pop("states")
        trajectory_id = "teacher-" + checksum(episode)[:20]
        trajectory_path = f"{trajectory_id}.json"
        trajectory_hash = _write_new(output_dir / trajectory_path, episode)
        manifest["trajectories"].append({"trajectory_id": trajectory_id,
            "seed": episode["seed"], "total_steps": episode["total_steps"],
            "path": trajectory_path, "sha256": trajectory_hash})
        for step, saved, pellets in states:
            state_id = "restart-" + saved["sha256"]
            state_path = f"{state_id}.json"
            state_hash = _write_new(output_dir / state_path, saved)
            manifest["restart_states"].append({"restart_state_id": state_id,
                "state_path": state_path, "state_file_sha256": state_hash,
                "saved_state_sha256": saved["sha256"], "trajectory_id": trajectory_id,
                "seed": episode["seed"], "env_step": step,
                "is_true_initial_state": step == 0,
                "remaining_steps": config.max_steps - step,
                "normal_pellets_remaining": pellets,
                "identity": saved["payload"]["identity"],
                "runtime": saved["payload"]["worker"]["runtime"]})
    manifest["bank_id"] = "bank-" + checksum(manifest)
    _write_new(output_dir / "manifest.json", manifest)
    return manifest


def load_restart_bank(bank_dir: str | Path) -> dict[str, Any]:
    """Validate the manifest and its successful-teacher evidence before use."""
    root = Path(bank_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_bytes())
    payload = {k: v for k, v in manifest.items() if k != "bank_id"}
    if manifest.get("schema") != BANK_SCHEMA or manifest.get("bank_id") != "bank-" + checksum(payload):
        raise ValueError("restart bank manifest checksum/schema mismatch")
    for trajectory in manifest["trajectories"]:
        path = (root / trajectory["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("teacher trajectory path escapes bank directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != trajectory["sha256"]:
            raise ValueError("teacher trajectory file checksum mismatch")
        episode = json.loads(raw)
        if episode.get("success") is not True or episode.get("terminal_reason") != "all_normal_pellets":
            raise ValueError("restart bank requires a successful teacher trajectory")
    return manifest


def load_restart_state(bank_dir: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Read and verify exactly the restart file named by a manifest entry."""
    root = Path(bank_dir).resolve()
    path = (root / entry["state_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("restart state path escapes bank directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["state_file_sha256"]:
        raise ValueError("restart state file checksum mismatch")
    saved = json.loads(raw)
    if checksum(saved["payload"]) != saved["sha256"] or saved["sha256"] != entry["saved_state_sha256"]:
        raise ValueError("restart state payload checksum mismatch")
    if entry["restart_state_id"] != "restart-" + saved["sha256"]:
        raise ValueError("restart state ID mismatch")
    episode = saved["payload"]["episode"]
    if episode["finished"] or episode["steps"] != entry["env_step"] or episode["seed"] != entry["seed"]:
        raise ValueError("restart state episode metadata mismatch")
    if saved["payload"]["identity"] != entry["identity"] or saved["payload"]["worker"]["runtime"] != entry["runtime"]:
        raise ValueError("restart state compatibility metadata mismatch")
    if entry["remaining_steps"] != entry["identity"]["max_steps"] - episode["steps"]:
        raise ValueError("restart state remaining horizon mismatch")
    return saved


def _wilson(successes: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate_probe_results(
    records: Iterable[Mapping[str, Any]], *, policy_version: str, minimum_samples: int = 24,
) -> list[dict[str, Any]]:
    """Aggregate complete episodes, never per-turn rows, under one policy.

    Each record needs restart_state_id, policy_version, group_id, sample_id,
    success (bool), reward, pellets_eaten. Each group is exactly 12 complete
    episodes, matching the existing GRPO contract. Optional advantage is the actual
    recorded episode advantage. A normalized-return proxy is labeled separately.
    """
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("current policy_version is required")
    _positive_integer(minimum_samples, "minimum_samples")
    candidates = defaultdict(list)
    seen = set()
    for row in records:
        if row.get("policy_version") != policy_version:
            raise ValueError("probe mixes policy versions")
        for field in ("restart_state_id", "group_id", "sample_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"probe requires {field}")
        key = (row["restart_state_id"], row["sample_id"])
        if key in seen:
            raise ValueError("duplicate complete-episode probe sample")
        seen.add(key)
        if type(row.get("success")) is not bool:
            raise ValueError("probe success must be boolean")
        for field in ("reward", "pellets_eaten"):
            if isinstance(row.get(field), bool) or not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]):
                raise ValueError(f"probe {field} must be finite")
        if row["pellets_eaten"] < 0:
            raise ValueError("probe pellets_eaten must be nonnegative")
        if row.get("advantage") is not None and (
            isinstance(row["advantage"], bool) or not isinstance(row["advantage"], (int, float))
            or not math.isfinite(row["advantage"])
        ):
            raise ValueError("probe advantage must be finite")
        candidates[row["restart_state_id"]].append(row)
    summaries = []
    for state_id, rows in sorted(candidates.items()):
        groups = defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
        group_stats = []
        for group_id, members in sorted(groups.items()):
            if len(members) != 12:
                raise ValueError("probe groups require exactly 12 complete episodes")
            rewards = [row["reward"] for row in members]
            reward_var = variance(rewards) if len(rewards) > 1 else None
            group_stats.append({"group_id": group_id, "samples": len(members),
                "reward_variance": reward_var,
                "normalized_return_advantage_variance": (
                    reward_var / (math.sqrt(reward_var) + 1e-5) ** 2
                    if reward_var is not None else None)})
        n = len(rows)
        wins = sum(row["success"] for row in rows)
        observed_vars = [group["reward_variance"] for group in group_stats
                         if group["reward_variance"] is not None]
        advantages = [row.get("advantage") for row in rows]
        summaries.append({"restart_state_id": state_id, "policy_version": policy_version,
            "samples": n, "successes": wins, "success_rate": wins / n,
            "success_rate_wilson95": _wilson(wins, n),
            "minimum_samples": minimum_samples, "adequately_sampled": n >= minimum_samples,
            "mean_reward": mean(row["reward"] for row in rows),
            "mean_pellets_eaten": mean(row["pellets_eaten"] for row in rows),
            "group_reward_variance": mean(observed_vars) if observed_vars else None,
            "advantage_variance": (variance(advantages) if n > 1 and all(v is not None for v in advantages) else None),
            "groups": group_stats})
    return summaries


def select_restart_state(
    candidates: Iterable[Mapping[str, Any]], summaries: Iterable[Mapping[str, Any]], *,
    policy_version: str, minimum_samples: int = 24, lower: float = 0.3, upper: float = 0.7,
) -> dict[str, Any]:
    """Choose earliest teacher timestep with current-policy success in range.

    Confidence intervals are reported, not used as an extra selection formula.
    A missing frontier is an explicit outcome; never silently pick a hard state.
    """
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("current policy_version is required")
    _positive_integer(minimum_samples, "minimum_samples")
    if not (0 <= lower <= upper <= 1):
        raise ValueError("invalid success-rate interval")
    candidates = list(candidates)
    indexed = {entry["restart_state_id"]: entry for entry in candidates}
    if len(indexed) != len(candidates):
        raise ValueError("duplicate restart state ID")
    eligible, seen = [], set()
    for summary in summaries:
        state_id = summary["restart_state_id"]
        if state_id in seen:
            raise ValueError("duplicate probe summary")
        seen.add(state_id)
        if state_id not in indexed:
            raise ValueError("probe references an unknown restart state")
        if summary["policy_version"] != policy_version:
            raise ValueError("frontier selection requires the current policy version")
        if summary["samples"] >= minimum_samples and lower <= summary["success_rate"] <= upper:
            eligible.append(indexed[state_id])
    if not eligible:
        raise NoLearnableRestartState("no current-policy restart state has enough samples and success in the requested interval")
    return dict(min(eligible, key=lambda entry: (entry["env_step"], entry["restart_state_id"])))
