"""slime custom generation: one game episode -> independent vision samples."""

from copy import copy
from dataclasses import dataclass
from functools import lru_cache
import logging
import json
import os
from pathlib import Path
import uuid

from pacman_env.paths import pacman_python_root
from pacman_recipe.level1.contracts import (
    audit_neutral_trajectory,
    neutral_trajectory,
    repository_identity,
    runner_row,
    validate_episode_record,
    write_json_new,
)
from pacman_recipe.level1.episode import ModelTurn, PacmanEpisodeRunner
from pacman_recipe.level1.trajectories import TrajectoryAuditError
from .config import load_config, runner_options
from .generation import SGLangGenerator
from .grouping import binary_reward


@dataclass
class EpisodeResult:
    reward: float
    decisions: list
    weight_version: str
    trajectory: dict | None
    terminal_reason: str


class EpisodeRunner(PacmanEpisodeRunner):
    def __init__(
        self, record, *, tokenizer, generate, config, expected_sources=None, **kwargs
    ):
        validate_episode_record(record, expected_sources=expected_sources)
        if record["environment"]["max_steps"] != config.max_steps:
            raise ValueError("episode horizon differs from the runner configuration")
        self.record = record
        self.decisions = []
        self.generate = generate
        expected_row = runner_row(record)

        def validate_row(row):
            if row != expected_row:
                raise ValueError("runner row changed after schema validation")

        super().__init__(
            tokenizer=tokenizer,
            row_validator=validate_row,
            **runner_options(config),
            **kwargs,
        )

    async def _call_model(self, messages, **options):
        constraint = options["objective_constraint"]
        decision = await self.generate(messages, constraint)
        constraint.option_for_tokens([decision.action_id])
        if decision.allowed_token_ids != constraint.allowed_token_ids:
            raise ValueError("generator changed the advertised action support")
        if (
            self.decisions
            and decision.weight_version != self.decisions[0].weight_version
        ):
            raise ValueError("weights changed inside a synchronous episode")
        self.decisions.append(decision)
        return ModelTurn(
            decision.completion, f"decision-{len(self.decisions) - 1}", messages
        )

    async def collect(self, *, empty_weight_version):
        await self.run(runner_row(self.record))
        if self.last_episode is None:
            if self.decisions:
                raise RuntimeError("episode disappeared after generation")
            # The existing runner explicitly returns None on initial refusal.
            # Retain this zero-decision episode in its GRPO group; never replace it.
            return EpisodeResult(
                0.0, [], str(empty_weight_version), None, "initial_safety_refusal"
            )
        reward = binary_reward(self.last_episode)
        if abs(self.last_episode["total_shaped_reward"] - reward) > 1e-6:
            raise ValueError("shared runner reward differs from binary contract")
        trajectory = neutral_trajectory(self.last_episode)
        audit_neutral_trajectory(trajectory)
        return EpisodeResult(
            reward,
            self.decisions,
            self.decisions[0].weight_version,
            trajectory,
            self.last_episode["terminal_reason"],
        )


def samples_from_episode(parent, episode, tokenizer):
    from slime.utils.types import Sample

    episode_id = parent.rollout_id if parent.rollout_id is not None else parent.index
    if episode_id is None or parent.group_index is None:
        raise ValueError("slime must assign episode and group indices")
    count = len(episode.decisions)
    samples = []
    for index, decision in enumerate(episode.decisions or [None]):
        sample = copy(parent)
        sample.rollout_id = episode_id
        sample.reward = episode.reward
        sample.status = Sample.Status.COMPLETED
        sample.response_length = 1
        sample.weight_versions = [episode.weight_version] if count else []
        sample.multimodal_inputs = None
        sample.multimodal_train_input_id = None
        if decision is None:
            # Scheduling placeholder, not a model decision. Zero mask keeps the
            # episode in the global denominator without inventing a policy action.
            token = tokenizer.eos_token_id
            if type(token) is not int:
                raise ValueError("empty-episode placeholder needs an EOS token")
            sample.tokens, sample.response, sample.prompt = [token, token], "", ""
            sample.loss_mask, sample.rollout_log_probs = [0], [0.0]
            sample.multimodal_train_inputs = None
        else:
            sample.tokens = decision.input_ids + [decision.action_id]
            sample.prompt, sample.response = decision.prompt, decision.completion
            sample.loss_mask, sample.rollout_log_probs = (
                [1],
                [decision.behavior_log_prob],
            )
            sample.multimodal_train_inputs = decision.multimodal_train_inputs
        sample.train_metadata = dict(
            group_id=parent.group_index,
            episode_id=episode_id,
            initial_state_id=parent.metadata["episode_record"]["id"],
            decision_count=count,
            decision_index=index if count else -1,
            weight_version=episode.weight_version,
            allowed_token_ids=decision.allowed_token_ids if decision else [],
            empty_episode=not count,
            image_sha256=decision.image_sha256 if decision else None,
        )
        samples.append(sample)
    return samples


@lru_cache(maxsize=1)
def current_sources():
    root = Path(__file__).resolve().parents[1]
    return {
        "pacman-recipe": repository_identity(root),
        "pacman-python": repository_identity(
            pacman_python_root() or root.parent / "pacman-python"
        ),
        "slime": repository_identity(
            os.environ.get("SLIME_ROOT", root.parent / "slime")
        ),
    }


def episode_workers():
    """Episodes run in this many worker processes (0 = in the rollout process).

    The Edward planner is pure Python, so episodes sharing one asyncio loop serialize
    on the GIL: profiled at ~0.4 s of planner CPU per decision with the GPUs idle.
    """
    workers = int(os.environ.get("PACMAN_EPISODE_WORKERS", "0"))
    if workers < 0:
        raise ValueError("PACMAN_EPISODE_WORKERS must be nonnegative")
    return workers


_EPISODE_POOL = None


def _init_episode_worker():
    # Episode workers are CPU-only and run dozens at a time: never initialize CUDA (the
    # Ray actor's environment exposes the training GPUs) and keep one thread each so
    # torch/OpenMP pools do not oversubscribe the host. Inherited by pygame subprocesses.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)


def _episode_pool(workers):
    global _EPISODE_POOL
    if _EPISODE_POOL is None:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # spawn: the rollout process is a threaded Ray actor, so fork is unsafe.
        _EPISODE_POOL = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_episode_worker,
        )
    return _EPISODE_POOL


@lru_cache(maxsize=2)
def _worker_tokenizer_and_processor(hf_checkpoint):
    # Same loaders as slime's GenerateState in the rollout process.
    from slime.utils.processing_utils import load_processor, load_tokenizer

    return (
        load_tokenizer(hf_checkpoint, trust_remote_code=True),
        load_processor(hf_checkpoint, trust_remote_code=True),
    )


def _multimodal_to_numpy(decisions):
    # Copy tensors by value through the result pipe; torch's default tensor pickling
    # would pass one file descriptor per storage (thousands per rollout). numpy has no
    # bfloat16: such tensors travel as their int16 bit pattern, tagged.
    import torch

    def to_numpy(value):
        if not hasattr(value, "numpy"):
            return value
        if value.dtype == torch.bfloat16:
            return ("bfloat16", value.view(torch.int16).numpy())
        return value.numpy()

    for decision in decisions:
        decision.multimodal_train_inputs = {
            key: to_numpy(value) for key, value in decision.multimodal_train_inputs.items()
        }


def _multimodal_to_torch(decisions):
    import numpy
    import torch

    def to_torch(value):
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "bfloat16":
            return torch.from_numpy(value[1]).view(torch.bfloat16)
        return torch.from_numpy(value) if isinstance(value, numpy.ndarray) else value

    for decision in decisions:
        decision.multimodal_train_inputs = {
            key: to_torch(value) for key, value in decision.multimodal_train_inputs.items()
        }


def write_death_candidates(path, boundaries, episode, source):
    """Persist pre-death decision boundaries (distances 4/8/16) of a death episode, atomically."""
    from .dynamic_bank import CANDIDATE_SCHEMA, death_candidates

    candidates = death_candidates(boundaries, episode.terminal_reason)
    if not candidates:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last = list(boundaries)[-1]
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "source": dict(source, terminal_reason=episode.terminal_reason, reward=episode.reward,
                       weight_version=episode.weight_version, decision_count=len(boundaries)),
        "seed": last["env_state"]["payload"]["episode"]["seed"],
        "last_choice_position": last["features"]["pacman_position"],
        "candidates": [dict(distance=d, boundary=b) for d, b in candidates],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)
    return path


AUDIT_RETRIES = 2


def dump_audit_failure(record, exc, attempt):
    """Write a rejected episode (trajectory audit failure) under $PACMAN_RUN_DIR/audit-failures/."""
    logging.getLogger(__name__).warning(
        "episode %s failed its trajectory audit (attempt %d): %s", record.get("id"), attempt, exc
    )
    run_dir = os.environ.get("PACMAN_RUN_DIR")
    if not run_dir:
        return None
    path = Path(run_dir) / "audit-failures" / f"{record.get('id')}-{uuid.uuid4().hex[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"record_id": record.get("id"), "seed": (record.get("environment") or {}).get("seed"),
            "attempt": attempt, "error": str(exc), "payload": exc.payload}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, default=str))
    os.replace(temporary, path)
    return path


def audit_failure_count():
    run_dir = os.environ.get("PACMAN_RUN_DIR")
    folder = Path(run_dir) / "audit-failures" if run_dir else None
    return len(list(folder.glob("*.json"))) if folder is not None and folder.is_dir() else 0


async def _collect_episode(
    record, *, endpoint, tokenizer, processor, config, expected_sources, version, capture=None
):
    import httpx
    from collections import deque

    from .dynamic_bank import RING_SIZE

    async with httpx.AsyncClient(
        timeout=120.0, limits=httpx.Limits(max_keepalive_connections=0)
    ) as client:
        generator = SGLangGenerator(
            processor=processor,
            endpoint=endpoint,
            client=client,
            max_input_tokens=config.max_input_tokens,
        )
        runner = EpisodeRunner(
            record,
            tokenizer=tokenizer,
            generate=generator,
            config=config,
            expected_sources=expected_sources,
            pacman_python_root=pacman_python_root(),
        )
        for attempt in range(AUDIT_RETRIES + 1):
            boundaries = None
            if capture is not None:
                boundaries = deque(maxlen=RING_SIZE)
                runner.decision_boundary_sink = boundaries.append
            try:
                episode = await runner.collect(empty_weight_version=version)
                break
            except TrajectoryAuditError as exc:
                # A rejected episode never reaches training. Keep it for diagnosis and play a fresh
                # episode from the same start so the group keeps its size; give up after a few.
                dump_audit_failure(record, exc, attempt)
                if attempt == AUDIT_RETRIES:
                    raise
                runner = EpisodeRunner(
                    record,
                    tokenizer=tokenizer,
                    generate=generator,
                    config=config,
                    expected_sources=expected_sources,
                    pacman_python_root=pacman_python_root(),
                )
    if capture is not None:
        write_death_candidates(capture["path"], boundaries, episode, capture["source"])
    return episode


def collect_episode_in_worker(
    record, endpoint, hf_checkpoint, config_path, expected_sources, version, capture=None,
    summary_only=False,
):
    """Worker-process entry point: one complete episode, tensors returned as numpy.

    capture: {"path", "source"} to write death candidates; summary_only drops decisions
    (probe/initialization episodes need only the outcome).
    """
    import asyncio

    tokenizer, processor = _worker_tokenizer_and_processor(hf_checkpoint)
    episode = asyncio.run(
        _collect_episode(
            record,
            endpoint=endpoint,
            tokenizer=tokenizer,
            processor=processor,
            config=load_config(config_path),
            expected_sources=expected_sources,
            version=version,
            capture=capture,
        )
    )
    if summary_only:
        episode.decisions = []
    _multimodal_to_numpy(episode.decisions)
    return episode


async def generate_episode(args, sample, sampling_params, evaluation=False):
    import asyncio

    from slime.rollout.sglang_rollout import GenerateState, get_model_url

    config = load_config(os.environ["PACMAN_SLIME_CONFIG"])
    record = sample.metadata["episode_record"]
    if evaluation and sample.group_index is None:
        # Upstream evaluation assigns sample.index but no group_index.
        sample.group_index = record["id"]
    state = GenerateState(args)
    version_value = getattr(args, "_rollout_weight_version", None)
    version = "" if version_value is None else str(version_value)
    run_dir = Path(os.environ["PACMAN_RUN_DIR"])
    artifact_id = f"{'eval' if evaluation else 'train'}-{sample.group_index}-{sample.index}-{uuid.uuid4().hex}"
    workers = episode_workers()
    capture = None
    capture_dir = getattr(args, "_pacman_capture_dir", None)
    if capture_dir is not None and not evaluation:
        # Dynamic bank: training episodes record pre-death decision boundaries.
        capture = dict(path=str(Path(capture_dir) / f"{artifact_id}.json"),
                       source=dict(artifact_id=artifact_id, start_id=record["id"],
                                   group_index=sample.group_index, sample_index=sample.index,
                                   rollout_id=getattr(args, "_pacman_rollout_id", None)))
    try:
        if workers:
            episode = await asyncio.get_running_loop().run_in_executor(
                _episode_pool(workers),
                collect_episode_in_worker,
                record,
                get_model_url(args, "policy"),
                args.hf_checkpoint,
                os.environ["PACMAN_SLIME_CONFIG"],
                current_sources(),
                version,
                capture,
            )
            _multimodal_to_torch(episode.decisions)
        else:
            episode = await _collect_episode(
                record,
                endpoint=get_model_url(args, "policy"),
                tokenizer=state.tokenizer,
                processor=state.processor,
                config=config,
                expected_sources=current_sources(),
                version=version,
                capture=capture,
            )
        if episode.decisions and not episode.weight_version:
            raise ValueError("episode has no verified weight version")
        write_json_new(
            run_dir / "episodes" / f"{artifact_id}.json",
            {
                "schema": "pacman-rollout-v1",
                "record": record,
                "reward": episode.reward,
                "weight_version": episode.weight_version,
                "terminal_reason": episode.terminal_reason,
                "trajectory": episode.trajectory,
                "decisions": [
                    {k: v for k, v in vars(d).items() if k != "multimodal_train_inputs"}
                    for d in episode.decisions
                ],
            },
        )
        return samples_from_episode(sample, episode, state.tokenizer)
    except Exception as exc:
        write_json_new(
            run_dir / "failures" / f"{artifact_id}.json",
            {
                "schema": "pacman-rollout-failure-v1",
                "episode_id": sample.index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.rollout.base_types import iter_samples
    from slime.observability import logging_utils

    episodes = {}
    groups = {}
    for sample in iter_samples(samples):
        meta = sample.train_metadata
        episodes[meta["episode_id"]] = float(sample.reward)
        groups.setdefault(meta["group_id"], {})[meta["episode_id"]] = float(
            sample.reward
        )
    metrics = dict(rollout_extra_metrics or {})
    metrics.update(
        {
            "rollout/win_rate": sum(episodes.values()) / len(episodes),
            "rollout/zero_variance_groups": sum(
                len(set(g.values())) == 1 for g in groups.values()
            )
            / len(groups),
            "rollout/episodes": len(episodes),
            "rollout/audit_failures_total": audit_failure_count(),
            "rollout/seconds": rollout_time,
            "rollout/step": rollout_id,
        }
    )
    logging.getLogger(__name__).info("Pacman rollout metrics: %s", metrics)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return True


def log_eval(rollout_id, args, data, extra_metrics=None):
    """Count each episode once even when its decision count differs."""
    from slime.rollout.base_types import iter_samples
    from slime.observability import logging_utils

    metrics = dict(extra_metrics or {})
    versions = set()
    for name, result in data.items():
        episodes = {}
        for sample in iter_samples(result["samples"]):
            versions.update(sample.weight_versions or [])
            episode_id = sample.train_metadata["episode_id"]
            reward = float(sample.reward)
            if episodes.setdefault(episode_id, reward) != reward:
                raise ValueError("inconsistent evaluation episode reward")
        if not episodes:
            raise ValueError("empty evaluation")
        metrics[f"eval/{name}/win_rate"] = sum(episodes.values()) / len(episodes)
        metrics[f"eval/{name}/episodes"] = len(episodes)
    if len(versions) > 1:
        raise ValueError("evaluation mixes weight versions")
    metrics["eval/step"] = rollout_id
    # Initial evaluation and post-update evaluation can share rollout_id=0.
    # Keep the actual served weight identity in the log and immutable artifact.
    evidence = {
        "schema": "pacman-eval-metrics-v1",
        "rollout_id": rollout_id,
        "weight_versions": sorted(versions),
        "metrics": metrics,
    }
    logging.getLogger(__name__).info("Pacman evaluation metrics: %s", evidence)
    if os.environ.get("PACMAN_RUN_DIR"):
        write_json_new(
            Path(os.environ["PACMAN_RUN_DIR"])
            / "metrics"
            / f"eval-{rollout_id}-{uuid.uuid4().hex}.json",
            evidence,
        )
    logging_utils.log(args, metrics, step_key="eval/step")
    return True
