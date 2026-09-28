"""slime custom generation: one game episode -> independent vision samples."""

from copy import copy
from dataclasses import dataclass
from functools import lru_cache
import logging
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


async def generate_episode(args, sample, sampling_params, evaluation=False):
    import httpx
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
    try:
        async with httpx.AsyncClient(
            timeout=120.0, limits=httpx.Limits(max_keepalive_connections=0)
        ) as client:
            generator = SGLangGenerator(
                processor=state.processor,
                endpoint=get_model_url(args, "policy"),
                client=client,
                max_input_tokens=config.max_input_tokens,
            )
            runner = EpisodeRunner(
                record,
                tokenizer=state.tokenizer,
                generate=generator,
                config=config,
                expected_sources=current_sources(),
                pacman_python_root=pacman_python_root(),
            )
            episode = await runner.collect(empty_weight_version=version)
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
