"""Initial-stage group12 collection; promotion only in the optimizer completion hook.

Deployment places this file in slime_pacman and formal_sampling/controller modules
on PYTHONPATH. No framework imports occur during offline contract tests.
"""

import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

_POOLS = {}


def worker_pool(context):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from .map_training.workers import worker_init

    workers = int(os.environ.get("PACMAN_FORMAL_EPISODE_WORKERS", "4"))
    if workers < 1:
        raise ValueError("episode worker count must be positive")
    key = context["ghosts"], workers
    if key not in _POOLS:
        _POOLS[key] = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
            initargs=(context["game_root"], context["ghosts"]),
        )
    return _POOLS[key]


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("immutable rollout artifact changed")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)


def validate_binding(task):
    record = task["episode_record"]
    if record.get("restart") or record.get("state_prefix_actions"):
        raise ValueError("formal rollout requires initial stage")
    if record["environment"]["level_revision"] != task["layout_sha256"]:
        raise ValueError("record and frozen layout identity differ")
    if task["ghosts"] not in (0, 1, 2, 4):
        raise ValueError("unsupported ghost count")
    if (task["ghosts"] == 0) != (record["environment"]["ghost_mode"] == "disabled"):
        raise ValueError("catalog effective ghost count disagrees with episode record")
    return record, dict(game_root=task["game_root"], ghosts=task["ghosts"])


class Runtime:
    def __init__(self, args):
        from .map_training.sampling import FormalSampler
        from .map_training.controller import TrainingPolicy

        self.args = args
        from .map_training.preflight import validate_training_args

        validate_training_args(args)
        config = json.loads(Path(os.environ["PACMAN_FORMAL_CONFIG"]).read_text())
        from .map_training.reward_filter import configured_policy
        self.reward_group_filter = configured_policy(config)
        self.catalog = config["catalog"]
        self.sampler = FormalSampler(
            self.catalog,
            TrainingPolicy(**config["policy"]),
            max_tier=config["max_tier"],
            min_tier=config.get("min_tier", 1),
            seed=config["seed"],
            environment_seed_mode=config.get("environment_seed_mode", "group_randomized"),
            reward_group_filter=self.reward_group_filter,
        )
        self.root = Path(os.environ["PACMAN_RUN_DIR"])
        self.identity = self.sampler.controller.identity
        self.pending = {}

    def freeze_plan(self, update):
        if self.pending:
            raise ValueError(
                "previous collection has not completed its optimizer update"
            )
        plan = self.sampler.plan(update)
        for group in plan["groups"]:
            validate_binding(group["task"])
        write_once(self.root / "map-plans" / f"rollout-{update:04d}.json", plan)
        return plan

    def collected(self, update, plan, episodes):
        if len(episodes) != 192 or update in self.pending:
            raise ValueError("expected one complete group12 x 16 collection")
        self.pending[update] = dict(plan=deepcopy(plan), episodes=deepcopy(episodes))
        write_once(
            self.root / "map-batches" / f"rollout-{update:04d}.json",
            self.pending[update],
        )

    def training_completed(self, update):
        batch = self.pending[update]
        result = self.sampler.complete(update, batch["episodes"], plan=batch["plan"])
        del self.pending[update]
        return result

    def save(self, update):
        """Called after synchronous model/optimizer, datasource and builder saves."""
        from .map_training.checkpoints import publish

        if self.pending:
            raise ValueError("cannot checkpoint an incomplete optimizer batch")
        root = Path(self.args.save)
        dataset = root / "rollout" / f"global_dataset_state_dict_{update}.pt"
        if not dataset.is_file():
            raise ValueError("matching datasource checkpoint is required")
        extras = [dataset]
        builder = root / "rollout" / f"builder_state_{update}.json"
        if self.args.rollout_data_transport == "straw":
            if not builder.is_file():
                raise ValueError("matching straw builder checkpoint is required")
            extras.append(builder)
        return publish(
            root,
            self.sampler.snapshot(),
            identity=self.identity,
            update=update,
            extra_files=extras,
        )

    def load(self, update):
        """Verify the joint checkpoint before the framework loads datasource state."""
        if update == -1:
            if (
                self.args.load
                and (
                    Path(self.args.load) / "latest_checkpointed_iteration.txt"
                ).exists()
            ):
                raise ValueError("checkpoint restore cannot use cold-start update")
            return
        if not self.args.load:
            if update not in (None, -1):
                raise ValueError("cold start cannot claim completed updates")
            return
        from .map_training.checkpoints import restore

        receipt = restore(self.args.load, identity=self.identity, update=update)
        snapshot = receipt["controller"]
        self.sampler.restore(
            snapshot,
            completed_update=receipt["update"],
            bank_version=snapshot["state"]["bank_version"],
        )

    checkpoint_saved = save
    checkpoint_loaded = load


def optimizer_completed_hook(args, rollout_id):
    return get_runtime(args).training_completed(rollout_id)


def checkpoint_saved_hook(args, rollout_id):
    return get_runtime(args).save(rollout_id)


def checkpoint_load_hook(args, rollout_id):
    return get_runtime(args).load(rollout_id)


def get_runtime(args):
    if not hasattr(args, "_pacman_formal_runtime"):
        args._pacman_formal_runtime = Runtime(args)
    return args._pacman_formal_runtime


async def collect(args, task, sample, update, group, replica, seed, evaluation):
    from slime.rollout.sglang_rollout import GenerateState, get_model_url
    from .map_training.workers import collect_task
    from .rollout import samples_from_episode, _multimodal_to_torch

    record, context = validate_binding(task)
    sample.metadata = deepcopy(sample.metadata or {})
    sample.metadata.update(episode_record=record, map_context=context)
    value = getattr(args, "_rollout_weight_version", None)
    version = "" if value is None else str(value)
    # Request seeds are hashed again per decision by SeededSmallMazeClient.
    # Evaluation omits model/update identity so its request schedules stay paired.
    seed_context = dict(
        purpose="validation" if evaluation else "train",
        record=record["id"],
        map_sha=task["layout_sha256"],
        ghosts=task["ghosts"],
        replica=replica,
        frozen_seed=seed,
    )
    if not evaluation:
        seed_context.update(update=update, group=group, version=version)
    episode = await asyncio.get_running_loop().run_in_executor(
        worker_pool(context),
        collect_task,
        record,
        context["game_root"],
        get_model_url(args, "policy"),
        args.hf_checkpoint,
        os.environ["PACMAN_SLIME_CONFIG"],
        version,
        seed_context,
    )
    if episode.decisions and not episode.weight_version:
        raise ValueError("missing served weight identity")
    _multimodal_to_torch(episode.decisions)
    artifact = f'formal-{"eval" if evaluation else "train"}-{update}-{group}-{replica}-{uuid.uuid4().hex}'
    won = episode.terminal_reason == "all_normal_pellets"
    evidence = dict(
        schema="pacman-rollout-v1",
        record=record,
        map_context=context,
        task_id=task["task_id"],
        split=task["split"],
        tier=task["tier"],
        update=update,
        group=group,
        replica=replica,
        frozen_seed=seed,
        reward=episode.reward,
        success_speed_bonus=episode.success_speed_bonus,
        success_life_bonus=episode.success_life_bonus,
        won=won,
        weight_version=episode.weight_version,
        environment_steps=episode.environment_steps,
        terminal_reason=episode.terminal_reason,
        trajectory=episode.trajectory,
        decisions=[
            {k: v for k, v in vars(d).items() if k != "multimodal_train_inputs"}
            for d in episode.decisions
        ],
    )
    write_once(
        Path(os.environ["PACMAN_RUN_DIR"]) / "episodes" / f"{artifact}.json", evidence
    )
    samples = samples_from_episode(sample, episode, GenerateState(args).tokenizer)
    return samples, dict(
        group=group,
        replica=replica,
        task_id=task["task_id"],
        won=won,
        artifact=artifact,
        reward=episode.reward,
        weight_version=episode.weight_version,
        panel=task.get("panel", "heldout_layouts"),
        map_sha=task["layout_sha256"],
    )


async def rollout_async(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import finalize_rollout_groups, RolloutFnEvalOutput
    from slime.utils.types import Sample

    runtime = get_runtime(args)
    if evaluation:
        rows = sorted(
            (r for r in runtime.catalog if r["split"] == "validation"
             and runtime.sampler.min_tier <= r["tier"] <= runtime.sampler.controller.max_tier),
            key=lambda r: r["task_id"],
        )
        if not rows:
            raise ValueError("independent validation catalog is empty")
        bindings = []
        for group, task in enumerate(rows):
            seeds = [
                int.from_bytes(
                    hashlib.sha256(
                        f'validation:{task["task_id"]}:{replica}'.encode()
                    ).digest()[:4],
                    "big",
                )
                for replica in range(12)
            ]
            bindings.append(dict(group=group, task=task, sampling_seeds=seeds))
        groups = [
            [Sample(group_index=i, index=i * 12 + r) for r in range(12)]
            for i in range(len(rows))
        ]
        plan = None
    else:
        if args.rollout_batch_size != 16 or args.n_samples_per_prompt != 12:
            raise ValueError("formal training requires group12 x 16")
        plan = runtime.freeze_plan(rollout_id)
        bindings = plan["groups"]
        groups = data_source.get_samples(16)
    if len(groups) != len(bindings) or any(len(g) != 12 for g in groups):
        raise ValueError("incomplete collection shape")
    async def launch(round_bindings, attempt):
        round_groups = groups if attempt == 0 else data_source.get_samples(len(round_bindings))
        if len(round_groups) != len(round_bindings) or any(len(g) != 12 for g in round_groups):
            raise ValueError('incomplete resampling allocation')
        return await asyncio.gather(*[
            asyncio.gather(*[
                collect(args, binding['task'], sample, rollout_id, binding['group'],
                        replica, binding['sampling_seeds'][replica], evaluation)
                for replica, sample in enumerate(group)
            ]) for binding, group in zip(round_bindings, round_groups, strict=True)
        ])

    first_pass = None
    filter_metrics = {}
    if not evaluation and runtime.reward_group_filter is not None:
        from .map_training.reward_filter import collect_effective
        results, first_pass, stats = await collect_effective(
            bindings, launch, max_rounds=runtime.reward_group_filter['max_rounds'],
            replacement=(lambda binding, attempt, history: runtime.sampler.replacement(
                binding, attempt, history, update=rollout_id))
            if runtime.reward_group_filter.get('retry_task') == 'same_tier_new_task' else None,
            audit=lambda attempt, payload: write_once(
                runtime.root / 'resampling' / f'rollout-{rollout_id:04d}-round-{attempt:03d}.json',
                payload),
        )
        filter_metrics = {f'rollout/reward_filter_{k}': v for k, v in stats.items()}
        write_once(runtime.root / 'effective-batches' / f'rollout-{rollout_id:04d}.json',
                   [row for group in results for _, row in group])
    else:
        results = await launch(bindings, 0)
    summaries = [result[1] for group in results for result in group]
    sample_groups = [
        [sample for result in group for sample in result[0]] for group in results
    ]
    if evaluation:
        versions = sorted({row["weight_version"] for row in summaries})
        if len(versions) != 1:
            raise ValueError("validation mixed served weight versions")
        version_key = hashlib.sha256(json.dumps(versions).encode()).hexdigest()[:16]
        write_once(
            runtime.root
            / "validation"
            / f"eval-{rollout_id:04d}-weights-{version_key}.json",
            summaries,
        )
        panel_metrics = {}
        for panel in sorted({row["panel"] for row in summaries}):
            subset = [row for row in summaries if row["panel"] == panel]
            panel_metrics[f"validation/{panel}/win_rate"] = sum(
                row["won"] for row in subset
            ) / len(subset)
            panel_metrics[f"validation/{panel}/mean_reward"] = sum(
                row["reward"] for row in subset
            ) / len(subset)
        return RolloutFnEvalOutput(
            data={
                "heldout_layouts": {
                    "rewards": [row["reward"] for row in summaries],
                    "samples": [sample for group in sample_groups for sample in group],
                }
            },
            metrics={
                **panel_metrics,
                "validation/episode_win_rate": sum(row["won"] for row in summaries)
                / len(summaries),
            },
        )
    controller_episodes = first_pass if first_pass is not None else summaries
    runtime.collected(rollout_id, plan, controller_episodes)
    return finalize_rollout_groups(
        args,
        rollout_id,
        sample_groups,
        metrics={
            **filter_metrics,
            "curriculum/tier": plan["tier"],
            "curriculum/bank_version": plan["bank_version"],
            "curriculum/episode_win_rate": sum(row["won"] for row in controller_episodes) / 192,
            "rollout/effective_episode_win_rate": sum(row['won'] for row in summaries) / 192,
        },
    )


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run

    return run(rollout_async(args, rollout_id, data_source, evaluation))
