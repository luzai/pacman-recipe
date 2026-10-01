"""Zero-variance group resampling without completion-order bias.

slime's dynamic sampling filters groups as they finish (asyncio FIRST_COMPLETED) and aborts the rest
once enough pass. For Pacman that prefers groups whose longest episode is short (deaths end early,
wins and timeouts run long), biasing the batch toward short failures. This module instead works in
rounds over fixed start slots: run one group per slot that still lacks an informative group, wait for
all of them, keep groups whose episodes do not all share one reward, and rerun the remaining slots
from the same start. Selection never depends on finish time. If the round cap is reached, a slot is
completed with one of its own zero-variance groups (seeded by the rollout id) and the shortfall is
reported in the rollout metrics. Training always receives one group per slot (4 x 12 episodes).

Enable with --rollout-function-path slime_pacman.sampling.generate_rollout (starts = the dataset
groups slime would have used) or slime_pacman.curriculum.generate_rollout (dynamic bank).
"""

import asyncio
from copy import copy
import json
import os
from pathlib import Path
import random

DEFAULT_MAX_ROUNDS = 4


def _episodes(group):
    return [item if isinstance(item, list) else [item] for item in group]


def group_rewards(group):
    rewards = []
    for samples in _episodes(group):
        values = {float(sample.reward) for sample in samples}
        if len(values) != 1:
            raise ValueError("samples of one episode carry different rewards")
        rewards.append(values.pop())
    return rewards


def is_zero_variance(group):
    return len(set(group_rewards(group))) <= 1


def group_index(group):
    return _episodes(group)[0][0].group_index


def max_rounds():
    value = int(os.environ.get("PACMAN_RESAMPLE_MAX_ROUNDS", DEFAULT_MAX_ROUNDS))
    if value < 1:
        raise ValueError("PACMAN_RESAMPLE_MAX_ROUNDS must be positive")
    return value


async def resample_slots(slots, launch, rounds, rng):
    """One informative group per slot; `launch(slot_ids)` returns one completed group per slot id.

    Returns (kept groups in slot order, dropped groups, stats, per-slot log). Pure selection logic.
    """
    kept, spare, log = {}, {slot: [] for slot in range(slots)}, {slot: [] for slot in range(slots)}
    stats = dict(rounds=0, generated_groups=0, zero_variance_groups=0, filled_with_zero_variance=0)
    while len(kept) < slots and stats["rounds"] < rounds:
        missing = [slot for slot in range(slots) if slot not in kept]
        groups = await launch(missing)
        if len(groups) != len(missing):
            raise RuntimeError("launch returned a different number of groups than slots")
        stats["rounds"] += 1
        stats["generated_groups"] += len(groups)
        for slot, group in zip(missing, groups):
            zero = is_zero_variance(group)
            log[slot].append(dict(group_index=group_index(group), zero_variance=zero,
                                  wins=sum(group_rewards(group))))
            if zero:
                spare[slot].append(group)
                stats["zero_variance_groups"] += 1
            else:
                kept[slot] = group
    for slot in range(slots):
        if slot not in kept:
            kept[slot] = spare[slot].pop(rng.randrange(len(spare[slot])))
            stats["filled_with_zero_variance"] += 1
            log[slot].append(dict(group_index=group_index(kept[slot]), filled_with_zero_variance=True))
    dropped = [group for slot in range(slots) for group in spare[slot]]
    return [kept[slot] for slot in range(slots)], dropped, stats, log


def _with_record(group, record):
    out = []
    for item in group:
        samples = item if isinstance(item, list) else [item]
        copies = []
        for sample in samples:
            sample = copy(sample)
            sample.metadata = dict(sample.metadata or {}, episode_record=record)
            copies.append(sample)
        out.append(copies if isinstance(item, list) else copies[0])
    return out


async def rollout_slots(args, rollout_id, get_samples, records, log_path=None):
    """Generate one informative group per start record (resampling zero-variance groups)."""
    from slime.rollout.base_types import finalize_rollout_groups
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.rollout_transport import discard_rollout_group

    state = GenerateState(args)

    async def launch(slot_ids):
        # Fresh slime samples (new indices) carrying the slot's start record.
        groups = [_with_record(group, records[slot]) for group, slot in zip(get_samples(len(slot_ids)), slot_ids)]
        state.submit_generate_tasks(groups)
        done, pending = await asyncio.wait(state.pendings)  # ALL_COMPLETED
        if pending:
            raise RuntimeError("resampling round left pending generation tasks")
        state.pendings = set()
        by_index = {group_index(task.result()): task.result() for task in done}
        return [by_index[group_index(group)] for group in groups]

    rng = random.Random(f"pacman-resample-{args.seed}-{rollout_id}")
    kept, dropped, stats, log = await resample_slots(len(records), launch, max_rounds(), rng)
    for group in dropped:
        await asyncio.to_thread(discard_rollout_group, group, args, "zero_variance_group")
    state.reset()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(dict(rollout_id=rollout_id, stats=stats, slots=log,
                                            starts=[r["id"] for r in records]), indent=1))
    metrics = {f"rollout/resample_{key}": value for key, value in stats.items()}
    return await asyncio.to_thread(finalize_rollout_groups, args, rollout_id, kept, metrics)


def check_supported(args):
    unsupported = dict(
        dynamic_sampling_filter_path=args.dynamic_sampling_filter_path,
        partial_rollout=getattr(args, "partial_rollout", False),
        rollout_sample_filter_path=args.rollout_sample_filter_path,
        rollout_all_samples_process_path=args.rollout_all_samples_process_path,
    )
    if any(unsupported.values()) or args.rollout_data_transport != "object-store":
        raise ValueError(f"zero-variance resampling replaces these rollout options: {unsupported}")


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout import sglang_rollout
    from slime.utils.async_utils import run

    if evaluation:
        return sglang_rollout.generate_rollout(args, rollout_id, data_source, evaluation=True)
    check_supported(args)

    async def go():
        starts = data_source.get_samples(args.rollout_batch_size)
        records = [_episodes(group)[0][0].metadata["episode_record"] for group in starts]
        run_dir = os.environ.get("PACMAN_RUN_DIR")
        log_path = Path(run_dir) / "resampling" / f"rollout-{rollout_id:04d}.json" if run_dir else None
        return await rollout_slots(args, rollout_id, data_source.get_samples, records, log_path)

    return run(go())
