"""Strict final-reward rejection sampling, fixed slots and completion-order neutral."""
from copy import deepcopy
import hashlib
import math


def configured_policy(config):
    """New formal jobs filter by default; explicit JSON null preserves legacy runs."""
    return validate_policy(config.get('reward_group_filter',
        {'mode': 'exact_final_reward', 'max_rounds': 16,
         'retry_task': 'same_tier_new_task'}))


def validate_policy(policy):
    if policy is None:
        return None
    if (not isinstance(policy, dict) or not {'mode', 'max_rounds'} <= set(policy)
            or set(policy) - {'mode', 'max_rounds', 'retry_task'}
            or policy.get('retry_task', 'fixed_task') not in ('fixed_task', 'same_tier_new_task')
            or policy['mode'] != 'exact_final_reward'
            or type(policy['max_rounds']) is not int or policy['max_rounds'] < 1):
        raise ValueError('invalid reward_group_filter policy')
    return deepcopy(policy)


class InsufficientEffectiveGroups(RuntimeError):
    """No training batch may be returned on exhaustion."""


async def collect_effective(bindings, launch, *, max_rounds, audit, replacement=None):
    """launch(bindings, attempt) returns slot-ordered groups of 12 (samples, summary).

    Keep slot tier quotas fixed. Optional replacement selects fresh tasks on retry;
    without it, legacy fixed-task semantics apply. First-pass summaries drive curriculum after optimizer success;
    rejection-conditioned win rates must not drive promotion. audit persists every
    round before retry/failure. Rejected sample payloads are not retained here.
    """
    if type(max_rounds) is not int or max_rounds < 1 or not bindings:
        raise ValueError('positive rounds and nonempty bindings required')
    if len({b['group'] for b in bindings}) != len(bindings):
        raise ValueError('duplicate slots')
    kept, first, versions, used_seeds = {}, [], set(), set()
    total = rejected = 0
    history = deepcopy(bindings)
    for attempt in range(max_rounds):
        pending = [deepcopy(b) for b in bindings if b['group'] not in kept]
        if attempt and replacement is not None:
            for i, b in enumerate(pending):
                new = replacement(deepcopy(b), attempt, deepcopy(history))
                if (new['group'] != b['group'] or new['task']['tier'] != b['task']['tier']
                        or new['task'].get('split') != 'train'):
                    raise ValueError('replacement changed slot, tier or training split')
                pending[i] = new
                history.append(deepcopy(new))
        for b in pending:
            if len(b['sampling_seeds']) != 12:
                raise ValueError('expected twelve sampling seeds')
            if attempt and replacement is None:
                # Deterministic and domain-separated; retain one environment per group.
                original = b['sampling_seeds']
                b['sampling_seeds'] = [int.from_bytes(hashlib.sha256(
                    f'final-reward-filter-v1:{b["group"]}:{attempt}:{r}:{seed}'.encode()
                ).digest()[:4], 'big') for r, seed in enumerate(original)]
            for seed in b['sampling_seeds']:
                key = b['group'], seed
                if key in used_seeds:
                    raise ValueError('resampling seed collision')
                used_seeds.add(key)
        results = await launch(pending, attempt)
        if len(results) != len(pending):
            raise ValueError('incomplete collection slots')
        records = []
        for b, group in zip(pending, results, strict=True):
            if len(group) != 12:
                raise ValueError('incomplete group12')
            rewards = []
            for replica, (samples, row) in enumerate(group):
                reward = row['reward']
                if (row['group'] != b['group'] or row['replica'] != replica
                        or row['task_id'] != b['task']['task_id']
                        or not math.isfinite(reward) or not samples
                        or any(s.reward != reward for s in samples)):
                    raise ValueError('invalid final episode reward or group binding')
                versions.add(row['weight_version'])
                rewards.append(reward)
            if len(versions) != 1 or not next(iter(versions)):
                raise ValueError('collection mixed or missing weight versions')
            summaries = [deepcopy(row) for _, row in group]
            if attempt == 0:
                first.extend(summaries)
            equal = all(r == rewards[0] for r in rewards)
            total += 1
            rejected += int(equal)
            records.append(dict(group=b['group'], task_id=b['task']['task_id'],
                                environment_seed=b['task'].get('game_seed'),
                                tier=b['task']['tier'], sampling_seeds=b['sampling_seeds'],
                                rejected=equal, episodes=summaries))
            if not equal:
                kept[b['group']] = group
        stats = dict(rounds=attempt + 1, generated_groups=total,
                     rejected_groups=rejected, effective_groups=len(kept),
                     target_groups=len(bindings))
        audit(attempt, dict(stats=stats, groups=records))
        if len(kept) == len(bindings):
            return [kept[b['group']] for b in bindings], first, stats
        # Drop rejected multimodal tensors before the next round starts.
        del results, group, samples
    raise InsufficientEffectiveGroups(
        f'effective groups {len(kept)}/{len(bindings)} after {max_rounds} rounds; '
        'no optimizer update; inspect resampling audit')
