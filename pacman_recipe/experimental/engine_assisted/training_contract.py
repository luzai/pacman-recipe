"""Source-bound engine-assisted single-life batch accounting.

Adapted from the 2026-11-02/03 experiment. All run identities, topology splits,
budgets and reward settings are supplied explicitly; no historical run is a
default. Technical faults mask their whole original group without replacement.
"""
import hashlib
import json
import math
import re

FAULT_CODES = frozenset({'backend_timeout', 'backend_connection', 'model_output_format'})


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_protocol(protocol):
    if protocol.get('harness_kind') != 'engine_truth_assisted' or protocol.get('training_life_mode') != 'single_death':
        raise ValueError('explicit engine-assisted single-life protocol required')
    if not isinstance(protocol.get('variant'), str) or not protocol['variant']:
        raise ValueError('explicit training variant required')
    for key in ('groups_per_batch', 'samples_per_group', 'attempts_per_batch', 'loss_denominator', 'horizon', 'max_batches'):
        if type(protocol.get(key)) is not int or protocol[key] <= 0:
            raise ValueError('positive integer protocol field required: ' + key)
    total = protocol['groups_per_batch'] * protocol['samples_per_group']
    if total != protocol['attempts_per_batch'] or total != protocol['loss_denominator']:
        raise ValueError('all original attempts remain in the loss denominator')
    if protocol.get('replacement_sampling') is not False or protocol.get('fault_reward', 'missing') is not None:
        raise ValueError('no replacement and null technical-fault rewards required')
    for key in ('target_inventory_sha256', 'paired_task_schedule_sha256', 'parent_game_sha256', 'frozen_latest_sha256'):
        if not isinstance(protocol.get(key), str) or not re.fullmatch('[0-9a-f]{64}', protocol[key]):
            raise ValueError('explicit frozen identity required: ' + key)
    splits = []
    for key in ('train_topology_ids', 'validation_topology_ids', 'test_topology_ids'):
        values = protocol.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
            raise ValueError('nonempty unique topology IDs required: ' + key)
        splits.append(set(values))
    if any(splits[a] & splits[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('train/validation/test topology splits must be disjoint')
    for key in ('success_reward', 'success_speed_bonus'):
        value = protocol.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative reward setting required: ' + key)
    return protocol


def terminal_reward(*, won, deaths, steps, source_step, horizon, success_reward, speed_bonus):
    if type(won) is not bool or type(deaths) is not int or deaths not in (0, 1):
        raise ValueError('single-life native outcome required')
    if type(horizon) is not int or horizon <= 0 or type(steps) is not int or type(source_step) is not int or not 0 <= source_step < horizon or not 0 < steps <= horizon - source_step:
        raise ValueError('absolute native step budget required')
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (success_reward, speed_bonus)):
        raise ValueError('finite nonnegative reward settings required')
    if won and deaths:
        raise ValueError('single-life success cannot follow death')
    return success_reward + speed_bonus * (1 - (source_step + steps) / horizon) if won else 0.0


def sample_indices(length, episode_id, batch, *, limit):
    """Stable uniform-ranking subset of actual choices, with explicit capacity."""
    if type(length) is not int or length <= 0 or type(limit) is not int or limit <= 0:
        raise ValueError('positive actual corpus length and subset limit required')
    ranked = sorted(range(length), key=lambda i: hashlib.sha256(f'{episode_id}|{batch}|{i}'.encode()).digest())
    return sorted(ranked[:limit])


def batch_plan(episodes, batch, protocol_sha, version, adapter_sha, protocol):
    validate_protocol(protocol)
    group_count = protocol['groups_per_batch']
    samples = protocol['samples_per_group']
    attempts = protocol['attempts_per_batch']
    if canonical_sha(protocol) != protocol_sha or len(episodes) != attempts or type(batch) is not int or not 0 <= batch < protocol['max_batches']:
        raise ValueError('exact source-bound original attempts required')
    if not re.fullmatch('[0-9a-f]{64}', adapter_sha):
        raise ValueError('adapter identity required')
    groups, seen = {}, set()
    for row in episodes:
        eid = row['episode_id']
        if eid in seen:
            raise ValueError('duplicate attempt')
        seen.add(eid)
        expected = dict(protocol_sha256=protocol_sha, batch=batch, weight_version=version,
                        adapter_sha256=adapter_sha, life_mode='single_death',
                        harness_kind='engine_truth_assisted', training_variant=protocol['variant'], split='train', stage='train',
                        paired_task_schedule_sha256=protocol['paired_task_schedule_sha256'])
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError('episode identity mismatch: ' + key)
        if row['topology_id'] not in protocol['train_topology_ids']:
            raise ValueError('heldout topology in training')
        if type(row['group']) is not int or type(row['sample']) is not int or type(row['fault']) is not bool:
            raise ValueError('typed group/sample/fault facts required')
        kind, step, steps = row['start_class'], row['source_step'], row['steps']
        if kind not in ('normal', 'restart') or type(step) is not int or not 0 <= step < protocol['horizon'] or (kind == 'normal') != (step == 0):
            raise ValueError('truthful normal/restart identity required')
        if row['normal_initial_state'] is not (kind == 'normal') or row['failure_state_restart'] is not (kind == 'restart'):
            raise ValueError('normal/restart label mismatch')
        if type(steps) is not int or not 0 <= steps <= protocol['horizon'] - step:
            raise ValueError('invalid native prefix length')
        if row['fault']:
            if row['fault_code'] not in FAULT_CODES or row['reward'] is not None or row['trainable'] is not False or row['full_native_outcome'] is not False or row['game_won'] is not False:
                raise ValueError('nontrainable fault prefix with null reward required')
        else:
            if row['fault_code'] is not None or row['trainable'] is not True or row['full_native_outcome'] is not True:
                raise ValueError('accepted full native outcome required')
            expected_reward = terminal_reward(won=row['game_won'], deaths=row['deaths'], steps=steps, source_step=step,
                                             horizon=protocol['horizon'], success_reward=protocol['success_reward'], speed_bonus=protocol['success_speed_bonus'])
            if type(row['reward']) not in (int, float) or not math.isfinite(row['reward']) or row['reward'] != expected_reward:
                raise ValueError('terminal reward differs from native outcome')
        groups.setdefault(row['group'], []).append(row)
    if set(groups) != set(range(group_count)):
        raise ValueError('complete original group set required')
    eligible, excluded, informative, proofs, advantages = [], [], [], [], {}
    for group, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row['sample'])
        if len(rows) != samples or [row['sample'] for row in rows] != list(range(samples)):
            raise ValueError('complete unique original samples required')
        for key in ('start_class', 'source_step', 'source_state_sha256', 'topology_id', 'tier', 'game_seed'):
            if len({row[key] for row in rows}) != 1:
                raise ValueError('group crosses ' + key)
        masked = any(row['fault'] for row in rows)
        rewards = [row['reward'] for row in rows]
        if masked:
            adv = [None] * samples
            excluded.append(group)
        else:
            eligible.append(group)
            mean = sum(rewards) / samples
            std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / samples)
            adv = [0.0] * samples if std < 1e-12 else [(value - mean) / (std + 1e-8) for value in rewards]
            if any(value != 0 for value in adv):
                informative.append(group)
        advantages.update({row['episode_id']: value for row, value in zip(rows, adv)})
        proofs.append(dict(group=group, start_class=rows[0]['start_class'], group_masked=masked, rewards=rewards, advantages=adv))
    faults = sorted(row['episode_id'] for row in episodes if row['fault'])
    for proof in proofs:
        proof['reference_kl_applied'] = proof['entropy_applied'] = bool(informative) and not proof['group_masked']
    return dict(protocol_sha256=protocol_sha, weight_version=version, adapter_sha256=adapter_sha,
                paired_task_schedule_sha256=protocol['paired_task_schedule_sha256'], harness_kind='engine_truth_assisted', training_variant=protocol['variant'],
                attempts=attempts, loss_denominator=attempts, native_valid_outcomes=attempts - len(faults),
                fault_episode_ids=faults, eligible_group_ids=eligible, excluded_group_ids=excluded,
                informative_group_ids=informative, advantages_by_episode=advantages, group_proofs=proofs,
                excluded_episode_ids=sorted(row['episode_id'] for row in episodes if row['group'] in excluded),
                loss_eligible_episodes=samples * len(eligible), loss_applied_episodes=samples * len(eligible) if informative else 0,
                skip_before_model_loss=not informative,
                skip_reason='no_complete_valid_groups' if not eligible else 'no_within_complete_group_advantage' if not informative else None)


def validate_manifest(manifest, plan):
    if manifest.get('complete_native_private_audit_passed') is not True:
        raise ValueError('explicit native replay/input audit required')
    for key in ('protocol_sha256', 'weight_version', 'adapter_sha256', 'paired_task_schedule_sha256',
                'harness_kind', 'training_variant', 'eligible_group_ids', 'excluded_group_ids', 'fault_episode_ids'):
        if manifest.get(key) != plan[key]:
            raise ValueError('audited manifest differs: ' + key)
    return True


def loss_episodes(episodes, plan, rank=0, world=1):
    if type(rank) is not int or type(world) is not int or world < 1 or not 0 <= rank < world:
        raise ValueError('valid disjoint rank required')
    if plan['skip_before_model_loss']:
        return []
    return [row for index, row in enumerate(sorted(episodes, key=lambda row: (row['group'], row['sample'])))
            if index % world == rank and row['group'] in set(plan['eligible_group_ids'])]
