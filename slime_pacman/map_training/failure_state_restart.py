"""On-policy training from full states preceding the agent's own deaths."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

from .checkpoints import atomic_json, sha256
from .identity import digest
from .sampling import FormalSampler

DEFAULT_POLICY = dict(fraction=0.8, bins=[[4, 7], [8, 15], [16, 32]],
                      max_entries_per_tier=64, max_age_updates=12, seed=20261027)


def configured_policy(config):
    protocols = {t.get('episode_record', {}).get('prompt', {}).get('action_protocol')
                 for t in config.get('catalog', [])}
    if any(p is not None and p != 'direct-open-action-token-v1' for p in protocols):
        if config.get('failure_state_restart') not in (None, False):
            raise ValueError('failure_state_restart requires Primitive action histories')
        return None
    value = config.get('failure_state_restart', True)
    if value is None or value is False:
        return None
    if value is True:
        return deepcopy(DEFAULT_POLICY)
    if not isinstance(value, dict) or set(value) - set(DEFAULT_POLICY):
        raise ValueError('invalid failure_state_restart policy')
    result = dict(deepcopy(DEFAULT_POLICY), **value)
    if result['fraction'] != 0.8 or result['bins'] != DEFAULT_POLICY['bins']:
        raise ValueError('failure_state_restart requires audited 80% / 4..32 policy')
    if any(type(result[k]) is not int or result[k] <= 0
           for k in ['max_entries_per_tier', 'max_age_updates']):
        raise ValueError('invalid restart bank limits')
    if type(result['seed']) is not int:
        raise ValueError('restart seed must be integer')
    return result


class RestartBank:
    def __init__(self, root, config, policy):
        self.root = Path(root) / 'failure_state_restart'
        self.config, self.policy = deepcopy(config), deepcopy(policy)

    def read(self):
        p = self.root / 'latest.json'
        return json.loads(p.read_text()) if p.exists() else dict(entries=[], outcomes={})

    def refresh(self, paths, update):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from .failure_state_snapshots import extract
        old = self.read(); paths = sorted(set(map(str, paths)))
        new = []
        if paths:
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                                     max_tasks_per_child=1) as pool:
                jobs = [(p, self.config, str(self.root), self.policy) for p in paths]
                new = [e for batch in pool.map(extract, jobs) for e in batch]
        for e in new:
            e['admitted_update'] = update
        entries = {e['id']: e for e in old['entries'] + new
                   if e['admitted_update'] >= update - self.policy['max_age_updates']}
        retained = []
        for tier in sorted({e['tier'] for e in entries.values()}):
            offsets = {n: sorted((e for e in entries.values() if e['tier'] == tier and e['lookback'] == n),
                                  key=lambda e: (e['admitted_update'], e['id']), reverse=True)
                       for n in range(4, 33)}
            chosen = []
            while len(chosen) < self.policy['max_entries_per_tier'] and any(offsets.values()):
                for es in offsets.values():
                    if es and len(chosen) < self.policy['max_entries_per_tier']:
                        chosen.append(es.pop(0))
            retained.extend(chosen)
        ids = {e['id'] for e in retained}
        value = dict(entries=retained, outcomes={k:v for k,v in old['outcomes'].items() if k in ids},
                     update=update, policy=self.policy)
        atomic_json(self.root / 'latest.json', value)
        return value

    def observe(self, run, update, plan, episodes):
        bank = self.read()
        for p in sorted((Path(run) / 'resampling').glob(f'rollout-{update:04d}-round-*.json')):
            for group in json.loads(p.read_text())['groups']:
                row = group['episodes'][0]
                x = json.loads((Path(run) / 'episodes' / (row['artifact'] + '.json')).read_text())
                restart = x['record'].get('restart')
                if restart:
                    bank['outcomes'][restart['id'].removeprefix('failure-')] = dict(
                        wins=sum(e['won'] for e in group['episodes']), update=update)
        atomic_json(self.root / 'latest.json', bank)
        initial = {g['group'] for g in plan['groups'] if g['start'] == 'initial'}
        paths = [Path(run) / 'episodes' / (e['artifact'] + '.json') for e in episodes
                 if e['group'] in initial and e['replica'] < 3]
        self.refresh(paths, update)

    def checkpoint(self, checkpoint_root, update):
        root = Path(checkpoint_root)
        dest = root / 'rollout' / f'failure_state_restart_{update}.json'
        value = deepcopy(self.read()); extras = []
        for e in value['entries']:
            source = Path(e['record']['restart']['path'])
            target = root / 'rollout' / f'failure_state_restart_states_{update}' / source.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            e['checkpoint_state'] = target.relative_to(root).as_posix()
            extras.append(target)
        atomic_json(dest, value)
        return [dest, *extras]

    def restore(self, checkpoint_root, update):
        root = Path(checkpoint_root).resolve()
        value = json.loads((root / 'rollout' / f'failure_state_restart_{update}.json').read_text())
        if value['policy'] != self.policy:
            raise ValueError('restart checkpoint policy mismatch')
        for e in value['entries']:
            source = (root / e.pop('checkpoint_state')).resolve()
            if not source.is_relative_to(root) or sha256(source) != e['record']['restart']['sha256']:
                raise ValueError('restart checkpoint state changed')
            target = self.root / 'states' / source.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            e['record']['restart']['path'] = str(target)
        atomic_json(self.root / 'latest.json', value)


class FailureStateRestartSampler(FormalSampler):
    def __init__(self, *args, restart_bank, **kwargs):
        super().__init__(*args, **kwargs)
        self.restart_bank = restart_bank
        self.controller.identity = digest([self.controller.identity, 'failure_state_restart-v2-normal-window', restart_bank.policy])

    def choose(self, tier, domain, excluded=()):
        bank = self.restart_bank.read()
        entries = [e for e in bank['entries'] if e['tier'] == tier]
        if not entries:
            return None
        rng = random.Random(digest(domain))
        bin_id = rng.randrange(3)
        es = [e for e in entries if e['bin_id'] == bin_id] or entries
        offset = rng.choice(sorted({e['lookback'] for e in es}))
        es = [e for e in es if e['lookback'] == offset]
        es = [e for e in es if e['id'] not in excluded] or es
        def weight(e):
            o = bank['outcomes'].get(e['id'])
            return 1 if o is None else (2 if 0 < o['wins'] < 12 else .1)
        return rng.choices(es, weights=[weight(e) for e in es], k=1)[0]

    def bind(self, binding, entry, update, attempt):
        b = deepcopy(binding)
        t = deepcopy(next(t for t in self.catalog if t['task_id'] == entry['task_id']))
        if t['split'] != 'train' or entry['split'] != 'train' or t['tier'] != binding['task']['tier']:
            raise ValueError('restart changed training split or tier')
        t['episode_record'] = deepcopy(entry['record'])
        t['game_seed'] = entry['record']['environment']['seed']
        b.update(task=t, start='restart', environment_seed=t['game_seed'], restart_id=entry['id'],
                 lookback=entry['lookback'], bin_id=entry['bin_id'])
        b['sampling_seeds'] = [int(digest([update,b['group'],attempt,entry['id'],r])[:8],16) for r in range(12)]
        return b

    def plan(self, update):
        p = super().plan(update); counts = dict(current=0, replay=0); used = []
        for k,b in enumerate(p['groups']):
            kind = 'current' if b['task']['tier'] == p['tier'] else 'replay'
            slot = counts[kind]; counts[kind] += 1
            requested = (update + slot) % 5 != 0
            b.update(start_class=kind, restart_requested=requested)
            if requested:
                entry = self.choose(b['task']['tier'], [update,k,'initial'], used)
                if entry is None:
                    b['restart_fallback'] = 'no_same_tier_state'
                else:
                    p['groups'][k] = self.bind(b, entry, update, 0); used.append(entry['id'])
        p['restart_bank_sha256'] = digest(self.restart_bank.read())
        return p

    def replacement(self, binding, attempt, history, *, update):
        if binding['start'] == 'initial':
            return super().replacement(binding, attempt, history, update=update)
        entry = self.choose(binding['task']['tier'], [update,binding['group'],attempt],
                            [b['restart_id'] for b in history if 'restart_id' in b])
        if entry is None:
            raise ValueError('restart bank disappeared during collection')
        return self.bind(binding, entry, update, attempt)

    def complete(self, update, episodes, plan=None):
        p = deepcopy(plan or self.plan(update))
        policy = self.controller.policy
        # A controller window of three full normal-start batches must not become
        # three 20%-normal batches. Accumulate the equivalent current-Tier sample
        # volume, preserving ordinary fallback behavior and every promotion gate.
        current_groups = sum(g['task']['tier'] == p['tier'] for g in p['groups'])
        target_games = max(policy.min_games, policy.window_batches * current_groups * 12)
        p['groups'] = [g for g in p['groups'] if g['start'] == 'initial']
        normal_games = 12 * sum(g['task']['tier'] == p['tier'] for g in p['groups'])
        window_games = [b['games'] for b in self.controller.state['window']] + [normal_games]
        while (len(window_games) > policy.window_batches
               and sum(window_games[1:]) >= target_games):
            window_games.pop(0)
        self.controller.policy = replace(
            policy, window_batches=max(policy.window_batches, len(window_games)),
            min_games=target_games)
        ids = {g['group'] for g in p['groups']}
        try:
            return self.controller.commit(update=update, plan=p,
                episodes=[e for e in episodes if e['group'] in ids], optimizer_completed=True)
        finally:
            self.controller.policy = policy


def make_sampler(config, root):
    from .controller import TrainingPolicy
    from .reward_filter import configured_policy as filter_policy
    restart = configured_policy(config)
    cls = FormalSampler if restart is None else FailureStateRestartSampler
    extra = {} if restart is None else dict(restart_bank=RestartBank(root, config, restart))
    return cls(config['catalog'], TrainingPolicy(**config['policy']), max_tier=config['max_tier'],
               min_tier=config.get('min_tier',1), seed=config['seed'],
               environment_seed_mode=config.get('environment_seed_mode','group_randomized'),
               reward_group_filter=filter_policy(config), **extra)
