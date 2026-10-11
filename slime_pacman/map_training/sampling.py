"""Deterministic fixed-quota sampling; state changes only after optimizer success."""
from copy import deepcopy
import hashlib
from .identity import digest
from .controller import TrainingWinController


class FormalSampler:
    def __init__(self, catalog, policy, *, max_tier, seed=20261009):
        self.catalog = deepcopy(catalog)
        self.seed = seed
        ids = [r['task_id'] for r in catalog]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate task id')
        splits = {}
        for r in catalog:
            sha = r['layout_sha256']
            if sha in splits and splits[sha] != r['split']:
                raise ValueError('layout split leakage')
            splits[sha] = r['split']
        equivalents = {}
        for row in catalog:
            equivalent = row.get('equivalence_sha256', row['layout_sha256'])
            if equivalent in equivalents and equivalents[equivalent] != row['split']:
                raise ValueError('equivalent layout split leakage')
            equivalents[equivalent] = row['split']
        self.by_tier = {}
        for tier in range(max_tier + 1):
            rows = sorted((r for r in catalog if r['split'] == 'train' and r['tier'] == tier),
                          key=lambda r: (r.get('game_seed', 0), r['task_id']))
            if len({r['layout_sha256'] for r in rows}) < 8:
                raise ValueError('each tier requires at least eight train layouts')
            self.by_tier[tier] = rows
        self.controller = TrainingWinController(policy, catalog_identity=digest([catalog, seed]),
                                                max_tier=max_tier)

    def plan(self, update):
        state = self.controller.state
        if update != state['last_update'] + 1:
            raise ValueError('plan must follow last completed optimizer update')
        tier = state['tier']
        current_count = 16 if tier == 0 else 13
        current = self.by_tier[tier]
        # Deterministic rotation; banks smaller than the group quota wrap around.
        selected = [current[(update * current_count + i) % len(current)]
                    for i in range(current_count)]
        if tier:
            earlier = [r for t in range(tier) for r in self.by_tier[t]]
            selected += [earlier[(update * 3 + i) % len(earlier)] for i in range(3)]
        groups = []
        for group, task in enumerate(selected):
            seeds = [int.from_bytes(hashlib.sha256(
                f'{self.seed}:{update}:{group}:{replica}:{task["task_id"]}'.encode()).digest()[:4], 'big')
                for replica in range(12)]
            if len(set(seeds)) != 12:
                raise ValueError('sampling seed collision')
            groups.append(dict(group=group, task=deepcopy(task), start='initial', sampling_seeds=seeds))
        return dict(identity=self.controller.identity, bank_version=state['bank_version'],
                    tier=tier, update=update, groups=groups)

    def complete(self, update, episodes, plan=None):
        return self.controller.commit(update=update, plan=plan or self.plan(update),
                                      episodes=episodes, optimizer_completed=True)

    def snapshot(self):
        return self.controller.snapshot()

    def restore(self, payload, *, completed_update, bank_version):
        self.controller.restore(payload, completed_update=completed_update, bank_version=bank_version)
