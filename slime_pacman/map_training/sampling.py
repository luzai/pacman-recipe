"""Deterministic fixed-quota sampling; state changes only after optimizer success."""
from copy import deepcopy
import hashlib
from .identity import digest
from .controller import TrainingWinController


class FormalSampler:
    def __init__(self, catalog, policy, *, max_tier, min_tier=1, seed=20261009,
                 environment_seed_mode='group_randomized', reward_group_filter=None):
        from .reward_filter import validate_policy
        reward_group_filter = validate_policy(reward_group_filter)
        if environment_seed_mode not in ('group_randomized', 'fixed_catalog'):
            raise ValueError('invalid environment seed mode')
        self.catalog = deepcopy(catalog)
        if type(min_tier) is not int or not 0 <= min_tier <= max_tier:
            raise ValueError('invalid min tier')
        self.min_tier = min_tier
        self.seed = seed
        self.environment_seed_mode = environment_seed_mode
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
        for tier in range(min_tier, max_tier + 1):
            rows = sorted((r for r in catalog if r['split'] == 'train' and r['tier'] == tier),
                          key=lambda r: (r.get('game_seed', 0), r['task_id']))
            if len({r['layout_sha256'] for r in rows}) < 8:
                raise ValueError('each tier requires at least eight train layouts')
            self.by_tier[tier] = rows
        identity_parts = [catalog, seed]
        if reward_group_filter is not None:
            identity_parts += ['final-reward-filter-v1', reward_group_filter]
        if environment_seed_mode != 'fixed_catalog':
            identity_parts += ['group-environment-seed-v1', environment_seed_mode]
        self.controller = TrainingWinController(policy, catalog_identity=digest(identity_parts),
                                                max_tier=max_tier, min_tier=min_tier)

    def plan(self, update):
        state = self.controller.state
        if update != state['last_update'] + 1:
            raise ValueError('plan must follow last completed optimizer update')
        tier = state['tier']
        current_count = 16 if tier == self.min_tier else 13
        current = self.by_tier[tier]
        # Deterministic rotation; banks smaller than the group quota wrap around.
        selected = [current[(update * current_count + i) % len(current)]
                    for i in range(current_count)]
        if tier > self.min_tier:
            earlier = [r for t in range(self.min_tier, tier) for r in self.by_tier[t]]
            selected += [earlier[(update * 3 + i) % len(earlier)] for i in range(3)]
        groups = []
        environment_seeds = set()
        for group, task in enumerate(selected):
            task = deepcopy(task)
            if self.environment_seed_mode == 'group_randomized':
                # One environment per GRPO group; diversity belongs between groups.
                nonce = 0
                while True:
                    environment_seed = int.from_bytes(hashlib.sha256(
                        f'{self.seed}:environment:{update}:{group}:{task["task_id"]}:{nonce}'
                        .encode()).digest()[:4], 'big')
                    if (environment_seed not in environment_seeds
                            and environment_seed != task.get('game_seed')):
                        break
                    nonce += 1
                environment_seeds.add(environment_seed)
                task['game_seed'] = environment_seed
                if 'episode_record' in task:
                    task['episode_record']['environment']['seed'] = environment_seed
            seeds = [int.from_bytes(hashlib.sha256(
                f'{self.seed}:{update}:{group}:{replica}:{task["task_id"]}'.encode()).digest()[:4], 'big')
                for replica in range(12)]
            if len(set(seeds)) != 12:
                raise ValueError('sampling seed collision')
            binding = dict(group=group, task=task, start='initial', sampling_seeds=seeds)
            if self.environment_seed_mode == 'group_randomized':
                binding['environment_seed'] = task['game_seed']
            groups.append(binding)
        return dict(identity=self.controller.identity, bank_version=state['bank_version'],
                    tier=tier, update=update, groups=groups)

    def replacement(self, binding, attempt, history, *, update):
        """Same-tier fresh layout when available; never remove tasks from the bank.

        Exhausted layout pools cycle deterministically with fresh environment seeds.
        The controller, catalog and future first-pass plans remain untouched.
        """
        tier = binding['task']['tier']
        pool = self.by_tier[tier]
        seen = {b['task']['layout_sha256'] for b in history if b['task']['tier'] == tier}
        previous = next(b for b in reversed(history) if b['group'] == binding['group'])
        candidates = [t for t in pool if t['layout_sha256'] not in seen]
        if not candidates:
            candidates = [t for t in pool
                          if t['layout_sha256'] != previous['task']['layout_sha256']]
        domain = f'{self.seed}:replacement-v2:{update}:{binding["group"]}:{attempt}'
        def seed(label):
            return int.from_bytes(hashlib.sha256(f'{domain}:{label}'.encode()).digest()[:4], 'big')
        task = deepcopy(candidates[seed('task') % len(candidates)])
        if self.environment_seed_mode == 'group_randomized':
            used = {b['task'].get('game_seed') for b in history}
            nonce = 0
            environment_seed = seed(f'environment:{nonce}')
            while environment_seed in used or environment_seed == task.get('game_seed'):
                nonce += 1
                environment_seed = seed(f'environment:{nonce}')
            task['game_seed'] = environment_seed
            if 'episode_record' in task:
                task['episode_record']['environment']['seed'] = environment_seed
        result = dict(group=binding['group'], task=task, start='initial',
                      sampling_seeds=[seed(f'policy:{i}') for i in range(12)])
        if self.environment_seed_mode == 'group_randomized':
            result['environment_seed'] = task['game_seed']
        return result

    def complete(self, update, episodes, plan=None):
        return self.controller.commit(update=update, plan=plan or self.plan(update),
                                      episodes=episodes, optimizer_completed=True)

    def snapshot(self):
        return self.controller.snapshot()

    def restore(self, payload, *, completed_update, bank_version):
        self.controller.restore(payload, completed_update=completed_update, bank_version=bank_version)
