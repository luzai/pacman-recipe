"""Offline training-only controller. Explicit policy; no production defaults.

Call commit only after the corresponding optimizer update succeeds. This module
does not implement atomic model/optimizer checkpoint publication.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from .identity import digest


@dataclass(frozen=True)
class TrainingPolicy:
    threshold: float
    window_batches: int
    min_games: int
    consecutive_checks: int
    min_maps: int

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError("invalid threshold")
        if any(type(v) is not int or v < 1 for v in
               (self.window_batches, self.min_games, self.consecutive_checks, self.min_maps)):
            raise ValueError("counts must be positive integers")


class TrainingWinController:
    def __init__(self, policy, *, catalog_identity, max_tier, min_tier=0):
        if type(max_tier) is not int or max_tier < 0:
            raise ValueError("invalid max tier")
        if type(min_tier) is not int or not 0 <= min_tier <= max_tier:
            raise ValueError("invalid min tier")
        self.min_tier = min_tier
        self.policy, self.max_tier = policy, max_tier
        identity_parts = [asdict(policy), catalog_identity, max_tier,
                          "training-win-controller-v2"]
        if min_tier:
            identity_parts += ['min-tier-v1', min_tier]
        self.identity = digest(identity_parts)
        self.state = dict(tier=min_tier, bank_version=0, last_update=-1, streak=0,
                          window=[], events=[])

    def commit(self, *, update, plan, episodes, optimizer_completed):
        """Validate a complete frozen group12 batch before changing any state.

        plan: identity, bank_version, tier, groups (group, task, start,
        sampling_seeds). episodes: group, replica [0..11], task_id, won bool.
        Callers must separately audit trajectory/source/served weight identity.
        """
        s = self.state
        if optimizer_completed is not True or update != s['last_update'] + 1:
            raise ValueError("require next successfully completed update")
        if (plan['identity'] != self.identity or plan['bank_version'] != s['bank_version']
                or plan['tier'] != s['tier'] or plan['update'] != update):
            raise ValueError("stale or mismatched frozen bank")
        groups = {}
        for g in plan['groups']:
            task = g['task']
            sha = task.get('layout_sha256', '')
            if (g['group'] in groups or g['start'] != 'initial'
                    or task['split'] != 'train' or not self.min_tier <= task['tier'] <= s['tier']
                    or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                    or len(g['sampling_seeds']) != 12
                    or len(set(g['sampling_seeds'])) != 12):
                raise ValueError("invalid frozen training group")
            groups[g['group']] = g
        if not groups:
            raise ValueError("empty batch")
        seen, wins, games = set(), 0, 0
        for e in episodes:
            key = e['group'], e['replica']
            g = groups.get(e['group'])
            if (g is None or key in seen or type(e['replica']) is not int
                    or not 0 <= e['replica'] < 12 or type(e['won']) is not bool
                    or e['task_id'] != g['task']['task_id']):
                raise ValueError("invalid or duplicate episode")
            seen.add(key)
            if g['task']['tier'] == s['tier']:
                games += 1
                wins += int(e['won'])
        if len(seen) != len(groups) * 12:
            raise ValueError("incomplete group12 batch")
        new = deepcopy(s)
        layouts = sorted({g['task']['layout_sha256'] for g in groups.values()
                          if g['task']['tier'] == s['tier']})
        new['window'] = (new['window'] + [dict(wins=wins, games=games, layouts=layouts)])[-self.policy.window_batches:]
        total = sum(b['games'] for b in new['window'])
        map_count = len({sha for b in new['window'] for sha in b['layouts']})
        ready = (len(new['window']) == self.policy.window_batches
                 and total >= self.policy.min_games and map_count >= self.policy.min_maps)
        rate = sum(b['wins'] for b in new['window']) / total if total else None
        passed = ready and rate >= self.policy.threshold
        new['streak'] = new['streak'] + 1 if passed else 0
        before = new['tier']
        if new['streak'] >= self.policy.consecutive_checks and before < self.max_tier:
            new['tier'] += 1
            new['bank_version'] += 1
            new['streak'], new['window'] = 0, []
        event = dict(update=update, games=total, maps=map_count, rate=rate, ready=ready,
                     passed=passed, tier_before=before, tier_after=new['tier'],
                     bank_version=new['bank_version'])
        new['events'].append(event)
        new['last_update'] = update
        self.state = new
        return event

    def snapshot(self):
        return deepcopy(dict(identity=self.identity, state=self.state))

    def restore(self, payload, *, completed_update, bank_version):
        if (payload['identity'] != self.identity
                or not self.min_tier <= payload['state']['tier'] <= self.max_tier
                or payload['state']['last_update'] != completed_update
                or payload['state']['bank_version'] != bank_version):
            raise ValueError("checkpoint/controller mismatch")
        self.state = deepcopy(payload['state'])
