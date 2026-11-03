"""Normal-start-only promotion for the separate engine-assisted density course."""
from collections import defaultdict
import math


def normal_start_counts(episodes, *, groups_per_batch, samples_per_group):
    if len(episodes) != groups_per_batch * samples_per_group:
        raise ValueError('all original attempts required')
    groups = defaultdict(list)
    for row in episodes:
        if row['life_mode'] != 'single_death' or row['split'] != 'train':
            raise ValueError('single-life training only')
        groups[row['group']].append(row)
    if set(groups) != set(range(groups_per_batch)):
        raise ValueError('complete original groups required')
    games, wins, maps = 0, 0, set()
    for rows in groups.values():
        if len(rows) != samples_per_group or {row['sample'] for row in rows} != set(range(samples_per_group)):
            raise ValueError('complete original samples required')
        for key in ('start_class', 'topology_id', 'tier'):
            if len({row[key] for row in rows}) != 1:
                raise ValueError('group crosses ' + key)
        if any(type(row['fault']) is not bool for row in rows):
            raise ValueError('typed fault facts required')
        if any(row['fault'] for row in rows):
            if any(row['reward'] is not None for row in rows if row['fault']):
                raise ValueError('fault reward must be null')
            continue
        if not all(row['native_complete'] is True for row in rows):
            raise ValueError('incomplete nonfault group')
        if rows[0]['start_class'] == 'restart':
            continue
        if rows[0]['start_class'] != 'normal' or any(type(row['won']) is not bool for row in rows):
            raise ValueError('normal/restart class and boolean outcome required')
        games += samples_per_group
        wins += sum(row['won'] for row in rows)
        maps.add(rows[0]['topology_id'])
    return dict(games=games, wins=wins, maps=sorted(maps))


class PromotionWindow:
    """Explicit gates; callers supply the course and sampling contract.

    This is not the ASCII T1--T7 controller. Historical engine experiments used
    density tiers (0.6, 0.8, 1.0), a three-batch/384-game/eight-map window and
    two consecutive passes. These values are intentionally not global defaults.
    """
    def __init__(self, *, groups_per_batch, samples_per_group, window_batches,
                 minimum_games, minimum_maps, success_threshold, consecutive_passes, max_tier):
        values = (groups_per_batch, samples_per_group, window_batches, minimum_games, minimum_maps, consecutive_passes)
        if any(type(value) is not int or value <= 0 for value in values) or type(max_tier) is not int or max_tier < 0:
            raise ValueError('explicit positive window/sampling bounds required')
        if type(success_threshold) not in (int, float) or not math.isfinite(success_threshold) or not 0 <= success_threshold <= 1:
            raise ValueError('finite success threshold required')
        self.groups_per_batch, self.samples_per_group = groups_per_batch, samples_per_group
        self.window_batches, self.minimum_games, self.minimum_maps = window_batches, minimum_games, minimum_maps
        self.success_threshold, self.required_passes, self.max_tier = success_threshold, consecutive_passes, max_tier
        self.tier, self.window, self.consecutive_passes = 0, [], 0

    def record(self, episodes):
        if any(row['tier'] != self.tier for row in episodes):
            raise ValueError('promotion must use current-tier attempts')
        counts = normal_start_counts(episodes, groups_per_batch=self.groups_per_batch, samples_per_group=self.samples_per_group)
        self.window = (self.window + [counts])[-self.window_batches:]
        games, wins = (sum(row[key] for row in self.window) for key in ('games', 'wins'))
        maps = {value for row in self.window for value in row['maps']}
        ready = len(self.window) == self.window_batches and games >= self.minimum_games and len(maps) >= self.minimum_maps
        passed = ready and wins / games >= self.success_threshold
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        promoted = self.consecutive_passes >= self.required_passes and self.tier < self.max_tier
        old = self.tier
        if promoted:
            self.tier += 1
            self.window, self.consecutive_passes = [], 0
        return dict(tier_before=old, tier_after=self.tier, games=games, wins=wins,
                    maps=len(maps), ready=ready, passed=passed, promoted=promoted)
