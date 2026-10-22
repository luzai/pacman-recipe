"""Optional bounded portal-bait route. Public kinematics, no hidden ghost plan.

Compatibility: native CAE protocol encodes this optional skill under an unused
A0..A3 / AVOID slot; route_menu explicitly identifies LURE_PORTAL. This is a
hypothesis about drawing pursuit, not proof ghosts take bait or global safety.
"""
from copy import deepcopy
from . import route_bound as B
from . import motion_guard as G
from .experimental_planner import ghost_path

def approaching(ghost, bait):
    if ghost.get('state') not in B.P._LETHAL_STATES or ghost.get('speed', 0) <= 0:
        return False
    row, col = ghost['position']
    dr, dc = G.D.get(ghost.get('direction'), (0, 0))
    return dr * (bait.row - row) + dc * (bait.col - col) > 0

def distance(level, ghost, cell):
    path = ghost_path(level, B.P.Position(*ghost['position']), cell)
    return len(path) if path is not None else None

class LurePortalPlanner(G.NoPathPlanner):

    def __init__(self, *args, enable_lure=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lure = enable_lure
        self.lure_cooldown = 0
        self.lure_uses = {}
        self.lure_routes = {}
        self.lure_active = None
        self.advertised_lure = None

    def _portal_edges(self):
        for row in range(self.level.height):
            for col in range(self.level.width):
                src = B.P.Position(row, col)
                if B.P._is_wall(self.level, src, actor='pacman'):
                    continue
                for action in 'UDLR':
                    dst = B.move(self.level, src, action)
                    if dst is not None and abs(dst.row - row) + abs(dst.col - col) > 1:
                        yield (src, action, dst)

    def advertised_candidates(self, state):
        self.advertised_lure = None
        base = super().advertised_candidates(state)
        if any((c.strategy == 'RISK_FALLBACK' for c in base)):
            return base
        if not self.enable_lure or self.lure_cooldown:
            return base
        unused = next((f'A{i}' for i in range(4) if f'A{i}' not in self.menu), None)
        if unused is None:
            return base
        state = G.clean(B.normalized(state))
        start = B.P.Position(*state['pacman_position'])
        choices = []
        for entry, action, exit_cell in self._portal_edges():
            key = (entry.row, entry.col, action)
            if self.lure_uses.get(key, 0) >= 2:
                continue
            threats = [g for g in state.get('ghosts', []) if approaching(g, entry)]
            gains = [(distance(self.level, g, entry), distance(self.level, g, exit_cell)) for g in threats]
            gains = [b - a for a, b in gains if a is not None and b is not None]
            if not gains or min(gains) < 2:
                continue
            prefix = B.pacman_path(self.level, start, entry)
            if prefix is None or len(prefix) > 3:
                continue
            for target in sorted(self.remaining):
                suffix = B.pacman_path(self.level, exit_cell, target)
                if suffix is None:
                    continue
                route = tuple(prefix) + (action,) + tuple(suffix)
                if not route or len(route) > min(12, state.get('remaining_steps', 12)) or route[0] not in state['open']:
                    continue
                payload = B.describe(self.level, start, (target.row, target.col), route, 'AVOID')
                if len(payload['identity']['portals']) != 1 or len(set(map(tuple, payload['cells']))) != len(payload['cells']):
                    continue
                self.safety_checks += 1
                if not G.short_safe(self.level, state, route) or not B.escape_safe(self.level, state, route, target):
                    continue
                choices.append((len(route), -min(gains), route, target, key, len(prefix), min(gains)))
        if not choices:
            return base
        _, _, route, target, key, cross_index, gain = min(choices)
        candidate = B.P.PlannerCandidate(option_id=unused, strategy='AVOID', target=(target.row, target.col), first_action=route[0], route_distance=len(route), commit_moves=len(route))
        candidate = self.register(candidate, state, route)
        rid = self.menu[unused]
        meta = dict(portal_key=key, cross_index=cross_index, target=list(candidate.target), budget=len(route), distance_gain=gain)
        self.lure_routes[rid] = meta
        self.advertised_lure = (unused, rid, deepcopy(meta))
        self.route_menu[unused] = dict(self.route_menu[unused], skill='LURE_PORTAL', stages=['approach', 'portal', 'collect'], ghost_distance_gain=gain, skill_budget=len(route), exit_condition='goal, risk, invalid route, lost approach before crossing, or budget')
        return base + (candidate,)

    def bind(self, option, state):
        super().bind(option, state)
        advertised = self.advertised_lure
        self.lure_active = deepcopy(advertised[2]) if advertised is not None and (option.option_id, self.active['route_id']) == advertised[:2] else None
        if self.lure_active:
            key = self.lure_active['portal_key']
            self.lure_uses[key] = self.lure_uses.get(key, 0) + 1
            self.lure_cooldown = 20
            self.events.append(dict(kind='skill_start', skill='LURE_PORTAL', route_id=self.active['route_id'], budget=self.lure_active['budget']))

    def _end_skill(self, reason):
        self.events.append(dict(kind='boundary', reason=reason, route_id=self.active['route_id'], cursor=self.cursor, skill='LURE_PORTAL', old_route_valid=None, short_term_safe=False))
        self.lure_active = None
        self.lure_cooldown = max(self.lure_cooldown, 20)
        return (None, 'invalidated')

    def continue_option(self, option, state):
        if not self.lure_active:
            return super().continue_option(option, state)
        s = G.clean(B.normalized(state))
        meta = self.lure_active
        action, status = super().continue_option(option, s)
        if action is None:
            self.lure_active = None
            self.lure_cooldown = max(self.lure_cooldown, 20)
            return (action, status)
        if self.cursor >= meta['budget']:
            return self._end_skill('skill_budget')
        pellets = [list(p) for p in s.get('normal_pellet_positions', []) + s.get('power_pellet_positions', [])]
        if meta['target'] not in pellets:
            return self._end_skill('skill_target_completed')
        if self.cursor <= meta['cross_index']:
            entry = B.P.Position(*meta['portal_key'][:2])
            if not any((approaching(g, entry) for g in s.get('ghosts', []))):
                return self._end_skill('skill_approach_lost')
        rest = tuple(self.active['identity']['actions'][self.cursor:])
        self.safety_checks += 1
        if not B.escape_safe(self.level, s, rest, B.P.Position(*meta['target'])):
            return self._end_skill('skill_escape_risk')
        self.events.append(dict(kind='skill_continue', skill='LURE_PORTAL', cursor=self.cursor, stage='approach' if self.cursor < meta['cross_index'] else 'portal' if self.cursor == meta['cross_index'] else 'collect'))
        return (action, status)

    def record_action(self, action):
        super().record_action(action)
        self.lure_cooldown = max(0, self.lure_cooldown - 1)
