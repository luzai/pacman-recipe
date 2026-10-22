"""Production Edward route continuity with public-kinematics safety.

No module patching, no hidden ghost path and no LURE. Safety is a bounded
motion envelope, not a guarantee; risk-ranked fallback can still lose lives.
"""
from collections import Counter, deque
from dataclasses import replace
import hashlib
import json
import math
from . import planner as P


FIELDS = ('id','state','position','pixel_position','direction','speed','edible_ticks')
D = {'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}

def pacman_path(level, start, target):
    return P._shortest_path(level, start, target, actor='pacman', use_portals=True)

def ghost_path(level, start, target):
    return P._shortest_path(level, start, target, actor='ghost', use_portals=False)

def move(level, pos, a, actor='pacman'):
    return P._transition(level, pos, P._ACTION_BY_TOKEN[a], actor=actor, use_portals=actor == 'pacman')

def multi_routes(level, start, target):
    """One shortest continuation per first direction, with no immediate return.
    Plus local junction waypoints; all transitions retain native portals.
    """
    routes = set()
    direct = pacman_path(level, start, target)
    if direct:
        routes.add(tuple(direct))
    for first in 'UDLR':
        nxt = move(level, start, first)
        if nxt is None:
            continue
        q = deque([(nxt, (first,))])
        seen = {start, nxt}
        while q:
            pos, route = q.popleft()
            if pos == target:
                routes.add(route)
                break
            for a in 'UDLR':
                dst = move(level, pos, a)
                if dst is not None and dst not in seen:
                    seen.add(dst)
                    q.append((dst, route + (a,)))
    q = deque([(start, ())])
    seen = {start}
    while q:
        pos, prefix = q.popleft()
        if prefix and P._degree(level, pos) >= 3:
            suffix = pacman_path(level, pos, target)
            if suffix:
                route = prefix + tuple(suffix)
                positions = P._route_positions(level, start, route)
                if start not in positions and len(set(positions)) == len(positions):
                    routes.add(route)
        if len(prefix) == 4:
            continue
        for a in 'UDLR':
            dst = move(level, pos, a)
            if dst is not None and dst not in seen:
                seen.add(dst)
                q.append((dst, prefix + (a,)))
    return sorted((r for r in routes if len(r) <= len(direct or ()) + 8), key=lambda r: (len(r), r))[:12]

def clean(state):
    s = dict(state)
    s['ghosts'] = [{k: g[k] for k in FIELDS if k in g} for g in state.get('ghosts', [])]
    return s

def advance(level, branches, speed, allow_turn=True):
    out = set()
    for y, x, d in branches:
        aligned = abs(y % 16) < 1e-08 and abs(x % 16) < 1e-08
        if aligned and allow_turn:
            cell = P.Position(round(y / 16), round(x / 16))
            legal = [a for a in D if move(level, cell, a, 'ghost') is not None]
            dirs = legal
        else:
            dirs = [d]
        for a in dirs:
            dr, dc = D[a]
            out.add((y + dr * speed, x + dc * speed, a))
    return out

def short_safe(level, state, route, horizon=2):
    pos = P.Position(*state['pacman_position'])
    ghosts = []
    for g in state.get('ghosts', []):
        vulnerable = g['state'] == 'vulnerable'
        if g['state'] not in P._LETHAL_STATES and not vulnerable:
            continue
        # Unknown expiry is treated conservatively. Never freeze a vulnerable
        # ghost out of the prediction when it can become lethal mid-action.
        expiry = g.get('edible_ticks', state.get('edible_ticks', 0)) if vulnerable else 0
        if type(expiry) not in (int, float) or not math.isfinite(expiry):
            return False
        expiry = max(0, expiry)
        if not all((k in g for k in ('pixel_position', 'direction', 'speed'))):
            return False
        y, x = map(float, g['pixel_position'])
        v = float(g['speed'])
        d = g['direction']
        if d not in D or not all((math.isfinite(z) for z in (y, x, v))) or v < 0:
            return False
        # Native expiry changes state only, not speed. Timer is decremented
        # before ghost movement/collision; each grid action advances 16 frames.
        ghosts.append(({(y, x, d)}, v, expiry))
    for step_index, a in enumerate(route[:horizon]):
        pos = move(level, pos, a)
        if pos is None:
            return False
        ty, tx = (pos.row * 16, pos.col * 16)
        for i, (branches, v, expiry) in enumerate(ghosts):
            for frame in range(16):
                elapsed = step_index * 16 + frame
                if elapsed >= expiry and any((abs(ty - y) < 16 and abs(tx - x) < 16 for y, x, d in branches)):
                    return False
                branches = advance(level, branches, v, allow_turn=not (step_index == 0 and frame == 0))
                if not branches or len(branches) > 512:
                    return False
                if elapsed + 1 >= expiry and any((abs(ty - y) < 16 and abs(tx - x) < 16 for y, x, d in branches)):
                    return False
            ghosts[i] = (branches, v, expiry)
    return True

def one_step(level, state, target):
    s = clean(state)
    pos = P.Position(*s['pacman_position'])
    for a in s.get('open', 'UDLR'):
        if move(level, pos, a) == target:
            return short_safe(level, s, (a,), 1)
    return False

def fallback(level, state, target):
    if any((not all((k in g for k in ('pixel_position', 'direction', 'speed'))) for g in state.get('ghosts', []) if g['state'] in P._LETHAL_STATES)):
        return 'unknown'
    return 'clear_estimate' if one_step(level, state, target) else 'collision_predicted'

def escape_safe(level, state, route, target):
    """Receding-horizon guard: audit exit when pellet is within 2 moves.
    Predict approach plus bounded escape to a junction, preserving portals.
    Observed ghost motion only, never simulator RNG. Not a long-term guarantee.
    """
    if len(route) > 2:
        return True
    pos = P.Position(*state['pacman_position'])
    normal = {P.Position(*x) for x in state['normal_pellet_positions']}
    for a in route:
        pos = move(level, pos, a)
        if pos is None:
            return False
        normal.discard(pos)
    if not normal:
        return True
    assert pos == target
    q = deque([(pos, (), frozenset([pos]))])
    while q:
        cell, tail, seen = q.popleft()
        if tail and P._degree(level, cell) >= 3:
            whole = tuple(route) + tail
            if short_safe(level, state, whole, horizon=len(whole)):
                return True
        if len(tail) >= 4:
            continue
        for a in 'UDLR':
            dst = move(level, cell, a)
            if dst is not None and dst not in seen:
                q.append((dst, tail + (a,), seen | {dst}))
    return False

def normalized(state):
    state = dict(state)
    state.setdefault('pacman_position', [state.get('row'), state.get('col')])
    return state

def describe(level, start, target, route, strategy):
    cells = [start]
    portals = []
    junctions = []
    turns = []
    for i, a in enumerate(route):
        src = cells[-1]
        dst = move(level, src, a)
        assert dst is not None
        if P._degree(level, src) >= 3:
            junctions.append([i, [src.row, src.col], a])
        if i and a != route[i - 1]:
            turns.append([i, [src.row, src.col], a])
        if abs(dst.row - src.row) + abs(dst.col - src.col) > 1:
            portals.append(dict(index=i, entry=[src.row, src.col], exit=[dst.row, dst.col], action=a))
        cells.append(dst)
    identity = dict(start=[start.row, start.col], target=list(target), actions=list(route), portals=portals, strategy=strategy)
    route_id = 'r' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cap = min(4, len(route))
    boundary = 'segment_cap' if len(route) > 4 else 'route_end'
    for i, dst in enumerate(cells[1:cap + 1], 1):
        if any((p['index'] == i - 1 for p in portals)):
            cap = i
            boundary = 'portal_exit'
            break
        if P._degree(level, dst) >= 3:
            cap = i
            boundary = 'junction'
            break
    summary = dict(route_id=route_id, key_junctions=junctions, turns=turns, portals=portals, approach_from={'U': 'south', 'D': 'north', 'L': 'east', 'R': 'west'}[route[-1]], approach_cell=[cells[-2].row, cells[-2].col], segment_moves=cap, boundary=boundary)
    return dict(route_id=route_id, identity=identity, cells=[[p.row, p.col] for p in cells], summary=summary, commit=cap)

class _RoutePlanner(P.EdwardPlanner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visits = Counter()
        self.registry = {}
        self.menu = {}
        self.route_menu = {}
        self.active = None
        self.cursor = 0
        self.events = []
        self.safety_checks = 0

    def observe(self, state):
        p = super().observe(state)
        if getattr(self, 'previous_cell', None) != p:
            self.visits[p] += 1
        self.previous_cell = p
        return p

    def register(self, candidate, state, route):
        start = P.Position(*state['pacman_position'])
        payload = describe(self.level, start, candidate.target, tuple(route), candidate.strategy)
        rid = payload['route_id']
        if rid in self.registry:
            assert self.registry[rid] == payload
        self.registry[rid] = payload
        self.menu[candidate.option_id] = rid
        self.route_menu[candidate.option_id] = payload['summary']
        return replace(candidate, commit_moves=payload['commit'])

    def advertised_candidates(self, state):
        state = normalized(state)
        base = super().advertised_candidates(state)
        pos = P.Position(*state['pacman_position'])
        choices = []
        for target in sorted(self.remaining):
            for route in multi_routes(self.level, pos, target):
                if route[0] not in state['open']:
                    continue
                self.safety_checks += 1
                if short_safe(self.level, state, route) and escape_safe(self.level, state, route, target):
                    choices.append((len(route) + 3 * self.visits[move(self.level, pos, route[0])], route, target))
        ordered = sorted(choices)
        selected = []
        seen = set()
        for item in ordered:
            key = (item[2], item[1][0])
            if key not in seen:
                selected.append(item)
                seen.add(key)
            if len(selected) == 4:
                break
        for item in ordered:
            if len(selected) >= 4:
                break
            if item not in selected:
                selected.append(item)
        pairs = []
        for i, (_, route, target) in enumerate(selected):
            c = P.PlannerCandidate(option_id=f'C{i}', strategy='COLLECT', target=(target.row, target.col), first_action=route[0], route_distance=len(route), commit_moves=1)
            pairs.append((c, route))
        for c in base:
            if c.strategy not in ('AVOID', 'ELIMINATE'):
                continue
            nxt = move(self.level, pos, c.first_action)
            suffix = pacman_path(self.level, nxt, P.Position(*c.target)) if nxt is not None else None
            if suffix is None:
                continue
            route = (c.first_action,) + tuple(suffix)
            if short_safe(self.level, state, route):
                pairs.append((c, route))
        pairs.sort(key=lambda p: ({'ELIMINATE': 0, 'COLLECT': 1, 'AVOID': 2}.get(p[0].strategy, 3), p[0].option_id))
        self.menu = {}
        self.route_menu = {}
        if not pairs:
            lethal = tuple((P._GhostThreat(g.get('id'), P.Position(*g['position'])) for g in state.get('ghosts', []) if g['state'] in P._LETHAL_STATES))
            fallback = self._risk_fallback_candidates(state, pos, lethal)
            return tuple((self.register(c, state, (c.first_action,)) for c in fallback))
        return tuple((self.register(c, state, route) for c, route in pairs))

    def bind(self, option, state):
        state = normalized(state)
        rid = self.menu[option.option_id]
        payload = self.registry[rid]
        assert payload['cells'][0] == state['pacman_position']
        assert payload['identity']['actions'][0] == option.first_action
        switch = self.active is not None and self.cursor < len(self.active['identity']['actions']) and (self.active['identity']['target'] != payload['identity']['target'] or self.active['identity']['actions'][self.cursor:] != payload['identity']['actions'])
        self.active = payload
        self.cursor = 0
        self.events.append(dict(kind='select', route_id=rid, option_id=option.option_id, route_switch=switch))

    def before_action(self, action):
        assert self.active is not None
        assert action == self.active['identity']['actions'][self.cursor], 'executed action differs from bound route'
        self.events.append(dict(kind='execute', route_id=self.active['route_id'], cursor=self.cursor, action=action))

    def record_action(self, action):
        super().record_action(action)
        if self.active is not None:
            self.cursor += 1

class _ContinuousPlanner(_RoutePlanner):

    def register(self, candidate, state, route):
        p = describe(self.level, P.Position(*state['pacman_position']), candidate.target, tuple(route), candidate.strategy)
        assert 0 < len(route) <= 512
        p['commit'] = len(route)
        p['summary'] = dict(p['summary'], segment_moves=len(route), boundary='route_end')
        rid = p['route_id']
        if rid in self.registry:
            assert self.registry[rid] == p
        self.registry[rid] = p
        self.menu[candidate.option_id] = rid
        self.route_menu[candidate.option_id] = p['summary']
        return replace(candidate, commit_moves=len(route))

    def bind(self, option, state):
        super().bind(option, state)
        self.last_checked_cursor = -1
        self.bound_lives = state.get('lives')
        self.events[-1]['full_route_moves'] = len(self.active['identity']['actions'])

    def continue_option(self, option, state):
        state = normalized(state)
        pos = self.observe(state)
        p = self.active
        reason = None
        valid = False
        safe = None
        rest = ()
        if p is None:
            reason = 'missing_route'
        elif not 0 <= self.cursor <= len(p['identity']['actions']):
            reason = 'invalid_cursor'
        elif list(option.target) != p['identity']['target'] or option.strategy != p['identity']['strategy']:
            reason = 'target_changed'
        elif [pos.row, pos.col] != p['cells'][self.cursor]:
            reason = 'position_invalidated'
        elif self.cursor >= len(p['identity']['actions']):
            reason = 'route_complete'
        elif self.bound_lives is not None and state.get('lives') != self.bound_lives:
            reason = 'life_changed'
        elif option.strategy == 'COLLECT' and list(option.target) not in [list(x) for x in state.get('normal_pellet_positions', []) + state.get('power_pellet_positions', [])]:
            reason = 'target_completed'
        else:
            rest = tuple(p['identity']['actions'][self.cursor:])
            cell = pos
            valid = True
            for offset, action in enumerate(rest, 1):
                cell = move(self.level, cell, action)
                if cell is None or [cell.row, cell.col] != p['cells'][self.cursor + offset]:
                    valid = False
                    reason = 'route_invalidated'
                    break
            if reason is None and rest[0] not in state['open']:
                valid = False
                reason = 'route_blocked'
            if reason is None:
                self.safety_checks += 1
                if not short_safe(self.level, state, rest):
                    safe = False
                    reason = 'short_risk'
                elif option.strategy == 'COLLECT' and (not escape_safe(self.level, state, rest, P.Position(*option.target))):
                    safe = False
                    reason = 'escape_risk'
                else:
                    safe = True
        event = dict(kind='continue' if reason is None else 'boundary', route_id=p['route_id'] if p else None, cursor=self.cursor, reason=reason, old_route_valid=valid, short_term_safe=safe, at_junction=P._degree(self.level, pos) >= 3 if 0 <= pos.row < self.level.height and 0 <= pos.col < self.level.width else False, next_action=rest[0] if rest else None)
        self.events.append(event)
        if reason is None:
            return (rest[0], 'active')
        if reason in ('route_complete', 'target_completed'):
            return (None, 'completed')
        return (None, 'invalidated')


clean_state = clean

def fallback_escape_route(level, state, first_action, horizon=4):
    """Find a bounded, non-repeating escape witness, never a safety guarantee.

    Evaluate the entire prefix from the original observation so ghost time is
    not reset at each search node. Returning from a dead end is allowed once.
    """
    start = P.Position(*state['pacman_position'])
    dst = move(level, start, first_action)
    if dst is None:
        return None
    queue = deque([(dst, (first_action,), frozenset([dst]))])
    while queue:
        cell, route, seen = queue.popleft()
        if not short_safe(level, state, route, horizon=len(route)):
            continue
        if len(route) == horizon:
            return route
        for action in D:
            nxt = move(level, cell, action)
            if nxt is not None and nxt not in seen:
                queue.append((nxt, route + (action,), seen | {nxt}))
    return None


class LatestEdwardPlanner(_ContinuousPlanner):
    """Route-continuous Edward; all safety dispatch stays instance-local."""
    clean_state = staticmethod(clean_state)

    def __init__(self, level=None, *, fallback_mode='risk_ranked'):
        super().__init__(level, fallback_mode=fallback_mode)

    def reset(self):
        super().reset()
        self.visits = Counter()
        self.registry = {}
        self.menu = {}
        self.route_menu = {}
        self.active = None
        self.cursor = 0
        self.events = []
        self.safety_checks = 0
        self.previous_cell = None

    def _one_step_safe(self, state, target):
        return one_step(self.level, state, target)

    def _motion_risk(self, state, target):
        return fallback(self.level, state, target)

    def _risk_fallback_candidates(self, state, player, lethal):
        candidates = super()._risk_fallback_candidates(state, player, lethal)
        result = []
        for candidate in candidates:
            witness = fallback_escape_route(self.level, state, candidate.first_action)
            risk = dict(candidate.risk)
            risk.update(escape_horizon=4, escape_route=list(witness) if witness else None,
                        escape_found=witness is not None)
            result.append(replace(candidate, risk=risk))
        # Information only: never filter, re-rank, rename, or force an action.
        self.events.append(dict(kind='fallback_escape_audit', information_only=True,
                                candidates=[dict(action=c.first_action, **c.risk) for c in result]))
        return tuple(result)

    def candidates(self, state):
        return super().candidates(clean_state(state))

    def advertised_candidates(self, state):
        return super().advertised_candidates(clean_state(state))

    def bind(self, option, state):
        return super().bind(option, clean_state(state))

    def continue_option(self, option, state):
        return super().continue_option(option, clean_state(state))
