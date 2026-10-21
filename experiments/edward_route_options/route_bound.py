"""Isolated route-bound Edward prototype; no production monkeypatches on disk."""
from collections import Counter
from dataclasses import replace
import hashlib, json
import pacman_env.planner as P
from .experimental_planner import move, multi_routes, pacman_path, short_safe
from .escape_guard import escape_safe

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

class NativeStep1(P.EdwardPlanner):
    """Only execution commitment changes; native generation and ranking retained."""

    def advertised_candidates(self, state):
        return tuple((replace(c, commit_moves=1) for c in super().advertised_candidates(state)))

class BoundPlanner(P.EdwardPlanner):

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
        BoundPlanner.last_instance = self

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

    def continue_option(self, option, state):
        state = normalized(state)
        pos = self.observe(state)
        p = self.active
        assert p is not None
        reason = None
        if [pos.row, pos.col] != p['cells'][self.cursor]:
            reason = 'position_invalidated'
        elif self.cursor >= len(p['identity']['actions']):
            reason = 'target_reached'
        elif self.cursor >= p['commit']:
            reason = p['summary']['boundary']
        else:
            rest = tuple(p['identity']['actions'][self.cursor:])
            self.safety_checks += 1
            if rest[0] not in state['open']:
                reason = 'route_blocked'
            elif not short_safe(self.level, state, rest):
                reason = 'short_risk'
            elif option.strategy == 'COLLECT' and (not escape_safe(self.level, state, rest, P.Position(*option.target))):
                reason = 'escape_risk'
            if reason is None:
                self.events.append(dict(kind='continue', route_id=p['route_id'], cursor=self.cursor))
                return (rest[0], 'active')
        self.events.append(dict(kind='boundary', route_id=p['route_id'], cursor=self.cursor, reason=reason))
        if reason == 'target_reached':
            return (None, 'completed')
        if reason in ('junction', 'portal_exit', 'segment_cap', 'route_end'):
            return (None, 'max_commit')
        return (None, 'invalidated')

def install_prompt(prompts, native):
    original = prompts.render_vision_edward_decision_prompt

    def renderer(state, candidates, constraint, **kwargs):
        text = original(state, candidates, constraint, **kwargs)
        menu = state.get('route_menu')
        if not menu:
            return text
        return text + '\nBound routes (coordinates row,col): ' + json.dumps(menu, separators=(',', ':')) + '\nThe harness follows the chosen route to the stated boundary, checking short-term risk each step. This does not guarantee safety for the full route.'
    prompts.render_vision_edward_decision_prompt = renderer
    native.render_vision_edward_decision_prompt = renderer

def audit_bound(episode, planner):
    actions = [e for e in planner.events if e['kind'] == 'execute']
    assert len(actions) == len(episode['trajectory'])
    for t, e in zip(episode['trajectory'], actions):
        p = planner.registry[e['route_id']]
        i = e['cursor']
        assert t['action'] == p['identity']['actions'][i]
        assert t['observation_context']['pacman_position'] == p['cells'][i]
        assert (i == 0) == t['model_called']
    return dict(passed=True, steps=len(actions), routes=len(planner.registry), safety_checks=planner.safety_checks, continuation_steps=sum((not t['model_called'] for t in episode['trajectory'])))
