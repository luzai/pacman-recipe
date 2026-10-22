"""V3 motion guard: no hidden routes, corridor reversal at future boundaries.

Current kinematics remain environment supplied, not pure vision. A bounded
motion envelope is not a full-game safety guarantee; fallback can be risky.
"""
from contextlib import contextmanager
import math
from . import route_bound as B
from . import escape_guard as E
from .continuity import ContinuousPlanner
FIELDS = ('id', 'state', 'position', 'pixel_position', 'direction', 'speed', 'edible_ticks')
D = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

def clean(state):
    s = dict(state)
    s['ghosts'] = [{k: g[k] for k in FIELDS if k in g} for g in state.get('ghosts', [])]
    return s

def advance(level, branches, speed, allow_turn=True):
    out = set()
    for y, x, d in branches:
        aligned = abs(y % 16) < 1e-08 and abs(x % 16) < 1e-08
        if aligned and allow_turn:
            cell = B.P.Position(round(y / 16), round(x / 16))
            legal = [a for a in D if B.move(level, cell, a, 'ghost') is not None]
            dirs = legal
        else:
            dirs = [d]
        for a in dirs:
            dr, dc = D[a]
            out.add((y + dr * speed, x + dc * speed, a))
    return out

def short_safe(level, state, route, horizon=2):
    pos = B.P.Position(*state['pacman_position'])
    ghosts = []
    for g in state.get('ghosts', []):
        if g['state'] not in B.P._LETHAL_STATES:
            continue
        if not all((k in g for k in ('pixel_position', 'direction', 'speed'))):
            return False
        y, x = map(float, g['pixel_position'])
        v = float(g['speed'])
        d = g['direction']
        if d not in D or not all((math.isfinite(z) for z in (y, x, v))) or v < 0:
            return False
        ghosts.append(({(y, x, d)}, v))
    for step_index, a in enumerate(route[:horizon]):
        pos = B.move(level, pos, a)
        if pos is None:
            return False
        ty, tx = (pos.row * 16, pos.col * 16)
        for i, (branches, v) in enumerate(ghosts):
            for frame in range(16):
                if any((abs(ty - y) < 16 and abs(tx - x) < 16 for y, x, d in branches)):
                    return False
                branches = advance(level, branches, v, allow_turn=not (step_index == 0 and frame == 0))
                if not branches or len(branches) > 512:
                    return False
                if any((abs(ty - y) < 16 and abs(tx - x) < 16 for y, x, d in branches)):
                    return False
            ghosts[i] = (branches, v)
    return True

def one_step(level, state, target):
    s = clean(state)
    pos = B.P.Position(*s['pacman_position'])
    for a in s.get('open', 'UDLR'):
        if B.move(level, pos, a) == target:
            return short_safe(level, s, (a,), 1)
    return False

def fallback(level, state, target):
    if any((not all((k in g for k in ('pixel_position', 'direction', 'speed'))) for g in state.get('ghosts', []) if g['state'] in B.P._LETHAL_STATES)):
        return 'unknown'
    return 'clear_estimate' if one_step(level, state, target) else 'collision_predicted'

@contextmanager
def guard_mode(mode):
    if mode not in ('no_path', 'final_base', 'final_lure'):
        yield
        return
    saved = (B.short_safe, E.short_safe, B.P._one_step_pixel_safe, B.P._fallback_motion_risk)
    B.short_safe = E.short_safe = short_safe
    B.P._one_step_pixel_safe = one_step
    B.P._fallback_motion_risk = fallback
    try:
        yield
    finally:
        B.short_safe, E.short_safe, B.P._one_step_pixel_safe, B.P._fallback_motion_risk = saved

class NoPathPlanner(ContinuousPlanner):

    def advertised_candidates(self, state):
        return super().advertised_candidates(clean(state))

    def continue_option(self, option, state):
        return super().continue_option(option, clean(state))

    def bind(self, option, state):
        return super().bind(option, clean(state))
