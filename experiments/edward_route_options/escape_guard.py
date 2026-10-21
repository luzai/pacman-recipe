from collections import deque
import pacman_env.planner as P
from .experimental_planner import move, short_safe

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
