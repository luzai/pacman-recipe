"""Opt-in CPU prototype. Uses observations only; never reads simulator/RNG state."""
from collections import Counter, deque
import pacman_env.planner as P

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

def short_safe(level, state, route, horizon=2):
    """Conservative 16-frame-per-action pixel projection, at most two actions.
    Follow advertised ghost paths. If a path expires, branch over legal ghost
    moves (no teleport). Reject if any predicted branch collides. Not an oracle.
    """
    pos = P.Position(*state['pacman_position'])
    ghosts = []
    for g in state.get('ghosts', []):
        if g['state'] not in P._LETHAL_STATES:
            continue
        if not all((k in g for k in ['pixel_position', 'direction', 'path_remaining', 'speed'])):
            return False
        y, x = g['pixel_position']
        ghosts.append(({(float(y), float(x), g['direction'], g['path_remaining'])}, float(g['speed'])))
    for a in route[:horizon]:
        pos = move(level, pos, a)
        if pos is None:
            return False
        ty, tx = (pos.row * 16, pos.col * 16)
        for idx, (branches, speed) in enumerate(ghosts):
            for frame in range(16):
                out = set()
                for y, x, d, path in branches:
                    if abs(ty - y) < 16 and abs(tx - x) < 16:
                        return False
                    dr, dc = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}.get(d, (0, 0))
                    yy, xx = (y + dr * speed, x + dc * speed)
                    if abs(ty - yy) < 16 and abs(tx - xx) < 16:
                        return False
                    if abs(yy % 16) < 1e-08 and abs(xx % 16) < 1e-08:
                        rest = path[1:] if path else ''
                        if rest:
                            out.add((yy, xx, rest[0], rest))
                        else:
                            cell = P.Position(int(yy / 16), int(xx / 16))
                            for nd in 'UDLR':
                                if move(level, cell, nd, 'ghost') is not None:
                                    out.add((yy, xx, nd, ''))
                    else:
                        out.add((yy, xx, d, path))
                branches = out
                if len(branches) > 512 or not branches:
                    return False
            ghosts[idx] = (branches, speed)
    return True
