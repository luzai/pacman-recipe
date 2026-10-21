from dataclasses import replace
from collections import Counter
import pacman_env.planner as P
from .experimental_planner import multi_routes, short_safe, move
OriginalPlanner = P.EdwardPlanner

class RepairedPlanner(OriginalPlanner):
    """Experimental candidates only. No LURE and no simulator/RNG access."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.visits = Counter()

    def observe(self, state):
        pos = super().observe(state)
        if getattr(self, 'previous_cell', None) != pos:
            self.visits[pos] += 1
        self.previous_cell = pos
        return pos

    def advertised_candidates(self, state):
        state = dict(state)
        state.setdefault('pacman_position', [state.get('row'), state.get('col')])
        base = super().advertised_candidates(state)
        pos = P.Position(*state['pacman_position'])
        choices = []
        for target in sorted(self.remaining):
            for route in multi_routes(self.level, pos, target):
                if route[0] in state['open'] and short_safe(self.level, state, route):
                    score = len(route) + 3 * self.visits[move(self.level, pos, route[0])]
                    choices.append((score, route, target))
        chosen = []
        seen = set()
        for score, route, target in sorted(choices):
            key = (target, route[0])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(P.PlannerCandidate(option_id='', strategy='COLLECT', target=(target.row, target.col), first_action=route[0], route_distance=len(route), commit_moves=1))
            if len(chosen) == 4:
                break
        extras = [replace(x, commit_moves=1) for x in base if x.strategy in ('ELIMINATE', 'AVOID') and short_safe(self.level, state, (x.first_action,), horizon=1)]
        candidates = ([x for x in extras if x.strategy == 'ELIMINATE'] + chosen + [x for x in extras if x.strategy == 'AVOID'])[:8]
        if candidates:
            counts = Counter()
            out = []
            for x in candidates:
                prefix = x.strategy[0]
                limit = 2 if prefix == 'E' else 4
                if counts[prefix] >= limit:
                    continue
                out.append(replace(x, option_id=f'{prefix}{counts[prefix]}'))
                counts[prefix] += 1
            return tuple(out)
        return base
