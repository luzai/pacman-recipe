"""Route continuity experiment: unchanged candidates/guards, full finite route commitment."""
from dataclasses import replace
from copy import deepcopy
from . import route_bound as B

class ContinuousPlanner(B.BoundPlanner):

    def register(self, candidate, state, route):
        p = B.describe(self.level, B.P.Position(*state['pacman_position']), candidate.target, tuple(route), candidate.strategy)
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
        state = B.normalized(state)
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
                cell = B.move(self.level, cell, action)
                if cell is None or [cell.row, cell.col] != p['cells'][self.cursor + offset]:
                    valid = False
                    reason = 'route_invalidated'
                    break
            if reason is None and rest[0] not in state['open']:
                valid = False
                reason = 'route_blocked'
            if reason is None:
                self.safety_checks += 1
                if not B.short_safe(self.level, state, rest):
                    safe = False
                    reason = 'short_risk'
                elif option.strategy == 'COLLECT' and (not B.escape_safe(self.level, state, rest, B.P.Position(*option.target))):
                    safe = False
                    reason = 'escape_risk'
                else:
                    safe = True
        event = dict(kind='continue' if reason is None else 'boundary', route_id=p['route_id'] if p else None, cursor=self.cursor, reason=reason, old_route_valid=valid, short_term_safe=safe, at_junction=B.P._degree(self.level, pos) >= 3 if 0 <= pos.row < self.level.height and 0 <= pos.col < self.level.width else False, next_action=rest[0] if rest else None)
        self.events.append(event)
        if reason is None:
            return (rest[0], 'active')
        if reason in ('route_complete', 'target_completed'):
            return (None, 'completed')
        return (None, 'invalidated')
