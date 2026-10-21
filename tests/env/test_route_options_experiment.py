"""Portable contract checks for the opt-in route experiment, not a win-rate test."""
from dataclasses import dataclass, replace
import copy
import pytest
import pacman_env.planner as P
from experiments.edward_route_options.route_bound import BoundPlanner, NativeStep1, describe
from experiments.edward_route_options.experimental_planner import move, short_safe
from experiments.edward_route_options.escape_guard import escape_safe

@dataclass(frozen=True)
class Grid:
    width: int = 7
    height: int = 5
    open_tiles: frozenset = frozenset(P.Position(2, c) for c in range(1, 6))
    pellets: frozenset = frozenset([P.Position(2, 5)])
    power_pellets: frozenset = frozenset()
    pacman_start: P.Position = P.Position(2, 1)
    portal: bool = False

    def is_wall(self, position, *, actor):
        return position not in self.open_tiles

    def portal_exit(self, position, action):
        if self.portal and position == P.Position(2, 2) and action == 'R':
            return P.Position(2, 5)
        return position

def state():
    return dict(pacman_position=[2,1], open=['R'], ghosts=[], normal_pellet_positions=[[2,5]])

def bound():
    planner = BoundPlanner(Grid(), fallback_mode='risk_ranked')
    s = state()
    candidate = next(c for c in planner.advertised_candidates(s) if c.strategy == 'COLLECT')
    planner.bind(candidate, s)
    return planner, candidate

def advance(planner, candidate):
    planner.before_action('R')
    planner.record_action('R')
    s = state()
    s.update(pacman_position=[2,2], open=['L','R'])
    return s

def test_native_step1_only_changes_commit():
    native = P.EdwardPlanner(Grid(), fallback_mode='risk_ranked')
    one = NativeStep1(Grid(), fallback_mode='risk_ranked')
    assert one.advertised_candidates(state()) == tuple(replace(c, commit_moves=1) for c in native.advertised_candidates(state()))

def test_complete_route_binding_and_continuation_without_regeneration():
    p, c = bound()
    assert p.active['identity']['actions'] == ['R'] * 4
    assert p.active['cells'][-1] == [2,5]
    assert c.commit_moves == 4
    s = advance(p,c)
    p.advertised_candidates = lambda *_: pytest.fail('unconditional replanning')
    assert p.continue_option(c,s) == ('R','active')

@pytest.mark.parametrize('kind', ['danger', 'position', 'blocked'])
def test_live_interrupts(kind):
    p,c = bound()
    s = advance(p,c)
    if kind == 'position': s['pacman_position'] = [0,0]
    elif kind == 'blocked': s['open'] = ['L']
    else:
        s['ghosts'] = [dict(state='normal',position=[2,3],pixel_position=[32,48],direction='L',path_remaining='LLL',speed=1)]
    assert p.continue_option(c,s)[1] == 'invalidated'

def test_mismatched_action_rejected():
    p,c = bound()
    with pytest.raises(AssertionError): p.before_action('L')

def test_portal_is_bound_and_ends_segment():
    level = Grid(portal=True)
    d = describe(level,P.Position(2,1),(2,5),('R',),'COLLECT')
    assert d['summary']['portals'] == [dict(index=0,entry=[2,1],exit=[2,5],action='R')]
    assert d['summary']['boundary'] == 'portal_exit'
    assert move(level,P.Position(2,1),'R') == P.Position(2,5)

def test_route_geometry_distinguishes_same_target_first_move_and_length():
    cells = frozenset(P.Position(r,c) for r in range(1,4) for c in range(1,6))
    level = Grid(open_tiles=cells)
    a = describe(level,P.Position(2,1),(2,4),tuple('RURRD'),'COLLECT')
    b = describe(level,P.Position(2,1),(2,4),tuple('RDRRU'),'COLLECT')
    assert a['route_id'] != b['route_id']
    sa, sb = copy.deepcopy(a['summary']), copy.deepcopy(b['summary'])
    sa.pop('route_id'); sb.pop('route_id')
    assert sa != sb

def test_nonfinal_dead_end_rejected_but_final_pellet_exempt():
    s = state(); s['pacman_position'] = [2,4]
    assert escape_safe(Grid(),s,('R',),P.Position(2,5))
    s['normal_pellet_positions'].append([2,1])
    assert not escape_safe(Grid(),s,('R',),P.Position(2,5))

def test_missing_ghost_motion_fails_closed():
    s = state(); s['ghosts'] = [dict(state='normal',position=[2,4])]
    assert not short_safe(Grid(),s,('R',))
