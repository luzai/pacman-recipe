"""Real package/topology continuity contracts; not model replay evidence."""
from dataclasses import dataclass, replace
from unittest.mock import patch
import pytest
import pacman_env.planner as P
from experiments.edward_route_options.continuity import ContinuousPlanner
from experiments.edward_route_options.motion_guard import NoPathPlanner, guard_mode
from experiments.edward_route_options import route_bound as B

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


def setup_route():
    level=Grid(open_tiles=Grid.open_tiles | {P.Position(1,2), P.Position(3,2)})
    p=ContinuousPlanner(level, fallback_mode="risk_ranked")
    s=state();s["lives"]=4
    c=next(c for c in p.advertised_candidates(s) if c.strategy=="COLLECT" and c.route_distance==4)
    p.bind(c,s);p.before_action("R");p.record_action("R")
    s.update(pacman_position=[2,2],open=["U","D","L","R"])
    return p,c,s

def test_junction_keeps_valid_route_without_new_decision():
    p,c,s=setup_route()
    p.advertised_candidates=lambda *_: pytest.fail("unexpected replanning")
    assert p.continue_option(c,s)==("R","active")
    assert p.events[-1]["at_junction"] and p.events[-1]["old_route_valid"]
    assert c.commit_moves==4
    assert p.active["summary"]["boundary"]=="route_end"

@pytest.mark.parametrize("reason",["short_risk","escape_risk","life_changed","target_completed","route_blocked","position_invalidated","invalid_cursor"])
def test_continuation_interrupts_are_explicit(reason):
    p,c,s=setup_route()
    if reason=="life_changed":s["lives"]=3
    if reason=="target_completed":s["normal_pellet_positions"]=[]
    if reason=="route_blocked":s["open"]=["L"]
    if reason=="position_invalidated":s["pacman_position"]=[1,2]
    if reason=="invalid_cursor":p.cursor=99
    with patch.object(B,"short_safe",return_value=reason!="short_risk"), patch.object(B,"escape_safe",return_value=reason!="escape_risk"):
        assert p.continue_option(c,s)[0] is None
    assert p.events[-1]["reason"]==reason

def test_portal_full_route_continues_until_complete():
    p=ContinuousPlanner(Grid(portal=True),fallback_mode="risk_ranked");s=state()
    c=next(c for c in p.advertised_candidates(s) if c.strategy=="COLLECT")
    p.bind(c,s);p.before_action(c.first_action);p.record_action(c.first_action)
    s.update(pacman_position=[2,5],open=["L"],normal_pellet_positions=[])
    assert p.continue_option(c,s)==(None,"completed")
    assert p.events[-1]["reason"]=="route_complete"

def test_hidden_path_changes_do_not_change_native_candidates():
    s=state();s["ghosts"]=[dict(id=0,state="normal",position=[0,0],pixel_position=[0,0],direction="R",speed=0)]
    results=[]
    with guard_mode("final_base"):
        for path in ("RRR","LLL"):
            s["ghosts"][0].update(path_remaining=path,currentPath=path,path_target=[999,999],rng=object())
            p=NoPathPlanner(Grid(),fallback_mode="risk_ranked")
            results.append((p.advertised_candidates(s),p.route_menu))
    assert results[0]==results[1]
