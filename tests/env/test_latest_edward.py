"""Real package/topology continuity contracts; not model replay evidence."""
from dataclasses import dataclass, replace
from unittest.mock import patch
import pytest
import pacman_env.planner as P
from pacman_env.latest_edward import LatestEdwardPlanner as ContinuousPlanner
from pacman_env.latest_edward import LatestEdwardPlanner as NoPathPlanner
from contextlib import nullcontext
def guard_mode(mode): return nullcontext()
from pacman_env import latest_edward as B

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


def test_production_matches_frozen_candidates_and_execution():
    from experiments.edward_route_options.motion_guard import NoPathPlanner as Frozen, guard_mode as frozen_guard
    from pacman_env.latest_edward import LatestEdwardPlanner
    for ghosts in ([], [dict(id=1,state="normal",position=[0,0],pixel_position=[0,0],direction="R",speed=0,path_remaining="RR")]):
        s=state();s["ghosts"]=ghosts;s["lives"]=4
        old=Frozen(Grid(),fallback_mode="risk_ranked")
        new=LatestEdwardPlanner(Grid())
        with frozen_guard("final_base"):
            a=old.advertised_candidates(s)
        b=new.advertised_candidates(s)
        # Production adds bounded escape evidence without changing fallback
        # identity/order/rank or execution on these non-vulnerable fixtures.
        comparable = tuple(replace(c, risk={k:v for k,v in c.risk.items()
                            if k not in {'escape_horizon','escape_found','escape_route'}})
                           if c.risk else c for c in b)
        assert a==comparable and old.route_menu==new.route_menu
        if not any(c.strategy=="COLLECT" for c in a):
            assert all(c.strategy=="RISK_FALLBACK" for c in a)
            continue
        ca=next(c for c in a if c.strategy=="COLLECT")
        cb=next(c for c in b if c.strategy=="COLLECT")
        old.bind(ca,s);new.bind(cb,s)
        for col in range(2,6):
            for p,c in [(old,ca),(new,cb)]:
                p.before_action("R");p.record_action("R")
            s.update(pacman_position=[2,col],open=["L","R"])
            with frozen_guard("final_base"):
                expected=old.continue_option(ca,s)
            assert new.continue_option(cb,s)==expected
        assert new.events==old.events

def test_instance_hooks_do_not_patch_legacy_or_other_instances():
    from pacman_env.latest_edward import LatestEdwardPlanner
    original=(P._one_step_pixel_safe,P._fallback_motion_risk)
    old=P.EdwardPlanner(Grid(),fallback_mode="risk_ranked")
    before=old.advertised_candidates(state())
    new=LatestEdwardPlanner(Grid());new.advertised_candidates(state())
    assert original==(P._one_step_pixel_safe,P._fallback_motion_risk)
    assert old.advertised_candidates(state())==before
    with patch.object(P,"_fallback_motion_risk",side_effect=AssertionError("legacy hidden prediction")),patch.object(P,"_one_step_pixel_safe",side_effect=AssertionError("legacy hidden prediction")):
        assert new._motion_risk(state(),P.Position(2,2))=="clear_estimate"
        assert new._one_step_safe(state(),P.Position(2,2))

def test_reset_removes_stale_routes_and_no_lure():
    p,c,s=setup_route();p.reset()
    assert p.active is None and not p.registry and not p.menu and not p.route_menu
    assert not p.events and p.cursor==0 and not p.visits
    assert all("skill" not in v for v in p.route_menu.values())


def test_production_motion_first_frame_and_future_reverse():
    assert B.advance(Grid(),{(32.,32.,"R")},1,False)=={(32.,33.,"R")}
    assert B.advance(Grid(),{(32.,32.,"R")},1)=={(32.,31.,"L"),(32.,33.,"R")}

def test_production_clean_never_reads_hidden_fields():
    class Poison(dict):
        def __getitem__(self,key):
            if key not in B.FIELDS: raise AssertionError(key)
            return super().__getitem__(key)
    s=state();s["ghosts"]=[Poison(id=0,state="normal",position=[2,4],pixel_position=[32,64],direction="L",speed=1,path_remaining="LLLL",currentPath="LLLL",rng="hidden")]
    cleaned=B.clean_state(s)
    assert not ({"path_remaining","currentPath","rng"}&set(cleaned["ghosts"][0]))
    p=ContinuousPlanner(Grid());p.advertised_candidates(s)

def test_production_live_ghost_collision_interrupts_without_mock():
    p,c,s=setup_route()
    s["ghosts"]=[dict(id=0,state="normal",position=[2,3],pixel_position=[32,48],direction="L",speed=1)]
    assert p.continue_option(c,s)==(None,"invalidated")
    assert p.events[-1]["reason"]=="short_risk"
