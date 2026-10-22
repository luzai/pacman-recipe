"""Production expiry guard and non-directive bounded escape evidence."""
from dataclasses import replace
from unittest.mock import patch
import pytest
from tests.env.test_latest_edward import Grid, state
from tests.env.test_edward_safety_candidate import danger
from pacman_env import latest_edward as C, planner as P
from pacman_recipe.level1.prompts import risk_ranked_edward_decision_prompt
from pacman_recipe.level1.token_constraints import ObjectiveTokenConstraint
from tests.test_edward_option_shuffle import Tokenizer
from tests.test_edward_risk_audit_permutation import fixture
from pacman_recipe.level1.trajectories import _audit_risk_fallback_step as audit

@pytest.mark.parametrize('ticks,expected', [(0,False),(1,False),(15,False),(16,False),(17,True),(33,True)])
def test_expiry_boundary_is_logic_frames(ticks, expected):
    assert C.short_safe(Grid(),danger(ticks),('R',),1) is expected


def test_expiry_clock_continues_across_moves():
    s=danger(20); s['ghosts'][0].update(position=[2,3],pixel_position=[32,48])
    assert C.short_safe(Grid(),s,('R',),1)
    assert not C.short_safe(Grid(),s,('R','R'),2)


def test_current_speed_preserved_across_expiry():
    s=danger(9); s['ghosts'][0].update(position=[2,5],pixel_position=[32,80],speed=1)
    speeds=[]
    real=C.advance
    def track(level, branches, speed, allow_turn=True):
        speeds.append(speed); return real(level,branches,speed,allow_turn)
    with patch.object(C,'advance',side_effect=track):
        C.short_safe(Grid(),s,('R',),1)
    assert len(speeds)==16 and set(speeds)=={1}


def test_unknown_expiry_is_conservative_and_eyes_ignored():
    s=danger(); s.pop('edible_ticks')
    assert not C.short_safe(Grid(),s,('R',))
    s['ghosts'][0]['state']='eyes'
    assert C.short_safe(Grid(),s,('R',))


def test_witness_checks_whole_prefix_and_preserves_portal():
    assert C.fallback_escape_route(Grid(),state(),'R')==('R',)*4
    with patch.object(C,'short_safe',side_effect=lambda l,s,r,horizon:len(r)<3):
        assert C.fallback_escape_route(Grid(),state(),'R') is None
    assert C.move(Grid(portal=True),P.Position(2,1),'R')==P.Position(2,5)


def test_no_hidden_path_reads():
    class Guard(dict):
        def get(self,key,*args):
            assert key not in {'path_remaining','path_target','currentPath','rng'}
            return super().get(key,*args)
    s=danger();s['ghosts']=[Guard(s['ghosts'][0])]
    assert not C.short_safe(Grid(),s,('R',))
    assert not {'path_remaining','currentPath'} & C.clean(s)['ghosts'][0].keys()


def test_witness_is_information_only_preserving_order_rank_and_actions():
    payload,step=fixture()
    cs=tuple(P.PlannerCandidate(**c) for c in step['observation_context']['planner_candidates'])
    p=C.LatestEdwardPlanner(Grid())
    with patch.object(P.EdwardPlanner,'_risk_fallback_candidates',return_value=cs), patch.object(C,'fallback_escape_route',side_effect=lambda l,s,a:('R',)*4 if a=='R' else None):
        got=p._risk_fallback_candidates(state(),P.Position(2,1),())
    assert [c.option_id for c in got]==[c.option_id for c in cs]
    assert [c.first_action for c in got]==list('UDLR')
    assert [c.risk['rank'] for c in got]==[1,2,3,4]
    assert got[-1].risk['escape_found'] and not got[0].risk['escape_found']
    for before,after in zip(cs,got):
        assert replace(after,risk=before.risk)==before
    assert p.events[-1]['information_only'] is True
    constraint=ObjectiveTokenConstraint.build(Tokenizer(),[c.option_id for c in cs])
    old=risk_ranked_edward_decision_prompt({},cs,constraint)
    new=risk_ranked_edward_decision_prompt({},got,constraint)
    assert new.startswith(old) and 'Bounded escape evidence' not in old
    assert 'not inevitable death' in new and 'replan after one move' in new


@pytest.mark.parametrize('extra', [dict(escape_found=True),dict(escape_horizon=4,escape_found=True,escape_route=['R']*4),dict(escape_horizon=4,escape_found=False,escape_route=['U']*4),dict(escape_horizon=True,escape_found=False,escape_route=None)])
def test_audit_rejects_malformed_witness(extra):
    payload,step=fixture();step['observation_context']['planner_candidates'][0]['risk'].update(extra)
    with pytest.raises(ValueError,match='escape evidence'):audit(payload,step)


def test_audit_accepts_matching_bounded_witness():
    payload,step=fixture();step['observation_context']['planner_candidates'][0]['risk'].update(escape_horizon=4,escape_found=True,escape_route=['U']*4)
    audit(payload,step)
