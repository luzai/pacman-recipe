import json
from collections import Counter
import pytest
from slime_pacman.diverse_bank_v2 import Policy, QuotaShortage, validate_starts
from slime_pacman.route_capture_v2 import route_candidates


def policy(early=True):
    p = Policy('v2-test', list(range(28)))
    manifest = {'states': {}}
    for i in range(28):
        key = f's{i:03d}'
        p.measure(key, i, f'b{i}', 'a'*64, 12, 24, 0, 'v0')
        manifest['states'][key] = dict(seed=i, file_sha256='a'*64, added_update=0,
            env_step=100, sources=[dict(artifact_id=f'route{i}',
                decision_index=20 if early else 90, route_decision_count=120)])
    p.annotate(manifest)
    p.freeze_representatives()
    return p


def commit(p, starts, wins=6):
    p.commit([dict(start=s, episodes=12, wins=wins) for s in starts])


def test_rotation_resume_and_quotas():
    p = policy(); reviews = []; openings = []
    for u in range(14):
        starts = p.select(allow_category_fallback=True)
        assert Counter(s['slot_role'] for s in starts[4:]) == {'review':4,'early':2,'recent_priority':6}
        reviews.extend(s['state_id'] for s in starts[4:] if s['slot_role']=='review')
        openings.extend(s['seed'] for s in starts[:4])
        restored = Policy.loads(p.dumps(), p.identity)
        assert restored.select() == starts
        commit(p, starts); commit(restored, starts)
        assert p.dumps() == restored.dumps()
    assert len(set(reviews[:16])) == 16
    assert Counter(openings) == {i:2 for i in range(28)}


def test_shortage_before_refresh_then_explicit_fallback():
    p = policy(False)
    with pytest.raises(QuotaShortage): p.select()
    assert not p.pending
    starts = p.select(allow_category_fallback=True)
    assert sum('early_pool_shortage' in s['fallback'] for s in starts[4:]) == 2
    commit(p, starts)
    assert p.fallback_ledger[-1]['deficits'] == {'early':2}


def test_joint_selection_does_not_greedily_consume_early():
    p = policy(False)
    for key in p.representatives[:2]: p.states[key]['early'] = True
    starts = p.select()
    assert {s['state_id'] for s in starts[4:] if s['slot_role']=='early'} == set(p.representatives[:2])


def test_merged_routes_count_and_receipt_validation():
    p = policy()
    for key in list(p.states)[:8]: p.states[key]['source_routes'].append('merged')
    starts = p.select()
    assert sum('merged' in s['source_routes'] for s in starts[4:]) <= 2
    for s in starts[4:7]: s['source_routes'] = [s['route_id'], 'hidden']
    with pytest.raises(ValueError, match='route cap'): validate_starts(starts, 0)


def test_joint_shortage_never_fakefills():
    p = policy()
    for s in p.states.values(): s['source_routes'].append('shared')
    with pytest.raises(QuotaShortage): p.select(allow_category_fallback=True, search_limit=64)
    assert not p.pending


def test_loss_priority_preserved_on_restore():
    p = policy(); starts = p.select(); commit(p, starts, 0)
    assert len(p.representatives) == 16
    lost = sorted(s['state_id'] for s in starts[4:])
    assert p.reprobe_order()[:len(lost)] == lost
    restored = Policy.loads(p.dumps(), p.identity)
    assert restored.reprobe_order() == p.reprobe_order()
    assert restored.fallback_ledger == p.fallback_ledger


def test_old_reprobe_does_not_become_recent():
    p = policy(); p.completed = 25
    for key, s in list(p.states.items()):
        p.measure(key,s['seed'],s['bucket'],s['file_sha'],0,24,25,'v25',eligible=False)
    with pytest.raises(QuotaShortage): p.select()
    starts = p.select(allow_category_fallback=True)
    assert all(s['added_update']==0 and s['qualification_age']==25 for s in starts[4:])
    assert all('retained_previous_qualification' in s['fallback'] for s in starts[4:])
    assert sum('recent_priority_pool_shortage' in s['fallback'] for s in starts[4:]) == 6


def test_v1_and_invalid_cursor_rejected():
    p = policy(); value = json.loads(p.dumps()); value['schema']='ascii-diverse-first192-v1'
    with pytest.raises(ValueError): Policy.loads(json.dumps(value),p.identity)
    value = json.loads(p.dumps()); value['representative_cursor']=16
    with pytest.raises(ValueError): Policy.loads(json.dumps(value),p.identity)


def boundary(i):
    return dict(env_state=dict(sha256=str(i),payload=dict(episode=dict(steps=i))),
                context=dict(planner={},runner={}))


@pytest.mark.parametrize('n',range(1,25))
def test_early_floor_and_death_offsets_deduplicate(n):
    candidates = route_candidates([boundary(i) for i in range(n)],'death')
    indices = {p['decision_index'] for c in candidates for p in c['positions']}
    expected = {i for i in (n//6,n//3) if 0<i<n} | {n-1-d for d in (4,8,16) if n-1-d>=0}
    assert indices == expected
    assert len({c['candidate_id'] for c in candidates}) == len(candidates)
    for c in candidates:
        for p in c['positions']:
            assert p['route_endpoint']==p['route_decision_count']==n
            assert p['distance']==n-1-p['decision_index']


def test_early_prefix_not_limited_to_last_ring():
    candidates = route_candidates([boundary(i) for i in range(120)],'death')
    assert {c['positions'][0]['decision_index'] for c in candidates} == {20,40,103,111,115}
    assert len(route_candidates([boundary(i) for i in range(120)],'all_normal_pellets')) == 2
