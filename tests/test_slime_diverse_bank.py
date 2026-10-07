"""Acceptance of combined quotas, fixed openings and serialized rotation."""
import json
from collections import Counter
import pytest
from slime_pacman.diverse_bank import Policy, validate_starts
from slime_pacman.grouping import group_advantages


def policy(n=28):
    p=Policy('test-run',list(range(28)));manifest={'states':{}}
    for i in range(n):
        k=f's{i:03d}'
        p.measure(k,i%28,f'b{i%8}','a'*64,12,24,0,'v0')
        manifest['states'][k]=dict(seed=i%28,file_sha256='a'*64,added_update=0,
            env_step=100+i,sources=[dict(artifact_id=f'route{i}')])
    p.annotate(manifest);p.freeze_representatives()
    return p


def commit(p,starts):
    p.commit([dict(start=s,wins=(0 if i%3==0 else 12 if i%3==1 else 6),episodes=12)
              for i,s in enumerate(starts)])


def test_two_rotation_cycles_and_resume_match():
    p=policy();seen=[]
    for update in range(14):
        starts=p.select();validate_starts(starts,update)
        assert p.select()==starts
        restored=Policy.loads(p.dumps(),p.identity)
        assert restored.select()==starts
        seen.extend(s['seed'] for s in starts if s['kind']=='initial')
        commit(p,starts);commit(restored,starts)
        assert p.dumps()==restored.dumps()
        p=restored
    assert Counter(seen)==Counter({i:2 for i in range(28)})


def test_extreme_and_expired_states_remain_with_qualification_age():
    p=policy();p.completed=25
    for s in p.states.values():s['wins']=0
    starts=p.select();bank=[s for s in starts if s['kind']=='bank']
    assert len(bank)==12
    assert all(s['qualification_age']==25 and not s['current_probe_mixed'] for s in bank)
    assert all('retained_previous_qualification' in s['fallback'] for s in bank)
    assert sum(s['target_category']=='review' for s in bank)==4
    assert all('recent_pool_shortage' in s['fallback'] for s in bank if s['target_category']!='review')


def test_opening_does_not_consume_bank_seed_cap():
    p=policy(12)
    for i,s in enumerate(p.states.values()):
        s['seed']=i//2;s['route_id']=f'r{i}'
    starts=p.select()
    assert Counter(s['seed'] for s in starts if s['kind']=='bank')[0]==2
    assert Counter(s['seed'] for s in starts)[0]==3


def test_joint_quota_cannot_fake_fill_or_drop_reviews():
    p=policy(12)
    p.representatives=[]
    with pytest.raises(ValueError,match='joint'):p.select()
    assert p.pending==[]
    p=policy(12)
    for s in p.states.values():s['seed']=0;s['route_id']='one-route'
    with pytest.raises(ValueError,match='joint'):p.select()


def test_never_qualified_states_not_available_as_fallback():
    p=policy(12);p.states['s000']['ever_qualified']=False
    with pytest.raises(ValueError,match='distinct'):p.select()


def test_contract_identity_and_pending_validation():
    p=policy();p.select();row=json.loads(p.dumps())
    row['pending'][0]['seed']=28
    with pytest.raises(ValueError):Policy.loads(json.dumps(row),p.identity)
    with pytest.raises(ValueError):Policy.loads(p.dumps(),'different')


@pytest.mark.parametrize('wins',range(13))
def test_binary_extremes_zero_despite_speed_differences(wins):
    rewards=[1+i*.001 for i in range(wins)]+[0.]*(12-wins)
    actual=group_advantages(rewards,success_speed_bonus=.05,zero_binary_extremes=True)
    if wins in (0,12):assert actual==[0.]*12
    else:
        assert actual==pytest.approx([r-sum(rewards)/12 for r in rewards])
        assert any(a!=0 for a in actual)


def test_previous_contract_keeps_speed_signal():
    assert any(group_advantages([1.01]*6+[1.02]*6,success_speed_bonus=.05))
