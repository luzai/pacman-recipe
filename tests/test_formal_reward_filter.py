import asyncio
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from slime_pacman.map_training.reward_filter import (
    collect_effective, InsufficientEffectiveGroups, validate_policy, configured_policy,
)


def test_new_job_default_explicit_disable_and_independent_policy():
    first = configured_policy({})
    assert first == {'mode': 'exact_final_reward', 'max_rounds': 16,
                     'retry_task': 'same_tier_new_task'}
    first['max_rounds'] = 1
    assert configured_policy({})['max_rounds'] == 16
    assert configured_policy({'reward_group_filter': None}) is None
    assert configured_policy({'reward_group_filter': {
        'mode': 'exact_final_reward', 'max_rounds': 3}})['max_rounds'] == 3


def bindings(n=16):
    return [dict(group=i, task={'task_id': f'map-{i}', 'tier': i % 2,
                               'game_seed': 100+i},
                 sampling_seeds=list(range(i*12, i*12+12))) for i in range(n)]


def group(b, rewards, version='v1'):
    return [([SimpleNamespace(reward=r)], dict(group=b['group'], replica=i,
             task_id=b['task']['task_id'], reward=r, weight_version=version,
             won=r >= 1)) for i, r in enumerate(rewards)]


def test_exact_shaped_reward_not_binary_win_filter_and_fixed_192():
    original=bindings(); frozen=deepcopy(original); calls=[]; audits=[]
    async def launch(bs, attempt):
        calls.append(deepcopy(bs))
        # Constant nonbinary reward is rejected; all-win but unequal stays.
        return [group(b, [1.125]*12 if attempt==0 and b['group']%2==0
                      else [1.125]*11+[1.12500000001]) for b in bs]
    kept, first, stats=asyncio.run(collect_effective(original, launch, max_rounds=3,
        audit=lambda a,p:audits.append(p)))
    assert original==frozen
    assert [len(c) for c in calls]==[16,8]
    assert len(first)==192 and sum(len(g) for g in kept)==192
    assert stats==dict(rounds=2,generated_groups=24,rejected_groups=8,effective_groups=16,target_groups=16)
    assert all(g[-1][1]['reward']!=g[0][1]['reward'] for g in kept)
    for b in calls[1]:
        assert b['task']==original[b['group']]['task']
        assert set(b['sampling_seeds']).isdisjoint(original[b['group']]['sampling_seeds'])
    # Deterministic retries and fixed order, independent of caller's RNG.
    again=[]
    asyncio.run(collect_effective(original, launch,max_rounds=3,audit=lambda a,p:again.append(p)))
    assert audits==again


def test_all_equal_exhaustion_never_falls_back_and_preserves_audit():
    audits=[]
    async def launch(bs,a):return [group(b,[0.]*12) for b in bs]
    with pytest.raises(InsufficientEffectiveGroups,match='no optimizer update'):
        asyncio.run(collect_effective(bindings(2),launch,max_rounds=3,
                                     audit=lambda a,p:audits.append(p)))
    assert len(audits)==3
    assert audits[-1]['stats']['generated_groups']==6
    assert audits[-1]['stats']['effective_groups']==0


@pytest.mark.parametrize('failure',['nan','inf','short','version','sample_reward','binding'])
def test_invalid_collection_is_rejected(failure):
    async def launch(bs,a):
        gs=[group(b,[0.]*11+[1.]) for b in bs]
        if failure=='short':gs[0].pop()
        elif failure=='version':gs[1][0][1]['weight_version']='v2'
        elif failure=='sample_reward':gs[0][0][0][0].reward=.5
        elif failure=='binding':gs[0][0][1]['task_id']='wrong'
        else:gs[0][0][1]['reward']=float(failure)
        return gs
    with pytest.raises(ValueError):
        asyncio.run(collect_effective(bindings(2),launch,max_rounds=2,audit=lambda *a:None))


@pytest.mark.parametrize('policy',[{}, {'mode':'binary','max_rounds':4},
                                  {'mode':'exact_final_reward','max_rounds':True},
                                  {'mode':'exact_final_reward','max_rounds':0}])
def test_bad_policy(policy):
    with pytest.raises(ValueError):validate_policy(policy)


@pytest.mark.parametrize('new_tasks', [False, True])
def test_formal_wiring_uses_first_pass_for_controller_and_kept_for_training(monkeypatch,tmp_path,new_tasks):
    import slime_pacman.formal_rollout as f
    bs=bindings(); plan={'groups':bs,'tier':5,'bank_version':0}
    pending=[];final=[];alloc=[]
    rt=SimpleNamespace(reward_group_filter={'mode':'exact_final_reward','max_rounds':2},
        root=tmp_path,freeze_plan=lambda u:plan,collected=lambda u,p,s:pending.append(s))
    if new_tasks:
        rt.reward_group_filter['retry_task']='same_tier_new_task'
        def replacement(b, attempt, history, *, update):
            b['task'].update(task_id='replacement-'+b['task']['task_id'],split='train')
            b['sampling_seeds']=[s+10000 for s in b['sampling_seeds']]
            return b
        rt.sampler=SimpleNamespace(replacement=replacement)
    monkeypatch.setattr(f,'get_runtime',lambda a:rt)
    base=SimpleNamespace(finalize_rollout_groups=lambda a,u,g,metrics:final.append((g,metrics)),
                         RolloutFnEvalOutput=lambda **kw:kw)
    monkeypatch.setitem(sys.modules,'slime.rollout.base_types',base)
    monkeypatch.setitem(sys.modules,'slime.utils.types',SimpleNamespace(Sample=SimpleNamespace))
    def get_samples(n):
        attempt=len(alloc);alloc.append(n)
        return [[SimpleNamespace(attempt=attempt) for _ in range(12)] for _ in range(n)]
    async def collect(a,task,s,u,g,r,seed,evaluation):
        reward=0. if s.attempt==0 else float(r==11)
        return ([SimpleNamespace(reward=reward)],dict(group=g,replica=r,
            task_id=task['task_id'],won=reward>=1,reward=reward,weight_version='v1'))
    monkeypatch.setattr(f,'collect',collect)
    args=SimpleNamespace(rollout_batch_size=16,n_samples_per_prompt=12)
    asyncio.run(f.rollout_async(args,0,SimpleNamespace(get_samples=get_samples)))
    assert alloc==[16,16]
    assert len(pending)==1 and sum(r['won'] for r in pending[0])==0
    assert len(final)==1 and len(final[0][0])==16
    assert sum(len(g) for g in final[0][0])==192
    assert final[0][1]['curriculum/episode_win_rate']==0
    assert final[0][1]['rollout/effective_episode_win_rate']==1/12
    if new_tasks:
        import json
        effective=json.loads((tmp_path/'effective-batches/rollout-0000.json').read_text())
        assert all(e['task_id'].startswith('replacement-') for e in effective)
        assert all(not e['task_id'].startswith('replacement-') for e in pending[0])
    # An exhausted collection cannot reach runtime.collected or finalization.
    pending.clear();final.clear();alloc.clear();rt.reward_group_filter['max_rounds']=1
    with pytest.raises(InsufficientEffectiveGroups):
        asyncio.run(f.rollout_async(args,1,SimpleNamespace(get_samples=get_samples)))
    assert not pending and not final


def test_filter_policy_changes_checkpoint_identity():
    from slime_pacman.map_training.sampling import FormalSampler
    from slime_pacman.map_training.controller import TrainingPolicy
    cat=[dict(task_id=str(i),tier=0,split='train',layout_sha256=f'{i:064x}') for i in range(8)]
    p=TrainingPolicy(.8,3,384,2,8)
    plain=FormalSampler(cat,p,max_tier=0,min_tier=0)
    filtered=FormalSampler(cat,p,max_tier=0,min_tier=0,
                          reward_group_filter={'mode':'exact_final_reward','max_rounds':16})
    assert plain.controller.identity!=filtered.controller.identity
    with pytest.raises(ValueError,match='checkpoint/controller mismatch'):
        filtered.restore(plain.snapshot(),completed_update=-1,bank_version=0)


def test_independent_validation_keeps_constant_reward_groups(monkeypatch,tmp_path):
    import slime_pacman.formal_rollout as f
    import slime_pacman.map_training.reward_filter as rf
    def forbidden(*a,**kw):raise AssertionError('validation must not filter or update')
    monkeypatch.setattr(rf,'collect_effective',forbidden)
    rt=SimpleNamespace(root=tmp_path,reward_group_filter={'mode':'exact_final_reward','max_rounds':2},
        catalog=[dict(task_id='validation-map',split='validation',tier=5)],
        sampler=SimpleNamespace(min_tier=5,controller=SimpleNamespace(max_tier=7)),
        collected=forbidden)
    monkeypatch.setattr(f,'get_runtime',lambda a:rt)
    monkeypatch.setitem(sys.modules,'slime.rollout.base_types',SimpleNamespace(
        finalize_rollout_groups=forbidden,RolloutFnEvalOutput=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'slime.utils.types',SimpleNamespace(Sample=SimpleNamespace))
    async def collect(a,task,s,u,g,r,seed,evaluation):
        assert evaluation
        return [s],dict(group=g,replica=r,task_id=task['task_id'],won=False,
                        reward=0.,weight_version='v1',panel='heldout_layouts')
    monkeypatch.setattr(f,'collect',collect)
    output=asyncio.run(f.rollout_async(SimpleNamespace(),0,None,evaluation=True))
    assert output['data']['heldout_layouts']['rewards']==[0.]*12
    assert not (tmp_path/'resampling').exists()


def test_same_tier_replacement_rescues_hard_slot_and_preserves_revisit():
    from slime_pacman.map_training.sampling import FormalSampler
    from slime_pacman.map_training.controller import TrainingPolicy
    cat=[dict(task_id=f't{t}-{i}',tier=t,split='train',layout_sha256=f'{t}-{i}',
              game_seed=i,episode_record={'environment':{'seed':i}})
         for t in (1,2) for i in range(8)]
    sampler=FormalSampler(cat,TrainingPolicy(.8,3,384,2,8),max_tier=2,
                          reward_group_filter=configured_policy({}))
    bs=sampler.plan(0)['groups']; hard=bs[0]['task']['layout_sha256']
    snapshot=sampler.snapshot(); original_catalog=deepcopy(sampler.catalog); calls=[]
    def replacement(b,a,h):return sampler.replacement(b,a,h,update=0)
    async def launch(pending,attempt):
        calls.append(deepcopy(pending))
        return [group(b,[0.]*12 if b['task']['layout_sha256']==hard
                      else [0.]*11+[1.]) for b in pending]
    kept,first,stats=asyncio.run(collect_effective(bs,launch,max_rounds=3,
                               audit=lambda *a:None,replacement=replacement))
    assert stats['effective_groups']==16 and stats['rounds']==2
    assert len(first)==192 and sum(len(g) for g in kept)==192
    assert sampler.snapshot()==snapshot and sampler.catalog==original_catalog
    assert sampler.plan(0)['groups']==bs
    for b in calls[1]:
        assert b['task']['layout_sha256']!=hard and b['task']['tier']==1
        assert b['task']['game_seed']==b['task']['episode_record']['environment']['seed']
        assert b['environment_seed'] not in {x['environment_seed'] for x in bs}
        assert len(set(b['sampling_seeds']))==12
    assert len({b['environment_seed'] for b in calls[1]})==len(calls[1])
    assert replacement(bs[0],1,bs)==replacement(bs[0],1,bs)
    b=deepcopy(bs[0]);b['task']=deepcopy(cat[-1])
    assert replacement(b,1,[b])['task']['tier']==2
    old=FormalSampler(cat,TrainingPolicy(.8,3,384,2,8),max_tier=2,
                      reward_group_filter={'mode':'exact_final_reward','max_rounds':16})
    assert old.controller.identity!=sampler.controller.identity
