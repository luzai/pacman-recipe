from copy import deepcopy
import pytest
from slime_pacman.map_training.controller import TrainingPolicy
from slime_pacman.map_training.sampling import FormalSampler
from slime_pacman.map_training.stages import begin_stage


def setup_stage():
    catalog=[dict(task_id=f't{t}-{i}',tier=t,split='train',
                  layout_sha256=f'{t*100+i:064x}',ghosts=t,game_seed=i)
             for t in range(2) for i in range(8)]
    sampler=FormalSampler(catalog,TrainingPolicy(.8,3,384,2,8),max_tier=1,min_tier=0)
    parent=dict(schema='pacman-joint-checkpoint-v1',identity='parent',update=1,
                controller=dict(identity='parent',state=dict(last_update=1,tier=1)))
    kw=dict(parent_identity='parent',parent_update=1,receipt_sha256='a'*64,
            expected_receipt_sha256='a'*64,optimizer_budget=200)
    return sampler,parent,kw


def test_stage_preserves_global_count_and_parent():
    sampler,parent,kw=setup_stage();before=deepcopy(parent)
    result=begin_stage(parent,sampler,**kw)
    assert result['next_rollout']==2 and result['remaining_updates']==198
    assert result['optimizer_reset'] is False and parent==before
    assert sampler.controller.state['tier']==0
    assert sampler.controller.state['window']==[]
    assert len(sampler.plan(2)['groups'])==16


@pytest.mark.parametrize('field,value',[('expected_receipt_sha256','b'*64),
                                      ('parent_identity','wrong'),
                                      ('parent_update',2),('optimizer_budget',2)])
def test_rejected_transition_does_not_mutate(field,value):
    sampler,parent,kw=setup_stage();before=sampler.snapshot();kw[field]=value
    with pytest.raises(ValueError):begin_stage(parent,sampler,**kw)
    assert sampler.snapshot()==before


def test_tier0_promotes_and_remains_in_replay():
    sampler,parent,kw=setup_stage();begin_stage(parent,sampler,**kw)
    for update in range(2,6):
        plan=sampler.plan(update)
        assert all(g['task']['tier']==0 for g in plan['groups'])
        episodes=[dict(group=g['group'],replica=r,task_id=g['task']['task_id'],won=True)
                  for g in plan['groups'] for r in range(12)]
        sampler.complete(update,episodes,plan)
    assert sampler.controller.state['tier']==1
    plan=sampler.plan(6)
    assert sum(g['task']['tier']==1 for g in plan['groups'])==13
    assert sum(g['task']['tier']==0 for g in plan['groups'])==3
    assert len({g['environment_seed'] for g in plan['groups']})==16
