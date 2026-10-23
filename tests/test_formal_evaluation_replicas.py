import asyncio
import sys
from types import SimpleNamespace

import pytest

from slime_pacman import formal_rollout as f


@pytest.mark.parametrize('configured,update,expected,vqa', [
    (None, 0, 12, False), (1, 0, 1, False), (1, 198, 1, True),
    (1, 199, 12, True), (None, 199, 12, False)])
def test_eval_replicas_and_opt_in_vqa(monkeypatch,tmp_path,configured,update,expected,vqa):
    rt=SimpleNamespace(root=tmp_path,catalog=[dict(task_id='v1',split='validation',tier=5)],
        sampler=SimpleNamespace(min_tier=5,controller=SimpleNamespace(max_tier=7)),vqa_probe=vqa)
    if configured is not None:rt.evaluation_replicas=configured
    monkeypatch.setattr(f,'get_runtime',lambda args:rt)
    monkeypatch.setitem(sys.modules,'slime.rollout.base_types',SimpleNamespace(
        finalize_rollout_groups=None,RolloutFnEvalOutput=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'slime.utils.types',SimpleNamespace(Sample=SimpleNamespace))
    probes=[]; calls=[]
    async def probe(args,rollout):probes.append(rollout)
    monkeypatch.setitem(sys.modules,'vqa_probe',SimpleNamespace(evaluate=probe))
    async def collect(args,task,sample,u,group,replica,seed,evaluation):
        calls.append((replica,seed,sample.index))
        return [sample],dict(won=True,reward=1.,weight_version='1',panel='heldout_layouts')
    monkeypatch.setattr(f,'collect',collect)
    result=asyncio.run(f.rollout_async(SimpleNamespace(num_rollout=200),update,None,True))
    assert len(result['data']['heldout_layouts']['rewards'])==expected
    assert [x[0] for x in calls]==list(range(expected))
    assert len({x[1] for x in calls})==expected
    assert probes==([update] if vqa else [])


@pytest.mark.parametrize('bad',[0,-1,True,1.5,'1'])
def test_invalid_eval_replica_count_fails_before_collect(monkeypatch,bad):
    monkeypatch.setattr(f,'get_runtime',lambda args:SimpleNamespace(evaluation_replicas=bad))
    monkeypatch.setitem(sys.modules,'slime.rollout.base_types',SimpleNamespace(
        finalize_rollout_groups=None,RolloutFnEvalOutput=None))
    monkeypatch.setitem(sys.modules,'slime.utils.types',SimpleNamespace(Sample=None))
    with pytest.raises(ValueError,match='replica'):
        asyncio.run(f.rollout_async(SimpleNamespace(num_rollout=200),0,None,True))
