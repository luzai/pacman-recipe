"""New jobs exclude tier zero, including after promotion and resume."""
from copy import deepcopy

import pytest

from slime_pacman.map_training.controller import TrainingPolicy
from slime_pacman.map_training.sampling import FormalSampler


def sampler(**kwargs):
    rows = [dict(task_id=f'{tier}-{i}', split='train', tier=tier,
                 layout_sha256=f'{tier * 16 + i:064x}')
            for tier in range(3) for i in range(16)]
    return FormalSampler(rows, TrainingPolicy(.8, 1, 12, 1, 8),
                         max_tier=2, **kwargs)


def test_default_starts_at_one_and_replay_never_samples_zero():
    s = sampler()
    plan = s.plan(0)
    assert plan['tier'] == 1
    assert len(plan['groups']) == 16
    assert {g['task']['tier'] for g in plan['groups']} == {1}
    episodes = [dict(group=g['group'], replica=r,
                     task_id=g['task']['task_id'], won=True)
                for g in plan['groups'] for r in range(12)]
    s.complete(0, episodes, plan)
    next_plan = s.plan(1)
    assert next_plan['tier'] == 2
    assert sum(g['task']['tier'] == 2 for g in next_plan['groups']) == 13
    assert sum(g['task']['tier'] == 1 for g in next_plan['groups']) == 3
    restored = sampler()
    restored.restore(s.snapshot(), completed_update=0, bank_version=1)
    assert restored.plan(1) == next_plan


def test_old_identity_cannot_resume_under_new_defaults():
    legacy = sampler(min_tier=0)
    assert legacy.plan(0)['tier'] == 0
    with pytest.raises(ValueError, match='mismatch'):
        sampler().restore(legacy.snapshot(), completed_update=-1, bank_version=0)


def test_excluded_review_tier_is_rejected_transactionally():
    s = sampler()
    plan = deepcopy(s.plan(0))
    plan['groups'][0]['task']['tier'] = 0
    before = s.snapshot()
    with pytest.raises(ValueError, match='invalid frozen'):
        s.complete(0, [], plan)
    assert s.snapshot() == before
