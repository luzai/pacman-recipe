import json
import pytest
from slime_pacman.map_training.sampling import FormalSampler
from slime_pacman.map_training.checkpoints import publish, restore
from slime_pacman.map_training.controller import TrainingPolicy


def sampler():
    rows = [dict(task_id=f'{tier}-{i}', split='train', tier=tier,
                 layout_sha256=f'{tier * 16 + i:064x}') for tier in range(2) for i in range(16)]
    return FormalSampler(rows, TrainingPolicy(.8, 3, 384, 2, 8), max_tier=1, min_tier=0)


def episodes(plan, won=True):
    return [dict(group=g['group'], replica=r, task_id=g['task']['task_id'], won=won)
            for g in plan['groups'] for r in range(12)]


def test_fixed_quota_promotion_restore():
    first = sampler()
    for update in range(4):
        plan = first.plan(update)
        assert len(plan['groups']) == 16
        assert len({g['task']['layout_sha256'] for g in plan['groups']}) == 16
        first.complete(update, episodes(plan), plan)
    assert first.controller.state['tier'] == 1
    next_plan = first.plan(4)
    assert sum(g['task']['tier'] == 1 for g in next_plan['groups']) == 13
    assert sum(g['task']['tier'] == 0 for g in next_plan['groups']) == 3
    second = sampler()
    second.restore(first.snapshot(), completed_update=3, bank_version=1)
    assert next_plan == second.plan(4)


def test_incomplete_batch_does_not_commit():
    instance = sampler()
    before = instance.snapshot()
    with pytest.raises(ValueError):
        instance.complete(0, episodes(instance.plan(0))[:-1])
    assert instance.snapshot() == before


def checkpoint(tmp_path):
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('0')
    model = tmp_path / 'iter_0000000'
    model.mkdir()
    (model / 'common.pt').write_bytes(b'model optimizer rng fixture')
    state = dict(identity='controller', state=dict(last_update=0, bank_version=0))
    return state, model


def test_joint_roundtrip_and_changed_payload(tmp_path):
    state, model = checkpoint(tmp_path)
    publish(tmp_path, state, identity='run', update=0)
    assert restore(tmp_path, identity='run')['controller'] == state
    (model / 'common.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='payload changed'):
        restore(tmp_path, identity='run')


def test_unpaired_model_is_rejected(tmp_path):
    state, _ = checkpoint(tmp_path)
    publish(tmp_path, state, identity='run', update=0)
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('1')
    with pytest.raises(ValueError, match='identity/update'):
        restore(tmp_path, identity='run')


def test_unfinished_publish_has_no_latest(tmp_path):
    state, _ = checkpoint(tmp_path)
    state['state']['last_update'] = -1
    with pytest.raises(ValueError, match='controller update'):
        publish(tmp_path, state, identity='run', update=0)
    assert not (tmp_path / 'joint/latest.json').exists()
