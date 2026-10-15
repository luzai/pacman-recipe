from copy import deepcopy

import pytest

from slime_pacman.map_training.controller import TrainingPolicy, TrainingWinController
from slime_pacman.map_training.identity import digest
from slime_pacman.map_training.sampling import FormalSampler


def catalog():
    rows = [dict(task_id=f't{tier}-map{i}', split='train', tier=tier,
                 layout_sha256=f'{tier * 8 + i:064x}', game_seed=2050 + tier,
                 episode_record=dict(id=f't{tier}-map{i}', environment=dict(seed=2050 + tier)))
            for tier in range(2) for i in range(8)]
    rows.append(dict(task_id='fixed-validation', split='validation', tier=1,
                     layout_sha256='f' * 64, game_seed=2054,
                     episode_record=dict(environment=dict(seed=2054))))
    return rows


def sampler(rows=None, **kwargs):
    return FormalSampler(rows or catalog(), TrainingPolicy(.8, 3, 384, 2, 8),
                         max_tier=1, min_tier=kwargs.pop('min_tier', 0), **kwargs)


def complete(s, plan):
    return s.complete(plan['update'], [dict(group=g['group'], replica=r,
                     task_id=g['task']['task_id'], won=False)
                     for g in plan['groups'] for r in range(12)], plan=plan)


def test_group_environment_is_shared_and_model_sampling_is_diverse():
    s = sampler()
    # Include three review groups as well as the current tier.
    s.controller.state['tier'] = 1
    plan = s.plan(0)
    assert len(plan['groups']) == 16
    assert sum(g['task']['tier'] == 1 for g in plan['groups']) == 13
    assert len({g['environment_seed'] for g in plan['groups']}) == 16
    for g in plan['groups']:
        assert len(g['sampling_seeds']) == len(set(g['sampling_seeds'])) == 12
        assert g['environment_seed'] == g['task']['game_seed']
        assert g['environment_seed'] == g['task']['episode_record']['environment']['seed']
    # The same map can recur in a batch, but gets a distinct group seed.
    assert len({g['task']['task_id'] for g in plan['groups']}) < 16


def test_catalog_and_validation_stay_frozen():
    rows = catalog()
    original = deepcopy(rows)
    s = sampler(rows)
    plan = s.plan(0)
    assert rows == original == s.catalog
    assert all(g['task']['split'] == 'train' for g in plan['groups'])
    for g in plan['groups']:
        source = next(r for r in original if r['task_id'] == g['task']['task_id'])
        assert g['environment_seed'] != source['game_seed']
        assert g['task']['layout_sha256'] == source['layout_sha256']


def test_retry_and_resume_reproduce_seeds_but_next_batch_changes_them():
    s = sampler()
    first = s.plan(0)
    assert first == s.plan(0)
    complete(s, first)
    saved = s.snapshot()
    second = s.plan(1)
    restored = sampler()
    restored.restore(saved, completed_update=0, bank_version=saved['state']['bank_version'])
    assert second == restored.plan(1)
    assert {g['environment_seed'] for g in first['groups']}.isdisjoint(
        g['environment_seed'] for g in second['groups'])


def test_legacy_mode_preserves_identity_and_cannot_mix_resume_modes():
    rows = catalog()
    fixed = sampler(rows, environment_seed_mode='fixed_catalog')
    assert fixed.controller.identity == TrainingWinController(TrainingPolicy(.8, 3, 384, 2, 8),
                            catalog_identity=digest([rows, 20261009]), max_tier=1).identity
    plan = fixed.plan(0)
    for g in plan['groups']:
        assert g['task'] == next(r for r in rows if r['task_id'] == g['task']['task_id'])
        assert 'environment_seed' not in g
    complete(fixed, plan)
    with pytest.raises(ValueError):
        sampler().restore(fixed.snapshot(), completed_update=0, bank_version=0)
    with pytest.raises(ValueError, match='seed mode'):
        sampler(environment_seed_mode='per_replica')


@pytest.mark.parametrize('mode', ['group_randomized', 'fixed_catalog'])
def test_runtime_config_preserves_validation_and_freezes_training(tmp_path, monkeypatch, mode):
    import json
    from types import SimpleNamespace
    from slime_pacman.formal_rollout import Runtime

    rows = catalog()
    for row in rows:
        row.update(ghosts=2, game_root='/private/game')
        row['episode_record']['environment'].update(
            level_revision=row['layout_sha256'], ghost_mode='normal')
    config = dict(catalog=rows, seed=20261009, max_tier=1,
                  environment_seed_mode=mode,
                  policy=dict(threshold=.8, window_batches=3, min_games=384,
                              consecutive_checks=2, min_maps=8))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('PACMAN_FORMAL_CONFIG', str(path))
    monkeypatch.setenv('PACMAN_RUN_DIR', str(tmp_path))
    rt = Runtime(SimpleNamespace())
    plan = rt.freeze_plan(0)
    assert rt.catalog == rows
    assert rt.sampler.environment_seed_mode == mode
    assert json.loads((tmp_path / 'map-plans/rollout-0000.json').read_text()) == plan
    if mode == 'group_randomized':
        assert all(g['task']['episode_record']['environment']['seed'] == g['environment_seed']
                   for g in plan['groups'])
