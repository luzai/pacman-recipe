"""Bounded probe concurrency keeps attribution independent of completion order."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pacman_recipe.level1 import backplay_runner as runner


@pytest.fixture
def probe_engine(tmp_path, monkeypatch):
    timeline, pending, cleared = [], [], []
    settings = {'malformed': None, 'version': 0, 'change_after_wait': False}
    def submit(row, workflow, kwargs, **unused):
        group = row['group']
        state = row['state']
        timeline.append(('submit', group))
        pending.append(group)
        directory = Path(kwargs['trajectory_dir'])
        directory.mkdir(parents=True)
        count = 11 if settings['malformed'] == 'missing' else 12
        for i in range(count):
            payload = {'restart_state': {'id': 'foreign' if settings['malformed'] == 'wrong' else state},
                       'trajectory_sample_id': group + '-' + str(0 if settings['malformed'] == 'duplicate' else i),
                       'won': state == 'a', 'total_shaped_reward': i,
                       'suffix_normal_pellets_eaten': 1, 'steps': 3}
            (directory / f'{i}.json').write_text(json.dumps(payload))
    def wait(*args, **kwargs):
        group = pending.pop()  # Deliberately complete in reverse submission order.
        timeline.append(('wait', group))
        if settings['change_after_wait']: settings['version'] = 1
        return group
    monkeypatch.setitem(sys.modules, 'areal.infra.utils.concurrent',
                        SimpleNamespace(run_async_task=lambda fn, value: fn(value)))
    monkeypatch.setitem(sys.modules, 'areal.trainer.rl_trainer',
                        SimpleNamespace(_clear_eval_result=lambda result: cleared.append(result)))
    monkeypatch.setattr(runner, 'bind_restart_row',
                        lambda template, entry, bank, group: {'state': entry['restart_state_id'], 'group': group})
    exp = runner.BackplayExperiment.__new__(runner.BackplayExperiment)
    exp.trainer = SimpleNamespace(eval_rollout=SimpleNamespace(
        get_version=lambda: settings['version'], submit=submit, wait=wait))
    exp.version = 0
    exp.probe_counter = 0
    exp.output_dir = tmp_path
    exp.probe_kwargs = {}
    exp.template = {}
    exp.bank_dir = tmp_path
    exp.workflow = 'native'
    exp.probe_samples = 24
    exp.probe_state_batch_size = 2
    exp.event = lambda *args, **kwargs: None
    return exp, settings, timeline, cleared


def test_four_groups_submitted_before_wait_and_odd_tail_keeps_attribution(probe_engine):
    exp, _, timeline, cleared = probe_engine
    summaries = exp.probe([{'restart_state_id': s} for s in ('a', 'b', 'c')], 'test')
    assert [kind for kind, _ in timeline] == ['submit'] * 4 + ['wait'] * 4 + ['submit'] * 2 + ['wait'] * 2
    assert len(cleared) == len(set(cleared)) == 6
    by_id = {s['restart_state_id']: s for s in summaries}
    assert {key: s['samples'] for key, s in by_id.items()} == {'a': 24, 'b': 24, 'c': 24}
    assert by_id['a']['successes'] == 24
    assert by_id['b']['successes'] == by_id['c']['successes'] == 0


@pytest.mark.parametrize('malformed,error', [('missing', RuntimeError), ('wrong', RuntimeError), ('duplicate', ValueError)])
def test_invalid_complete_groups_rejected_after_every_result_cleared(probe_engine, malformed, error):
    exp, settings, _, cleared = probe_engine
    settings['malformed'] = malformed
    with pytest.raises(error): exp.probe([{'restart_state_id': s} for s in ('a', 'b')], 'bad')
    assert len(cleared) == len(set(cleared)) == 4


def test_wrong_policy_rejected_before_submission(probe_engine):
    exp, settings, timeline, cleared = probe_engine
    settings['version'] = 1
    with pytest.raises(RuntimeError, match='policy version'):
        exp.probe([{'restart_state_id': 'a'}], 'stale')
    assert not timeline and not cleared


def test_policy_change_during_batch_clears_all_results_then_rejects(probe_engine):
    exp, settings, _, cleared = probe_engine
    settings['change_after_wait'] = True
    with pytest.raises(RuntimeError, match='policy version changed'):
        exp.probe([{'restart_state_id': s} for s in ('a', 'b', 'c')], 'changed')
    assert len(cleared) == 4


def test_duplicate_candidate_rejected_before_submission(probe_engine):
    exp, _, timeline, _ = probe_engine
    with pytest.raises(ValueError, match='duplicate probe candidate'):
        exp.probe([{'restart_state_id': 'a'}] * 2, 'duplicate')
    assert not timeline


def test_original_one_state_mode_remains_serial(probe_engine):
    exp, _, timeline, _ = probe_engine
    exp.probe_state_batch_size = 1
    exp.probe([{'restart_state_id': s} for s in ('a', 'b')], 'serial')
    assert [kind for kind, _ in timeline] == (['submit'] * 2 + ['wait'] * 2) * 2


def test_larger_sample_count_never_exceeds_four_outstanding_groups(probe_engine):
    exp, _, timeline, _ = probe_engine
    exp.probe_samples = 48
    exp.probe([{'restart_state_id': s} for s in ('a', 'b')], 'larger')
    outstanding = 0
    for kind, _ in timeline:
        outstanding += 1 if kind == 'submit' else -1
        assert 0 <= outstanding <= 4
    assert outstanding == 0


@pytest.mark.parametrize('operation', ['submit', 'wait'])
def test_transport_failure_cannot_publish_probe_completion(probe_engine, operation):
    exp, _, _, _ = probe_engine
    events = []
    exp.event = lambda *args, **kwargs: events.append((args, kwargs))
    def fail(*args, **kwargs): raise RuntimeError('transport failed')
    setattr(exp.trainer.eval_rollout, operation, fail)
    with pytest.raises(RuntimeError, match='transport failed'):
        exp.probe([{'restart_state_id': s} for s in ('a', 'b')], 'failed')
    assert not events
    assert not list(exp.output_dir.rglob('summary.json'))
