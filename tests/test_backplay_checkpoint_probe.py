"""Frozen checkpoint probes select only requested states and record failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/level1/evaluate/probe_backplay_checkpoint.py'
spec = importlib.util.spec_from_file_location('checkpoint_probe', SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_only_explicit_candidates_are_probed_and_source_version_is_preserved():
    calls = []
    exp = SimpleNamespace(candidates=[{'restart_state_id': s} for s in ('initial', 'missing')],
        report={}, persist=lambda: None,
        probe=lambda entries, label: calls.append((entries, label)) or [{'success_rate': .5}])
    result = probe.run_probe(exp, source_policy_version='15', candidate_ids=['missing'])
    assert calls[0][0] == [{'restart_state_id': 'missing'}]
    assert exp.report['optimizer_updates'] == 0
    assert exp.report['source_policy_version'] == '15'
    assert exp.report['status'] == 'checkpoint_probe_completed'
    assert exp.report['summaries'] == result


def test_unknown_candidate_fails_before_submitting_any_rollout():
    exp = SimpleNamespace(candidates=[{'restart_state_id': 'known'}])
    with pytest.raises(ValueError, match='known candidate'):
        probe.run_probe(exp, source_policy_version='15', candidate_ids=['unknown'])


def test_explicit_order_is_preserved_and_duplicates_are_rejected():
    calls = []
    exp = SimpleNamespace(candidates=[{'restart_state_id': s} for s in ('a', 'b', 'c')],
        report={}, persist=lambda: None, probe=lambda entries, label: calls.append(entries) or [])
    probe.run_probe(exp, source_policy_version='15', candidate_ids=['c', 'a'])
    assert [e['restart_state_id'] for e in calls[0]] == ['c', 'a']
    with pytest.raises(ValueError, match='duplicate'):
        probe.run_probe(exp, source_policy_version='15', candidate_ids=['a', 'a'])


def test_failed_probe_is_persisted_and_reraised():
    def fail(*args):
        raise RuntimeError('incomplete group')
    states = []
    exp = SimpleNamespace(candidates=[{'restart_state_id': 'known'}], report={}, probe=fail)
    exp.persist = lambda: states.append(dict(exp.report))
    with pytest.raises(RuntimeError, match='incomplete group'):
        probe.run_probe(exp, source_policy_version='15', candidate_ids=['known'])
    assert states[-1]['status'] == 'failed'
