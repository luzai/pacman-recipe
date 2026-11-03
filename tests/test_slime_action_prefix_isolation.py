"""Request isolation must preserve current input/support and prove zero cache hits."""
import asyncio
import json

import pytest

from slime_pacman.config import PacmanConfig
from slime_pacman.generation import SGLangGenerator
from test_slime_generation import Client, Processor, CONSTRAINT, messages


def generator(client, mode=None):
    return SGLangGenerator(processor=Processor(), endpoint='http://test/generate',
                           client=client, action_prefix_isolation=mode)


def test_isolation_has_fresh_namespace_per_request_without_changing_input(monkeypatch):
    monkeypatch.delenv('PACMAN_CACHE_CONTRACT', raising=False)
    monkeypatch.setenv('PACMAN_ACTION_PREFIX_ISOLATION', 'unique-extra-key-v1')
    client = Client(lambda result: result['meta_info'].update(cached_tokens=0))
    isolated = generator(client, 'unique-extra-key-v1')
    a = asyncio.run(isolated(messages('red'), CONSTRAINT))
    b = asyncio.run(isolated(messages('red'), CONSTRAINT))
    first, second = client.requests
    assert first['extra_key'] == 'pacman-action-prefix-v1-' + first['rid']
    assert first['extra_key'] != second['extra_key']
    assert first['text'] == second['text']
    assert first['image_data'] == second['image_data']
    assert first['sampling_params'] == second['sampling_params']
    assert first['token_ids_logprob'] == second['token_ids_logprob'] == [66, 70]
    assert a.behavior_log_prob == b.behavior_log_prob
    assert a.candidate_log_probs == b.candidate_log_probs


@pytest.mark.parametrize('cached', [None, 1, -1, False, '0'])
def test_isolation_rejects_missing_or_nonzero_cache_evidence(monkeypatch, cached):
    monkeypatch.delenv('PACMAN_CACHE_CONTRACT', raising=False)
    monkeypatch.setenv('PACMAN_ACTION_PREFIX_ISOLATION', 'unique-extra-key-v1')
    def damage(result):
        if cached is not None:
            result['meta_info']['cached_tokens'] = cached
    with pytest.raises(ValueError, match='explicit zero cached tokens'):
        asyncio.run(generator(Client(damage), 'unique-extra-key-v1')(messages('red'), CONSTRAINT))


@pytest.mark.parametrize('mode,environment', [
    ('unique-extra-key-v1', 'off'), ('off', 'unique-extra-key-v1'), ('off', 'unknown'),
])
def test_config_environment_mismatch_is_rejected(monkeypatch, mode, environment):
    monkeypatch.setenv('PACMAN_ACTION_PREFIX_ISOLATION', environment)
    with pytest.raises(ValueError, match='config/environment mismatch'):
        generator(Client(), mode)


def test_default_off_and_independent_legacy_caller_remain_unmodified(monkeypatch):
    monkeypatch.delenv('PACMAN_CACHE_CONTRACT', raising=False)
    monkeypatch.delenv('PACMAN_ACTION_PREFIX_ISOLATION', raising=False)
    client = Client()
    asyncio.run(generator(client, 'off')(messages('red'), CONSTRAINT))
    assert 'extra_key' not in client.requests[-1]
    # An independent sensor/diagnostic caller has not opted into the policy mode.
    monkeypatch.setenv('PACMAN_ACTION_PREFIX_ISOLATION', 'unique-extra-key-v1')
    asyncio.run(generator(client)(messages('blue'), CONSTRAINT))
    assert 'extra_key' not in client.requests[-1]
    assert PacmanConfig().action_prefix_isolation == 'off'


def test_fixed_cache_arm_cannot_be_combined_with_isolation(monkeypatch, tmp_path):
    contract = tmp_path/'fixed.json'
    contract.write_text(json.dumps({'mode': 'fixed'}))
    monkeypatch.setenv('PACMAN_CACHE_CONTRACT', str(contract))
    monkeypatch.setenv('PACMAN_ACTION_PREFIX_ISOLATION', 'unique-extra-key-v1')
    with pytest.raises(ValueError, match='conflicts with fixed-prefix caching'):
        generator(Client(), 'unique-extra-key-v1')


@pytest.mark.parametrize('mode', [True, 1, 'unique', ''])
def test_unknown_config_mode_rejected(mode):
    with pytest.raises(ValueError, match='unsupported action_prefix_isolation'):
        PacmanConfig(action_prefix_isolation=mode)
    with pytest.raises(ValueError, match='unsupported action_prefix_isolation'):
        generator(Client(), mode)
