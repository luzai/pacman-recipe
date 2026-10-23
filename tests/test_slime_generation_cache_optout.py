"""VQA can use a different prefix only with explicit evidence of zero cache hits."""
import asyncio
import json
import math
import os

import pytest

from slime_pacman.generation import SGLangGenerator
from test_slime_generation import Client, Processor, CONSTRAINT, messages


def make(client, **kwargs):
    return SGLangGenerator(processor=Processor(), endpoint='http://test/generate',
                           client=client, **kwargs)


def test_explicit_optout_preserves_image_and_probability_contract(monkeypatch, tmp_path):
    path = tmp_path / 'policy-only.json'
    path.write_text('not a VQA contract')
    monkeypatch.setenv('PACMAN_CACHE_CONTRACT', str(path))
    client = Client(lambda r: r['meta_info'].update(cached_tokens=0))
    generator = make(client, cache_contract_enabled=False)
    red = asyncio.run(generator(messages('red'), CONSTRAINT))
    blue = asyncio.run(generator(messages('blue'), CONSTRAINT))
    assert red.image_sha256 != blue.image_sha256
    assert red.behavior_log_prob == blue.behavior_log_prob == -math.log(2)
    assert red.candidate_log_probs == [-math.log(2)] * 2
    assert all(r['token_ids_logprob'] == [66, 70] for r in client.requests)
    assert client.requests[0]['image_data'] != client.requests[1]['image_data']
    assert os.environ['PACMAN_CACHE_CONTRACT'] == str(path)
    # The independent instance does not disable validation for default policy calls.
    with pytest.raises(json.JSONDecodeError):
        make(Client())


@pytest.mark.parametrize('cached', [None, 512, -1, False, '0'])
def test_optout_requires_explicit_integer_zero(cached):
    def evidence(r):
        if cached is not None:
            r['meta_info']['cached_tokens'] = cached
    with pytest.raises(ValueError, match='explicit zero cached tokens'):
        asyncio.run(make(Client(evidence), cache_contract_enabled=False)(messages('red'), CONSTRAINT))


def test_optout_still_rejects_wrong_legal_support():
    def evidence(r):
        r['meta_info'].update(cached_tokens=0, output_token_ids_logprobs=[[[-math.log(2), 66, 'B']]])
    with pytest.raises(ValueError, match='probability support differs'):
        asyncio.run(make(Client(evidence), cache_contract_enabled=False)(messages('red'), CONSTRAINT))


def test_optout_flag_requires_boolean():
    with pytest.raises(ValueError, match='explicit boolean'):
        make(Client(), cache_contract_enabled='false')
