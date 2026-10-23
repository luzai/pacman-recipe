import asyncio

import pytest

from slime_pacman.fixed_prefix import declare_prefix
from slime_pacman.prefix_warmup import warm_engines


@pytest.mark.parametrize('prewarm_hit', [0, 512])
def test_warmup_records_observed_hit_without_rejecting_same_version_rewarm(prewarm_hit):
    prefix = declare_prefix([list(range(540)) + [x] for x in (900, 901)],
                            fixed_boundary_lengths=[524, 524])

    class Response:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Client:
        async def get(self, url): return Response({'weight_version': '2'})
        async def post(self, url, *, json):
            count = len(json['input_ids'])
            hit = prewarm_hit if count == 512 else 512
            return Response({'meta_info': {'weight_version': '2', 'cached_tokens': hit}})

    receipt = asyncio.run(warm_engines(Client(), ['http://engine'],
        {'mode': 'fixed', 'prefix': prefix}, '2', canary_ids=list(range(540))))[0]
    assert receipt == dict(endpoint='http://engine', version='2',
        prewarm_cached_tokens=prewarm_hit, cached_tokens=512,
        prefix_sha256=prefix['token_ids_sha256'])


@pytest.mark.parametrize('bad_hit', [None, True, -1, 513, '0'])
def test_missing_or_invalid_prewarm_evidence_fails_closed(bad_hit):
    prefix = declare_prefix([list(range(540)) + [x] for x in (900, 901)],
                            fixed_boundary_lengths=[524, 524])

    class Response:
        def __init__(self, payload): self.payload = payload
        def raise_for_status(self): pass
        def json(self): return self.payload

    class Client:
        async def get(self, url): return Response({'weight_version': '2'})
        async def post(self, url, *, json):
            return Response({'meta_info': {'weight_version': '2', 'cached_tokens': bad_hit}})

    with pytest.raises(ValueError, match='prewarm'):
        asyncio.run(warm_engines(Client(), ['http://engine'],
            {'mode': 'fixed', 'prefix': prefix}, '2', canary_ids=list(range(540))))
