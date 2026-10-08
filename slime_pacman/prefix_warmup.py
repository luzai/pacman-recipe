"""Warm every explicit engine after weight/cache changes and before any games."""
import asyncio
import json
from pathlib import Path
from .fixed_prefix import validate_prefix


async def warm_engines(client, endpoints, contract, version, *, canary_ids):
    tokens=validate_prefix(contract['prefix'])
    if contract['mode']=='off': return []
    if (not endpoints or len(set(endpoints))!=len(endpoints)
            or tuple(canary_ids[:len(tokens)])!=tokens or len(canary_ids)<=len(tokens)):
        raise ValueError('Explicit distinct engines and a real frozen prompt canary required')
    async def warm(endpoint):
        endpoint=endpoint.rstrip('/');endpoint=endpoint.removesuffix('/generate')
        info=await client.get(endpoint+'/get_model_info');info.raise_for_status()
        if str(info.json()['weight_version'])!=str(version):raise ValueError('Warmup policy version differs')
        params=dict(temperature=0,max_new_tokens=1,ignore_eos=True)
        body=dict(input_ids=list(tokens),sampling_params=params)
        response=await client.post(endpoint+'/generate',json=body);response.raise_for_status()
        if str(response.json()['meta_info']['weight_version'])!=str(version):raise ValueError('Policy changed during warmup')
        body['input_ids']=list(canary_ids[:len(tokens)+1])
        response=await client.post(endpoint+'/generate',json=body);response.raise_for_status()
        meta=response.json()['meta_info']
        if meta.get('cached_tokens')!=len(tokens) or str(meta['weight_version'])!=str(version):
            raise ValueError('Fixed node was not warmed at the required policy version')
        return dict(endpoint=endpoint,version=str(version),cached_tokens=len(tokens),prefix_sha256=contract['prefix']['token_ids_sha256'])
    return await asyncio.gather(*(warm(e) for e in endpoints))


async def warm_router(args, version):
    """Call from the frozen data-source prepare boundary before initialization/eval/rollout."""
    import os
    import httpx
    path=os.environ.get('PACMAN_CACHE_CONTRACT')
    if not path:return []
    contract=json.loads(Path(path).read_text())
    if contract['mode']=='off':return []
    from slime.rollout.sglang_rollout import get_model_url
    from slime.utils.http_utils import get_rollout_num_engines
    router=get_model_url(args,'policy').removesuffix('/generate')
    async with httpx.AsyncClient(timeout=120) as client:
        response=await client.get(router+'/workers');response.raise_for_status()
        workers=response.json()['workers']
        endpoints=[w['url'] for w in workers]
        if len(endpoints)!=get_rollout_num_engines(args):
            raise ValueError('Cannot prove every declared rollout engine was warmed')
        return await warm_engines(client,endpoints,contract,version,canary_ids=contract['canary_ids'])
