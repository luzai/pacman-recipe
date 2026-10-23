"""Frozen runtime boundary: prove common served version and warm every engine.

Called after KV onload and before each eval/generate. No cached 'already warmed'
shortcut: offload/flush may remove the node without changing weight version.
"""
import asyncio
import json
import os
from pathlib import Path
import time
import uuid


async def prepare(args, boundary, rollout_id):
    import httpx
    from slime.rollout.sglang_rollout import get_model_url
    from slime.utils.http_utils import get_rollout_num_engines
    from slime_pacman.prefix_warmup import warm_router

    path = os.environ.get('PACMAN_CACHE_CONTRACT')
    if not path:
        raise ValueError('New cache-aware run requires an explicit cache contract')
    contract = json.loads(Path(path).read_text())
    router = get_model_url(args, 'policy').removesuffix('/generate')
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(router + '/workers')
        response.raise_for_status()
        endpoints = [worker['url'].rstrip('/') for worker in response.json()['workers']]
        if not endpoints or len(set(endpoints)) != len(endpoints) or len(endpoints) != get_rollout_num_engines(args):
            raise ValueError('Cannot account for every rollout engine')
        responses = await asyncio.gather(*(client.get(endpoint + '/get_model_info') for endpoint in endpoints))
        versions = []
        for response in responses:
            response.raise_for_status()
            versions.append(str(response.json()['weight_version']))
        if len(set(versions)) != 1 or not versions[0].isascii() or not versions[0].isdigit():
            raise ValueError('Mixed or missing served weight versions at cache boundary')
    version = versions[0]
    rows = await warm_router(args, version)
    if contract['mode'] != 'off':
        if len(rows) != len(endpoints) or {row['endpoint'].rstrip('/') for row in rows} != set(endpoints):
            raise ValueError('Warmup receipt does not cover the discovered engines')
        if any(str(row['version']) != version for row in rows):
            raise ValueError('Warmup receipt version mismatch')
    args._rollout_weight_version = int(version)
    receipt = dict(boundary=boundary, rollout_id=rollout_id, served_version=version,
                   mode=contract['mode'], engines=endpoints, warmup=rows,
                   timestamp=time.time(), training=False)
    output = Path(os.environ['PACMAN_RUN_DIR']) / 'cache-boundaries'
    output.mkdir(parents=True, exist_ok=True)
    with (output / (uuid.uuid4().hex + '.json')).open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
    print('CACHE_BOUNDARY', json.dumps(receipt), flush=True)
    return receipt
