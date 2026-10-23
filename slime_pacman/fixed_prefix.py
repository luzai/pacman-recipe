"""Immutable ASCII cache declaration; never change or pad the policy prompt."""
import hashlib
import json
from pathlib import Path


def digest_tokens(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def declare_prefix(tokenized_requests, *, fixed_boundary_lengths, chunk_size=64):
    """Derive a prefix bounded by the actual fixed text, not coincidentally shared state."""
    rows = [list(x) for x in tokenized_requests]
    boundaries = list(fixed_boundary_lengths)
    if len(rows) < 2 or len(rows) != len(boundaries) or chunk_size != 64:
        raise ValueError('Multiple real prompts and one fixed boundary per prompt required')
    if any(type(t) is not int or t < 0 for row in rows for t in row):
        raise ValueError('Nonnegative integer token IDs required')
    if any(type(n) is not int or not 0 < n <= len(row) for row,n in zip(rows,boundaries)):
        raise ValueError('Invalid fixed-text boundary')
    shared = min(map(len, rows))
    for row in rows[1:]:
        shared = min(shared, next((i for i,(a,b) in enumerate(zip(rows[0],row)) if a!=b),len(row)))
    length = min(shared, min(boundaries)) // chunk_size * chunk_size
    if not length:
        raise ValueError('No complete cache chunk within fixed prefix')
    ids = rows[0][:length]
    return dict(schema='pacman-fixed-prefix-v1', token_ids=ids, length=length,
                chunk_size=chunk_size, shared_length=shared,
                fixed_boundary_lengths=sorted(set(boundaries)), token_ids_sha256=digest_tokens(ids))


def validate_prefix(value):
    ids = value['token_ids']; length = value['length']
    if (value.get('schema')!='pacman-fixed-prefix-v1' or type(length) is not int
            or length<=0 or len(ids)!=length or value.get('chunk_size')!=64 or length%64
            or any(type(t) is not int or t<0 for t in ids)
            or value.get('token_ids_sha256')!=digest_tokens(ids)
            or length>value['shared_length'] or length>min(value['fixed_boundary_lengths'])):
        raise ValueError('Fixed prefix identity or boundary differs')
    return tuple(ids)


def cache_contract(prefix, *, mode, patch_path, startup_args, deterministic, workers=192, max_running_requests=32):
    validate_prefix(prefix)
    if mode not in ('off','fixed') or workers!=192 or max_running_requests not in (32,64):
        raise ValueError('Explicit off/fixed arm, 192 workers and 32/64 concurrency required')
    args = list(startup_args)
    if ('--enable-deterministic-inference' in args)!=deterministic:
        raise ValueError('Determinism declaration differs from startup arguments')
    if ('--disable-radix-cache' in args)!=(mode=='off'):
        raise ValueError('Cache declaration differs from startup arguments')
    if args[args.index('--max-running-requests')+1]!=str(max_running_requests):
        raise ValueError('Actual concurrency differs')
    patch_sha = hashlib.sha256(Path(patch_path).read_bytes()).hexdigest()
    return dict(schema='pacman-fixed-cache-runtime-v1', mode=mode, prefix=prefix,
                patch_sha256=patch_sha, startup_args=args, deterministic=deterministic,
                episode_workers=workers, max_running_requests=max_running_requests,
                warmup='each_engine_start_and_every_weight_version_before_dispatch',
                other_prefixes='never_insert_or_hit', mutable_within_run=False)


def assert_same_contract(frozen, actual):
    if frozen != actual:
        raise ValueError('Cache configuration cannot change within an experiment line')


def vision_cache_contract(prefix, *, patch_path, startup_args, canary_ids,
                          vision_start_token_id, workers=16):
    """Separate Vision contract; never interpret shared image placeholders as text."""
    tokens = validate_prefix(prefix)
    args = list(startup_args)
    for flag, expected in [('--max-running-requests', '8'),
                           ('--context-length', '8192'),
                           ('--max-total-tokens', '65536')]:
        if args.count(flag) != 1 or args.index(flag)+1 >= len(args) or args[args.index(flag)+1] != expected:
            raise ValueError('Vision cache startup differs: '+flag)
    if '--disable-radix-cache' in args or workers != 16:
        raise ValueError('Vision fixed cache requires enabled radix and 16 episode workers')
    value = dict(schema='pacman-vision-fixed-cache-v1', mode='fixed',
                 observation_mode='image', prefix=prefix, canary_ids=list(canary_ids),
                 vision_start_token_id=vision_start_token_id,
                 patch_sha256=hashlib.sha256(Path(patch_path).read_bytes()).hexdigest(),
                 startup_args=args, episode_workers=workers, max_running_requests=8,
                 warmup='each_engine_start_and_every_weight_version_before_dispatch',
                 other_prefixes='never_insert_or_hit', mutable_within_run=False)
    validate_vision_request(value, canary_ids)
    if len(tokens)+1 >= list(canary_ids).index(vision_start_token_id):
        raise ValueError('Warmup canary must remain strictly before the image')
    return value


def validate_vision_request(contract, ids):
    tokens = validate_prefix(contract['prefix'])
    if (contract.get('schema') != 'pacman-vision-fixed-cache-v1'
            or contract.get('observation_mode') != 'image' or contract.get('mode') != 'fixed'
            or type(contract.get('vision_start_token_id')) is not int):
        raise ValueError('Explicit Vision fixed-cache contract required')
    ids = list(ids)
    if (tuple(ids[:len(tokens)]) != tokens
            or contract['vision_start_token_id'] not in ids
            or ids.index(contract['vision_start_token_id']) < len(tokens)):
        raise ValueError('Vision prefix identity or actual image boundary differs')
    return tokens
