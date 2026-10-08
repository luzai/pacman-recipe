"""Opt-in guard for pinned MambaRadixCache: only one exact fixed-text node.

Mount at sglang/srt/mem_cache/pacman_fixed_prefix_cache.py and append
`from .pacman_fixed_prefix_cache import install; install(MambaRadixCache)`.
PACMAN_FIXED_PREFIX_FILE points at the frozen declaration. No variable-length
Mamba state is relabelled as the fixed prefix; only an exact warmup can insert.
"""
import dataclasses
import json
import os
from pathlib import Path


def install(cls):
    declaration_path = os.environ.get('PACMAN_FIXED_PREFIX_FILE')
    if not declaration_path:
        return
    from slime_pacman.fixed_prefix import validate_prefix
    tokens = validate_prefix(json.loads(Path(declaration_path).read_text()))
    original_init, original_match = cls.__init__, cls.match_prefix
    original_insert, original_finished = cls.insert, cls.cache_finished_req

    def allowed(key, exact=False):
        ids = tuple(key.raw_token_ids())
        return key.extra_key is None and len(ids)>=len(tokens) and ids[:len(tokens)]==tokens and (not exact or len(ids)==len(tokens))

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if (self.disable or self.mamba_cache_chunk_size!=64 or self.page_size!=1
                or not self.enable_mamba_extra_buffer or self.int8_ckpt_pool is not None):
            raise ValueError('Fixed prefix requires enabled radix, 64 chunks, page 1, FP32 extra Mamba buffer')

    def match(self, params):
        # Empty key refuses partial and foreign-prefix matches, without splitting nodes.
        key = params.key[:len(tokens)] if allowed(params.key) else params.key[:0]
        return original_match(self, dataclasses.replace(params, key=key))

    def insert(self, params):
        if not allowed(params.key, exact=True) or params.chunked:
            raise ValueError('Attempt to insert a non-fixed cache node')
        return original_insert(self, params)

    def finished(self, req, is_insert=True):
        # GDN state is only valid for its exact tracked length. Ordinary decisions
        # use upstream's no-insert cleanup, freeing suffix KV and transient SSM.
        warm = (tuple(req.origin_input_ids)==tokens and req.extra_key is None
                and req.mamba_last_track_seqlen==len(tokens))
        return original_finished(self, req, is_insert=bool(is_insert and warm))

    def unfinished(self, req, chunked=False):
        if chunked:
            raise ValueError('Fixed-prefix experiment forbids chunked prefill')
        # Upstream no-insert path preserves local prefill KV for scheduling.
        indices=self.req_to_token_pool.req_to_token[req.req_pool_idx,:req.extend_range.end]
        import torch
        req.prefix_indices=indices.to(dtype=torch.int64,copy=True)

    cls.__init__, cls.match_prefix, cls.insert = initialize, match, insert
    cls.cache_finished_req, cls.cache_unfinished_req = finished, unfinished
