"""Exercise insertion/matching and cleanup guards without a model or GPU."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
from slime_pacman.fixed_prefix import declare_prefix


def guard(monkeypatch,tmp_path):
    prefix=declare_prefix([list(range(620))+[x] for x in (800,801)],fixed_boundary_lengths=[615,615])
    path=tmp_path/'prefix.json';path.write_text(json.dumps(prefix))
    monkeypatch.setenv('PACMAN_FIXED_PREFIX_FILE',str(path))
    class Key:
        def __init__(self,ids,extra_key=None):self.ids,self.extra_key=ids,extra_key
        def raw_token_ids(self):return self.ids
        def __getitem__(self,index):return Key(self.ids[index],self.extra_key)
    stub=types.ModuleType('sglang.srt.mem_cache.radix_cache');stub.RadixKey=Key
    monkeypatch.setitem(sys.modules,stub.__name__,stub)
    @dataclasses.dataclass
    class Params:
        key: object
        chunked: bool=False
        cow_mamba: bool=True
        req: object=None
    class Cache:
        def __init__(self):
            self.disable=False;self.mamba_cache_chunk_size=64;self.page_size=1
            self.enable_mamba_extra_buffer=True;self.int8_ckpt_pool=None
        def match_prefix(self,p):return p
        def insert(self,p):return p
        def cache_finished_req(self,req,is_insert=True):return is_insert
    spec=importlib.util.spec_from_file_location('prefix_guard',Path(__file__).resolve().parents[1]/'patches/sglang/fixed_prefix_cache.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.install(Cache)
    return Cache(),Key,Params


def test_matching_never_leaks_partial_foreign_or_extra_prefix(monkeypatch,tmp_path):
    cache,Key,Params=guard(monkeypatch,tmp_path)
    for ids,salt in [(list(range(575)),None),([999]+list(range(1,700)),None),(list(range(700)),'foreign')]:
        assert cache.match_prefix(Params(Key(ids,salt))).key.ids==[]
    req=object(); p=cache.match_prefix(Params(Key(list(range(700))),req=req))
    assert p.key.ids==list(range(576)) and p.cow_mamba and p.req is req


def test_only_exact_prefix_can_insert_and_only_exact_state_can_be_donated(monkeypatch,tmp_path):
    cache,Key,Params=guard(monkeypatch,tmp_path)
    assert cache.insert(Params(Key(list(range(576))))).key.ids==list(range(576))
    for ids in [list(range(575)),list(range(577)),[999]+list(range(1,576))]:
        with pytest.raises(ValueError):cache.insert(Params(Key(ids)))
    with pytest.raises(ValueError):cache.insert(Params(Key(list(range(576))),chunked=True))
    req=types.SimpleNamespace(origin_input_ids=list(range(576)),extra_key=None,mamba_last_track_seqlen=576)
    assert cache.cache_finished_req(req)
    req.mamba_last_track_seqlen=640
    assert not cache.cache_finished_req(req)
    req.mamba_last_track_seqlen=576;req.origin_input_ids.append(900)
    assert not cache.cache_finished_req(req)


def test_matching_preserves_pinned_array_storage(monkeypatch,tmp_path):
    from array import array
    cache,Key,Params=guard(monkeypatch,tmp_path)
    result=cache.match_prefix(Params(Key(array('q',range(700)))))
    assert isinstance(result.key.ids,array)
    assert list(result.key.ids)==list(range(576))
