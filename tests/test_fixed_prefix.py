import copy
import pytest
from slime_pacman.fixed_prefix import declare_prefix, validate_prefix, cache_contract, assert_same_contract


def test_fixed_text_boundary_caps_coincidentally_shared_dynamic_state():
    rows=[list(range(700))+[x] for x in (900,901)]
    value=declare_prefix(rows,fixed_boundary_lengths=[615,615])
    assert value['length']==576 and value['shared_length']==700
    assert validate_prefix(value)==tuple(range(576))
    assert rows[0]==list(range(700))+[900]


def test_changed_token_or_illegal_alignment_is_rejected():
    value=declare_prefix([list(range(620))+[x] for x in (800,801)],fixed_boundary_lengths=[615,615])
    for change in ('tokens','length'):
        bad=copy.deepcopy(value)
        if change=='tokens': bad['token_ids'][0]=88
        else: bad['length']=575
        with pytest.raises(ValueError):validate_prefix(bad)


def test_frozen_cache_arm_and_actual_startup_flags(tmp_path):
    prefix=declare_prefix([list(range(620))+[x] for x in (800,801)],fixed_boundary_lengths=[615,615])
    patch=tmp_path/'patch.py';patch.write_text('guard')
    fixed=cache_contract(prefix,mode='fixed',patch_path=patch,
        startup_args=['--max-running-requests','32','--enable-deterministic-inference'],deterministic=True)
    assert_same_contract(fixed,copy.deepcopy(fixed))
    changed=copy.deepcopy(fixed);changed['mode']='off'
    with pytest.raises(ValueError):assert_same_contract(fixed,changed)
    with pytest.raises(ValueError):cache_contract(prefix,mode='fixed',patch_path=patch,
        startup_args=['--max-running-requests','32','--disable-radix-cache'],deterministic=False)


def test_warm_every_engine_and_fail_on_stale_version_or_wrong_hit():
    import asyncio
    from slime_pacman.prefix_warmup import warm_engines
    prefix=declare_prefix([list(range(620))+[x] for x in (800,801)],fixed_boundary_lengths=[615,615])
    class Response:
        def __init__(self,value):self.value=value
        def raise_for_status(self):pass
        def json(self):return self.value
    class Client:
        def __init__(self):self.calls=[];self.version='4';self.hit=576
        async def get(self,url):return Response(dict(weight_version=self.version))
        async def post(self,url,*,json):
            self.calls.append((url,len(json['input_ids'])))
            return Response(dict(meta_info=dict(weight_version=self.version,cached_tokens=self.hit)))
    client=Client();contract=dict(mode='fixed',prefix=prefix)
    rows=asyncio.run(warm_engines(client,['http://one','http://two'],contract,'4',canary_ids=list(range(620))))
    assert len(rows)==2 and sorted(client.calls)==[('http://one/generate',576),('http://one/generate',577),('http://two/generate',576),('http://two/generate',577)]
    client.version='3'
    with pytest.raises(ValueError):asyncio.run(warm_engines(client,['http://one'],contract,'4',canary_ids=list(range(620))))
    client.version='4';client.hit=640
    with pytest.raises(ValueError):asyncio.run(warm_engines(client,['http://one'],contract,'4',canary_ids=list(range(620))))
