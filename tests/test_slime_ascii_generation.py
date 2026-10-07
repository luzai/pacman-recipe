"""No-image request acceptance for the opt-in ASCII generator."""
import asyncio
from types import SimpleNamespace
import pytest
import torch
from slime_pacman.generation import process_text_request,SGLangGenerator

class Tokenizer:
    def apply_chat_template(self,chat,**kw):
        assert kw==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert all(isinstance(m['content'],str) for m in chat)
        return 'ascii prompt'
    def __call__(self,prompt,**kw):
        assert kw==dict(return_tensors='pt',truncation=False)
        return {'input_ids':torch.tensor([[1,2,3]])}

MESSAGES=[dict(role='system',content='rules'),dict(role='user',content=[dict(type='text',text='MAP')])]

def test_text_processor_never_invokes_image_processor():
    class Processor:
        tokenizer=Tokenizer()
        def __call__(self,*a,**k):raise AssertionError('image processor must not run')
    prompt,ids,mm,image=process_text_request(Processor(),MESSAGES,2048)
    assert ids==[1,2,3] and mm=={} and image is None
    with pytest.raises(ValueError,match='budget'):process_text_request(Processor(),MESSAGES,2)
    bad=[MESSAGES[0],dict(role='user',content=[dict(type='image_url',image_url={'url':'data:image/png;base64,AA'})])]
    with pytest.raises(ValueError,match='refuses'):process_text_request(Processor(),bad,2048)

def test_sglang_text_request_has_no_image_payload(monkeypatch):
    from slime_pacman.probability import PacmanLogitProcessor
    monkeypatch.setattr(PacmanLogitProcessor,'to_str',lambda:'mask')
    class Client:
        async def post(self,endpoint,json):
            assert 'image_data' not in json and json['text']=='ascii prompt'
            assert json['sampling_params']['custom_params']['pacman_allowed_token_ids']==[7]
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(text='B',meta_info=dict(
                output_token_logprobs=[[0.,7]],output_token_ids_logprobs=[[[0.,7]]],weight_version='frozen',prompt_tokens=3)))
    constraint=SimpleNamespace(allowed_token_ids=[7],option_for_tokens=lambda tokens:'option',code_for_option=lambda option:'B')
    result=asyncio.run(SGLangGenerator(processor=Tokenizer(),endpoint='local',client=Client(),observation_mode='ascii')(MESSAGES,constraint))
    assert result.multimodal_train_inputs=={} and result.image_sha256 is None

def test_ascii_boundary_identity_rejects_image_history():
    from pacman_recipe.level1.episode import _boundary_verify
    constraint=SimpleNamespace(allowed_token_ids=[7])
    image=_boundary_verify(b'real-png',[],constraint,'system','user')
    text=_boundary_verify(b'',[],constraint,'system','user')
    assert text['png_sha256'] is None and text['observation_mode']=='ascii'
    assert text!=image
    assert _boundary_verify(b'',[],constraint,'system','changed-history')!=text


def test_ascii_config_new_identity():
    from slime_pacman.config import PacmanConfig,runner_options
    from pacman_recipe.level1.prompts import prompt_contract_metadata
    cfg=PacmanConfig(groups_per_update=16,clip=.05,observation_mode='ascii',edward_fallback_mode='risk_ranked')
    options=runner_options(cfg)
    assert options['edward_options'] is True and options['enable_thinking'] is False
    assert options['image_prompt_style']=='ascii_edward_v1'
    old=prompt_contract_metadata('live_state_v3',edward_options=True,fallback_mode='risk_ranked')
    new=prompt_contract_metadata('ascii_edward_v1',edward_options=True,fallback_mode='risk_ranked')
    assert new['action_protocol']==old['action_protocol']
    assert new['prompt_template_sha256']!=old['prompt_template_sha256']

def test_actual_ascii_runner_restore_without_png(monkeypatch,tmp_path):
    from pathlib import Path
    from copy import deepcopy
    import hashlib,json
    from pacman_recipe.level1.contracts import make_episode_record
    from slime_pacman.config import PacmanConfig
    from slime_pacman.rollout import EpisodeRunner
    from test_slime_pacman import Tokenizer as ActionTokenizer
    from slime_pacman.generation import Decision
    from pacman_recipe.level1 import episode as source
    root=Path(__file__).resolve().parents[1]
    record=make_episode_record(0,split='test',recipe_root=root,game_root=root.parent/'pacman-python',backend_root=root.parent/'slime',max_steps=32,observation_mode='ascii',edward_fallback_mode='risk_ranked')
    cfg=PacmanConfig(max_steps=32,groups_per_update=16,clip=.05,observation_mode='ascii',edward_fallback_mode='risk_ranked')
    def forbidden(*args,**kwargs):raise AssertionError('ASCII runner must never encode PNG')
    monkeypatch.setattr(source,'encode_png',forbidden)
    def play(record):
        boundaries=[];observed=[]
        async def generate(messages,constraint):
            assert all(part['type']=='text' for m in messages if isinstance(m['content'],list) for part in m['content'])
            token=constraint.allowed_token_ids[0]
            observed.append((messages[1]['content'][0]['text'],tuple(constraint.allowed_token_ids)))
            return Decision('teacher',[],token,chr(token),constraint.allowed_token_ids,0.,'scripted',{},None,[])
        runner=EpisodeRunner(record,tokenizer=ActionTokenizer(),generate=generate,config=cfg)
        runner.decision_boundary_sink=boundaries.append
        result=asyncio.run(runner.collect(empty_weight_version='scripted'))
        return boundaries,observed,result
    boundaries,observed,result=play(record)
    assert len(boundaries)>2
    index=len(boundaries)//2
    saved=dict(boundaries[index]['env_state'],boundary_context=boundaries[index]['context'])
    path=tmp_path/'saved.json';path.write_text(json.dumps(saved))
    resumed=deepcopy(record);resumed['restart']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),id='ascii-replay')
    _,after,second=play(resumed)
    assert after==observed[index:] and second.terminal_reason==result.terminal_reason
    for row in result.trajectory['episode']['trajectory']:
        assert row['observation_png_sha256'] is None and row['observation_mode']=='ascii'

def test_actual_ascii_probe_config_and_rejection():
    from pathlib import Path
    from slime_pacman.config import load_config,PacmanConfig
    import os
    path=Path(os.environ.get('PACMAN_ASCII_PROBE_CONFIG',str(Path(__file__).resolve().parents[2]/'reports/slime-migration-20260928/round2/ascii-edward/probe_config.yaml')))
    cfg=load_config(path)
    assert (cfg.observation_mode,cfg.groups_per_update,cfg.group_size,cfg.clip)==('ascii',16,12,.05)
    for groups,clip in ((4,.05),(16,.2),(8,.05)):
        with pytest.raises(ValueError):PacmanConfig(observation_mode='ascii',groups_per_update=groups,clip=clip)
    assert PacmanConfig().groups_per_update==4 and PacmanConfig().clip==.2
    with pytest.raises(ValueError):PacmanConfig(groups_per_update=16,clip=.05)
