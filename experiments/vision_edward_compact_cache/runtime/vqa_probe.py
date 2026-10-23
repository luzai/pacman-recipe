"""Fixed wall/pellet cell diagnostic; not a full visual-reading qualification."""
import base64
import json
import math
from pathlib import Path
import httpx
from transformers import AutoProcessor
from pacman_recipe.level1.vision_prompt import configure_image_processor,GROUNDING_FIXED_TEXT,GROUNDING_SYSTEM
from slime_pacman.generation import SGLangGenerator

class Answers:
    def __init__(self,tokenizer):
        self.codes=['A','B']
        ids=[tokenizer.encode(s,add_special_tokens=False) for s in self.codes]
        assert all(len(x)==1 for x in ids)
        self.allowed_token_ids=[x[0] for x in ids]
        assert len(set(self.allowed_token_ids))==2
        assert all(tokenizer.decode(x)==s for x,s in zip(ids,self.codes))
    def option_for_tokens(self,ids):
        assert len(ids)==1 and ids[0] in self.allowed_token_ids
        return self.codes[self.allowed_token_ids.index(ids[0])]
    def code_for_option(self,s):return s

async def evaluate(args,rollout_id):
    from slime.rollout.sglang_rollout import get_model_url
    processor=configure_image_processor(AutoProcessor.from_pretrained(args.hf_checkpoint,local_files_only=True))
    constraint=Answers(processor.tokenizer)
    root=Path('/acceptance/vqa')
    cases=json.loads((root/'exam.json').read_text())
    rows=[]
    async with httpx.AsyncClient(timeout=120.,limits=httpx.Limits(max_keepalive_connections=0)) as client:
        # VQA uses its own system prefix. The shared fixed-node guard must return
        # zero hits; the generator verifies that evidence without changing policy cache.
        generate=SGLangGenerator(processor=processor,endpoint=get_model_url(args,'policy'),client=client,
                                 cache_contract_enabled=False)
        for case in cases:
            url='data:image/png;base64,'+base64.b64encode((root/case['image']).read_bytes()).decode()
            question=f"What is visible in cell [{case['row']},{case['col']}]? A = wall, B = normal pellet. Return exactly A or B."
            messages=[dict(role='system',content=GROUNDING_SYSTEM),dict(role='user',content=[
                dict(type='text',text=GROUNDING_FIXED_TEXT),dict(type='image_url',image_url=dict(url=url)),
                dict(type='text',text='[QUESTION]\n'+question)])]
            result=await generate(messages,constraint)
            choice=constraint.codes[max(range(2),key=lambda i:result.candidate_log_probs[i])]
            rows.append(dict(case,choice=choice,correct=choice==case['answer'],weight_version=result.weight_version,
                             log_probs=result.candidate_log_probs))
    versions={r['weight_version'] for r in rows}
    assert len(versions)==1
    version=next(iter(versions))
    out=Path('/run/vqa');out.mkdir(exist_ok=True)
    accuracy=sum(r['correct'] for r in rows)/len(rows)
    result=dict(rollout_id=rollout_id,weight_version=version,accuracy=accuracy,cases=rows,
                scope='48 fixed visible wall/pellet cells on eight initial frames; not complete VQA')
    path=out/f'probe-{rollout_id:04d}-weights-{version}.json'
    if path.exists():assert json.loads(path.read_text())==result
    else:path.write_text(json.dumps(result,indent=2))
    baseline=out/'baseline.json'
    if not baseline.exists():baseline.write_text(json.dumps(result,indent=2))
    initial=json.loads(baseline.read_text())['accuracy']
    print('VQA_CELL_PROBE',rollout_id,version,accuracy,flush=True)
    if initial-accuracy>.10:
        raise RuntimeError(f'fixed cell reading accuracy dropped: {initial:.3f} -> {accuracy:.3f}')
