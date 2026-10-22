"""Presentation permutations must never change option semantics or game RNG."""
import json
import random
from types import SimpleNamespace
import pytest
from pacman_recipe.level1.token_constraints import ObjectiveTokenConstraint
from pacman_recipe.level1.ascii_observation import ASCII_MAP_HEADER
from pacman_recipe.level1.prompts import render_vision_edward_decision_prompt, compact_ascii_edward_decision_prompt

class Tokenizer:
    def encode(self, text, **kw): return [ord(c) for c in text]
    def decode(self, ids, **kw): return ''.join(map(chr, ids))

def test_permutation_reproducible_bijective_and_rng_independent():
    ids=['C0','C1','A0','A1','E0'];before=random.getstate()
    layouts=[]
    for k in range(24):
        key=f'820029:{k}'
        c=ObjectiveTokenConstraint.build(Tokenizer(),ids,shuffle_key=key)
        assert c==ObjectiveTokenConstraint.build(Tokenizer(),reversed(ids),shuffle_key=key)
        assert set(c.option_ids)==set(ids)
        assert len(set(c.rendered_choices))==len(ids)
        for oid in ids:
            code=c.code_for_option(oid)
            assert c.option_for_completion(code)==oid
            assert c.option_for_tokens([ord(code)])==oid
        layouts.append((c.option_ids,c.rendered_choices))
    assert random.getstate()==before
    assert len(set(layouts))>20

@pytest.mark.parametrize('ascii_mode',[False,True])
def test_prompt_has_single_letter_choices_no_internal_ids_and_true_risk(ascii_mode):
    cs=[SimpleNamespace(option_id=i,strategy='RISK_FALLBACK',target=(1,n),first_action=a,
          route_distance=1,commit_moves=1,safety_margin=None,future_safe_exits=None,entity_id=None,
          risk=dict(rank=n,motion='clear_estimate',ghost_clearance=3,route_margin=2,
                    safe_next_cells=0,dead_end=bool(n==1),reverse=False))
        for i,n,a in [('A0',1,'R'),('A1',2,'L')]]
    constraint=ObjectiveTokenConstraint.build(Tokenizer(),['A0','A1'],shuffle_key='1:9')
    cs=sorted(cs,key=lambda c:constraint.option_ids.index(c.option_id))
    context={'option_shuffle_key':'1:9','route_menu':{'A0':{'route_id':'route-x'},'A1':{'route_id':'route-y'}}}
    if ascii_mode:
        text=compact_ascii_edward_decision_prompt(context,cs,constraint,ASCII_MAP_HEADER+'\n###',fallback_mode='risk_ranked')
    else:
        text=render_vision_edward_decision_prompt(context,cs,constraint,fallback_mode='risk_ranked')
    # Notice must also avoid leaking the old A0..A3 fixed ranking convention.
    assert 'A0' not in text and 'A1' not in text
    assert '[code,strategy' in text and '[code,id,strategy' not in text
    assert 'shuffled each decision' in text
    assert 'not safety guarantees or commands' in text
    rows=text.split('[CANDIDATE OBJECTIVES]\n')[1].split('\n\n[OUTPUT]')[0].splitlines()
    for row,candidate in zip(rows,cs):
        assert json.loads(row)[:2]==[constraint.code_for_option(candidate.option_id),'RISK_FALLBACK']
    risk=json.loads(text.split('Fallback risk by code: ')[1].splitlines()[0])
    assert risk[constraint.code_for_option('A0')][0]==1
    assert risk[constraint.code_for_option('A1')][0]==2
