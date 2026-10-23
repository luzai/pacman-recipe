import unittest,json,copy
from types import SimpleNamespace as S
from pacman_recipe.level1.compact_vision_edward import compact_render,route_summary
class Tests(unittest.TestCase):
 def setUp(self):
  self.cs=[S(option_id='C0',strategy='COLLECT',target=(1,2),first_action='L',route_distance=3,commit_moves=3,safety_margin=0,future_safe_exits=None,entity_id=None),S(option_id='C1',strategy='COLLECT',target=(1,2),first_action='L',route_distance=3,commit_moves=3,safety_margin=0,future_safe_exits=None,entity_id=None)]
  self.con=S(code_for_option=lambda x:{'C0':'B','C1':'F'}[x],rendered_choices=['B','F'])
  self.meta=dict(route_id='abcdef',key_junctions=[[1,[2,3],'D']],turns=[[1,[2,3],'D']],portals=[dict(entry=[1,0],exit=[1,9],action='L',index=2)],segment_moves=3,boundary='route_end')
  self.ctx=dict(route_menu={'C0':copy.deepcopy(self.meta),'C1':copy.deepcopy(self.meta)},_route_actions={'C0':['L','D','L'],'C1':['L','D','R']})
 def original(self,*a,**k):return '[CURRENT STATE]\n'+json.dumps(dict(p=[1,1],last=None,edible_ticks=0,ghosts=[]))+'\n\n[CANDIDATE OBJECTIVES]\nold\n\n[OUTPUT]\nReturn code; nothing else.\nFallback risk by code: {"B":[1,"unknown",0,0,0,true,false]}\nBounded escape evidence by code: {"B":{"escape_found":false,"escape_route":[],"escape_horizon":4}}\nInformation only.\nBound routes (coordinates row,col): junk'
 def test_omit_ids_null(self):
  t=compact_render(self.original,self.ctx,self.cs,self.con)
  for x in ['C0','C1','null','abcdef','route_id']:self.assertNotIn(x,t)
 def test_safety_exact(self):
  t=compact_render(self.original,self.ctx,self.cs,self.con)
  self.assertIn('"escape_found":false',t);self.assertIn('"B":[1,"unknown",0,0,0,true,false]',t);self.assertIn('"safety":0',t)
 def test_route_collision(self):
  t=compact_render(self.original,self.ctx,self.cs,self.con);self.assertIn('"actions":"LDL"',t);self.assertIn('"actions":"LDR"',t)
 def test_dedup(self):self.assertEqual(route_summary(self.meta)['waypoints'],[[1,[2,3],'D']])
 def test_immutable(self):
  before=copy.deepcopy(self.ctx);compact_render(self.original,self.ctx,self.cs,self.con);self.assertEqual(before,self.ctx)
 def test_portal(self):self.assertEqual(route_summary(self.meta)['portals'],self.meta['portals'])
 def test_missing_actions_rejected(self):
  self.ctx.pop('_route_actions')
  with self.assertRaises(AssertionError):compact_render(self.original,self.ctx,self.cs,self.con)
 def test_commit(self):
  self.cs[0].commit_moves=2;self.ctx['route_menu']['C0']['segment_moves']=2
  self.assertIn('"commit":2',compact_render(self.original,self.ctx,self.cs,self.con))
if __name__=='__main__':unittest.main()


def test_explicit_config_and_unique_prompt_identity():
 from slime_pacman.config import PacmanConfig,runner_options,record_options
 from pacman_recipe.level1.prompts import prompt_contract_metadata
 standard=runner_options(PacmanConfig())
 compact=runner_options(PacmanConfig(edward_prompt_variant='compact_v1',groups_per_update=16,clip=.05,max_input_tokens=4096))
 assert standard['edward_prompt_variant']=='standard'
 assert compact['edward_prompt_variant']=='compact_v1'
 assert record_options(PacmanConfig(edward_prompt_variant='compact_v1',groups_per_update=16,clip=.05,max_input_tokens=4096))['edward_prompt_variant']=='compact_v1'
 assert standard['prompt_version']!=compact['prompt_version']
 a=prompt_contract_metadata('live_state_v3',edward_options=True,fallback_mode='risk_ranked')
 b=prompt_contract_metadata('live_state_v3',edward_options=True,fallback_mode='risk_ranked',edward_prompt_variant='compact_v1')
 from pacman_recipe.level1.compact_vision_edward import system_prompt
 assert 'four total lives' in system_prompt('risk_ranked')
 assert a['prompt_template_sha256']!=b['prompt_template_sha256']


def test_compact_invalid_combinations_fail_closed():
 import pytest
 from slime_pacman.config import PacmanConfig
 from pacman_recipe.level1.prompts import prompt_contract_metadata
 with pytest.raises(ValueError):PacmanConfig(edward_harness='legacy',edward_prompt_variant='compact_v1')
 with pytest.raises(ValueError):prompt_contract_metadata('ascii_edward_v1',edward_options=True,edward_prompt_variant='compact_v1')
 with pytest.raises(ValueError):prompt_contract_metadata('live_state_v3',edward_options=False,edward_prompt_variant='compact_v1')
 with pytest.raises(ValueError):prompt_contract_metadata('live_state_v3',edward_options=True,edward_prompt_variant='typo')


def test_compact_image_layout_exact_text():
 from pacman_recipe.level1.compact_vision_edward import FIXED,layout
 instruction=FIXED+'[CURRENT STATE]\n{}'
 content=layout(b'png',instruction)
 assert [x['type'] for x in content]==['text','image_url','text']
 assert ''.join(x['text'] for x in content if x['type']=='text')==instruction


def test_actual_fallback_and_escape_facts_preserved():
 from pacman_recipe.level1 import prompts
 from pacman_recipe.level1.compact_vision_edward import render
 c=S(option_id='A0',strategy='RISK_FALLBACK',target=(1,2),first_action='R',route_distance=1,commit_moves=1,safety_margin=0,future_safe_exits=0,entity_id=None,risk=dict(rank=1,motion='unknown',ghost_clearance=0,route_margin=0,safe_next_cells=0,dead_end=True,reverse=False,escape_horizon=4,escape_found=False,escape_route=[]))
 constraint=S(code_for_option=lambda _: 'B',rendered_choices=['B'])
 state=dict(pacman_position=[1,1])
 base=prompts.render_vision_edward_decision_prompt(state,[c],constraint,fallback_mode='risk_ranked')
 small=render(state,[c],constraint,fallback_mode='risk_ranked')
 assert base.split('Fallback risk by code:',1)[1]==small.split('Fallback risk by code:',1)[1]
 assert 'A0' not in small and 'null' not in small
 assert '"escape_found":false' in small and '"safety":0' in small


def test_compact_budget_only_does_not_relax_standard():
 import pytest
 from slime_pacman.config import PacmanConfig
 c=PacmanConfig(edward_prompt_variant='compact_v1',groups_per_update=16,clip=.05,max_input_tokens=4096)
 assert c.group_size*c.groups_per_update==192
 with pytest.raises(ValueError):PacmanConfig(max_input_tokens=4096)
 with pytest.raises(ValueError):PacmanConfig(groups_per_update=16,clip=.05)
 with pytest.raises(ValueError):PacmanConfig(edward_prompt_variant='compact_v1',groups_per_update=16,clip=.05,max_input_tokens=4097)
