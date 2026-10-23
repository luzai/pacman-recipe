import unittest,json,copy
from types import SimpleNamespace as S
from compact_prompt import compact_render,route_summary
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
