"""Synthetic topology/guard contract tests, not native-game success evidence."""
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace
from experiments.edward_route_options import portal_skill as L

P=L.B.P.Position
class SkillTests(unittest.TestCase):
 def planner(self):
  p=object.__new__(L.LurePortalPlanner)
  p.advertised_lure=None;p.enable_lure=True;p.lure_cooldown=0;p.lure_uses={};p.lure_routes={};p.lure_active=None;p.menu={};p.route_menu={};p.registry={};p.remaining={P(9,3)};p.level=object();p.safety_checks=0;p.events=[];p.cursor=0
  p._portal_edges=lambda:iter([(P(1,1),'U',P(9,1))])
  def register(c,s,r):
   payload={'route_id':'test','identity':{'actions':list(r),'target':list(c.target)},'cells':[[1,1],[9,1],[9,2],[9,3]]}
   p.registry['test']=payload;p.menu[c.option_id]='test';p.route_menu[c.option_id]={'route_id':'test'};return c
  p.register=register
  return p
 def state(self):
  return {'pacman_position':[1,1],'open':['U'],'normal_pellet_positions':[[9,3]],'power_pellet_positions':[],'ghosts':[{'id':0,'state':'normal','position':[1,3],'direction':'L','speed':1,'pixel_position':[16,48]}]}
 def mocks(self,stack,base=()):
  stack.enter_context(patch.object(L.G.NoPathPlanner,'advertised_candidates',return_value=base))
  stack.enter_context(patch.object(L.B,'pacman_path',side_effect=lambda level,start,target:() if start==target else ('R','R')))
  stack.enter_context(patch.object(L,'distance',side_effect=lambda level,g,c:2 if c==P(1,1) else 10))
  stack.enter_context(patch.object(L.B,'describe',return_value={'identity':{'portals':[{}]},'cells':[[1,1],[9,1],[9,2],[9,3]]}))
  stack.enter_context(patch.object(L.G,'short_safe',return_value=True))
  stack.enter_context(patch.object(L.B,'escape_safe',return_value=True))
 def test_route_and_summary(self):
  p=self.planner()
  with ExitStack() as s:
   self.mocks(s);cs=p.advertised_candidates(self.state())
  self.assertEqual(len(cs),1);self.assertEqual(cs[0].option_id,'A0');self.assertEqual(p.registry['test']['identity']['actions'],['U','R','R']);self.assertEqual(p.route_menu['A0']['skill'],'LURE_PORTAL')
 def test_off_exact_base(self):
  p=self.planner();p.enable_lure=False;base=(SimpleNamespace(strategy='AVOID'),)
  with patch.object(L.G.NoPathPlanner,'advertised_candidates',return_value=base):self.assertIs(p.advertised_candidates(self.state()),base)
 def test_no_heading_no_skill(self):
  p=self.planner();state=self.state();state['ghosts'][0]['direction']='R'
  with ExitStack() as s:
   self.mocks(s);self.assertEqual(p.advertised_candidates(state),())
 def test_blocked_exit_guard(self):
  p=self.planner()
  with ExitStack() as s:
   self.mocks(s)
   with patch.object(L.G,'short_safe',return_value=False):self.assertEqual(p.advertised_candidates(self.state()),())
 def test_escape_guard_applied(self):
  p=self.planner()
  with ExitStack() as s:
   self.mocks(s)
   with patch.object(L.B,'escape_safe',return_value=False):self.assertEqual(p.advertised_candidates(self.state()),())
 def test_no_free_slot_no_displacement(self):
  p=self.planner();p.menu={f'A{i}':'x' for i in range(4)};base=(SimpleNamespace(strategy='AVOID'),)
  with ExitStack() as s:
   self.mocks(s,base);self.assertIs(p.advertised_candidates(self.state()),base)
 def test_hidden_fields_not_read(self):
  class Poison(dict):
   def __getitem__(self,k):
    if k in ('path_remaining','path_target','currentPath','rng'):raise AssertionError(k)
    return super().__getitem__(k)
   def get(self,k,*a):
    if k in ('path_remaining','path_target','currentPath','rng'):raise AssertionError(k)
    return super().get(k,*a)
  p=self.planner();state=self.state();state['ghosts']=[Poison(state['ghosts'][0],path_remaining='BAD',rng='BAD')]
  with ExitStack() as s:
   self.mocks(s);self.assertEqual(len(p.advertised_candidates(state)),1)
 def active(self):
  p=self.planner();p.active={'route_id':'x','identity':{'actions':['U','R','R']}};p.lure_active={'portal_key':(1,1,'U'),'cross_index':0,'budget':3,'target':[9,3]};return p
 def test_lost_approach_interrupt(self):
  p=self.active();state=self.state();state['ghosts'][0]['direction']='R'
  with patch.object(L.G.NoPathPlanner,'continue_option',return_value=('U','active')):self.assertEqual(p.continue_option(None,state),(None,'invalidated'))
  self.assertEqual(p.events[-1]['reason'],'skill_approach_lost')
 def test_goal_removed_interrupt(self):
  p=self.active();state=self.state();state['normal_pellet_positions']=[]
  with patch.object(L.G.NoPathPlanner,'continue_option',return_value=('U','active')):self.assertEqual(p.continue_option(None,state),(None,'invalidated'))
  self.assertEqual(p.events[-1]['reason'],'skill_target_completed')
 def test_continuation_escape_guard(self):
  p=self.active();p.cursor=1
  with patch.object(L.G.NoPathPlanner,'continue_option',return_value=('R','active')),patch.object(L.B,'escape_safe',return_value=False):self.assertEqual(p.continue_option(None,self.state()),(None,'invalidated'))
  self.assertEqual(p.events[-1]['reason'],'skill_escape_risk')
 def test_base_risk_preserved(self):
  p=self.active()
  with patch.object(L.G.NoPathPlanner,'continue_option',return_value=(None,'invalidated')):self.assertEqual(p.continue_option(None,self.state()),(None,'invalidated'))
  self.assertGreaterEqual(p.lure_cooldown,20)
 def test_budget_and_repeat_bound(self):
  p=self.planner();p.lure_uses[(1,1,'U')]=2
  with ExitStack() as s:
   self.mocks(s);self.assertEqual(p.advertised_candidates(self.state()),())
  p.lure_uses={};state=self.state();state['remaining_steps']=2
  with ExitStack() as s:
   self.mocks(s);self.assertEqual(p.advertised_candidates(state),())

 def test_stale_same_route_not_skill(self):
  p=self.planner()
  with ExitStack() as s:
   self.mocks(s);skill=p.advertised_candidates(self.state())[0]
  p.active=p.registry['test']
  with patch.object(L.G.NoPathPlanner,'bind'):
   p.bind(skill,self.state())
  self.assertIsNotNone(p.lure_active)
  active_copy=p.lure_active
  p.lure_routes['test']['budget']=999
  self.assertEqual(active_copy['budget'],3)
  # Same route id, now a normal option in a freshly generated cooldown menu.
  with ExitStack() as s:
   self.mocks(s,(skill,));base=p.advertised_candidates(self.state())
  self.assertIsNone(p.advertised_lure)
  with patch.object(L.G.NoPathPlanner,'bind'):p.bind(base[0],self.state())
  self.assertIsNone(p.lure_active)
  self.assertEqual(sum(e['kind']=='skill_start' for e in p.events),1)
 def test_fallback_only_preserved(self):
  p=self.planner();base=(SimpleNamespace(strategy='RISK_FALLBACK'),)
  with ExitStack() as s:
   self.mocks(s,base);self.assertIs(p.advertised_candidates(self.state()),base)
  self.assertIsNone(p.advertised_lure)

if __name__=='__main__':unittest.main()
