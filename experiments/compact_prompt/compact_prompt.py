"""Experimental presentation only. No planner, binding or safety changes."""
import json
from copy import deepcopy
J=lambda x:json.dumps(x,separators=(',',':'),allow_nan=False)
FIXED=('Use the latest screenshot and authoritative state. Coordinates=[row,column], starting at 0. '
       'Candidates are keyed by the output code. target=destination; first=first move; steps=route length; '
       'commit=max executed moves (equals steps when omitted); safety/exits: larger is better; entity=ghost id. '
       'waypoints=[step_index,cell,action] merges junctions and turns; portals state entry/exit; approach states target approach. '
       'Omitted optional fields are unavailable, not evidence of safety. Select one advertised code.\n\n[CURRENT IMAGE]\n')

def route_summary(meta):
 d={k:deepcopy(v) for k,v in meta.items() if k not in ('route_id','key_junctions','turns','segment_moves') and v is not None}
 merged={J(x):deepcopy(x) for k in ('key_junctions','turns') for x in meta.get(k,[])}
 if merged:d['waypoints']=sorted(merged.values(),key=lambda x:(x[0],J(x)))
 return d

def compact_render(original,context,candidates,constraint,**kwargs):
 # Parse the production output, preserving its exact safety suffix.
 baseline=original(context,candidates,constraint,**kwargs)
 state_text=baseline.split('[CURRENT STATE]\n',1)[1].split('\n\n[CANDIDATE OBJECTIVES]',1)[0]
 state=json.loads(state_text)
 state={k:v for k,v in state.items() if v is not None}
 output=baseline.split('[OUTPUT]\n',1)[1]
 safety=output.split('; nothing else.',1)[1]
 safety=safety.split('\nBound routes (coordinates row,col): ',1)[0]
 safety=safety.replace('ranked A0..A3 by increasing estimated risk','with explicit estimated risk ranks')
 menu=context.get('route_menu') or {}; rows={}
 for c in candidates:
  code=constraint.code_for_option(c.option_id)
  d={'strategy':c.strategy,'target':list(c.target),'first':c.first_action,'steps':c.route_distance}
  for k,v in [('commit',c.commit_moves if c.commit_moves!=c.route_distance else None),('safety',c.safety_margin),('exits',c.future_safe_exits),('entity',c.entity_id)]:
   if v is not None:d[k]=v
  if c.option_id in menu:
   meta=menu[c.option_id]; assert meta.get('segment_moves',c.commit_moves)==c.commit_moves
   d['route']=route_summary(meta)
  rows[code]=d
 # Identical summaries must not hide different full routes; use exact actions.
 for code,d in rows.items():
  same=[k for k,v in rows.items() if J(v)==J(d)]
  if len(same)>1:
   for k in same:
    oid=next(c.option_id for c in candidates if constraint.code_for_option(c.option_id)==k)
    actions=context.get('_route_actions',{}).get(oid)
    assert actions is not None, 'Ambiguous route summary without full bound actions'
    rows[k]['actions']=''.join(actions)
 return FIXED+'[CURRENT STATE]\n'+J(state)+'\n\n[CANDIDATE OBJECTIVES]\n'+'\n'.join(k+': '+J(v) for k,v in rows.items())+'\n\n[OUTPUT]\nReturn exactly one code from ['+','.join(constraint.rendered_choices)+']; nothing else.'+safety+ ('\nThe harness follows the selected bound route; short-term safety checks occur every step and may interrupt. No full-route safety guarantee.' if menu else '')

def install(prompts,native):
 original=prompts.render_vision_edward_decision_prompt
 def render(*args,**kw):return compact_render(original,*args,**kw)
 # Preserve full internal route menu. Only the textual renderer/layout varies.
 prompts.VISION_EDWARD_FIXED_PREFIX=FIXED
 prompts.render_vision_edward_decision_prompt=render
 native.render_vision_edward_decision_prompt=render
 return original
