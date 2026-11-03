"""Pure-Python closed target inventory; no torch, model, game or truth imports."""
import hashlib,json,re
PATH=re.compile(r'^(?:model\.)?language_model(?:\.model)?\.layers\.(\d+)\.(self_attn|linear_attn|mlp)\.(.+)$')
LEAVES={'self_attn':{'q_proj','k_proj','v_proj','o_proj'},'linear_attn':{'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','in_proj_qkvz','in_proj_ba','out_proj'},'mlp':{'gate_proj','up_proj','down_proj'}}
def digest(inventory):return hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_inventory(inventory,expected_layers=None):
 if not inventory or len({r['name'] for r in inventory})!=len(inventory):raise ValueError('unique nonempty actual module inventory required')
 layers={};heads=[]
 for row in inventory:
  if row['kind']=='output':
   if row['name'] not in ('lm_head','model.lm_head'):raise ValueError('explicit actual output embedding required')
   heads.append(row);continue
  match=PATH.fullmatch(row['name'])
  if not match:raise ValueError('outside language decoder layers: '+row['name'])
  index,category,leaf=match.groups()
  if category!=row['kind'] or leaf not in LEAVES[category]:raise ValueError('unreviewed projection: '+row['name'])
  layers.setdefault(int(index),{}).setdefault(category,set()).add(leaf)
  if min(row['in_features'],row['out_features'])<1:raise ValueError('invalid projection dimensions')
 if len(heads)!=1:raise ValueError('exactly one explicit low-rank output adapter required')
 if set(layers)!=set(range(expected_layers or len(layers))):raise ValueError('missing decoder layer')
 for index,parts in layers.items():
  if parts.get('mlp')!=LEAVES['mlp']:raise ValueError('missing dense MLP projection')
  attention=set(parts)-{'mlp'}
  if attention=={'self_attn'}:
   if parts['self_attn']!=LEAVES['self_attn']:raise ValueError('missing full-attention projection')
  elif attention=={'linear_attn'}:
   linear=parts['linear_attn']
   if linear not in ({'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'},{'in_proj_qkvz','in_proj_ba','out_proj'}):raise ValueError('incomplete or unexpected hybrid linear-attention projections')
  else:raise ValueError('exactly one attention family per decoder layer required')
 return dict(layers=len(layers),full_attention_layers=sum('self_attn' in p for p in layers.values()),linear_attention_layers=sum('linear_attn' in p for p in layers.values()),projections=len(inventory),inventory_sha256=digest(inventory))
