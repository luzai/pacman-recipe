import copy,unittest
from pacman_recipe.experimental.engine_assisted.module_contract import validate_inventory,digest
def fixture():
 rows=[]
 for i,kind,leaves in [(0,'linear_attn',['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']),(1,'self_attn',['q_proj','k_proj','v_proj','o_proj'])]:
  for family,names in [(kind,leaves),('mlp',['gate_proj','up_proj','down_proj'])]:
   rows += [dict(name=f'model.language_model.layers.{i}.{family}.{name}',kind=family,in_features=8,out_features=8) for name in names]
 return rows+[dict(name='lm_head',kind='output',in_features=8,out_features=31)]
class Contract(unittest.TestCase):
 def test_complete_hybrid_and_output(self):
  r=validate_inventory(fixture(),2);self.assertEqual((r['layers'],r['full_attention_layers'],r['linear_attention_layers']),(2,1,1))
 def test_no_vision(self):
  r=fixture();r[0]['name']='model.visual.blocks.0.attn.q_proj'
  with self.assertRaises(ValueError):validate_inventory(r)
 def test_no_missing_linear_gates(self):
  r=[x for x in fixture() if not x['name'].endswith('in_proj_b')]
  with self.assertRaises(ValueError):validate_inventory(r)
 def test_no_missing_output(self):
  with self.assertRaises(ValueError):validate_inventory(fixture()[:-1])
 def test_no_missing_layer(self):
  with self.assertRaises(ValueError):validate_inventory(fixture(),3)
 def test_no_extra_projection(self):
  r=fixture();r[0]['name']='model.language_model.layers.0.linear_attn.secret_proj'
  with self.assertRaises(ValueError):validate_inventory(r)
 def test_packed_linear_gates(self):
  r=[x for x in fixture() if '.0.linear_attn.' not in x['name']]
  r += [dict(name='model.language_model.layers.0.linear_attn.'+n,kind='linear_attn',in_features=8,out_features=8) for n in ('in_proj_qkvz','in_proj_ba','out_proj')];validate_inventory(r,2)
 def test_duplicate_rejected(self):
  r=fixture();r.append(copy.deepcopy(r[0]))
  with self.assertRaises(ValueError):validate_inventory(r)
 def test_inventory_sha_dimension_sensitive(self):
  r=fixture();h=digest(r);r[0]['out_features']+=1;self.assertNotEqual(h,digest(r))
if __name__=='__main__':unittest.main()
