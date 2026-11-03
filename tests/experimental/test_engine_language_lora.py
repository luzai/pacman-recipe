"""CPU numerical contract. Requires torch; a skip is NOT numerical acceptance."""
import importlib.util,unittest,types,io
HAVE_TORCH=importlib.util.find_spec('torch') is not None
if HAVE_TORCH:
 import torch
 from torch import nn
 from pacman_recipe.experimental.engine_assisted.language_lora import install_language_adapters,LowRankProjection,actual_inventory
 class FakeModel(nn.Module):
  def __init__(self):
   super().__init__();self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=2));self.model=nn.Module();self.model.visual=nn.Linear(8,8);self.model.language_model=nn.Module();self.model.language_model.embed_tokens=nn.Embedding(31,8);self.model.language_model.layers=nn.ModuleList()
   for family,names in [('linear_attn',('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')),('self_attn',('q_proj','k_proj','v_proj','o_proj'))]:
    layer=nn.Module();attention=nn.Module();layer.add_module(family,attention)
    for name in names:attention.add_module(name,nn.Linear(8,8))
    layer.mlp=nn.Module()
    for name in ('gate_proj','up_proj','down_proj'):layer.mlp.add_module(name,nn.Linear(8,8))
    self.model.language_model.layers.append(layer)
   self.lm_head=nn.Linear(8,31)
  def get_output_embeddings(self):return self.lm_head
  def forward(self,x):
   for layer in self.model.language_model.layers:
    for module in layer.modules():
     if isinstance(module,(nn.Linear,LowRankProjection)) and not any(module is q.base for q in layer.modules() if isinstance(q,LowRankProjection)):x=torch.tanh(module(x))
   return self.lm_head(x)
@unittest.skipUnless(HAVE_TORCH,'torch unavailable; numerical gate remains required')
class AdapterTests(unittest.TestCase):
 def test_zero_delta_dtype_rng_and_sensor(self):
  torch.manual_seed(8);m=FakeModel().to(torch.bfloat16).eval();x=torch.randn(2,8,dtype=torch.bfloat16);before=m(x).detach();rng=torch.get_rng_state().clone();ctl=install_language_adapters(m);self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(before,m(x)));self.assertEqual(m(x).dtype,torch.bfloat16)
  with torch.no_grad():
   for p in ctl.parameters():p.add_(.03)
  self.assertFalse(torch.equal(before,m(x)))
  with ctl.frozen_perception():self.assertTrue(torch.equal(before,m(x)))
  ctl.assert_only_adapters_trainable()
 def test_gradients_and_adam_reload_all_layers(self):
  torch.manual_seed(9);m=FakeModel().to(torch.bfloat16).eval();ctl=install_language_adapters(m);optimizer=torch.optim.AdamW(ctl.parameters(),lr=1e-5,weight_decay=0);x=torch.randn(2,8,dtype=torch.bfloat16);loss=m(x).float().square().mean();loss.backward();ctl.assert_only_adapters_trainable()
  self.assertTrue(all(p.grad is None for p in m.parameters() if not p.requires_grad));self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ctl.parameters()));self.assertTrue(all(float(q.up.grad.norm())>0 for q in ctl.adapters.values()));self.assertTrue(all(q.calls>0 for q in ctl.adapters.values()))
  before=ctl.fingerprint();optimizer.step();self.assertNotEqual(before,ctl.fingerprint());expected=m(x).detach().clone();checkpoint=io.BytesIO();torch.save(dict(adapter=ctl.state_dict(),optimizer=optimizer.state_dict()),checkpoint);checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=True)
  for p in ctl.parameters():p.data.zero_()
  ctl.load_state_dict(saved['adapter']);optimizer.load_state_dict(saved['optimizer']);self.assertTrue(torch.equal(expected,m(x)));self.assertTrue(all(int(s['step'])==1 for s in optimizer.state.values()));self.assertTrue(all(s['exp_avg'].dtype==torch.float32 for s in optimizer.state.values()))
 def test_disabled_exception_restores_every_flag(self):
  m=FakeModel();ctl=install_language_adapters(m)
  try:
   with ctl.frozen_perception():raise RuntimeError('probe')
  except RuntimeError:pass
  self.assertTrue(all(m.enabled for m in ctl.adapters.values()))
 def test_train_install_requires_reviewed_inventory(self):
  with self.assertRaises(ValueError):install_language_adapters(FakeModel(),qualification_only=False)
if __name__=='__main__':unittest.main()
