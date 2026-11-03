"""Fresh FP32 language + explicit output LoRA. All base/vision weights frozen."""
from contextlib import contextmanager
import hashlib,math
import torch
from torch import nn
from .module_contract import PATH,validate_inventory,digest

class LowRankProjection(nn.Module):
 def __init__(self,base,*,rank=16,alpha=16,dropout=0.,seed=314159):
  super().__init__()
  if not isinstance(base,nn.Linear) or rank!=16 or alpha!=16 or dropout!=0.:raise ValueError('qualified dense Linear, rank16/alpha16/dropout0 required')
  self.base=base;self.rank=rank;self.alpha=float(alpha);self.enabled=True;self.calls=0
  for p in base.parameters():p.requires_grad_(False)
  generator=torch.Generator(device='cpu').manual_seed(seed)
  down=torch.randn(rank,base.in_features,generator=generator,dtype=torch.float32)/math.sqrt(base.in_features)
  self.down=nn.Parameter(down.to(base.weight.device));self.up=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 @property
 def in_features(self):return self.base.in_features
 @property
 def out_features(self):return self.base.out_features
 def forward(self,x):
  self.calls+=1;result=self.base(x)
  if not self.enabled:return result
  delta=((x.float()@self.down.t())@self.up.t())*(self.alpha/self.rank)
  return (result.float()+delta).to(result.dtype)

def actual_inventory(model):
 output=model.get_output_embeddings();rows=[];output_names=[]
 for name,module in model.named_modules():
  if module is output:output_names.append(name)
  match=PATH.fullmatch(name)
  if isinstance(module,nn.Linear) and match:
   rows.append(dict(name=name,kind=match.group(2),in_features=module.in_features,out_features=module.out_features))
 if len(output_names)!=1 or not isinstance(output,nn.Linear):raise ValueError('unique actual plain Linear output layer required')
 rows.append(dict(name=output_names[0],kind='output',in_features=output.in_features,out_features=output.out_features))
 rows.sort(key=lambda r:r['name']);config=getattr(model.config,'text_config',model.config);validate_inventory(rows,expected_layers=config.num_hidden_layers);return rows

class LanguageAdapters:
 def __init__(self,model,inventory,adapters):self.model=model;self.inventory=inventory;self.adapters=adapters;self.inventory_sha256=digest(inventory)
 def parameters(self):return [p for name in sorted(self.adapters) for p in (self.adapters[name].down,self.adapters[name].up)]
 def assert_only_adapters_trainable(self):
  allowed={id(p) for p in self.parameters()};actual={id(p) for p in self.model.parameters() if p.requires_grad}
  if actual!=allowed:raise AssertionError('only explicit language/output LoRA may train')
  assert all(p.dtype==torch.float32 for p in self.parameters())
 @contextmanager
 def frozen_perception(self):
  flags={k:v.enabled for k,v in self.adapters.items()}
  modes={module:module.training for module in self.model.modules()}
  try:
   self.model.eval()
   for v in self.adapters.values():v.enabled=False
   yield
  finally:
   for k,v in self.adapters.items():v.enabled=flags[k]
   for module,training in modes.items():module.training=training
 def state_dict(self):return dict(schema='public-language-output-lora-v1',rank=16,alpha=16,dropout=0.,inventory=self.inventory,inventory_sha256=self.inventory_sha256,adapters={name:dict(down=m.down.detach().cpu().clone(),up=m.up.detach().cpu().clone()) for name,m in self.adapters.items()})
 def load_state_dict(self,state):
  if (state.get('schema'),state.get('rank'),state.get('alpha'),state.get('dropout'),state.get('inventory_sha256'))!=('public-language-output-lora-v1',16,16,0.,self.inventory_sha256) or state['inventory']!=self.inventory or digest(state['inventory'])!=self.inventory_sha256 or set(state['adapters'])!=set(self.adapters):raise ValueError('exact qualified target inventory and adapter schema required')
  with torch.no_grad():
   for name,module in self.adapters.items():
    for key in ('down','up'):
     tensor=state['adapters'][name][key];parameter=getattr(module,key)
     if tensor.dtype!=torch.float32 or tensor.shape!=parameter.shape or not torch.isfinite(tensor).all():raise ValueError('invalid FP32 adapter tensor')
     parameter.copy_(tensor)
 def adapter_state(self):return self.state_dict()
 def load_adapter_state(self,state):return self.load_state_dict(state)
 def fingerprint(self):
  h=hashlib.sha256(self.inventory_sha256.encode())
  for name in sorted(self.adapters):
   for p in (self.adapters[name].down,self.adapters[name].up):h.update(p.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 def enable_gradient_checkpointing(self):
  # Reentrant checkpointing needs a differentiable boundary even though embedding
  # PARAMETERS stay frozen. The output activation hook never unfreezes embedding.
  self.model.train()
  for path in ('model.visual','visual','model.vision_tower','vision_tower'):
   try:self.model.get_submodule(path).eval()
   except AttributeError:pass
  self.model.enable_input_require_grads();self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});self.assert_only_adapters_trainable()

def install_language_adapters(model,*,expected_inventory_sha256=None,qualification_only=True):
 inventory=actual_inventory(model)
 if not qualification_only and not expected_inventory_sha256:raise ValueError('training requires reviewed actual module inventory SHA')
 if expected_inventory_sha256 and digest(inventory)!=expected_inventory_sha256:raise ValueError('actual model targets differ from frozen qualification inventory')
 for p in model.parameters():p.requires_grad_(False)
 adapters={}
 for row in inventory:
  name=row['name'];parent_name,_,leaf=name.rpartition('.');parent=model.get_submodule(parent_name) if parent_name else model;base=getattr(parent,leaf)
  seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16);module=LowRankProjection(base,seed=seed);setattr(parent,leaf,module);adapters[name]=module
 controller=LanguageAdapters(model,inventory,adapters);controller.assert_only_adapters_trainable();model.eval();return controller
