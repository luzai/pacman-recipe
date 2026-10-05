"""CPU math and real pinned response/reducer regression tests for regularization."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_slime_upstream_contract import load_body
from slime_pacman.probability import custom_loss, legal_distribution_kl
from slime_pacman.reference_policy import reference_outputs


@pytest.fixture
def runtime(monkeypatch):
    for key in ('PACMAN_EXPERIMENTAL_ENTROPY_COEF', 'PACMAN_EXPERIMENTAL_KL_COEF', 'PACMAN_EXPERIMENTAL_PPO_CLIP'):
        monkeypatch.delenv(key, raising=False)
    mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1, get_tensor_model_parallel_world_size=lambda: 1)
    core = ModuleType('megatron.core')
    core.mpu = mpu
    loss = ModuleType('slime.backends.megatron_utils.loss')
    loss.get_responses = load_body('slime/backends/megatron_utils/loss.py', 'get_responses', mpu)
    monkeypatch.setitem(sys.modules, 'megatron.core', core)
    monkeypatch.setitem(sys.modules, 'slime.backends.megatron_utils.loss', loss)
    args = SimpleNamespace(use_rollout_logprobs=True, calculate_per_token_loss=False, rollout_temperature=.7,
                           eps_clip=.2, kl_coef=0, kl_loss_coef=0, entropy_coef=0, use_kl_loss=False,
                           num_steps_per_rollout=1, global_batch_size=48, rollout_batch_size=4, n_samples_per_prompt=12)
    batch = dict(response_lengths=[1]*4, total_lengths=[3,4,3,2],
                 unconcat_tokens=[torch.tensor(v) for v in ([3,2,1],[3,2,2,0],[2,2,1],[0,0])],
                 metadata=[dict(allowed_token_ids=[0,1]),dict(allowed_token_ids=[0,2]),dict(allowed_token_ids=[1]) ,dict(empty_episode=True)],
                 rollout_log_probs=[torch.tensor([0.])]*4, advantages=[torch.tensor([0.])]*4)
    reducer = load_body('slime/backends/megatron_utils/cp_utils.py','get_sum_of_sample_mean',mpu)(
        batch['total_lengths'], batch['response_lengths'], [torch.tensor([1.])]*3+[torch.tensor([0.])], torch.tensor([1.,2.,2.,1.]))
    return args, batch, reducer


@pytest.mark.parametrize('kind', ['entropy','kl'])
def test_regularizer_actual_slice_reducer_gradient(runtime, monkeypatch, kind):
    args,batch,reducer=runtime
    torch.manual_seed(17)
    logits=torch.randn(1,12,4,requires_grad=True)
    reference_logits=torch.randn_like(logits,requires_grad=True)
    output=reference_outputs(reference_logits,metadata=batch['metadata'],args=args,
        **{k:batch[k] for k in ('unconcat_tokens','total_lengths','response_lengths')})
    assert [len(x) for x in output['allowed_log_probs']]==[2,2,1,1]
    assert all(not x.requires_grad for x in output['allowed_log_probs'])
    batch['ref_allowed_log_probs']=output['allowed_log_probs']
    if kind=='entropy':
        monkeypatch.setenv('PACMAN_EXPERIMENTAL_ENTROPY_COEF','0.01')
        args.entropy_coef=.01
    else:
        monkeypatch.setenv('PACMAN_EXPERIMENTAL_KL_COEF','0.01')
        args.use_kl_loss=True
        args.kl_loss_coef=.01
    actual,metrics=custom_loss(args,batch,logits,reducer)
    (actual/2).backward()
    oracle_logits=logits.detach().clone().requires_grad_()
    terms=[]
    for i,pos in enumerate((1,5,8)):
        lp=torch.log_softmax(oracle_logits[0,pos,batch['metadata'][i]['allowed_token_ids']]/.7,0)
        terms.append(.01*(lp.exp()*lp).sum() if kind=='entropy' else .01*(lp.exp()*(lp-output['allowed_log_probs'][i])).sum())
    oracle=(terms[0]+(terms[1]+terms[2])/2)/2
    oracle.backward()
    torch.testing.assert_close(actual/2,oracle)
    torch.testing.assert_close(logits.grad,oracle_logits.grad)
    assert reference_logits.grad is None
    assert logits.grad[0,10:].count_nonzero()==0
    assert logits.grad[0,:,3].count_nonzero()==0
    assert metrics['policy_loss']==0


def test_exact_kl_identity_gradient_and_reference_frozen():
    x=torch.tensor([2.,-1.,100.],requires_grad=True)
    q=torch.log_softmax(x[:2].detach()/.7,0).requires_grad_()
    value=legal_distribution_kl(x,[0,1],q)
    value.backward()
    assert abs(value.item())<1e-7
    torch.testing.assert_close(x.grad,torch.zeros_like(x),atol=1e-7,rtol=0)
    assert q.grad is None
    y=torch.tensor([-1.,2.,-999.],requires_grad=True)
    value=legal_distribution_kl(y,[0,1],q)
    value.backward()
    assert value>0 and y.grad[1]>0 and y.grad[0]<0 and y.grad[2]==0
    with pytest.raises(ValueError):
        legal_distribution_kl(y,[0,1],torch.tensor([0.,0.]))


def test_fail_closed_regularization(runtime,monkeypatch):
    args,batch,reducer=runtime
    logits=torch.zeros(1,12,4)
    args.entropy_coef=.01
    with pytest.raises(ValueError): custom_loss(args,batch,logits,reducer)
    monkeypatch.setenv('PACMAN_EXPERIMENTAL_ENTROPY_COEF','0.01')
    monkeypatch.setenv('PACMAN_EXPERIMENTAL_KL_COEF','0.01')
    args.use_kl_loss=True
    args.kl_loss_coef=.01
    with pytest.raises(ValueError): custom_loss(args,batch,logits,reducer)
    monkeypatch.delenv('PACMAN_EXPERIMENTAL_ENTROPY_COEF')
    args.entropy_coef=0
    with pytest.raises(ValueError,match='missing masked'): custom_loss(args,batch,logits,reducer)
    args.ref_update_interval=1
    with pytest.raises(ValueError,match='frozen'): custom_loss(args,batch,logits,reducer)


def test_launch_inherits_regularization_declarations(monkeypatch,tmp_path):
    from slime_pacman.launch import build_environment
    names=('PACMAN_EXPERIMENTAL_PPO_CLIP','PACMAN_EXPERIMENTAL_KL_COEF','PACMAN_EXPERIMENTAL_ENTROPY_COEF')
    for name in names: monkeypatch.setenv(name,'sentinel')
    env=build_environment(slime_root=tmp_path,config=tmp_path/'config',run_dir=tmp_path)
    assert all(env[name]=='sentinel' for name in names)


def test_reference_hook_transport_restore_and_actor_passthrough(monkeypatch):
    from slime_pacman.reference_policy import install, make_reference_actor
    monkeypatch.setenv('PACMAN_EXPERIMENTAL_KL_COEF','0.01')
    args=SimpleNamespace(use_kl_loss=True,kl_loss_coef=.01,kl_coef=0,ref_load='/frozen',
        tensor_model_parallel_size=1,pipeline_model_parallel_size=1,context_parallel_size=1,
        micro_batch_size=1,use_dynamic_batch_size=False,compute_advantages_and_returns=True)
    package=ModuleType('slime.backends.megatron_utils')
    actor=ModuleType(package.__name__+'.actor')
    model=ModuleType(package.__name__+'.model')
    class Actor:
        def compute_log_prob(self,*args): return 'original'
    actor.MegatronTrainRayActor=Actor
    # Ray captures a distinct actor class before worker init calls install.
    import ray
    import ray.cloudpickle
    # Exercise Ray's actual class wrapping and serialization before worker init.
    remote_class = ray.remote(make_reference_actor(Actor))
    CapturedActor = ray.cloudpickle.loads(ray.cloudpickle.dumps(
        remote_class.__ray_metadata__.modified_class))
    seen=[]
    def get_batch(iterator,keys,*args,**kwargs):
        seen.append(list(keys))
        return {'metadata':['first']}
    model.get_batch=get_batch
    def fail_forward(callback,args,model_arg,iterator,num_microbatches,**kwargs):
        model.get_batch(iterator,['tokens'])
        raise RuntimeError('forward failure')
    model.forward_only=fail_forward
    package.actor=actor
    package.model=model
    monkeypatch.setitem(sys.modules,package.__name__,package)
    monkeypatch.setitem(sys.modules,actor.__name__,actor)
    monkeypatch.setitem(sys.modules,model.__name__,model)
    install(args)
    wrapped=model.get_batch
    wrapped(None,['tokens','ref_log_probs'])
    assert 'ref_allowed_log_probs' in seen[-1]
    obj=CapturedActor()
    obj.args=args
    obj.model=[]
    assert obj.compute_log_prob([],[])=='original'
    with pytest.raises(RuntimeError,match='forward failure'):
        obj.compute_log_prob([],[],store_prefix='ref_')
    assert 'metadata' in seen[-1]
    assert model.get_batch is wrapped


    install(args)
    assert model.get_batch is wrapped
    from slime_pacman import reference_policy
    captured=[]
    def capture(logits, *, metadata, **kwargs):
        captured.append(metadata)
        lp = torch.log_softmax(torch.tensor(logits), 0)
        return {'allowed_log_probs':[lp], 'log_probs':[lp[:1]]}
    monkeypatch.setattr(reference_policy,'reference_outputs',capture)
    def success_forward(callback,args,model_arg,iterator,num_microbatches,**kwargs):
        model.get_batch(iterator,['tokens'])
        model.get_batch(iterator,['tokens'])
        first_loss, first=callback(torch.tensor([-.2,-1.7]))
        second_loss, second=callback(torch.tensor([0.]))
        # Megatron forward_step_calc_loss divides the returned loss tensor.
        first_loss /= 2
        second_loss /= 2
        assert first_loss.numel() == second_loss.numel() == 0
        return {'ref_allowed_log_probs':first['allowed_log_probs']+second['allowed_log_probs'],
                'ref_log_probs':first['log_probs']+second['log_probs']}
    model.forward_only=success_forward
    result=obj.compute_log_prob([],[2],store_prefix='ref_')
    assert captured==[['first'],['first']]
    assert [len(x) for x in result['ref_allowed_log_probs']]==[2,1]
    assert model.get_batch is wrapped


def test_reference_callback_forbidden_logits_invariance(runtime):
    args,batch,_=runtime
    logits=torch.randn(1,12,4)
    named=dict(metadata=batch['metadata'],args=args,**{k:batch[k] for k in ('unconcat_tokens','total_lengths','response_lengths')})
    first=reference_outputs(logits,**named)
    changed=logits.clone()
    for i,pos in enumerate((1,5,8)):
        banned=[j for j in range(4) if j not in batch['metadata'][i]['allowed_token_ids']]
        changed[0,pos,banned]=100000
    second=reference_outputs(changed,**named)
    for key in first:
        for x,y in zip(first[key],second[key],strict=True): torch.testing.assert_close(x,y)
    for i,pos in enumerate((1,5,8)):
        support=batch['metadata'][i]['allowed_token_ids']
        token=int(batch['unconcat_tokens'][i][-1])
        torch.testing.assert_close(first['log_probs'][i][0],first['allowed_log_probs'][i][support.index(token)])



def test_driver_injects_reference_actor_before_ray_capture(monkeypatch):
    from slime_pacman.reference_policy import install_driver, ReferencePolicyMixin
    package = ModuleType('slime.ray')
    placement = ModuleType('slime.ray.placement_group')
    actor = ModuleType('slime.backends.megatron_utils.actor')
    class Base:
        pass
    actor.MegatronTrainRayActor = Base
    seen = []
    def original(args, pgs, manager, actor_cls=None):
        seen.append(actor_cls)
        return 'created'
    placement.create_training_models = original
    package.placement_group = placement
    monkeypatch.setitem(sys.modules, 'slime.ray', package)
    monkeypatch.setitem(sys.modules, placement.__name__, placement)
    monkeypatch.setitem(sys.modules, actor.__name__, actor)
    install_driver()
    wrapper = placement.create_training_models
    assert wrapper(SimpleNamespace(use_kl_loss=True), None, None) == 'created'
    assert issubclass(seen[0], Base) and issubclass(seen[0], ReferencePolicyMixin)
    install_driver()
    assert placement.create_training_models is wrapper
    with pytest.raises(ValueError):
        wrapper(SimpleNamespace(use_kl_loss=True), None, None, actor_cls=Base)


def test_actual_native_reference_callback_return_contract():
    """Execute the pinned callback body, stubbing only distributed math helpers."""
    mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    native = load_body('slime/backends/megatron_utils/loss.py',
                       'get_log_probs_and_entropy', mpu)
    native.__globals__.update(
        _build_shifted_tokens=lambda *args: torch.zeros(3, dtype=torch.long),
        calculate_log_probs_and_entropy=lambda *args, **kw: (torch.zeros(3, 1), None),
        _extract_per_sample=lambda *args: ([torch.tensor([0.])], []),
    )
    result = native(torch.zeros(1, 3, 4),
                    args=SimpleNamespace(rollout_temperature=.7, log_probs_chunk_size=8,
                                         allgather_cp=False),
                    unconcat_tokens=[torch.tensor([1, 2, 3])],
                    total_lengths=[3], response_lengths=[1])
    loss, data = result
    loss /= 2
    assert isinstance(result, tuple) and loss.shape == (0,)
    assert list(data) == ['log_probs'] and len(data['log_probs']) == 1
