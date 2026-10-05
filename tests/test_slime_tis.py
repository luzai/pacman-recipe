"""Numerical and gradient acceptance for detached legal-action TIS."""
from types import SimpleNamespace

import pytest
import torch

from slime_pacman.probability import tis_weights, clipped_policy_terms, validate_single_update


def test_tis_bounds_no_rejection_and_detached_gradient():
    old = torch.tensor([-.1, -2., -.3], requires_grad=True)
    behavior = torch.tensor([-5., -1., -.3], requires_grad=True)
    weights = tis_weights(old, behavior)
    torch.testing.assert_close(weights, torch.tensor([2., 0.36787944, 1.]))
    assert not weights.requires_grad
    current = old.detach().clone().requires_grad_()
    advantage = torch.tensor([1., -1., 2.])
    loss = (weights * clipped_policy_terms(current, old, advantage, .05)).sum()
    loss.backward()
    torch.testing.assert_close(current.grad, -weights * advantage)
    assert old.grad is None and behavior.grad is None


def test_clip_and_tis_use_distinct_denominators():
    old = torch.tensor([-.5])
    current = (old + torch.log(torch.tensor(1.2))).requires_grad_()
    behavior = old - torch.log(torch.tensor(1.5))
    loss = (tis_weights(old, behavior) * clipped_policy_terms(current, old, torch.ones(1), .05)).sum()
    torch.testing.assert_close(loss, torch.tensor(-1.5 * 1.05))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(1))


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_tis_rejects_nonfinite(bad):
    with pytest.raises(ValueError):
        tis_weights(torch.tensor([bad]), torch.tensor([-1.]))


def test_tis_avoids_exponential_overflow():
    torch.testing.assert_close(tis_weights(torch.tensor([0.]), torch.tensor([-1000.])), torch.tensor([2.]))


@pytest.mark.parametrize('changes', [dict(num_steps_per_rollout=2), dict(global_batch_size=24), dict(hidden_dropout=.1)])
def test_single_update_contract(changes):
    args = SimpleNamespace(num_steps_per_rollout=1, global_batch_size=48, rollout_batch_size=4, n_samples_per_prompt=12)
    validate_single_update(args)
    for k, v in changes.items():
        setattr(args, k, v)
    with pytest.raises(ValueError, match='one optimizer step'):
        validate_single_update(args)
