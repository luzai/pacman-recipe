import pytest

torch = pytest.importorskip('torch', reason='PyTorch CPU numerical checks require the optional training dependency')
from pacman_recipe.experimental.engine_assisted.choice_loss import clipped_choice_loss, support_log_probs, engine_choice_loss


def test_exact_support_normalization_and_clipped_surrogate_gradient():
    logits = torch.tensor([0., 1., 2., 3.], requires_grad=True)
    logs = support_log_probs(logits, [1, 3], temperature=.7)
    assert torch.allclose(logs.exp().sum(), torch.tensor(1.))
    old = torch.log(torch.tensor([.5, .5])); reference = logs.detach().clone()
    loss = clipped_choice_loss(logs, old, reference, 1, 1., clip=.1, kl_beta=.02)
    assert torch.allclose(loss, torch.tensor(-1.1), atol=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0] == logits.grad[2] == 0


def test_old_and_reference_are_detached_and_invalid_support_rejected():
    new = torch.log_softmax(torch.tensor([.2, -.2], requires_grad=True), dim=0)
    old = torch.log_softmax(torch.tensor([.1, -.1], requires_grad=True), dim=0); old.retain_grad()
    ref = torch.log_softmax(torch.tensor([0., 0.], requires_grad=True), dim=0); ref.retain_grad()
    clipped_choice_loss(new, old, ref, 0, -.5).backward()
    assert old.grad is None and ref.grad is None
    with pytest.raises(ValueError): support_log_probs(torch.zeros(3), [1, 1])
    with pytest.raises(ValueError): clipped_choice_loss(torch.tensor([0., 0.]), torch.zeros(2), torch.zeros(2), 0, 1.)


def test_engine_entropy_is_explicit_and_does_not_replace_behavior_logs():
    new = torch.log_softmax(torch.tensor([.2, -.2]), dim=0)
    old = torch.log(torch.tensor([.5, .5])); reference = old.clone()
    baseline = clipped_choice_loss(new, old, reference, 0, 1., clip=.1, kl_beta=.02)
    actual = engine_choice_loss(new, old, reference, 0, 1., clip=.1, kl_beta=.02, entropy_bonus=.01)
    assert torch.allclose(actual, baseline + .01*(new.exp()*new).sum())
    assert torch.equal(old, reference)
