"""CPU execution of pinned upstream reducer bodies, without a CUDA import stack.

The actual source functions are extracted with AST; only the TP/CP topology
provider is replaced. This does not exercise a Megatron model or distributed GPU.
"""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterator

import pytest
import torch

from slime_pacman.probability import custom_loss, masked_log_prob, clipped_policy_terms

UPSTREAM = Path(__file__).resolve().parents[2] / "slime"


def load_body(relative, name, mpu):
    path = UPSTREAM / relative
    if not path.is_file():
        pytest.skip("pinned slime checkout unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = dict(
        torch=torch,
        mpu=mpu,
        Namespace=SimpleNamespace,
        Iterator=Iterator,
        Callable=Callable,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def test_actual_causal_slice_and_episode_reducer_match_gradient_oracle(monkeypatch):
    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    get_responses = load_body(
        "slime/backends/megatron_utils/loss.py", "get_responses", mpu
    )
    get_reducer = load_body(
        "slime/backends/megatron_utils/cp_utils.py", "get_sum_of_sample_mean", mpu
    )
    core = ModuleType("megatron.core")
    core.mpu = mpu
    module = ModuleType("slime.backends.megatron_utils.loss")
    module.get_responses = get_responses
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", module)
    args = SimpleNamespace(
        use_rollout_logprobs=True,
        calculate_per_token_loss=False,
        rollout_temperature=0.7,
        eps_clip=0.2,
    )
    # Episode 0: one decision. Episode 1: two decisions. Last sample is padding.
    batch = dict(
        response_lengths=[1] * 4,
        total_lengths=[3, 4, 3, 2],
        unconcat_tokens=[
            torch.tensor(v) for v in ([3, 2, 1], [3, 2, 2, 0], [2, 2, 1], [0, 0])
        ],
        metadata=[dict(allowed_token_ids=[0, 1], empty_episode=False)] * 3
        + [dict(empty_episode=True)],
        rollout_log_probs=[torch.tensor([-0.7])] * 3 + [torch.tensor([0.0])],
        advantages=[
            torch.tensor([1.0]),
            torch.tensor([-1.0]),
            torch.tensor([-1.0]),
            torch.tensor([1.0]),
        ],
    )
    masks = [torch.tensor([1.0])] * 3 + [torch.tensor([0.0])]
    reducer = get_reducer(
        batch["total_lengths"],
        batch["response_lengths"],
        masks,
        torch.tensor([1.0, 2.0, 2.0, 1.0]),
    )
    torch.manual_seed(5)
    logits = torch.randn(1, 12, 4, requires_grad=True)
    loss, _ = custom_loss(args, batch, logits, reducer)
    (loss / 2).backward()
    actual_grad = logits.grad.clone()
    reference = logits.detach().clone().requires_grad_()
    selected = torch.stack(
        [
            masked_log_prob(reference[0, position], [0, 1], action)[0]
            for position, action in ((1, 1), (5, 0), (8, 1))
        ]
    )
    terms = clipped_policy_terms(
        selected, torch.full((3,), -0.7), torch.tensor([1.0, -1.0, -1.0])
    )
    oracle = (terms[0] + terms[1:].mean()) / 2
    oracle.backward()
    torch.testing.assert_close(loss / 2, oracle)
    torch.testing.assert_close(actual_grad, reference.grad)
    assert actual_grad[0, 10:].count_nonzero() == 0


def test_metadata_survives_both_upstream_transport_boundaries():
    for relative, function in (
        ("slime/rollout/batch_builder.py", "split_by_dp"),
        ("slime/backends/megatron_utils/model.py", "train_one_step"),
    ):
        path = UPSTREAM / relative
        if not path.is_file():
            pytest.skip("pinned slime checkout unavailable")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        assert any(
            isinstance(n, ast.Constant) and n.value == "metadata"
            for n in ast.walk(body)
        )
