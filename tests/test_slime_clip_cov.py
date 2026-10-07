"""Clip-Cov whole-update selection and its masked policy-loss contract."""
from types import SimpleNamespace

import pytest
import torch

from slime_pacman import clip_cov
from slime_pacman.clip_cov import KEY, annotate, declaration, detach_mask, select
from slime_pacman.probability import custom_loss
from test_slime_regularization import runtime  # noqa: F401  (pytest fixture)

ENV = clip_cov.ENV


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_declaration_parsing(monkeypatch):
    assert declaration() is None
    monkeypatch.setenv(ENV, "0.002,1,5")
    assert declaration() == (0.002, 1.0, 5.0)
    for bad in ("0.002", "0,1,5", "0.2,1,5", "0.01,5,1", "0.01,-1,5", "nan,1,5", "a,b,c"):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError):
            declaration()


def test_select_matches_definition_and_is_deterministic():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(5000, generator=g)
    lp = -torch.rand(5000, generator=g) * 6
    mask, cov = select(a, lp, ratio=0.01, low=0.2, high=5, seed=7)
    expected = (a.double() - a.double().mean()) * (lp.double() - lp.double().mean())
    torch.testing.assert_close(cov, expected)
    eligible = (expected > 0.2) & (expected < 5)
    assert int(mask.sum()) == min(50, int(eligible.sum()))
    assert not (mask & ~eligible).any()
    again, _ = select(a, lp, ratio=0.01, low=0.2, high=5, seed=7)
    other, _ = select(a, lp, ratio=0.01, low=0.2, high=5, seed=8)
    assert torch.equal(mask, again) and not torch.equal(mask, other)


def test_select_floor_cap_and_empty():
    a, lp = torch.tensor([1.0, -1.0, 0.0]), torch.tensor([-0.1, -3.0, -1.0])
    # cov: [(1)(+1.267), (-1)(-1.633), 0] -> two eligible above 1; ratio floor selects one
    mask, _ = select(a, lp, ratio=1e-4, low=1, high=5, seed=0)
    assert int(mask.sum()) == 1 and not mask[2]
    mask, _ = select(a, lp, ratio=0.1, low=10, high=20, seed=0)
    assert not mask.any()
    with pytest.raises(ValueError):
        select(torch.tensor([1.0, float("nan")]), torch.zeros(2), ratio=0.1, low=0, high=1, seed=0)


def make_data():
    rewards = [0.75, -0.25, -0.25, -0.25, 0.0]
    log_probs = [[-0.01], [-4.0], [-0.02], [-6.0], [0.0]]
    metadata = [dict(episode_id=e, empty_episode=False) for e in (0, 1, 1, 2)]
    metadata.append(dict(episode_id=3, empty_episode=True))
    return dict(rewards=rewards, rollout_log_probs=log_probs,
                loss_masks=[[1]] * 4 + [[0]], metadata=metadata)


def test_annotate_excludes_padding_and_records_summary(monkeypatch):
    args = SimpleNamespace(seed=1234, advantage_estimator="grpo", normalize_advantages=False)
    data = make_data()
    assert annotate(args, data) is None and all(KEY not in m for m in data["metadata"])
    # cov over the four real decisions: [1.873, 0.373, -0.622, 0.873]
    monkeypatch.setenv(ENV, "0.1,0.5,5")
    summary = annotate(args, data)
    assert summary["decisions"] == 4 and summary["eligible"] == 2 and summary["selected"] == 1
    assert data["metadata"][4][KEY] is False
    chosen = [i for i, m in enumerate(data["metadata"]) if m[KEY]]
    assert chosen in ([0], [3])
    repeat = make_data()
    annotate(args, repeat)
    assert [m[KEY] for m in repeat["metadata"]] == [m[KEY] for m in data["metadata"]]
    args.advantage_estimator = "ppo"
    with pytest.raises(ValueError, match="GRPO"):
        annotate(args, make_data())


def test_detach_mask_fails_closed(monkeypatch):
    ref = torch.zeros(2)
    assert detach_mask([{}, {}], ref) is None
    with pytest.raises(ValueError, match="without its declaration"):
        detach_mask([{KEY: True}, {}], ref)
    monkeypatch.setenv(ENV, "0.01,1,5")
    with pytest.raises(ValueError, match="did not annotate"):
        detach_mask([{KEY: True}, {}], ref)
    torch.testing.assert_close(detach_mask([{KEY: True}, {KEY: False}], ref), torch.tensor([0.0, 1.0]))


def test_loss_gradient_equals_zeroed_advantage_oracle(runtime, monkeypatch):  # noqa: F811
    args, batch, reducer = runtime
    torch.manual_seed(3)
    logits = torch.randn(1, 12, 4)
    batch["advantages"] = [torch.tensor([v]) for v in (0.8, -0.6, 0.4, 0.0)]
    batch["rollout_log_probs"] = [torch.tensor([v]) for v in (-0.3, -0.9, 0.0, 0.0)]

    monkeypatch.setenv(ENV, "0.01,1,5")
    for meta, flag in zip(batch["metadata"], (False, True, False, False)):
        meta[KEY] = flag
    actual_logits = logits.clone().requires_grad_()
    actual, metrics = custom_loss(args, batch, actual_logits, reducer)
    actual.backward()
    torch.testing.assert_close(metrics["clip_cov_detach_fraction"], torch.tensor(0.5))

    monkeypatch.delenv(ENV)
    for meta in batch["metadata"]:
        del meta[KEY]
    batch["advantages"][1] = torch.tensor([0.0])
    oracle_logits = logits.clone().requires_grad_()
    oracle, oracle_metrics = custom_loss(args, batch, oracle_logits, reducer)
    oracle.backward()
    assert "clip_cov_detach_fraction" not in oracle_metrics
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(actual_logits.grad, oracle_logits.grad)
    assert actual_logits.grad[0, 5].count_nonzero() == 0
    assert actual_logits.grad[0, 1].count_nonzero() > 0


def test_launch_inherits_clip_cov_declaration(monkeypatch, tmp_path):
    from slime_pacman.launch import build_environment
    monkeypatch.setenv(ENV, "0.002,1,5")
    env = build_environment(slime_root=tmp_path, config=tmp_path / "config", run_dir=tmp_path)
    assert env[ENV] == "0.002,1,5"


def test_real_upstream_conversion_annotates_and_writes_summary(monkeypatch, tmp_path):
    Sample = pytest.importorskip("slime.utils.types").Sample
    pytest.importorskip("slime.rollout.batch_builder")
    from slime_pacman.batching import convert_samples

    samples = []
    for episode in range(48):
        for index in range(3):
            samples.append(Sample(
                index=episode, group_index=episode // 12, rollout_id=episode,
                tokens=[1, 2, 66], response="B", response_length=1, loss_mask=[1],
                reward=float(episode % 3 == 0), rollout_log_probs=[-0.05 - 3.0 * ((episode + index) % 4 == 0)],
                status=Sample.Status.COMPLETED, metadata={"source": "pacman"},
                train_metadata=dict(group_id=episode // 12, episode_id=episode,
                                    initial_state_id=f"seed-{episode // 12}", decision_count=3,
                                    decision_index=index, weight_version="v0",
                                    allowed_token_ids=[66, 70], empty_episode=False)))
    args = SimpleNamespace(
        micro_batch_size=1, use_dynamic_batch_size=False, tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, context_parallel_size=1, actor_num_nodes=1,
        actor_num_gpus_per_node=8, global_batch_size=48, n_samples_per_prompt=12,
        custom_convert_samples_to_train_data_path="slime_pacman.batching.convert_samples",
        custom_reward_post_process_path="slime_pacman.grouping.post_process_rewards",
        seed=1234, advantage_estimator="grpo", normalize_advantages=False)
    monkeypatch.setenv(ENV, "0.01,0.1,5")
    monkeypatch.setenv("PACMAN_RUN_DIR", str(tmp_path))
    data = convert_samples(args, samples)
    flags = [m[KEY] for m in data["metadata"]]
    assert len(flags) == 144 and sum(flags) == 1
    (written,) = (tmp_path / "clip-cov").glob("*.json")
    assert '"selected": 1' in written.read_text()
    keep = detach_mask(data["metadata"], torch.zeros(1))
    assert keep.shape == (144,) and keep.sum() == 143
