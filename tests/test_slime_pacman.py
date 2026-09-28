"""CPU acceptance for the independent-image slime adapter."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pacman_recipe.level1.contracts import (
    make_episode_record,
    runner_row,
    validate_episode_record,
)
from slime_pacman.config import PacmanConfig, load_config
from slime_pacman.generation import Decision
from slime_pacman.grouping import binary_reward, group_advantages, post_process_rewards
from slime_pacman.probability import (
    PacmanLogitProcessor,
    clipped_policy_terms,
    masked_log_prob,
)
from slime_pacman.rollout import EpisodeRunner, samples_from_episode

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    all_special_ids = [0]
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(text)]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def test_default_contract_and_disjoint_seeds():
    config = load_config(ROOT / "configs/slime/c2.yaml")
    assert config.group_size == 12 and config.max_steps == 512
    assert len(config.train_seeds) == 40 and len(config.validation_seeds) == 4
    with pytest.raises(ValueError, match="overlap"):
        PacmanConfig(validation_seed_start=28)


@pytest.mark.parametrize("support", [[], [2, 2], [-1], [9], [True]])
def test_invalid_probability_support_fails(support):
    with pytest.raises(ValueError):
        masked_log_prob(torch.zeros(9), support, 2)


def test_masked_softmax_oracle_and_forbidden_gradient():
    logits = torch.tensor([100.0, 2.0, -1.0, 3.0, 99.0], requires_grad=True)
    actual, entropy = masked_log_prob(logits, [1, 3], 3)
    oracle = torch.log_softmax(torch.tensor([2.0, 3.0]) / 0.7, 0)[1]
    torch.testing.assert_close(actual, oracle)
    (-actual).backward()
    assert torch.equal(logits.grad[[0, 2, 4]], torch.zeros(3))
    assert logits.grad[3] < 0 and logits.grad[1] > 0 and entropy > 0


def test_singleton_has_zero_loss_gradient_and_entropy():
    logits = torch.randn(12, requires_grad=True)
    value, entropy = masked_log_prob(logits, [4], 4)
    (value + entropy).backward()
    assert value == 0 and entropy == 0 and torch.equal(logits.grad, torch.zeros(12))


def test_sglang_mask_and_temperature_match_training_for_mixed_batch():
    raw = torch.tensor([[11.0, 3.0, -2.0, 1.0], [2.0, 4.0, 9.0, 8.0]])
    params = [
        {"pacman_allowed_token_ids": [1, 3], "pacman_temperature": 0.7},
        {"pacman_allowed_token_ids": [0], "pacman_temperature": 0.7},
    ]
    masked = PacmanLogitProcessor()(raw.clone(), params)
    expected, _ = masked_log_prob(raw[0], [1, 3], 3)
    torch.testing.assert_close(masked.log_softmax(-1)[0, 3], expected)
    assert masked.log_softmax(-1)[1, 0] == 0
    assert torch.isneginf(masked[0, 0]) and torch.isneginf(masked[1, 1])


@pytest.mark.parametrize("rewards", [[0.0] * 12, [1.0] * 12])
def test_zero_variance_groups_are_retained_with_zero_advantages(rewards):
    assert group_advantages(rewards) == [0.0] * 12


def samples_fixture():
    samples = []
    for episode in range(12):
        count = episode % 3 + 1
        for index in range(count):
            samples.append(
                SimpleNamespace(
                    rollout_id=episode,
                    reward=float(episode >= 6),
                    train_metadata={
                        "group_id": 7,
                        "episode_id": episode,
                        "initial_state_id": "seed28",
                        "decision_count": count,
                        "decision_index": index,
                        "weight_version": "version-1",
                    },
                )
            )
    return samples


def test_normalization_uses_twelve_episodes_before_expansion():
    samples = samples_fixture()
    _, advantages = post_process_rewards(
        SimpleNamespace(n_samples_per_prompt=12), samples
    )
    reference = group_advantages([0.0] * 6 + [1.0] * 6)
    assert advantages == [reference[s.rollout_id] for s in samples]


def test_zero_decision_refusals_do_not_invent_weight_versions_or_get_replaced():
    samples = [
        SimpleNamespace(
            rollout_id=i,
            reward=0.0,
            train_metadata=dict(
                group_id=0,
                episode_id=i,
                initial_state_id="seed28",
                decision_count=0,
                decision_index=-1,
                weight_version="",
            ),
        )
        for i in range(12)
    ]
    raw, advantages = post_process_rewards(
        SimpleNamespace(n_samples_per_prompt=12), samples
    )
    assert raw == advantages == [0.0] * 12
    assert len(samples) == 12
    samples[0].reward = 1.0
    with pytest.raises(ValueError, match="zero-decision"):
        post_process_rewards(SimpleNamespace(n_samples_per_prompt=12), samples)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "state", "version", "reward"]
)
def test_grouping_fails_closed(damage):
    samples = samples_fixture()
    if damage == "missing":
        samples.pop()
    if damage == "duplicate":
        samples.append(deepcopy(samples[-1]))
    if damage == "state":
        samples[0].train_metadata["initial_state_id"] = "other"
    if damage == "version":
        samples[0].train_metadata["weight_version"] = "stale"
    if damage == "reward":
        samples[-1].reward = 0.5
    with pytest.raises(ValueError):
        post_process_rewards(SimpleNamespace(n_samples_per_prompt=12), samples)


def test_unequal_episode_lengths_partition_and_padding_preserve_gradient():
    # Explicit oracle: two equal episode means, with 1 and 3 decisions.
    def objective(partitions):
        logits = torch.tensor([[1.0, 2.0, 20.0]] * 4, requires_grad=True)
        old = torch.full((4,), -0.5)
        advantage = torch.tensor([1.0, -1.0, -1.0, -1.0])
        loss = logits.sum() * 0
        for partition in partitions:
            new = torch.stack(
                [masked_log_prob(logits[i], [0, 1], 1)[0] for i in partition]
            )
            term = clipped_policy_terms(new, old[partition], advantage[partition])
            denominators = torch.tensor([1 if i == 0 else 3 for i in partition])
            loss = loss + (term / denominators).sum() / 2
        loss.backward()
        return loss.detach(), logits.grad

    expected = objective([[0, 1, 2, 3]])
    actual = objective([[3], [0], [2, 1]])
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert torch.equal(actual[1][:, 2], torch.zeros(4))


@pytest.mark.parametrize(
    "won,reason,remaining,reward",
    [
        (True, "all_normal_pellets", 0, 1.0),
        (False, "death", 12, 0.0),
        (False, "max_steps", 1, 0.0),
        (False, "safety_refusal", 9, 0.0),
    ],
)
def test_binary_game_outcomes(won, reason, remaining, reward):
    payload = dict(
        episode_life_mode="single_death",
        won=won,
        terminal_reason=reason,
        normal_pellets_remaining=remaining,
    )
    assert binary_reward(payload) == reward
    payload["parse_failures"] = 1
    with pytest.raises(ValueError):
        binary_reward(payload)


def test_schema_records_actual_backend_and_rejects_prompt_drift():
    record = make_episode_record(
        28,
        split="test",
        recipe_root=ROOT,
        game_root=ROOT.parent / "pacman-python",
        backend_root=ROOT.parent / "slime",
        max_steps=32,
    )
    validate_episode_record(record, expected_sources=record["source_revisions"])
    assert set(record["source_revisions"]) == {
        "slime",
        "pacman-recipe",
        "pacman-python",
    }
    assert "maapacman" not in json.dumps(record)
    assert runner_row(record)["training_backend"] == "slime"
    record["prompt"]["prompt_version"] = "edward-option-code-v1"
    with pytest.raises(ValueError, match="prompt"):
        validate_episode_record(record)


def test_real_game_runs_without_areal_and_yields_audited_binary_episode():
    config = PacmanConfig(max_steps=32)
    record = make_episode_record(
        28,
        split="test",
        recipe_root=ROOT,
        game_root=ROOT.parent / "pacman-python",
        backend_root=ROOT.parent / "slime",
        max_steps=32,
    )
    requests = []

    async def generate(messages, constraint):
        assert len(messages) == 2
        images = [
            part
            for m in messages
            if isinstance(m["content"], list)
            for part in m["content"]
            if part["type"] == "image_url"
        ]
        assert len(images) == 1
        requests.append(images[0]["image_url"]["url"])
        token = constraint.allowed_token_ids[0]
        return Decision(
            "prompt",
            [1, 2],
            token,
            chr(token),
            constraint.allowed_token_ids,
            -float(np.log(len(constraint.allowed_token_ids))),
            "v0",
            {},
            "0" * 64,
            [],
        )

    runner = EpisodeRunner(
        record, tokenizer=Tokenizer(), generate=generate, config=config
    )
    result = asyncio.run(runner.collect(empty_weight_version="v0"))
    assert result.reward in (0.0, 1.0) and result.trajectory is not None
    assert result.trajectory["episode"]["episode_life_mode"] == "single_death"
    assert len(requests) == len(result.decisions) > 1
    assert len(set(requests)) > 1


def test_upstream_samples_preserve_episode_identity_and_vision_fields():
    Sample = pytest.importorskip("slime.utils.types").Sample
    from slime_pacman.rollout import EpisodeResult

    decision = Decision(
        "p",
        [1, 2],
        66,
        "B",
        [66],
        0.0,
        "v0",
        {"pixel_values": torch.ones(2, 3)},
        "a" * 64,
        [0.0],
    )
    parent = Sample(
        index=5, group_index=3, metadata={"episode_record": {"id": "seed28"}}
    )
    samples = samples_from_episode(
        parent,
        EpisodeResult(1.0, [decision, decision], "v0", None, "all_normal_pellets"),
        Tokenizer(),
    )
    assert [s.rollout_id for s in samples] == [5, 5]
    assert all(s.tokens == [1, 2, 66] and s.loss_mask == [1] for s in samples)
    assert all(
        s.multimodal_train_inputs is decision.multimodal_train_inputs for s in samples
    )
    empty = samples_from_episode(
        parent,
        EpisodeResult(0.0, [], "v0", None, "initial_safety_refusal"),
        Tokenizer(),
    )
    assert (
        len(empty) == 1
        and empty[0].loss_mask == [0]
        and empty[0].train_metadata["empty_episode"]
    )
