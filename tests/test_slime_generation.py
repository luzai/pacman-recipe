"""Exercise the request boundary, including stale-history rejection."""

import asyncio
import base64
from copy import deepcopy
from io import BytesIO
import math
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from pacman_recipe.level1.token_constraints import ObjectiveTokenConstraint
from slime_pacman.generation import SGLangGenerator, process_request
from slime_pacman.batching import pad_schedule


def messages(color):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return [
        {"role": "system", "content": "Choose a code."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "B or F"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    ]


class Processor:
    def apply_chat_template(self, chat, **kwargs):
        assert kwargs == dict(
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return "system user image assistant"

    def __call__(self, *, text, images, **kwargs):
        assert kwargs == dict(return_tensors="pt", truncation=False)
        assert len(images) == len(text) == 1
        return dict(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            pixel_values=torch.tensor(
                [list(images[0].getpixel((0, 0)))], dtype=torch.float32
            ),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 1, 0]]),
        )


CONSTRAINT = ObjectiveTokenConstraint(("C0", "C1"), ("B", "F"), ((66,), (70,)))


class Client:
    def __init__(self, damage=None):
        self.requests = []
        self.damage = damage

    async def post(self, endpoint, json):
        self.requests.append(deepcopy(json))
        result = {
            "text": "B",
            "meta_info": {
                "weight_version": 0,
                "prompt_tokens": 3,
                "output_token_logprobs": [[-math.log(2), 66, "B"]],
                "output_token_ids_logprobs": [
                    [[-math.log(2), 66, "B"], [-math.log(2), 70, "F"]]
                ],
            },
        }
        if self.damage:
            self.damage(result)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: result)


def test_each_request_contains_only_current_image_and_fresh_identity():
    pytest.importorskip("slime.utils.processing_utils")
    client = Client()
    generator = SGLangGenerator(
        processor=Processor(), endpoint="http://localhost/generate", client=client
    )

    async def run():
        return [
            await generator(messages(color), CONSTRAINT)
            for color in ("red", "blue", "red")
        ]

    decisions = asyncio.run(run())
    a, b, a2 = client.requests
    assert len({r["rid"] for r in client.requests}) == 3
    assert a["image_data"] == a2["image_data"] != b["image_data"]
    assert all(
        len(r["image_data"]) == 1 and "session_params" not in r for r in client.requests
    )
    assert all(r["sampling_params"]["temperature"] == 1 for r in client.requests)
    assert (
        decisions[0].image_sha256
        == decisions[2].image_sha256
        != decisions[1].image_sha256
    )
    assert all(
        d.weight_version == "0" and d.behavior_log_prob == -math.log(2)
        for d in decisions
    )
    assert not torch.equal(
        decisions[0].multimodal_train_inputs["pixel_values"],
        decisions[1].multimodal_train_inputs["pixel_values"],
    )


@pytest.mark.parametrize(
    "damage",
    [
        lambda r: r["meta_info"].update(weight_version=None),
        lambda r: r["meta_info"].update(prompt_tokens=4),
        lambda r: r["meta_info"].update(output_token_ids_logprobs=[]),
        lambda r: r["meta_info"].update(output_token_logprobs=[[0.0, 66, "B"]]),
        lambda r: r["meta_info"].update(
            output_token_logprobs=[[float("nan"), 66, "B"]]
        ),
        lambda r: r.update(text="F"),
    ],
)
def test_bad_server_evidence_aborts(damage):
    pytest.importorskip("slime.utils.processing_utils")
    generator = SGLangGenerator(
        processor=Processor(),
        endpoint="http://localhost/generate",
        client=Client(damage),
    )
    with pytest.raises(ValueError):
        asyncio.run(generator(messages("red"), CONSTRAINT))


def test_mm_token_type_ids_are_not_forwarded_to_training():
    # Megatron GPTModel.forward rejects this processor output (smoke v7).
    _, _, mm, _ = process_request(Processor(), messages("red"), CONSTRAINT, 2048)
    assert set(mm) == {"pixel_values", "image_grid_thw"}


def test_history_second_image_and_truncation_are_rejected():
    current = messages("red")
    with pytest.raises(ValueError, match="fresh"):
        process_request(
            Processor(),
            current + [{"role": "assistant", "content": "B"}],
            CONSTRAINT,
            2048,
        )
    current[1]["content"].append(deepcopy(current[1]["content"][-1]))
    with pytest.raises(ValueError, match="exactly one"):
        process_request(Processor(), current, CONSTRAINT, 2048)
    with pytest.raises(ValueError, match="budget"):
        process_request(Processor(), messages("red"), CONSTRAINT, 2)


def test_padding_preserves_episode_denominator_and_real_decisions():
    schedule = pytest.importorskip("slime.utils.dp_schedule").build_dp_schedule
    # 48 episodes, 49 decisions: cannot be scheduled on DP8 without padding.
    ids = [0] + list(range(48))
    data = dict(
        tokens=[[1, 2, 66] for _ in ids],
        response_lengths=[1] * 49,
        rollout_ids=ids.copy(),
        loss_masks=[[1] for _ in ids],
        rollout_log_probs=[[-0.5] for _ in ids],
        rollout_mask_sums=[2 if i == 0 else 1 for i in ids],
        metadata=[dict(episode_id=i, empty_episode=False) for i in ids],
        multimodal_train_inputs=[None] * 49,
    )
    pad_schedule(data, 8)
    assert len(data["tokens"]) == 56
    assert data["rollout_ids"][:49] == ids and len(set(data["rollout_ids"])) == 48
    assert sum(sum(m) for m in data["loss_masks"]) == 49
    assert data["rollout_mask_sums"] == [
        2 if i == 0 else 1 for i in data["rollout_ids"]
    ]
    args = SimpleNamespace(
        use_dynamic_batch_size=False,
        micro_batch_size=1,
        balance_data=False,
        balance_by_flops=False,
    )
    parts, batches, micro, sizes = schedule(
        args,
        dict(dp_size=8, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1),
        [len(t) for t in data["tokens"]],
        global_batch_size=48,
        rollout_indices=data["rollout_ids"],
    )
    assert sizes == [48] and micro == [7]
    assert sorted(i for part in parts for i in part) == list(range(56))
    assert all(len(batch) == 1 for rank in batches for batch in rank)


def test_real_upstream_conversion_keeps_support_and_episode_weights():
    Sample = pytest.importorskip("slime.utils.types").Sample
    pytest.importorskip("slime.rollout.batch_builder")
    from slime_pacman.batching import convert_samples

    samples = []
    for episode in range(48):
        count = 2 if episode == 0 else 1
        for index in range(count):
            samples.append(
                Sample(
                    index=episode,
                    group_index=episode // 12,
                    rollout_id=episode,
                    tokens=[1, 2, 66],
                    response="B",
                    response_length=1,
                    loss_mask=[1],
                    reward=float(episode % 2),
                    rollout_log_probs=[-0.5],
                    status=Sample.Status.COMPLETED,
                    metadata={"source": "pacman"},
                    train_metadata=dict(
                        group_id=episode // 12,
                        episode_id=episode,
                        initial_state_id=f"seed-{episode // 12}",
                        decision_count=count,
                        decision_index=index,
                        weight_version="v0",
                        allowed_token_ids=[66, 70],
                        empty_episode=False,
                    ),
                )
            )
    args = SimpleNamespace(
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        global_batch_size=48,
        n_samples_per_prompt=12,
        custom_convert_samples_to_train_data_path="slime_pacman.batching.convert_samples",
        custom_reward_post_process_path="slime_pacman.grouping.post_process_rewards",
    )
    data = convert_samples(args, samples)
    assert len(data["tokens"]) == 56 and len(set(data["rollout_ids"])) == 48
    assert data["metadata"][1]["allowed_token_ids"] == [66, 70]
    assert data["rollout_mask_sums"][:3] == [2, 2, 1]
    assert data["rewards"][0] == data["rewards"][1] == -data["rewards"][2]
    assert all(m == [0] for m in data["loss_masks"][49:])
