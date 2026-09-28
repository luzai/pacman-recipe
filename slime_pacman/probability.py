"""The same finite action support and temperature on both sides of GRPO."""

import json
import math

import torch


def validate_support(support, vocab_size):
    if not support or any(
        type(i) is not int or not 0 <= i < vocab_size for i in support
    ):
        raise ValueError("support must contain valid token IDs")
    if len(set(support)) != len(support):
        raise ValueError("support contains duplicate token IDs")


def masked_log_prob(logits, support, action, temperature=0.7):
    if logits.ndim != 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("expected one vocabulary vector and a positive temperature")
    validate_support(support, logits.numel())
    if action not in support:
        raise ValueError("sampled action is outside support")
    values = logits[support].float() / temperature
    if not torch.isfinite(values).all():
        raise ValueError("non-finite allowed logits")
    log_probs = torch.log_softmax(values, dim=0)
    return log_probs[support.index(action)], -(log_probs.exp() * log_probs).sum()


class PacmanLogitProcessor:
    """SGLang callable: mask and scale before either log-probability path.

    Request temperature must be 1.0. Effective policy temperature is applied
    here so SGLANG_RETURN_ORIGINAL_LOGPROB cannot silently change its meaning.
    """

    @classmethod
    def to_str(cls):
        import dill

        return json.dumps({"callable": dill.dumps(cls).hex()})

    def __call__(self, logits, custom_param_list=None):
        if (
            logits.ndim != 2
            or logits.dtype != torch.float32
            or custom_param_list is None
            or len(custom_param_list) != len(logits)
        ):
            raise ValueError(
                "custom masking requires FP32 logits and per-request parameters"
            )
        for row, params in zip(logits, custom_param_list, strict=True):
            support = params["pacman_allowed_token_ids"]
            validate_support(support, row.numel())
            temperature = params["pacman_temperature"]
            if temperature != 0.7:
                raise ValueError("unsupported policy temperature")
            selected = row[support].clone() / temperature
            if not torch.isfinite(selected).all():
                raise ValueError("non-finite allowed logits")
            row.fill_(-float("inf"))
            row[support] = selected
        return logits


def clipped_policy_terms(new_log_probs, old_log_probs, advantages, clip=0.2):
    if (
        new_log_probs.shape != old_log_probs.shape
        or new_log_probs.shape != advantages.shape
    ):
        raise ValueError("policy inputs must have identical shapes")
    if not all(
        torch.isfinite(x).all() for x in (new_log_probs, old_log_probs, advantages)
    ):
        raise ValueError("non-finite policy inputs")
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    if not torch.isfinite(ratio).all():
        raise ValueError("policy ratio overflow")
    return torch.maximum(
        -advantages.detach() * ratio,
        -advantages.detach() * ratio.clamp(1 - clip, 1 + clip),
    )


def custom_loss(args, batch, logits, sum_of_sample_mean):
    """slime custom loss; its outer reducer averages whole episodes globally."""
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_responses

    if (
        mpu.get_tensor_model_parallel_world_size() != 1
        or mpu.get_context_parallel_world_size() != 1
    ):
        raise ValueError("first Pacman adapter requires TP=CP=1")
    if not args.use_rollout_logprobs or args.calculate_per_token_loss:
        raise ValueError("Pacman requires behavior log-probs and per-episode reduction")
    if (
        args.rollout_temperature != 0.7
        or args.eps_clip != 0.2
        or getattr(args, "eps_clip_high", 0.2) != 0.2
    ):
        raise ValueError("Pacman temperature/clip contract changed")
    if any(
        getattr(args, name, False)
        for name in (
            "normalize_advantages",
            "use_kl_loss",
            "use_tis",
            "use_score_centering",
        )
    ):
        raise ValueError("unsupported additional policy correction or normalization")
    if any(
        getattr(args, name, 0) != 0
        for name in ("kl_coef", "kl_loss_coef", "entropy_coef")
    ):
        raise ValueError("Pacman requires zero KL and entropy coefficients")
    metadata = batch.get("metadata")
    if metadata is None or len(metadata) != len(batch["response_lengths"]):
        raise ValueError(
            "missing support metadata: apply the pinned slime transport patch"
        )
    new, entropies = [], []
    responses = get_responses(
        logits.float(),
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=False,
    )
    for (row, tokens), meta in zip(responses, metadata, strict=True):
        if len(tokens) != 1:
            raise ValueError("each Pacman decision must have exactly one output token")
        if meta.get("empty_episode"):
            value = row.sum() * 0
            entropy = value
        else:
            value, entropy = masked_log_prob(
                row[0],
                meta["allowed_token_ids"],
                int(tokens[0]),
                args.rollout_temperature,
            )
        new.append(value)
        entropies.append(entropy)
    new = torch.stack(new)
    old = torch.cat(batch["rollout_log_probs"]).to(new)
    advantages = torch.cat(batch["advantages"]).to(new)
    loss = sum_of_sample_mean(clipped_policy_terms(new, old, advantages, args.eps_clip))
    return loss, {
        "loss": loss.detach(),
        "masked_entropy": sum_of_sample_mean(torch.stack(entropies)).detach(),
        "masked_logprob_abs_diff": sum_of_sample_mean((new - old).abs()).detach(),
    }
