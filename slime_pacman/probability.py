"""The same finite action support and temperature on both sides of GRPO."""

import json
import math
import os

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


def legal_distribution_kl(logits, support, reference_log_probs, temperature=0.7):
    """Exact KL(actor || frozen reference) over this decision's legal actions."""
    validate_support(support, logits.numel())
    log_p = torch.log_softmax(logits[support].float() / temperature, dim=0)
    log_q = reference_log_probs.detach().to(log_p)
    if (log_q.shape != log_p.shape or not torch.isfinite(log_q).all()
            or not torch.isfinite(log_p).all()
            or not torch.allclose(log_q.logsumexp(0), log_q.new_zeros(()), atol=1e-5, rtol=0)):
        raise ValueError("reference must contain normalized finite legal log probabilities")
    return (log_p.exp() * (log_p - log_q)).sum()


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


def validate_policy_clip(args):
    """Require the default C2 clip, or the explicitly declared 0.05 experiment."""
    declaration = os.environ.get("PACMAN_EXPERIMENTAL_PPO_CLIP")
    if declaration not in (None, "0.05"):
        raise ValueError("unsupported Pacman experimental clip declaration")
    expected = 0.2 if declaration is None else 0.05
    if (args.rollout_temperature != 0.7 or args.eps_clip != expected
            or getattr(args, "eps_clip_high", expected) != expected):
        raise ValueError("Pacman temperature/clip contract changed")


def tis_weights(old_log_probs, behavior_log_probs, upper_bound=2.0):
    """Detached per-decision IS weights, upper truncated without sample rejection."""
    if (old_log_probs.shape != behavior_log_probs.shape
            or not math.isfinite(upper_bound) or upper_bound < 1
            or not torch.isfinite(old_log_probs).all()
            or not torch.isfinite(behavior_log_probs).all()):
        raise ValueError("invalid TIS probabilities or bound")
    log_ratio = old_log_probs.detach().float() - behavior_log_probs.detach().float()
    if not torch.isfinite(log_ratio).all():
        raise ValueError("non-finite TIS log ratio")
    return torch.exp(log_ratio.clamp(max=math.log(upper_bound)))


def validate_single_update(args):
    """Same-forward old probabilities are valid only before one optimizer step."""
    if (getattr(args, "num_steps_per_rollout", None) != 1
            or getattr(args, "global_batch_size", None) != (
                getattr(args, "rollout_batch_size", 0) * getattr(args, "n_samples_per_prompt", 0))
            or getattr(args, "attention_dropout", 0) != 0
            or getattr(args, "hidden_dropout", 0) != 0
            or getattr(args, "use_critic", False)
            or getattr(args, "use_opd", False)):
        raise ValueError("Pacman decoupled TIS requires one optimizer step per rollout and no dropout/critic/OPD")


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
    validate_policy_clip(args)
    validate_single_update(args)
    if any(
        getattr(args, name, False)
        for name in (
            "normalize_advantages",
            "use_tis",
            "use_score_centering",
        )
    ):
        raise ValueError("unsupported additional policy correction or normalization")
    entropy_declaration = os.environ.get("PACMAN_EXPERIMENTAL_ENTROPY_COEF")
    if entropy_declaration not in (None, "0.01"):
        raise ValueError("unsupported Pacman experimental entropy declaration")
    expected_entropy = 0.0 if entropy_declaration is None else 0.01
    if getattr(args, "entropy_coef", 0) != expected_entropy:
        raise ValueError("Pacman entropy coefficient requires matching declaration")
    kl_declaration = os.environ.get("PACMAN_EXPERIMENTAL_KL_COEF")
    if kl_declaration not in (None, "0.01"):
        raise ValueError("unsupported Pacman experimental KL declaration")
    expected_kl = 0.0 if kl_declaration is None else 0.01
    if (getattr(args, "kl_coef", 0) != 0
            or getattr(args, "kl_loss_coef", 0) != expected_kl
            or bool(getattr(args, "use_kl_loss", False)) != bool(expected_kl)
            or (expected_kl and expected_entropy)
            or (expected_kl and getattr(args, "ref_update_interval", None) is not None)):
        raise ValueError("Pacman requires one declared regularizer and frozen reference")
    metadata = batch.get("metadata")
    if metadata is None or len(metadata) != len(batch["response_lengths"]):
        raise ValueError(
            "missing support metadata: apply the pinned slime transport patch"
        )
    references = batch.get("ref_allowed_log_probs")
    if expected_kl and (references is None or len(references) != len(metadata)):
        raise ValueError("missing masked reference distributions")
    new, entropies, kls = [], [], []
    responses = get_responses(
        logits.float(),
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=False,
    )
    for index, ((row, tokens), meta) in enumerate(zip(responses, metadata, strict=True)):
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
        if expected_kl:
            kls.append(value * 0 if meta.get("empty_episode") else legal_distribution_kl(
                row[0], meta["allowed_token_ids"], references[index], args.rollout_temperature
            ))
    new = torch.stack(new)
    behavior = torch.cat(batch["rollout_log_probs"]).to(new).detach()
    # All gradient-accumulation microbatches precede the only optimizer step.
    # Thus this forward uses the old actor weights; reuse its detached output.
    # Never use upstream full-vocabulary batch['log_probs'] as the masked old.
    old = new.detach()
    weights = tis_weights(old, behavior)
    advantages = torch.cat(batch["advantages"]).to(new)
    policy_loss = sum_of_sample_mean(weights * clipped_policy_terms(new, old, advantages, args.eps_clip))
    entropy = sum_of_sample_mean(torch.stack(entropies))
    kl = sum_of_sample_mean(torch.stack(kls)) if expected_kl else policy_loss * 0
    loss = policy_loss - expected_entropy * entropy + expected_kl * kl
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "entropy_bonus": (expected_entropy * entropy).detach(),
        "masked_reference_kl": kl.detach(),
        "masked_entropy": entropy.detach(),
        "masked_logprob_abs_diff": sum_of_sample_mean((old - behavior).abs()).detach(),
        "tis_weight_mean": sum_of_sample_mean(weights).detach(),
        "tis_truncate_fraction": sum_of_sample_mean(((old - behavior) > math.log(2.0)).float()).detach(),
        "ppo_ratio_mean": sum_of_sample_mean(torch.exp(new - old)).detach(),
    }
