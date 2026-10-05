"""Opt-in frozen reference inference over Pacman's exact legal action support."""

from collections import deque
import logging
import os

import torch

from .probability import masked_log_prob, validate_support


def reference_outputs(logits, *, metadata, args, unconcat_tokens, total_lengths,
                      response_lengths, with_entropy=False, non_loss_data=True, **kwargs):
    from slime.backends.megatron_utils.loss import get_responses
    if not non_loss_data or kwargs:
        raise ValueError("unsupported reference callback mode")
    responses = get_responses(logits.float(), args=args, unconcat_tokens=unconcat_tokens,
                              total_lengths=total_lengths, response_lengths=response_lengths,
                              apply_temperature=False)
    selected, allowed = [], []
    with torch.no_grad():
        for (row, tokens), meta in zip(responses, metadata, strict=True):
            if len(tokens) != 1:
                raise ValueError("reference requires single action token")
            if meta.get("empty_episode"):
                selected.append(row.new_zeros(1))
                allowed.append(row.new_zeros(1))
                continue
            support = meta["allowed_token_ids"]
            validate_support(support, row.shape[-1])
            value, _ = masked_log_prob(row[0], support, int(tokens[0]), args.rollout_temperature)
            selected.append(value.reshape(1).detach())
            allowed.append(torch.log_softmax(row[0, support].float() / args.rollout_temperature, 0).detach())
    return {"log_probs": selected, "allowed_log_probs": allowed}


def install(args):
    """Use as --custom-megatron-init-path slime_pacman.reference_policy.install."""
    if (os.environ.get("PACMAN_EXPERIMENTAL_KL_COEF") != "0.01"
            or not args.use_kl_loss or args.kl_loss_coef != 0.01 or args.kl_coef != 0
            or getattr(args, "ref_update_interval", None) is not None
            or not getattr(args, "ref_load", None)
            or args.tensor_model_parallel_size != 1
            or args.pipeline_model_parallel_size != 1
            or args.context_parallel_size != 1
            or not getattr(args, "compute_advantages_and_returns", False)
            or args.micro_batch_size != 1 or args.use_dynamic_batch_size):
        raise ValueError("masked reference requires declared frozen reference and static TP=PP=CP=1")
    from slime.backends.megatron_utils import model
    if getattr(model.get_batch, "_pacman_reference_installed", False):
        return
    original_get_batch = model.get_batch

    def training_batch(iterator, keys, *positional, **named):
        if "ref_log_probs" in keys:
            keys = list(dict.fromkeys([*keys, "ref_allowed_log_probs"]))
        return original_get_batch(iterator, keys, *positional, **named)

    training_batch._pacman_reference_installed = True
    model.get_batch = training_batch


class ReferencePolicyMixin:
    """Serialize this override with the actor, before Ray captures its methods."""

    def compute_log_prob(self, data_iterator, num_microbatches, store_prefix=""):
        if store_prefix != "ref_":
            return super().compute_log_prob(data_iterator, num_microbatches, store_prefix)
        from slime.backends.megatron_utils import model
        training_batch = model.get_batch
        if not getattr(training_batch, "_pacman_reference_installed", False):
            raise RuntimeError("masked reference transport was not installed")
        pending = deque()
        def reference_batch(iterator, keys, *positional, **named):
            batch = training_batch(iterator, list(dict.fromkeys([*keys, "metadata"])), *positional, **named)
            pending.append(batch["metadata"])
            return batch
        def callback(logits, **named):
            outputs = reference_outputs(logits, metadata=pending.popleft(), **named)
            # Megatron's forward-only schedule still expects the loss tensor
            # and collected data tuple, matching get_log_probs_and_entropy.
            return torch.empty((0,), device=logits.device), outputs
        previous = model.get_batch
        model.get_batch = reference_batch
        try:
            result = model.forward_only(callback, self.args, self.model, data_iterator,
                                        num_microbatches, store_prefix=store_prefix,
                                        use_rollout_top_p_replay=False)
            if pending:
                raise RuntimeError("reference metadata was not fully consumed")
            vectors = result.get("ref_allowed_log_probs", [])
            selected = result.get("ref_log_probs", [])
            if len(vectors) != sum(num_microbatches) or len(selected) != len(vectors):
                raise RuntimeError("masked reference output count does not match static microbatches")
            for vector, value in zip(vectors, selected, strict=True):
                if (vector.ndim != 1 or not vector.numel() or value.numel() != 1
                        or vector.requires_grad or value.requires_grad
                        or not torch.isfinite(vector).all() or not torch.isfinite(value).all()
                        or not torch.allclose(vector.logsumexp(0), vector.new_zeros(()), atol=1e-5, rtol=0)):
                    raise RuntimeError("invalid masked reference probability output")
            if getattr(self.args, "rank", 0) == 0:
                lengths = [vector.numel() for vector in vectors]
                histogram = {size: lengths.count(size) for size in sorted(set(lengths))}
                logging.getLogger(__name__).info(
                    "Pacman masked reference verified: decisions=%s support_lengths=%s finite=true detached=true ref_load=%s",
                    len(vectors), histogram, self.args.ref_load,
                )
            return result
        finally:
            model.get_batch = previous


def make_reference_actor(base_class):
    class PacmanReferenceActor(ReferencePolicyMixin, base_class):
        pass
    return PacmanReferenceActor


def install_driver():
    """Inject the explicit actor class through slime's supported factory argument."""
    from slime.ray import placement_group
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    original = placement_group.create_training_models
    if getattr(original, "_pacman_reference_installed", False):
        return
    actor_class = make_reference_actor(MegatronTrainRayActor)

    def create_training_models(args, pgs, rollout_manager, actor_cls=None):
        if actor_cls is not None:
            raise ValueError("reference actor cannot replace another custom actor")
        if not args.use_kl_loss:
            raise ValueError("reference actor requires KL experiment")
        return original(args, pgs, rollout_manager, actor_cls=actor_class)

    create_training_models._pacman_reference_installed = True
    placement_group.create_training_models = create_training_models
