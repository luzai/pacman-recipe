"""Clip-Cov (Cui et al. 2025, arXiv:2505.22617) over one whole Pacman update.

verl computes the covariance inside each loss microbatch. Pacman trains with
microbatch=1 decision, where that covariance is identically zero, so the
selection is made once per update at batch conversion instead: over every real
decision, from its behavior log-prob and its episode advantage. Under the
single-optimizer-step contract the actor's current log-prob equals old, so the
only difference from verl's input is the logged rollout/train mismatch.

Selected decisions keep their forward value but contribute no policy term,
exactly as verl multiplies their pg loss by zero. PPO-clipped decisions are not
excluded because the ratio is identically one here.
"""

import hashlib
import math
import os

import torch

ENV = "PACMAN_EXPERIMENTAL_CLIP_COV"
KEY = "clip_cov_detach"


def declaration():
    """Return (ratio, low, high) from 'ratio,low,high', or None when off."""
    raw = os.environ.get(ENV)
    if raw is None:
        return None
    try:
        ratio, low, high = (float(part) for part in raw.split(","))
    except ValueError:
        raise ValueError(f"{ENV} must be 'ratio,low,high'") from None
    if (not all(map(math.isfinite, (ratio, low, high)))
            or not 0 < ratio <= 0.1 or not 0 <= low < high):
        raise ValueError(f"unsupported {ENV} declaration")
    return ratio, low, high


def select(advantages, log_probs, *, ratio, low, high, seed):
    """Detach mask and covariance; at least one eligible decision when any exist."""
    a = torch.as_tensor(advantages, dtype=torch.float64)
    lp = torch.as_tensor(log_probs, dtype=torch.float64)
    if a.ndim != 1 or a.shape != lp.shape or not a.numel():
        raise ValueError("expected matching non-empty advantage and log-prob vectors")
    if not torch.isfinite(a).all() or not torch.isfinite(lp).all():
        raise ValueError("non-finite Clip-Cov input")
    cov = (a - a.mean()) * (lp - lp.mean())
    eligible = torch.nonzero((cov > low) & (cov < high)).flatten()
    count = min(max(int(ratio * a.numel()), 1), eligible.numel())
    generator = torch.Generator().manual_seed(seed)
    chosen = eligible[torch.randperm(eligible.numel(), generator=generator)[:count]]
    mask = torch.zeros(a.numel(), dtype=torch.bool)
    mask[chosen] = True
    return mask, cov


def annotate(args, data):
    """Mark this update's selected decisions in metadata; returns summary or None."""
    spec = declaration()
    if spec is None:
        return None
    if getattr(args, "advantage_estimator", None) != "grpo" or getattr(args, "normalize_advantages", False):
        raise ValueError("Clip-Cov requires episode GRPO rewards to be the final advantages")
    real = [
        i for i, (_, meta) in enumerate(zip(data["loss_masks"], data["metadata"], strict=True))
        if not meta.get("empty_episode")
    ]
    if any(list(data["loss_masks"][i]) != [1] or len(data["rollout_log_probs"][i]) != 1 for i in real):
        raise ValueError("Clip-Cov expects exactly one trained token per decision")
    for meta in data["metadata"]:
        meta[KEY] = False
    if not real:
        raise ValueError("Clip-Cov update has no real decisions")
    ratio, low, high = spec
    identity = ",".join(sorted({str(data["metadata"][i]["episode_id"]) for i in real}))
    seed = int.from_bytes(hashlib.sha256(f"{args.seed}|{identity}".encode()).digest()[:8], "little")
    mask, cov = select(
        [float(data["rewards"][i]) for i in real],
        [float(data["rollout_log_probs"][i][0]) for i in real],
        ratio=ratio, low=low, high=high, seed=seed,
    )
    for i, chosen in zip(real, mask.tolist(), strict=True):
        data["metadata"][i][KEY] = chosen
    positive = cov.clamp(min=0)
    return {
        "ratio": ratio, "low": low, "high": high, "seed": seed,
        "decisions": len(real),
        "eligible": int(((cov > low) & (cov < high)).sum()),
        "selected": int(mask.sum()),
        "selected_share_of_positive_cov": float(positive[mask].sum() / positive.sum()) if positive.sum() > 0 else 0.0,
        "cov_quantiles": {
            str(q): float(cov.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999)
        },
    }


def detach_mask(metadata, reference):
    """Per-decision float keep-mask for the loss; fail closed on a missing annotation."""
    declared = declaration() is not None
    if not declared:
        if any(KEY in meta for meta in metadata):
            raise ValueError("Clip-Cov annotation present without its declaration")
        return None
    if any(KEY not in meta for meta in metadata):
        raise ValueError("Clip-Cov declared but batch conversion did not annotate")
    return torch.tensor([0.0 if meta[KEY] else 1.0 for meta in metadata], dtype=reference.dtype, device=reference.device)
