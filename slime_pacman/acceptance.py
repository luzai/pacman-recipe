"""Numerical evidence helpers; passing one comparison is not GPU acceptance."""

import math

import torch


IDENTITY_FIELDS = (
    "fixture_sha256",
    "weights_sha256",
    "processor_sha256",
    "input_ids",
    "image_sha256",
    "image_grid_thw",
    "allowed_token_ids",
    "temperature",
)


def compare_probabilities(reference, candidate, *, max_abs_logp, max_kl):
    """Compare complete conditional action distributions with explicit tolerances.

    Records must identify the actual loaded weights, not just a server version.
    Backend and request/cache metadata should be retained by the caller.
    """
    for limit in (max_abs_logp, max_kl):
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("tolerances must be finite and nonnegative")
    for key in IDENTITY_FIELDS:
        if (
            key not in reference
            or key not in candidate
            or reference[key] != candidate[key]
        ):
            raise ValueError(f"comparison identity differs or is missing: {key}")
    for key in ("fixture_sha256", "weights_sha256", "processor_sha256", "image_sha256"):
        value = reference[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"invalid SHA256: {key}")
    support = reference["allowed_token_ids"]
    ids = reference["input_ids"]
    grid = reference["image_grid_thw"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or i < 0 for i in ids)
    ):
        raise ValueError("invalid input IDs")
    if (
        not isinstance(grid, list)
        or len(grid) != 1
        or not isinstance(grid[0], list)
        or len(grid[0]) != 3
        or any(type(i) is not int or i <= 0 for i in grid[0])
    ):
        raise ValueError("expected one valid image grid")
    if (
        not support
        or any(type(i) is not int or i < 0 for i in support)
        or len(set(support)) != len(support)
    ):
        raise ValueError("invalid action support")
    if not math.isfinite(reference["temperature"]) or reference["temperature"] <= 0:
        raise ValueError("invalid temperature")
    values = []
    for record in (reference, candidate):
        if record.get("outside_support_probability") != 0:
            raise ValueError("outside-support probability must be measured as zero")
        logp = torch.tensor(record["log_probs"], dtype=torch.float64)
        if logp.shape != (len(support),) or not torch.isfinite(logp).all():
            raise ValueError("expected every allowed action's finite log probability")
        if abs(torch.logsumexp(logp, 0).item()) > 2e-4:
            raise ValueError("probabilities are not normalized on action support")
        values.append(logp)
    old, new = values
    delta = new - old
    absolute = delta.abs()
    old_normalized = old - torch.logsumexp(old, 0)
    new_normalized = new - torch.logsumexp(new, 0)
    p, q = old_normalized.exp(), new_normalized.exp()
    # Normalize tiny serialization error before KL; do not alter logp differences.
    kl = (p * (old_normalized - new_normalized)).sum().clamp_min(0).item()
    ratio = delta.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("importance ratio overflow")
    result = {
        "action_count": len(support),
        "abs_delta_logp": {
            name: torch.quantile(absolute, quantile).item()
            for name, quantile in (
                ("p50", 0.5),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1.0),
            )
        },
        "kl_reference_candidate": kl,
        "max_probability_difference": (p - q).abs().max().item(),
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
        "ratio_outside_clip_fraction": ((ratio < 0.8) | (ratio > 1.2))
        .double()
        .mean()
        .item(),
        "tolerances": {"max_abs_logp": max_abs_logp, "max_kl": max_kl},
    }
    result["passed"] = absolute.max().item() <= max_abs_logp and kl <= max_kl
    return result


def probe_sequence_isolation(module, a, b, *, atol, rtol):
    """Exercise the real module's cu_seqlens boundary and backward path.

    A-only loss isolates contamination of A by B. Returned gradients include
    all parameters and both inputs. No optimizer step or .grad mutation occurs.
    Caller supplies an eval-mode GDN module and same-device [1, length, hidden]
    inputs; CPU toy modules are useful only to test this probe itself.
    """
    if module.training:
        raise ValueError("use eval mode to exclude dropout")
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("invalid tolerances")
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape[0] != 1
        or b.shape[0] != 1
        or a.shape[2] != b.shape[2]
        or min(a.shape[1], b.shape[1]) < 1
        or a is b
        or a.device != b.device
        or a.dtype != b.dtype
    ):
        raise ValueError("expected nonempty [1, length, hidden] inputs")
    named_parameters = tuple(
        (name, p) for name, p in module.named_parameters() if p.requires_grad
    )
    parameters = tuple(p for _, p in named_parameters)

    def run(order):
        inputs = [x.detach().clone().requires_grad_(True) for x in order]
        lengths = [x.shape[1] for x in inputs]
        boundaries = (
            torch.tensor([0, *lengths], device=a.device, dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
        )
        output = module(torch.cat(inputs, dim=1), cu_seqlens=boundaries)
        index = next(i for i, x in enumerate(order) if x is a)
        start = sum(lengths[:index])
        selected = output[:, start : start + a.shape[1]]
        # Nonuniform fixed coefficients make token swaps visible to backward.
        coefficients = torch.linspace(
            0.1, 1.0, selected.numel(), device=a.device
        ).reshape(selected.shape)
        gradients = torch.autograd.grad(
            (selected.float() * coefficients).sum(),
            (*inputs, *parameters),
            allow_unused=True,
        )
        tensors = {"output": selected.detach(), "input_a": gradients[index]}
        for (name, _), gradient in zip(
            named_parameters, gradients[len(inputs) :], strict=True
        ):
            tensors[f"parameter:{name}"] = gradient
        leakage = [
            g
            for i, g in enumerate(gradients[: len(inputs)])
            if i != index and g is not None
        ]
        return tensors, leakage

    baseline, _ = run([a])
    cases = {}
    for name, order in (("repeat_a", [a]), ("a_b", [a, b]), ("b_a", [b, a])):
        actual, leakage = run(order)
        checks = {}
        for key, expected in baseline.items():
            observed = actual[key]
            if expected is None or observed is None:
                checks[key] = {
                    "passed": expected is None and observed is None,
                    "max_abs": None,
                }
            else:
                finite = bool(
                    torch.isfinite(expected).all() and torch.isfinite(observed).all()
                )
                checks[key] = {
                    "passed": finite
                    and torch.allclose(expected, observed, atol=atol, rtol=rtol),
                    "max_abs": float((expected.float() - observed.float()).abs().max())
                    if finite
                    else None,
                }
        no_leak = all(
            bool(torch.isfinite(g).all()) and bool((g == 0).all()) for g in leakage
        )
        cases[name] = {
            "passed": all(x["passed"] for x in checks.values()) and no_leak,
            "zero_cross_sample_gradient": no_leak,
            "comparisons": checks,
        }
    return {
        "passed": all(x["passed"] for x in cases.values()),
        "cases": cases,
        "atol": atol,
        "rtol": rtol,
    }
