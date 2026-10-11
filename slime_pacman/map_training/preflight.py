"""Fail-closed CPU contracts; no claim of GPU runtime acceptance."""

import hashlib
import json
from pathlib import Path


def validate_training_args(args):
    for name, expected in [
        ("rollout_batch_size", 16),
        ("n_samples_per_prompt", 12),
        ("global_batch_size", 192),
    ]:
        if getattr(args, name, expected) != expected:
            raise ValueError(f"{name} must be {expected}")
    for name in ["use_kl_loss", "use_kl", "normalize_advantages"]:
        if getattr(args, name, False):
            raise ValueError(f"{name} is disabled for initial-stage map training")
    if getattr(args, "kl_coef", 0) != 0:
        raise ValueError("reference KL must be disabled")


def verify_files(root, manifest):
    root = Path(root).resolve()
    for relative, expected in manifest.items():
        file = (root / relative).resolve()
        if not file.is_relative_to(root):
            raise ValueError("manifest path outside source root")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source mismatch: {relative}")


def verify_runtime_receipts(directory, image_id):
    root = Path(directory)
    for name in ["IMAGE_CONTENT_VERIFIED.json", "RUNTIME_CPU_VERIFIED.json"]:
        receipt = json.loads((root / name).read_text())
        if receipt.get("image_id") != image_id or receipt.get("passed") is not True:
            raise ValueError(f"unverified runtime receipt: {name}")
    return True
