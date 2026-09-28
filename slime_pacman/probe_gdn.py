"""Run pinned slime's actual GDN layer on CUDA; no optimizer or service launch."""

import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path

import torch

from .acceptance import probe_sequence_isolation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("actual GDN isolation requires CUDA and the pinned runtime")
    from transformers import AutoConfig
    from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet
    from .preflight import check_patch

    recipe_root = Path(__file__).resolve().parents[1]
    import slime

    patch_sha = check_patch(Path(slime.__file__).resolve().parents[1], recipe_root)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config = getattr(config, "text_config", config)
    config.dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    module = Qwen3_5GatedDeltaNet(config, layer_idx=0).cuda().bfloat16().eval()
    results = []
    # Cover both sides of the 64-token kernel boundary and unequal lengths.
    for lengths in ((17, 65), (64, 129), (127, 33)):
        a, b = [
            torch.randn(1, n, config.hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.1
            for n in lengths
        ]
        result = probe_sequence_isolation(module, a, b, atol=args.atol, rtol=args.rtol)
        results.append({"lengths": list(lengths), **result})
    report = {
        "schema": "pacman-gdn-layer-probe-v1",
        "scope": "random_initialized_real_gdn_layer_not_checkpoint_or_full_model_acceptance",
        "passed": all(r["passed"] for r in results),
        "seed": args.seed,
        "dtype": "bfloat16",
        "backend": "fla",
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "slime_patch_sha256": patch_sha,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "cases": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
