"""Compare one pair of saved action distributions; never certifies a whole run."""

import argparse
import hashlib
import json
from pathlib import Path

from .acceptance import compare_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = {
        name: getattr(args, name).read_bytes()
        for name in ("reference", "candidate", "tolerances")
    }
    limits = json.loads(raw["tolerances"])
    if not limits.get("calibration_evidence"):
        raise ValueError("tolerances must name prior calibration evidence")
    result = compare_probabilities(
        json.loads(raw["reference"]),
        json.loads(raw["candidate"]),
        max_abs_logp=limits["max_abs_logp"],
        max_kl=limits["max_kl"],
    )
    report = {
        "schema": "pacman-probability-comparison-v1",
        "scope": "one_pair_only_not_full_gpu_acceptance",
        "input_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in raw.items()
        },
        "calibration_evidence": limits["calibration_evidence"],
        **result,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
