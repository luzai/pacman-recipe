"""Host-side image identity and CPU checks for a new, idle container.

Run before starting Ray/SGLang. This does not perform GPU serving acceptance.
Patches are installed only when --install-patches is explicitly supplied.
"""

import argparse
import json
from pathlib import Path
import subprocess

from .checkpoints import atomic_json


def docker_json(*args):
    return json.loads(subprocess.check_output(["docker", *args], text=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument(
        "--lock", required=True, help="image_id, versions, runtime_files"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--slime-root", required=True, help="absolute path inside container"
    )
    parser.add_argument("--install-patches", action="store_true")
    args = parser.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    container = docker_json("inspect", args.container)[0]
    if container["Image"] != lock["image_id"]:
        raise ValueError("container image differs from locked image ID")
    image = docker_json("image", "inspect", lock["image_id"])[0]
    if image["Id"] != lock["image_id"]:
        raise ValueError("image identity mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "IMAGE_CONTENT_VERIFIED.json",
        dict(
            passed=True,
            image_id=image["Id"],
            layers=image["RootFS"]["Layers"],
            container_id=container["Id"],
        ),
    )
    atomic_json(
        output / "RUNTIME_CPU_VERIFIED.json", dict(passed=False, image_id=image["Id"])
    )
    if args.install_patches:
        for name in ["sglang_logprob_fix.py", "sglang_sampler_endpoint_fix.py"]:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    args.container,
                    "python",
                    str(Path(args.slime_root) / "tools" / name),
                ],
                check=True,
            )
    # Exact hashes prove sampler/overlap installation; versions prove CPU dependencies.
    code = """import json,sys,hashlib,importlib.metadata
from pathlib import Path
lock=json.loads(sys.argv[1])
versions={k:importlib.metadata.version(k) for k in lock['versions']}
if versions != lock['versions']: raise ValueError('dependency version mismatch')
for path, expected in lock['runtime_files'].items():
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected: raise ValueError('runtime file mismatch: '+path)
import torch,numpy,pygame,transformers
if torch.cuda.is_initialized(): raise ValueError('CPU check unexpectedly initialized CUDA')
print(json.dumps(dict(versions=versions,runtime_files=lock['runtime_files'],cuda_initialized=False)))
"""
    result = json.loads(
        subprocess.check_output(
            [
                "docker",
                "exec",
                args.container,
                "python",
                "-B",
                "-c",
                code,
                json.dumps(lock),
            ],
            text=True,
        ).splitlines()[-1]
    )
    atomic_json(
        output / "RUNTIME_CPU_VERIFIED.json",
        dict(passed=True, image_id=image["Id"], **result),
    )


if __name__ == "__main__":
    main()
