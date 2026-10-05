"""Launch pinned slime with explicit environment inheritance for Ray workers."""

import argparse
import os
from pathlib import Path
import runpy
import sys


PROCESSOR_ENV = {
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "slime_pacman.sglang_processors"
}
INHERITED_ENV = (
    "PACMAN_EXPERIMENTAL_PPO_CLIP", "PACMAN_EXPERIMENTAL_KL_COEF",
    "PACMAN_EXPERIMENTAL_ENTROPY_COEF",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
    "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT",
)


def build_environment(*, slime_root, config, run_dir):
    """Use absolute worker paths while retaining configured dependency paths."""
    root = Path(__file__).resolve().parents[1]
    paths = [str(root), str(Path(slime_root).expanduser().resolve())]
    paths.extend(
        str(Path(path).expanduser().resolve())
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    env = {name: os.environ[name] for name in INHERITED_ENV if name in os.environ}
    for name in ("PACMAN_PYTHON_ROOT", "MAAPACMAN_PACMAN_ROOT", "MAAPACMAN_PACMAN_PYTHON_ROOT"):
        if env.get(name):
            env[name] = str(Path(env[name]).expanduser().resolve())
    env.update(PROCESSOR_ENV)
    env.update(
        PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)),
        SLIME_ROOT=str(Path(slime_root).expanduser().resolve()),
        PACMAN_SLIME_CONFIG=str(Path(config).expanduser().resolve()),
        PACMAN_RUN_DIR=str(Path(run_dir).expanduser().resolve()),
    )
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    if not training_args:
        parser.error("slime training arguments are required after --")
    environment = build_environment(
        slime_root=args.slime_root, config=args.config, run_dir=args.run_dir
    )
    entry = Path(environment["SLIME_ROOT"]) / "train.py"
    if not entry.is_file() or not Path(environment["PACMAN_SLIME_CONFIG"]).is_file():
        raise ValueError("slime train.py and Pacman config must exist")
    import ray

    if ray.is_initialized():
        raise RuntimeError("Ray is already initialized; launch in a fresh driver process")
    os.environ.update(environment)
    # PYTHONPATH affects children; update this already-running driver as well.
    sys.path[:0] = [p for p in environment["PYTHONPATH"].split(os.pathsep) if p not in sys.path]
    ray.init(runtime_env={"env_vars": environment})
    try:
        if environment.get("PACMAN_EXPERIMENTAL_KL_COEF") is not None:
            from .reference_policy import install_driver
            install_driver()
        sys.argv = [str(entry), *training_args]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
