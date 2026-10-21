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
    "PACMAN_FORMAL_CONFIG", "PACMAN_FORMAL_EPISODE_WORKERS",
    "SMALL_MAZE_GHOST_COUNT", "PACMAN_CUSTOM_WALL_RENDER",
    "SMALL_MAZE_REQUEST_SEED_DOMAIN", "SDL_VIDEODRIVER", "SDL_AUDIODRIVER",
    "PACMAN_EXPERIMENTAL_PPO_CLIP", "PACMAN_EXPERIMENTAL_KL_COEF",
    "PACMAN_EXPERIMENTAL_ENTROPY_COEF", "PACMAN_EXPERIMENTAL_CLIP_COV",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
    "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
    "PACMAN_CACHE_CONTRACT", "PACMAN_FIXED_PREFIX_FILE",
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
    if Path(config).expanduser().is_file():
        # Rollout workers and SGLang's image processor read the contract from here.
        from pacman_recipe.level1.vision_prompt import VISION_CONTRACT_ENV
        from .config import load_config

        cfg = load_config(config)
        env[VISION_CONTRACT_ENV] = cfg.vision_image_contract
        # Explicit legacy environment overrides remain supported and are recorded.
        env.setdefault("PACMAN_EXPERIMENTAL_CLIP_COV", (
            f"{cfg.clip_cov_ratio},{cfg.clip_cov_low},{cfg.clip_cov_high}" if cfg.clip_cov else "off"))
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tracking-child", action="store_true", help=argparse.SUPPRESS)
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
    from .config import load_config
    cfg = load_config(args.config)
    if not args.tracking_child and cfg.swanlab_mode != "disabled":
        from .tracking import run_logged
        command = [sys.executable, "-m", "slime_pacman.launch", "--tracking-child",
                   "--slime-root", str(args.slime_root), "--config", str(args.config),
                   "--run-dir", str(args.run_dir), "--", *training_args]
        raise SystemExit(run_logged(command, cfg, environment, training_args))
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
