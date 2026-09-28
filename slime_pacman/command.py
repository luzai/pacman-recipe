"""Render the GPU smoke command for review; this module never launches jobs."""

import argparse
import json
from pathlib import Path
import shlex

from .config import load_config
from .launch import PROCESSOR_ENV, build_environment


def build_command(*, slime_root, model, dataset, run_dir, config, updates, resume=None):
    if updates not in (1, 2):
        raise ValueError(
            "only one/two-update acceptance commands are available before GPU validation"
        )
    if config.max_steps != 512 or config.max_input_tokens != 2048:
        raise ValueError("GPU acceptance uses the production horizon and input budget")
    source = (Path(slime_root) / "scripts/models/qwen3.5-9B.sh").read_text()
    tokens = shlex.split(
        source.split("MODEL_ARGS=(", 1)[1].rsplit(")", 1)[0], comments=True
    )
    if tokens[:3] != ["--spec", "slime_plugins.models.qwen3_5", "get_qwen3_5_spec"]:
        raise ValueError("upstream model arguments changed")
    tokens[1:3] = ["slime_plugins.models.qwen3_5_vl", "get_qwen3_5_vl_model_provider"]
    run_dir, dataset = Path(run_dir), Path(dataset)
    command = ["python", str(Path(slime_root) / "train.py"), *tokens]
    options = {
        "train-backend": "megatron",
        "hf-checkpoint": model,
        "load": resume or model,
        "save": run_dir / "checkpoints",
        "save-hf": run_dir / "hf-{}",
        "save-interval": 1,
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": 8,
        "rollout-num-gpus-per-engine": 1,
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "micro-batch-size": 1,
        "global-batch-size": 48,
        "rollout-batch-size": 4,
        "n-samples-per-prompt": 12,
        "num-rollout": updates,
        "num-steps-per-rollout": 1,
        "prompt-data": dataset / "train.jsonl",
        "input-key": "prompt",
        "metadata-key": "metadata",
        "rollout-max-response-len": 1,
        "rollout-max-prompt-len": 2048,
        "rollout-temperature": 0.7,
        "rollout-top-p": 1,
        "rollout-top-k": -1,
        "custom-generate-function-path": "slime_pacman.rollout.generate_episode",
        "custom-reward-post-process-path": "slime_pacman.grouping.post_process_rewards",
        "custom-convert-samples-to-train-data-path": "slime_pacman.batching.convert_samples",
        "custom-rollout-log-function-path": "slime_pacman.rollout.log_rollout",
        "custom-eval-rollout-log-function-path": "slime_pacman.rollout.log_eval",
        "loss-type": "custom_loss",
        "custom-loss-function-path": "slime_pacman.probability.custom_loss",
        "advantage-estimator": "grpo",
        "eps-clip": 0.2,
        "eps-clip-high": 0.2,
        "kl-coef": 0,
        "kl-loss-coef": 0,
        "entropy-coef": 0,
        "optimizer": "adam",
        "lr": config.learning_rate,
        "lr-decay-style": "constant",
        "weight-decay": 0,
        "clip-grad": 1,
        "adam-beta1": 0.9,
        "adam-beta2": 0.999,
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "attention-dropout": 0,
        "hidden-dropout": 0,
        "attention-backend": "flash",
        "eval-interval": 1,
        "n-samples-per-eval-prompt": 12,
        "eval-temperature": 0.7,
        "sglang-mem-fraction-static": 0.7,
        "sglang-max-running-requests": 8,
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    command.extend(["--eval-prompt-data", "pacman", str(dataset / "validation.jsonl")])
    command.extend(
        [
            "--colocate",
            "--bf16",
            "--use-distributed-optimizer",
            "--use-rollout-logprobs",
            "--rollout-shuffle",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--sglang-enable-custom-logit-processor",
        ]
    )
    return command


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-root", type=Path, default=root.parent / "slime")
    parser.add_argument("--config", type=Path, default=root / "configs/slime/c2.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--updates", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume")
    args = parser.parse_args()
    args.slime_root = args.slime_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.run_dir = str(Path(args.run_dir).expanduser().resolve())
    command = build_command(
        slime_root=args.slime_root,
        model=args.model,
        dataset=args.dataset,
        run_dir=args.run_dir,
        config=load_config(args.config),
        updates=args.updates,
        resume=args.resume,
    )
    environment = build_environment(
        slime_root=args.slime_root, config=args.config, run_dir=args.run_dir
    )
    command = [
        command[0], "-m", "slime_pacman.launch",
        "--slime-root", str(args.slime_root), "--config", str(args.config),
        "--run-dir", args.run_dir, "--", *command[2:],
    ]
    print(
        json.dumps(
            {
                "status": "command_only_gpu_gates_pending",
                "argv": command,
                "environment": environment,
                "environment_required": [],
                "bash": shlex.join(
                    ["env", *(f"{k}={v}" for k, v in environment.items()), *command]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
