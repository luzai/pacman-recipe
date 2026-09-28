from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from pacman_recipe.actions import ActionParseError, parse_action
from pacman_recipe.level1.level1_dataset import validate_episode_row
from pacman_recipe.level1.prompts import (
    build_image_messages,
    crop_pacman_local_view,
    encode_png,
)
from pacman_recipe.level1.workflow import PacmanNativeVisionWorkflow
from pacman_env.env import Action, PygamePacmanEnv, PygamePacmanEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy wall-hit evaluation on held-out single-step states."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        default="wall_avoidance_v1",
        choices=(
            "wall_avoidance_v1",
            "wall_avoidance_local_v2",
            "wall_avoidance_axis_v3",
        ),
    )
    parser.add_argument(
        "--constrain-actions",
        action="store_true",
        help=(
            "Renormalize generation over U/D/L/R. Leave disabled when "
            "evaluating the same full-vocabulary policy used by PPO."
        ),
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_episode_row(row)
        if row.get("decision_steps") != 1:
            raise ValueError("evaluation requires single-step rows")
    return rows


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    torch.manual_seed(args.seed)
    rows = load_rows(args.dataset)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    allowed_token_ids: list[int] = []
    for token in ("U", "D", "L", "R"):
        token_ids = processor.tokenizer.encode(
            token, add_special_tokens=False
        )
        if len(token_ids) != 1:
            raise ValueError(f"action {token!r} is not one tokenizer token")
        allowed_token_ids.append(int(token_ids[0]))

    records: list[dict[str, Any]] = []
    state_policy_records: list[dict[str, Any]] = []
    for row in rows:
        requested = row["env"]
        config = PygamePacmanEnvConfig(
            level=int(requested["level"]),
            ghost_mode=requested["ghost_mode"],
            max_steps=int(requested["max_steps"]),
            video_driver="dummy",
            audio_driver="dummy",
        )
        with PygamePacmanEnv(config) as env:
            image, info = env.reset(seed=int(requested["seed"]))
            for token in row["state_prefix_actions"]:
                image, _, terminated, truncated, info = env.step(Action(token))
                if terminated or truncated or info["wall_collision"]:
                    raise RuntimeError(f"invalid state prefix for {row['id']}")
            model_image = (
                crop_pacman_local_view(image)
                if args.prompt_style
                in ("wall_avoidance_local_v2", "wall_avoidance_axis_v3")
                else image
            )
            messages = build_image_messages(
                encode_png(model_image),
                prompt_style=args.prompt_style,
            )
            image_obj, chat_messages = (
                PacmanNativeVisionWorkflow._pil_and_chat_messages(messages)
            )
            text = processor.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(
                text=[text],
                images=[image_obj],
                padding=False,
                return_tensors="pt",
            ).to(args.device)
            open_actions = set(env.snapshot().get("open") or [])
            with torch.inference_mode():
                next_token_logits = model(**inputs).logits[0, -1]
                action_id_tensor = torch.tensor(
                    allowed_token_ids,
                    device=next_token_logits.device,
                    dtype=torch.long,
                )
                policy_temperature = (
                    args.temperature if args.samples > 1 else 1.0
                )
                if args.constrain_actions:
                    action_probabilities = torch.softmax(
                        next_token_logits[action_id_tensor].float()
                        / policy_temperature,
                        dim=-1,
                    ).cpu()
                else:
                    action_probabilities = torch.softmax(
                        next_token_logits.float() / policy_temperature,
                        dim=-1,
                    )[action_id_tensor].cpu()
                action_probability_map = {
                    token: float(probability)
                    for token, probability in zip(
                        ("U", "D", "L", "R"),
                        action_probabilities,
                        strict=True,
                    )
                }
                valid_move_probability = sum(
                    probability
                    for token, probability in action_probability_map.items()
                    if token in open_actions
                )
                expected_wall_probability = 1.0 - valid_move_probability
                state_policy_records.append(
                    {
                        "id": row["id"],
                        "open_actions": sorted(open_actions),
                        "action_probabilities": action_probability_map,
                        "expected_wall_probability": (
                            expected_wall_probability
                        ),
                    }
                )
                outputs = model.generate(
                    **inputs,
                    do_sample=args.samples > 1,
                    temperature=(
                        args.temperature if args.samples > 1 else None
                    ),
                    num_return_sequences=args.samples,
                    min_new_tokens=1,
                    max_new_tokens=1,
                    prefix_allowed_tokens_fn=(
                        lambda _batch_id, _input_ids: allowed_token_ids
                    )
                    if args.constrain_actions
                    else None,
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            for sample_index, output in enumerate(outputs):
                completion = processor.tokenizer.decode(
                    output[prompt_length:].tolist(),
                    skip_special_tokens=True,
                )
                parse_failed = False
                wall_collision = False
                action_token = None
                try:
                    action = parse_action(completion)
                    action_token = action.value
                    if action == Action.STAY:
                        wall_collision = True
                    else:
                        wall_collision = action.value not in open_actions
                except ActionParseError:
                    parse_failed = True
                    wall_collision = True
                records.append(
                    {
                        "id": row["id"],
                        "sample_index": sample_index,
                        "prefix_length": len(row["state_prefix_actions"]),
                        "completion": completion,
                        "action": action_token,
                        "parse_failed": parse_failed,
                        "wall_collision": wall_collision,
                    }
                )

    count = len(records)
    wall_hits = sum(record["wall_collision"] for record in records)
    parse_failures = sum(record["parse_failed"] for record in records)
    expected_wall_rate = sum(
        record["expected_wall_probability"]
        for record in state_policy_records
    ) / len(state_policy_records)
    result = {
        "model": args.model,
        "dataset": str(args.dataset),
        "samples": count,
        "states": len(rows),
        "samples_per_state": args.samples,
        "temperature": args.temperature if args.samples > 1 else 0.0,
        "seed": args.seed,
        "prompt_style": args.prompt_style,
        "constrain_actions": args.constrain_actions,
        "wall_hits": wall_hits,
        "wall_hit_rate": wall_hits / count,
        "valid_move_rate": (count - wall_hits) / count,
        "expected_wall_rate": expected_wall_rate,
        "expected_valid_move_rate": 1.0 - expected_wall_rate,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / count,
        "state_policies": state_policy_records,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
