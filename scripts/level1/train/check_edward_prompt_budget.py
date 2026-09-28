"""Fail closed when the worst Edward request exceeds the Qwen context budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pacman_env.planner import PlannerCandidate

from pacman_recipe.level1.prompts import (
    build_image_messages,
    edward_system_prompt,
    encode_png,
)
from pacman_recipe.level1.token_constraints import (
    OPTION_IDS,
    ObjectiveTokenConstraint,
)
from pacman_recipe.level1.image_transport import pil_and_chat_messages
from pacman_recipe.level1.prompts import render_edward_decision_prompt


OBSERVATION_SHAPE = (400, 336, 3)


def worst_case_messages(
    tokenizer: object, *, fallback_mode: str = "refuse", risk_fallback: bool = False
) -> tuple[list[dict], str]:
    candidates = tuple(
        PlannerCandidate(
            option_id=option_id,
            strategy={"C": "COLLECT", "A": "AVOID", "E": "ELIMINATE"}[
                option_id[0]
            ],
            target=(24, 20),
            first_action=("U", "D", "L", "R")[index % 4],
            route_distance=999,
            commit_moves={"C": 8, "A": 3, "E": 6}[option_id[0]],
            safety_margin=1_000_000,
            future_safe_exits=4,
            entity_id=index % 4 if option_id.startswith("E") else None,
        )
        for index, option_id in enumerate(OPTION_IDS)
    )
    if risk_fallback:
        if fallback_mode != "risk_ranked":
            raise ValueError("risk fallback budget requires risk_ranked mode")
        candidates = tuple(
            PlannerCandidate(
                option_id=f"A{index}", strategy="RISK_FALLBACK",
                target=(24, 20), first_action=action, route_distance=1, commit_moves=1,
                risk={
                    "rank": index + 1, "motion": "collision_predicted",
                    "ghost_clearance": 999, "route_margin": -999,
                    "safe_next_cells": 4, "dead_end": True, "reverse": True,
                },
            )
            for index, action in enumerate(("U", "D", "L", "R"))
        )
    state_context = {
        "pacman_position": [24, 20],
        "facing": "R",
        "pellets_remaining": 196,
        "maze_size": [25, 21],
        "ghosts": [
            {"id": index, "state": "vulnerable", "position": [24, 20]}
            for index in range(4)
        ],
        "edible_ticks": 360,
        "last_action": "R",
        "episode_life_mode": "original_three_lives",
    }
    if fallback_mode != "refuse":
        state_context["edward_fallback_mode"] = fallback_mode
    constraint = ObjectiveTokenConstraint.build(
        tokenizer, (candidate.option_id for candidate in candidates)
    )
    user_prompt = render_edward_decision_prompt(
        state_context, candidates, constraint, fallback_mode=fallback_mode
    )
    image = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    messages = build_image_messages(
        encode_png(image),
        prompt_style="live_state_v3",
        state_context=state_context,
    )
    messages[0]["content"] = edward_system_prompt(fallback_mode)
    messages[1]["content"][0]["text"] = user_prompt
    return messages, user_prompt


def measure(
    processor: object, *, fallback_mode: str = "refuse", risk_fallback: bool = False
) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    messages, user_prompt = worst_case_messages(
        tokenizer, fallback_mode=fallback_mode, risk_fallback=risk_fallback
    )
    image, chat_messages = pil_and_chat_messages(
        messages
    )
    text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    processed = processor(
        text=[text], images=[image], padding=False, truncation=False, return_tensors="pt"
    )
    return {
        "input_tokens": int(processed["input_ids"].shape[-1]),
        "system_characters": len(messages[0]["content"]),
        "user_characters": len(user_prompt),
        "candidate_count": 4 if risk_fallback else len(OPTION_IDS),
        "ghost_count": 4,
        "image_height": OBSERVATION_SHAPE[0],
        "image_width": OBSERVATION_SHAPE[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    mode_args = parser.add_mutually_exclusive_group()
    mode_args.add_argument("--config", type=Path)
    mode_args.add_argument("--fallback-mode", choices=("refuse", "risk_ranked"), default="refuse")
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    fallback_mode = args.fallback_mode
    if args.config is not None:
        import yaml
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        fallback_mode = raw.get("edward_fallback_mode", "refuse")
    results = {"normal": measure(processor, fallback_mode=fallback_mode)}
    if fallback_mode == "risk_ranked":
        results["risk_fallback"] = measure(
            processor, fallback_mode=fallback_mode, risk_fallback=True
        )
    print(json.dumps(results if fallback_mode == "risk_ranked" else results["normal"], sort_keys=True))
    for scenario, result in results.items():
        if result["input_tokens"] >= args.max_input_tokens:
            raise SystemExit(
                f"Edward {scenario} worst-case prompt exceeds the model budget: "
                f"{result['input_tokens']} >= {args.max_input_tokens}"
            )


if __name__ == "__main__":
    main()
