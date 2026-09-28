"""Measure C1 direct prompts with a real screenshot and the model processor.

This conservative boundary suite is not an exhaustive proof over every runtime
state. It checks input IDs after the real image processor/chat template, with
truncation explicitly disabled. It does not load model weights or test inference.
No Edward planner is constructed or called.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from io import BytesIO
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from pacman_recipe.level1.prompts import (
    build_image_messages,
    encode_png,
    png_sha256,
    prompt_contract_metadata,
    text_sha256,
)

MOVEMENT_ACTIONS = ("U", "D", "L", "R")
OBSERVATION_SHAPE = (400, 336, 3)
MAZE_SHAPE = (25, 21)
MAX_PELLETS = 196


def conservative_contexts() -> list[tuple[str, dict[str, Any]]]:
    """Cover bounded rendered fields without depending on an Edward harness.

    The workflow deduplicates per-cell exit history to at most four directions;
    recent paths and exit counts are not rendered by live_state_v3. Preferred
    directions here deliberately include the full open set, a conservative
    superset of the actual least-used/anti-reverse preference. Some boundary
    combinations need not be reachable cells in the fixed maze.
    """
    base = {
        "pacman_position": [24, 20],
        "facing": "R",
        "pellets_remaining": MAX_PELLETS,
        "maze_size": list(MAZE_SHAPE),
        "ghosts": [],
        "edible_ticks": 0,
        "open_actions": list(MOVEMENT_ACTIONS),
        "blocked_actions": [],
        "current_cell_exit_history": list(MOVEMENT_ACTIONS),
        "preferred_open_actions": list(MOVEMENT_ACTIONS),
        "last_action": "R",
    }
    cases = []

    def add(label: str, **changes: Any) -> None:
        context = deepcopy(base)
        context.update(changes)
        cases.append((label, context))

    # All nonempty open/blocked partitions, in the runtime's U,D,L,R order.
    for mask in range(1, 16):
        opened = [a for i, a in enumerate(MOVEMENT_ACTIONS) if mask & (1 << i)]
        add(
            f"open-mask-{mask:02d}",
            open_actions=opened,
            blocked_actions=[a for a in MOVEMENT_ACTIONS if a not in opened],
            preferred_open_actions=opened,
        )
    for index, history in enumerate(permutations(MOVEMENT_ACTIONS)):
        add(f"history-order-{index:02d}", current_cell_exit_history=list(history))
    add("initial-empty-history", current_cell_exit_history=[], last_action=None)
    for length in (1, 2, 3):
        add(
            f"history-length-{length}",
            current_cell_exit_history=list(MOVEMENT_ACTIONS[:length]),
        )
    for row, col in ((0, 0), (0, 20), (24, 0), (9, 9), (10, 10), (24, 20)):
        add(f"position-{row}-{col}", pacman_position=[row, col])
    for remaining in (0, 1, 9, 10, 99, 100, MAX_PELLETS):
        add(f"pellets-{remaining}", pellets_remaining=remaining)
    for action in (*MOVEMENT_ACTIONS, "S"):
        add(f"facing-{action}", facing=action)
    for action in MOVEMENT_ACTIONS:
        add(f"last-action-{action}", last_action=action)
    return cases


def native_messages(png: bytes, context: dict[str, Any]) -> tuple[Any, list[dict], str]:
    """Mirror the native inline-PNG conversion without importing AReaL runtime.

    A regression test compares this conversion with the actual workflow method.
    Keeping this CPU preflight independent avoids importing GPU/uvloop packages.
    """
    from PIL import Image

    messages = build_image_messages(
        png, prompt_style="live_state_v3", state_context=context
    )
    if (
        len(messages) != 2
        or messages[0]["role"] != "system"
        or messages[1]["role"] != "user"
        or [item["type"] for item in messages[1]["content"]] != ["text", "image_url"]
    ):
        raise ValueError(
            "direct multimodal message structure changed; update budget preflight"
        )
    image = Image.open(BytesIO(png)).convert("RGB")
    user_text = messages[1]["content"][0]["text"]
    chat_messages = [
        messages[0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
    ]
    return image, chat_messages, user_text


def capture_c1_image(pacman_python_root: Path | None = None) -> tuple[np.ndarray, dict]:
    """Capture the actual headless Level 1 renderer with ghosts disabled."""
    from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig

    env = PygamePacmanEnv(
        PygamePacmanEnvConfig(
            pacman_python_root=pacman_python_root,
            ghost_mode="disabled",
            max_steps=512,
        )
    )
    try:
        observation, info = env.reset(seed=28)
        snapshot = env.snapshot()
        if snapshot.get("ghosts") or int(snapshot.get("edible_ticks", 0)):
            raise ValueError(
                "C1 prompt budget capture must have no ghosts or edible timer"
            )
        if (int(snapshot["height"]), int(snapshot["width"])) != MAZE_SHAPE:
            raise ValueError("maze bounds changed; update conservative direct contexts")
        if int(info["pellets_remaining"]) != MAX_PELLETS:
            raise ValueError(
                "initial pellet bound changed; update conservative direct contexts"
            )
        return observation, {
            "source": "headless-pacman-python-renderer",
            "seed": 28,
            "ghost_mode": "disabled",
            "maze_shape": list(MAZE_SHAPE),
            "initial_pellets": MAX_PELLETS,
        }
    finally:
        env.close()


def measure(
    processor: Any, observation: np.ndarray, *, max_input_tokens: int = 1024
) -> dict:
    """Count untruncated multimodal IDs for every boundary request."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    if observation.shape != OBSERVATION_SHAPE or observation.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB observation {OBSERVATION_SHAPE}")
    png = encode_png(observation)
    cases = []
    for label, context in conservative_contexts():
        image, messages, user_text = native_messages(png, context)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        processed = processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
            truncation=False,
        )
        missing = {"input_ids", "pixel_values"} - set(processed)
        if missing:
            raise ValueError(
                f"processor omitted required VLM fields: {sorted(missing)}"
            )
        if (
            processed.get("mm_token_type_ids") is None
            and processed.get("token_type_ids") is None
        ):
            raise ValueError("processor omitted multimodal token type IDs")
        shape = tuple(processed["input_ids"].shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
            raise ValueError(f"unexpected single-request input_ids shape: {shape}")
        count = int(shape[1])
        cases.append(
            {
                "scenario": label,
                "input_tokens": count,
                "user_characters": len(user_text),
                "user_prompt_sha256": text_sha256(user_text),
            }
        )
    largest = max(cases, key=lambda item: item["input_tokens"])
    return {
        "check": "c1-direct-live-state-v3-prompt-budget",
        "scope": "processor-only; no model inference or gameplay completion claim",
        "coverage": "conservative boundary suite; not exhaustive over all runtime contexts",
        "prompt_contract": prompt_contract_metadata(
            "live_state_v3", edward_options=False
        ),
        "processor_class": type(processor).__name__,
        "image_shape": list(observation.shape),
        "png_sha256": png_sha256(png),
        "truncation": False,
        "max_input_tokens": max_input_tokens,
        "input_tokens": largest["input_tokens"],
        "largest_scenario": largest["scenario"],
        "headroom_tokens": max_input_tokens - largest["input_tokens"],
        "passed": largest["input_tokens"] < max_input_tokens,
        "comparison": "input_tokens < max_input_tokens (one token of headroom required)",
        "scenarios": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local tokenizer/processor directory; no weight load or download",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1024,
        help="Fail when any untruncated request reaches this bound",
    )
    parser.add_argument(
        "--pacman-python-root",
        type=Path,
        help="Fixed pacman-python checkout; otherwise use environment/default discovery",
    )
    args = parser.parse_args()
    if args.max_input_tokens <= 0:
        parser.error("--max-input-tokens must be positive")
    if not args.model_path.is_dir():
        parser.error("--model-path must be an existing local processor directory")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    observation, source = capture_c1_image(args.pacman_python_root)
    result = measure(processor, observation, max_input_tokens=args.max_input_tokens)
    result["image_source"] = source
    result["processor_path"] = str(args.model_path.resolve())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(
            "C1 direct prompt budget exceeded; no automatic truncation or recipe change: "
            f"{result['input_tokens']} >= {args.max_input_tokens} "
            f"({result['largest_scenario']})"
        )


if __name__ == "__main__":
    main()
