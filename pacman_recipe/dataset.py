"""Backward-compatible imports for synthetic dataset generation."""

from .synthetic.dataset import (
    GHOST_LEGAL_DISTANCE_JSON_SYSTEM_PROMPT,
    GHOST_LEGAL_JSON_CONCISE_SYSTEM_PROMPT,
    GHOST_LEGAL_JSON_FAST_THINK_SYSTEM_PROMPT,
    GHOST_LEGAL_JSON_FIRST_SYSTEM_PROMPT,
    GHOST_LEGAL_JSON_SYSTEM_PROMPT,
    GHOST_LEGAL_REASON_SYSTEM_PROMPT,
    GHOST_LEGAL_ROUTE_JSON_SYSTEM_PROMPT,
    GHOST_LEGAL_STRICT_SYSTEM_PROMPT,
    GHOST_LEGAL_SYSTEM_PROMPT,
    LEGAL_SYSTEM_PROMPT,
    LAYOUTS,
    SPLIT_SIZES,
    SYSTEM_PROMPT,
    TEACHER_HINT_SYSTEM_PROMPT,
    generate_episode_specs,
    generate_examples,
    generate_multi_maze_episode_specs,
    main,
    system_prompt,
    write_jsonl,
)

__all__ = [
    "GHOST_LEGAL_DISTANCE_JSON_SYSTEM_PROMPT",
    "GHOST_LEGAL_JSON_CONCISE_SYSTEM_PROMPT",
    "GHOST_LEGAL_JSON_FAST_THINK_SYSTEM_PROMPT",
    "GHOST_LEGAL_JSON_FIRST_SYSTEM_PROMPT",
    "GHOST_LEGAL_JSON_SYSTEM_PROMPT",
    "GHOST_LEGAL_REASON_SYSTEM_PROMPT",
    "GHOST_LEGAL_ROUTE_JSON_SYSTEM_PROMPT",
    "GHOST_LEGAL_STRICT_SYSTEM_PROMPT",
    "GHOST_LEGAL_SYSTEM_PROMPT",
    "LEGAL_SYSTEM_PROMPT",
    "LAYOUTS",
    "SPLIT_SIZES",
    "SYSTEM_PROMPT",
    "TEACHER_HINT_SYSTEM_PROMPT",
    "generate_episode_specs",
    "generate_examples",
    "generate_multi_maze_episode_specs",
    "main",
    "system_prompt",
    "write_jsonl",
]


if __name__ == "__main__":
    main()
