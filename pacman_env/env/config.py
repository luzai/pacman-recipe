"""Compatibility metadata for the original-pygame environment."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class PacmanEnvSpec:
    api_version: str
    env_id: str
    ruleset_revision: str
    level_revision: str
    renderer_revision: str
    observation_shape: tuple[int, int, int]
    observation_dtype: str
    action_tokens: tuple[str, ...]
    deterministic: bool
