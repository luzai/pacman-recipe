"""Strict model-completion parsing for the Pacman production recipe."""

from __future__ import annotations

from pacman_env.env import Action


class ActionParseError(ValueError):
    """Raised when a completion is not one canonical Pacman token."""


def parse_action(completion: str) -> Action:
    """Parse exactly one canonical token after trimming surrounding whitespace."""
    if not isinstance(completion, str):
        raise ActionParseError("completion must be a string")
    token = completion.strip()
    try:
        return Action(token)
    except ValueError as exc:
        raise ActionParseError(
            f"invalid completion {completion!r}; expected exactly U, D, L, R, or S"
        ) from exc
