"""Small coordinate type used by level-1 verification tooling."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Position:
    row: int
    col: int
