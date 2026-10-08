"""Per-cell spaced variant of the current-frame ASCII map (prompt style ascii_edward_spaced_v1).

Qwen's tokenizer merges runs such as `.....` or `###` into multi-character tokens, so in the
packed map a column index must be counted out of merged runs. Here every cell is a single
symbol followed by one space, and empty cells are written `_`, so each cell is its own token.
The board content and actor overlap priority are exactly those of `render_ascii_map`; this module
only re-lays out its output (and inverts that layout), leaving the packed renderer and its
contract fingerprint unchanged.
"""
from typing import Any, Mapping

from .ascii_observation import ASCII_MAP_HEADER, render_ascii_map

ASCII_SPACED_MAP_HEADER = (
    "MAP (row 0 at top; header shows column mod 10; cells separated by single spaces):"
)
EMPTY = "_"


def space_ascii_map(packed: str) -> str:
    """Packed `render_ascii_map` text -> one cell per token; empty cells become `_`."""
    head, header, *rows = packed.split("\n")
    if head != ASCII_MAP_HEADER or not rows:
        raise ValueError("expected rendered current-frame ASCII map")
    digits = len(header) - len(header.lstrip(" ")) - 1
    columns = header[digits + 1:]
    if digits < 1 or not columns.isdigit():
        raise ValueError("unexpected ASCII column header")
    out = [ASCII_SPACED_MAP_HEADER, " " * (digits + 1) + " ".join(columns)]
    for line in rows:
        label, cells = line[:digits], line[digits + 1:]
        if line[digits:digits + 1] != " " or len(cells) != len(columns) or EMPTY in cells:
            raise ValueError("unexpected ASCII map row")
        out.append(f"{label} " + " ".join(EMPTY if ch == " " else ch for ch in cells))
    return "\n".join(out)


def unspace_ascii_map(spaced: str) -> str:
    """Exact inverse of `space_ascii_map`."""
    head, header, *rows = spaced.split("\n")
    if head != ASCII_SPACED_MAP_HEADER or not rows:
        raise ValueError("expected spaced current-frame ASCII map")
    digits = len(header) - len(header.lstrip(" ")) - 1
    out = [ASCII_MAP_HEADER, " " * (digits + 1) + header[digits + 1:][::2]]
    for line in rows:
        label, cells = line[:digits], line[digits + 1:]
        if any(cells[i] != " " for i in range(1, len(cells), 2)):
            raise ValueError("spaced ASCII row must alternate cell and single space")
        out.append(f"{label} " + "".join(" " if ch == EMPTY else ch for ch in cells[::2]))
    packed = "\n".join(out)
    if space_ascii_map(packed) != spaced:
        raise ValueError("spaced ASCII map is not canonical")
    return packed


def render_ascii_map_spaced(level: Any, snapshot: Mapping[str, Any]) -> str:
    return space_ascii_map(render_ascii_map(level, snapshot))
