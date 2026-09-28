"""Parser and integrity checks for bundled pacman-python level resources."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pacman_env.errors import InvalidConfigurationError, LevelLoadError

from .state import Position

BLANK = 0
GHOST_DOOR = 1
PELLET = 2
POWER_PELLET = 3
PACMAN_START = 4
GHOST_IDS = frozenset({10, 11, 12, 13})
HORIZONTAL_DOOR = 20
VERTICAL_DOOR = 21
WALL_RANGE = range(100, 200)


@dataclass(frozen=True)
class LevelDefinition:
    number: int
    width: int
    height: int
    tiles: tuple[tuple[int, ...], ...]
    pacman_start: Position
    ghost_spawns: tuple[Position, ...]
    pellets: frozenset[Position]
    power_pellets: frozenset[Position]
    horizontal_doors: tuple[Position, Position]
    vertical_doors: tuple[Position, Position]
    background_color: tuple[int, int, int]
    edge_light_color: tuple[int, int, int]
    edge_shadow_color: tuple[int, int, int]
    fill_color: tuple[int, int, int]
    pellet_color: tuple[int, int, int]
    revision: str

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.row < self.height and 0 <= position.col < self.width

    def tile_at(self, position: Position) -> int:
        if not self.in_bounds(position):
            return BLANK
        return self.tiles[position.row][position.col]

    def is_wall(self, position: Position, *, actor: str = "pacman") -> bool:
        if actor not in {"pacman", "ghost", "vulnerable", "eyes"}:
            raise ValueError(f"unknown maze actor: {actor!r}")
        if not self.in_bounds(position):
            return True
        tile = self.tile_at(position)
        if tile == GHOST_DOOR:
            return actor == "pacman"
        return tile in WALL_RANGE

    def portal_exit(self, position: Position, action: str) -> Position:
        tile = self.tile_at(position)
        if tile == HORIZONTAL_DOOR:
            other = next(item for item in self.horizontal_doors if item != position)
            if action == "L":
                return Position(other.row, other.col - 1)
            if action == "R":
                return Position(other.row, other.col + 1)
        elif tile == VERTICAL_DOOR:
            other = next(item for item in self.vertical_doors if item != position)
            if action == "U":
                return Position(other.row - 1, other.col)
            if action == "D":
                return Position(other.row + 1, other.col)
        return position


def _parse_color(parts: list[str], name: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value) for value in parts)
    except ValueError as exc:
        raise LevelLoadError(f"invalid {name} value") from exc
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise LevelLoadError(f"{name} must contain three byte values")
    return values


def parse_level_text(text: str, *, number: int, revision: str) -> LevelDefinition:
    width = height = 0
    background_color = (0, 0, 0)
    edge_light_color = (255, 255, 0)
    edge_shadow_color = (255, 150, 0)
    fill_color = (0, 255, 255)
    pellet_color = (255, 255, 255)
    rows: list[list[int]] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "# startleveldata":
            in_data = True
            continue
        if line == "# endleveldata":
            in_data = False
            break
        if line.startswith("#"):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "lvlwidth":
                width = int(parts[2])
            elif len(parts) >= 3 and parts[1] == "lvlheight":
                height = int(parts[2])
            elif len(parts) >= 5 and parts[1] == "bgcolor":
                background_color = _parse_color(parts[2:5], "bgcolor")
            elif len(parts) >= 5 and parts[1] == "edgecolor":
                edge_light_color = _parse_color(parts[2:5], "edgecolor")
                edge_shadow_color = edge_light_color
            elif len(parts) >= 5 and parts[1] == "edgelightcolor":
                edge_light_color = _parse_color(parts[2:5], "edgelightcolor")
            elif len(parts) >= 5 and parts[1] == "edgeshadowcolor":
                edge_shadow_color = _parse_color(parts[2:5], "edgeshadowcolor")
            elif len(parts) >= 5 and parts[1] == "fillcolor":
                fill_color = _parse_color(parts[2:5], "fillcolor")
            elif len(parts) >= 5 and parts[1] == "pelletcolor":
                pellet_color = _parse_color(parts[2:5], "pelletcolor")
            continue
        if in_data:
            try:
                rows.append([int(value) for value in line.split()])
            except ValueError as exc:
                raise LevelLoadError("level data contains a non-integer tile") from exc

    if width <= 0 or height <= 0:
        raise LevelLoadError("level dimensions are missing or invalid")
    if len(rows) != height:
        raise LevelLoadError(f"expected {height} level rows, found {len(rows)}")
    if any(len(row) != width for row in rows):
        raise LevelLoadError(f"every level row must contain {width} tiles")

    pacman_starts: list[Position] = []
    ghost_by_id: dict[int, Position] = {}
    pellets: set[Position] = set()
    power_pellets: set[Position] = set()
    horizontal: list[Position] = []
    vertical: list[Position] = []
    normalized: list[list[int]] = []

    for row_index, row in enumerate(rows):
        normalized_row: list[int] = []
        for col_index, tile in enumerate(row):
            position = Position(row_index, col_index)
            if tile == PACMAN_START:
                pacman_starts.append(position)
                normalized_row.append(BLANK)
            elif tile in GHOST_IDS:
                if tile in ghost_by_id:
                    raise LevelLoadError(f"duplicate ghost tile {tile}")
                ghost_by_id[tile] = position
                normalized_row.append(BLANK)
            else:
                normalized_row.append(tile)
                if tile == PELLET:
                    pellets.add(position)
                elif tile == POWER_PELLET:
                    power_pellets.add(position)
                elif tile == HORIZONTAL_DOOR:
                    horizontal.append(position)
                elif tile == VERTICAL_DOOR:
                    vertical.append(position)
        normalized.append(normalized_row)

    if len(pacman_starts) != 1:
        raise LevelLoadError(
            f"expected exactly one Pacman start, found {len(pacman_starts)}"
        )
    if set(ghost_by_id) != GHOST_IDS:
        raise LevelLoadError("level must contain ghost spawn IDs 10, 11, 12, and 13")
    if len(horizontal) != 2 or horizontal[0].row != horizontal[1].row:
        raise LevelLoadError("level must contain one horizontal door pair")
    if len(vertical) != 2 or vertical[0].col != vertical[1].col:
        raise LevelLoadError("level must contain one vertical door pair")

    return LevelDefinition(
        number=number,
        width=width,
        height=height,
        tiles=tuple(tuple(row) for row in normalized),
        pacman_start=pacman_starts[0],
        ghost_spawns=tuple(ghost_by_id[tile] for tile in sorted(GHOST_IDS)),
        pellets=frozenset(pellets),
        power_pellets=frozenset(power_pellets),
        horizontal_doors=(horizontal[0], horizontal[1]),
        vertical_doors=(vertical[0], vertical[1]),
        background_color=background_color,
        edge_light_color=edge_light_color,
        edge_shadow_color=edge_shadow_color,
        fill_color=fill_color,
        pellet_color=pellet_color,
        revision=revision,
    )


def load_bundled_level(number: int = 1) -> LevelDefinition:
    if number != 1:
        raise InvalidConfigurationError(
            f"unsupported level {number!r}; bundled oracle data supports only level 1"
        )
    resource_dir = Path(__file__).resolve().parent / "levels"
    level_path = resource_dir / f"{number}.txt"
    manifest_path = resource_dir / "manifest.json"
    try:
        raw = level_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LevelLoadError(f"could not load bundled level {number}") from exc
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    revision = hashlib.sha256(canonical).hexdigest()
    expected = str(manifest.get("levels", {}).get(str(number), {}).get("sha256", ""))
    if not expected or revision != expected:
        raise LevelLoadError(
            f"level {number} integrity check failed: expected {expected!r}, got {revision}"
        )
    try:
        text = canonical.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LevelLoadError(f"level {number} is not UTF-8") from exc
    return parse_level_text(text, number=number, revision=revision)
