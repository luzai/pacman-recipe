"""Portable frozen map generation and shared runtime level binding."""

from collections import deque
import hashlib
import json
from pathlib import Path
import random

from pacman_env.env.level import load_bundled_level, parse_level_text
from pacman_env.env.state import Position
from pacman_env.paths import pacman_python_root


def load_runtime_level(number, root=None):
    root = root or pacman_python_root()
    if not root:
        return load_bundled_level(number)
    text = (Path(root) / f"pacman/res/levels/{number}.txt").read_text(encoding="utf-8")
    return parse_level_text(
        text, number=number, revision=hashlib.sha256(text.encode()).hexdigest()
    )


def parse_grid(text):
    head, rest = text.split("# startleveldata\n")
    body, tail = rest.split("# endleveldata")
    return (
        head,
        [[int(v) for v in line.split()] for line in body.strip().splitlines()],
        tail,
    )


def canonical(grid):
    variants = []
    for mirrored in (grid, [list(reversed(row)) for row in grid]):
        current = mirrored
        for _ in range(4):
            variants.append(json.dumps(current, separators=(",", ":")))
            current = [list(row) for row in zip(*current[::-1])]
    return hashlib.sha256(min(variants).encode()).hexdigest()


def gate(text):
    sha = hashlib.sha256(text.encode()).hexdigest()
    level = parse_level_text(text, number=1, revision=sha)
    seen = {level.pacman_start}
    queue = deque(seen)
    while queue:
        position = queue.popleft()
        for action, dr, dc in [("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1)]:
            target = Position(position.row + dr, position.col + dc)
            if level.is_wall(target):
                continue
            target = level.portal_exit(target, action)
            if level.is_wall(target):
                continue
            if target not in seen:
                seen.add(target)
                queue.append(target)
    if not level.pellets or not (level.pellets | level.power_pellets) <= seen:
        return None
    return dict(
        layout_sha256=sha,
        width=level.width,
        height=level.height,
        pellet_count=len(level.pellets),
        power_pellet_count=len(level.power_pellets),
    )


def generate(template, *, seed, counts, swaps=4, excluded=(), attempts=10000,
             small_floor_pellets=False):
    """Return layouts without writes. Freeze splits before any model evaluation.

    Supply a template for each desired size; protect the central ghost house.
    counts maps train/validation/test to the number of distinct layouts required.
    """
    if set(counts) != {"train", "validation", "test"} or any(
        type(v) is not int or v < 1 for v in counts.values()
    ):
        raise ValueError("positive train/validation/test layout counts required")
    head, original, tail = parse_grid(template)
    height, width = len(original), len(original[0])
    radius = 1 if height == 11 else 3
    protected = range(height // 2 - radius, height // 2 + radius + 1)
    walls = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (height < 11 or r not in protected) and 100 <= original[r][c] < 200
    ]
    pellets = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (height < 11 or r not in protected) and original[r][c] == 2
    ]
    if small_floor_pellets and height >= 11:
        raise ValueError("small floor mutation is restricted to maps below 11")
    if swaps < 1 or min(len(walls), len(pellets)) < swaps:
        raise ValueError("template lacks mutation cells")
    floor_cells = [(r, c) for r in range(1, height - 1)
                   for c in range(1, width - 1) if original[r][c] == 0]
    used = set(excluded) | {canonical(original)}
    rows = []
    schedule = [
        split for split in ("train", "validation", "test") for _ in range(counts[split])
    ]
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        grid = [row[:] for row in original]
        active_swaps = rng.randint(1, swaps) if height < 11 else swaps
        for r, c in rng.sample(walls, active_swaps):
            grid[r][c] = 2
        for r, c in rng.sample(pellets, active_swaps):
            grid[r][c] = 100
        if small_floor_pellets:
            # Only add ordinary pellets to existing traversable empty floor.
            # Spawn, door, portal, power and fruit tiles remain byte-identical.
            for r, c in floor_cells:
                if rng.random() < 0.5:
                    grid[r][c] = 2
        equivalent = canonical(grid)
        if equivalent in used:
            continue
        text = (
            head
            + "# startleveldata\n"
            + "\n".join(" ".join(map(str, row)) for row in grid)
            + "\n# endleveldata"
            + tail
        )
        check = gate(text)
        if check is None:
            continue
        used.add(equivalent)
        rows.append(
            dict(
                split=schedule[len(rows)],
                equivalence_sha256=equivalent,
                text=text,
                generation_attempt=attempt,
                **check,
            )
        )
        if len(rows) == len(schedule):
            return rows
    raise ValueError("could not generate enough connected distinct layouts")
