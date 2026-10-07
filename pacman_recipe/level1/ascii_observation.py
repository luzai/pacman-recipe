"""Current-frame ASCII observation; never consume ghost trajectories or pixels."""
from typing import Any, Mapping

from pacman_env.env.level import GHOST_DOOR, HORIZONTAL_DOOR, VERTICAL_DOOR, WALL_RANGE

ASCII_MAP_HEADER = "MAP (row 0 at top; header shows column mod 10):"


def render_ascii_map(level: Any, snapshot: Mapping[str, Any]) -> str:
    """Render live collectibles and actors, preserving every cell and trailing space.

    Overlap priority is Pac-Man > normal ghost > vulnerable ghost > eyes >
    power pellet > normal pellet > terrain. Gone ghosts are not displayed.
    Required live pellet lists deliberately have no initial-level fallback.
    """
    height, width = level.height, level.width
    if (snapshot["height"], snapshot["width"]) != (height, width):
        raise ValueError("snapshot and level dimensions differ")
    board = [
        ["#" if tile in WALL_RANGE else "-" if tile == GHOST_DOOR else
         "=" if tile in (HORIZONTAL_DOOR, VERTICAL_DOOR) else " " for tile in row]
        for row in level.tiles
    ]

    def put(position: Any, symbol: str) -> None:
        row, col = position
        if type(row) is not int or type(col) is not int or not (0 <= row < height and 0 <= col < width):
            raise ValueError("invalid current-frame actor/collectible position")
        board[row][col] = symbol

    for position in snapshot["normal_pellet_positions"]:
        put(position, ".")
    for position in snapshot["power_pellet_positions"]:
        put(position, "o")
    symbols = {"normal": "G", "vulnerable": "V", "eyes": "E"}
    ghosts = snapshot["ghosts"]
    if any(ghost["state"] not in {*symbols, "gone"} for ghost in ghosts):
        raise ValueError("unknown current-frame ghost state")
    for state in ("eyes", "vulnerable", "normal"):
        for ghost in ghosts:
            if ghost["state"] == state:
                put(ghost["position"], symbols[state])
    put((snapshot["row"], snapshot["col"]), "P")
    digits = len(str(height - 1))
    return "\n".join([
        ASCII_MAP_HEADER,
        " " * (digits + 1) + "".join(str(col % 10) for col in range(width)),
        *(f"{row:>{digits}} " + "".join(cells) for row, cells in enumerate(board)),
    ])
