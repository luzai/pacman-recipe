"""Maze layouts for visual-grounding SFT: native levels, their wall variants, and procedural
skeletons of any odd size up to 25x21. All share the level-1 palette and tile codes.

Tile codes (pacman-python level files): 100-199 wall sprites (retile picks the sprite from the
wall neighbours), 0 empty floor, 2 pellet, 3 power pellet, 4 Pac-Man start, 1 ghost-house door,
10-13 ghost spawns, 20 horizontal portal, 21 vertical portal.

- native(path): a 21-wide native level padded to 25 rows with exactly one horizontal and one
  vertical portal pair on the real boundary (the env/planner contract); other portals walled off.
- variant(base, seed): flips up to `changes` random wall/pellet cells, keeping every pellet,
  power pellet, start, ghost and portal reachable (pellets with Pac-Man's portal moves, the rest
  with ordinary moves). Flips are unconstrained in shape, so walls come out fragmented.
- procedural(seed, rows, cols): randomized spanning tree on a half-width lattice (mirrored unless
  asymmetric), braided, with the level-1 ghost house centred at (rows//2 - 1, cols//2), the
  start five rows below it, power pellets near the corners and one portal pair per axis.
Nothing is repaired silently: a disconnected result raises.
"""
import random
from collections import deque
from pathlib import Path

WALL = 120
WALL_TILE = {0: 120, 1: 110, 2: 113, 4: 112, 8: 111, 3: 101, 12: 100, 10: 107, 6: 108, 9: 105, 5: 106,
             14: 130, 13: 133, 7: 131, 11: 132, 15: 140}
SIZES = [(r, c) for r in range(11, 26, 2) for c in range(11, 22, 2)]
MAX_ROWS, MAX_COLS = 25, 21


def is_wall(t): return 100 <= t < 200


def dims(a): return len(a), len(a[0])


def raw_level(path):
    text = Path(path).read_text(encoding='utf-8'); body = text.split('# startleveldata')[1].split('# endleveldata')[0]
    return [[int(x) for x in line.split()] for line in body.splitlines() if line.strip()]


def text(a):
    R, C = dims(a)
    return (f'# lvlwidth {C}\n# lvlheight {R}\n# bgcolor 0 0 0\n# edgelightcolor 0 200 200\n'
            '# edgeshadowcolor 0 120 120\n# fillcolor 0 0 0\n# pelletcolor 255 255 180\n# fruittype 1\n'
            '# startleveldata\n' + '\n'.join(' '.join(map(str, row)) for row in a) + '\n# endleveldata\n# sprites\n')


def retile(a):
    R, C = dims(a); a = [row[:] for row in a]
    for r in range(R):
        for c in range(C):
            if is_wall(a[r][c]):
                mask = sum(bit for dr, dc, bit in ((-1, 0, 1), (1, 0, 2), (0, -1, 4), (0, 1, 8))
                           if 0 <= r+dr < R and 0 <= c+dc < C and is_wall(a[r+dr][c+dc]))
                a[r][c] = WALL_TILE[mask]
    return a


def _bfs(a, teleport, blocked):
    R, C = dims(a)
    start = next((r, c) for r, row in enumerate(a) for c, t in enumerate(row) if t == 4)
    q, seen = deque([start]), {start}
    while q:
        r, c = q.popleft()
        for x, y in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if 0 <= x < R and 0 <= y < C and (x, y) not in seen and not is_wall(a[x][y]) and a[x][y] not in blocked:
                seen.add((x, y)); q.append((x, y))
        if teleport and a[r][c] in (20, 21):
            for other in [(x, y) for x, row in enumerate(a) for y, t in enumerate(row) if t == a[r][c] and (x, y) != (r, c)]:
                if other not in seen: seen.add(other); q.append(other)
    return seen


def reachable(a):
    """Every pellet, power pellet and the start, with Pac-Man's portal moves (door blocked)."""
    seen = _bfs(a, True, (1,))
    return {(r, c) for r, row in enumerate(a) for c, t in enumerate(row) if t in (2, 3, 4)} <= seen


def ordinary_reachable(a):
    """Ghosts and fruit path without Pac-Man's teleport edges."""
    seen = _bfs(a, False, ())
    return {(r, c) for r, row in enumerate(a) for c, t in enumerate(row) if t in (2, 3, 4, 10, 11, 12, 13, 20, 21)} <= seen


def _horizontal(a, prefer=()):
    """One horizontal pair at columns 0/C-1, row nearest the middle (existing tunnel rows first);
    the carved shafts (at most 3 cells) must end on corridor cells Pac-Man can walk to."""
    R, C = dims(a); live = _bfs(a, False, (1,))
    for r in list(prefer) + sorted(range(1, R-1), key=lambda r: (abs(r-R//2), r)):
        left = next((c for c in range(1, 5) if (r, c) in live), None)
        right = next((c for c in range(C-2, C-6, -1) if (r, c) in live), None)
        if left is None or right is None or left >= right: continue
        if any(a[r][c] in (1, 10, 11, 12, 13, 21) for c in list(range(0, left)) + list(range(right+1, C))): continue
        for c in range(1, left): a[r][c] = 0
        for c in range(right+1, C-1): a[r][c] = 0
        a[r][0] = a[r][C-1] = 20
        return r
    raise ValueError('no row admits a boundary horizontal portal')


def _vertical(a, center):
    """One vertical pair at rows 0/R-1, column nearest `center`; shafts as for _horizontal."""
    R, C = dims(a); live = _bfs(a, False, (1,))
    for c in sorted(range(1, C-1), key=lambda c: (abs(c-center), c)):
        top = next((r for r in range(1, 5) if (r, c) in live), None)
        bot = next((r for r in range(R-2, R-6, -1) if (r, c) in live), None)
        if top is None or bot is None or top >= bot: continue
        if any(a[r][c] in (1, 10, 11, 12, 13, 20) for r in list(range(0, top)) + list(range(bot+1, R))): continue
        for r in range(1, top): a[r][c] = 0
        for r in range(bot+1, R-1): a[r][c] = 0
        a[0][c] = a[R-1][c] = 21
        return c
    raise ValueError('no column admits a boundary vertical portal')


def native(path):
    a = raw_level(path)
    if len(a[0]) != MAX_COLS or len(a) > MAX_ROWS: raise ValueError('only 21-wide levels of at most 25 rows')
    a = [row[:] for row in a] + [[WALL]*MAX_COLS for _ in range(MAX_ROWS-len(a))]   # padded rows are solid
    R, C = dims(a)
    rows_h = sorted({r for r in range(R) if a[r][0] == 20 or a[r][C-1] == 20}, key=lambda r: abs(r-R//2))
    for r in range(R):
        for c in range(C):
            if a[r][c] in (20, 21): a[r][c] = WALL
    _horizontal(a, prefer=rows_h)
    _vertical(a, C//2)
    a = retile(a)
    if not (reachable(a) and ordinary_reachable(a)): raise ValueError(f'normalized native level {path} is disconnected')
    return a


def variant(base, seed, changes=40):
    rng = random.Random(seed); a = [row[:] for row in base]; R, C = dims(a)
    protected = {(r, c) for r, row in enumerate(a) for c, t in enumerate(row) if t in (1, 3, 4, 10, 11, 12, 13, 20, 21)}
    # Keep the house, spawns, portals and their immediate surroundings.
    protected |= {(r+dr, c+dc) for r, c in list(protected) for dr in range(-2, 3) for dc in range(-2, 3)}
    options = [(r, c) for r in range(2, R-2) for c in range(2, C-2) if (r, c) not in protected and (a[r][c] == 2 or is_wall(a[r][c]))]
    rng.shuffle(options); done = 0
    for r, c in options:
        old = a[r][c]; a[r][c] = 2 if is_wall(old) else WALL
        if reachable(a) and ordinary_reachable(a): done += 1
        else: a[r][c] = old
        if done == changes: break
    if done < changes * 4 // 5: raise ValueError(f'variant seed {seed}: only {done} cells changed')
    return retile(a)


def procedural(seed, rows, cols, asymmetric=False, braid=0.8):
    R, C = rows, cols
    if (R, C) not in SIZES: raise ValueError(f'unsupported size {R}x{C}')
    rng = random.Random(seed * 1000 + R * 31 + C)
    a = [[WALL]*C for _ in range(R)]
    hr, hc = R//2 - 1, C//2
    half = C if asymmetric else C//2 + 1
    cells = [(r, c) for r in range(1, R-1, 2) for c in range(1, half, 2)]
    seen = {cells[0]}; stack = [cells[0]]; a[cells[0][0]][cells[0][1]] = 2
    while stack:                                   # randomized DFS spanning tree
        r, c = stack[-1]
        nbrs = [(r+dr, c+dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 1 <= r+dr < R-1 and 1 <= c+dc < half and (r+dr, c+dc) not in seen]
        if not nbrs: stack.pop(); continue
        nr, nc = rng.choice(nbrs); a[(r+nr)//2][(c+nc)//2] = 2; a[nr][nc] = 2; seen.add((nr, nc)); stack.append((nr, nc))
    for r, c in cells:                             # braid: open a wall at most dead ends
        if sum(a[r+dr][c+dc] == 2 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 1 and rng.random() < braid:
            opts = [(r+dr, c+dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 1 <= r+2*dr < R-1 and 1 <= c+2*dc < half and is_wall(a[r+dr][c+dc])]
            if opts: rr, cc = rng.choice(opts); a[rr][cc] = 2
    if not asymmetric:
        for r in range(R):
            for c in range(C//2 + 1, C): a[r][c] = a[r][C-1-c]
        for r in range(1, R-1):                    # join halves across the centre column
            if a[r][hc-1] == 2 and rng.random() < 0.5: a[r][hc] = 2
    for r in range(hr-2, hr+4):                    # clear a ring around the ghost house
        for c in range(hc-4, hc+5): a[r][c] = 0 if r in (hr-2, hr+3) or c in (hc-4, hc-3, hc+3, hc+4) else a[r][c]
    house = {(0, -2): WALL, (0, -1): WALL, (0, 0): 1, (0, 1): WALL, (0, 2): WALL,
             (1, -2): WALL, (1, -1): 11, (1, 0): 12, (1, 1): 13, (1, 2): WALL,
             (2, -2): WALL, (2, -1): WALL, (2, 0): WALL, (2, 1): WALL, (2, 2): WALL, (-1, 0): 10}
    for (dr, dc), t in house.items(): a[hr+dr][hc+dc] = t
    for r in (hr-2, hr+3):
        for c in range(hc-4, hc+5): a[r][c] = 2
    for r in range(hr-2, hr+4):
        a[r][hc-4] = a[r][hc+4] = 2
    sr = hr + 5
    if sr > R-2: raise ValueError('no room for the Pac-Man start')
    a[sr][hc] = 4
    for r, c in ((sr-1, hc), (sr+1, hc), (sr, hc-1), (sr, hc+1)):
        if 1 <= r < R-1: a[r][c] = 2
    for r, c in ((3, 1), (3, C-2), (R-4, 1), (R-4, C-2)):
        if a[r][c] != 4: a[r][c] = 3
    tr = hr + 1                                    # horizontal tunnel row through the house's middle row
    for c in range(1, hc-4): a[tr][c] = a[tr][C-1-c] = 2
    a[tr][0] = a[tr][C-1] = 20
    for r in (1, R-2):
        if a[r][hc] != 4: a[r][hc] = 2
    _vertical(a, hc)
    a = retile(a)
    if not (reachable(a) and ordinary_reachable(a)): raise ValueError(f'procedural seed {seed} {R}x{C} disconnected')
    return a
