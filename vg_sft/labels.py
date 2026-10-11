"""Pixel-faithful labels and the ten VQA questions per screenshot, for any board up to 25x21.

Rules:
1. No fixed HUD rows: a cell is `?` only where pixels hide it. An open cell is `?` when non-maze
   pixels cover its centre 4x4 (where a pellet is drawn) and no pellet pixel shows; a wall/door
   cell when at least half its pixels are covered. Sprites, fruit, score pop-ups and the HUD are
   all treated alike.
2. Each pixel belongs to the sprite drawn last (pacman.pyw: ghosts 0..3, then fruit, then
   Pac-Man; the HUD is drawn after every sprite). An actor sits in the cell holding most of its
   visible pixels (ties: engine centre); actors with fewer than MIN_ACTOR_PX visible are omitted.
   If actors share a cell the map shows the one with more pixels there, i.e. the one on top;
   the entities answer lists every visible actor.
3. Map and crop answers write one symbol per cell separated by single spaces.
4. Scoring: strict exact for every task, plus for maps the cell accuracy and a tolerant score
   where an actor letter may sit in any cell its sprite box touches.

`grid` is the board (`# - . o _ ?`); `overlay` adds the actor letters P G V E.
"""
import json, random
from collections import Counter
import numpy as np

C = 16
BLACK, EDGE_L, EDGE_S, PELLET = (0, 0, 0), (0, 200, 200), (0, 120, 120), (255, 255, 180)
DOOR_COLORS = {(132, 0, 0), (255, 0, 0)}
SYMBOLS = set('#-.oPGVE_?')
ACTOR = {'pacman': 'P', 'normal': 'G', 'vulnerable': 'V', 'eyes': 'E'}
MIN_ACTOR_PX = 10
SYSTEM = 'You answer questions about the visible Pacman screenshot precisely.'


def contract(rows, cols):
    return (
        f'The image is a {rows}-row by {cols}-column board of 16-pixel cells, starting at image top-left. '
        'Grid coordinates are 0-based [row,col], row downward, col rightward. '
        'Pixel coordinates [x,y] are absolute image pixels, never grid coordinates. '
        'Symbols: # wall tile, - red ghost-house door, . visible small pellet, o visible power pellet, '
        'P Pacman, G normal ghost, V vulnerable blue/white ghost, E eyes-only ghost, _ empty. '
        'A character is written at the cell holding most of its visible sprite; if two characters share a cell, '
        'write the one you see on top. ? means hidden: something drawn over the cell (a character, fruit, score '
        'text or the bottom score/lives bar) covers it so its contents cannot be seen. '
        'A blinking power pellet absent in this frame is empty. Portals look empty. '
        'Empty cells are underscores; spaces only separate cells. Return only the requested JSON, without explanation. '
    )


def sep(row): return ' '.join(row)


def occluder_mask(frame, tiles):
    """True where a pixel is not part of the static maze palette (sprites, fruit, text, HUD)."""
    f = frame[..., :3].astype(np.int16)
    bg = np.zeros(f.shape[:2], bool)
    for col in (BLACK, EDGE_L, EDGE_S, PELLET):
        bg |= np.all(f == col, -1)
    door = np.zeros_like(bg)
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            if t == 1: door[r*C:(r+1)*C, c*C:(c+1)*C] = True
    for col in DOOR_COLORS:
        bg |= door & np.all(f == col, -1)
    return ~bg


def hud_boxes(state, rows):
    """pacman.pyw DrawScore, measured from the screen bottom: score digits (9 px pitch) at
    (40, Y-24), one life icon per life at (40+10i, Y-12), fruit icon at (20, Y-20), level number
    at (0, Y-12). Generous glyph sizes."""
    Y = rows*C; n = len(str(int(state.get('score', 0)))); lv = len(str(int(state.get('level', 1))))
    return [(Y-24, Y-12, 40, 40+9*n), (Y-12, Y, 40, 40+10*max(0, state.get('lives', 0))+2), (Y-20, Y-4, 20, 36), (Y-12, Y, 0, 9*lv)]


def actor_list(state, occ):
    H, W = occ.shape[0]//C, occ.shape[1]//C
    items = [(g['state'], g['pixel_position'], i) for i, g in enumerate(state['ghosts']) if g['state'] != 'gone']
    items.append(('pacman', state['pacman_pixel_position'], 99))   # drawn last: on top of every ghost
    owner = np.full(occ.shape, -1, np.int16)
    for k, (kind, (y, x), _) in enumerate(items):
        if kind not in ACTOR: raise ValueError(f'unknown actor state {kind}')
        ys, xs = slice(max(0, y), min(H*C, y+C)), slice(max(0, x), min(W*C, x+C))
        owner[ys, xs] = np.where(occ[ys, xs], k, owner[ys, xs])
    if state['fruit']['active']:   # fruit is drawn after the ghosts, before Pac-Man
        fy, fx = state['fruit']['pixel_position']; fy -= state['fruit'].get('bounce_offset', 0)
        ys, xs = slice(max(0, fy), min(H*C, fy+C)), slice(max(0, fx), min(W*C, fx+C))
        sub = owner[ys, xs]; owner[ys, xs] = np.where(occ[ys, xs] & (sub != len(items)-1), -2, sub)
    for y0, y1, x0, x1 in hud_boxes(state, H):   # DrawScore runs after every sprite: HUD covers Pac-Man too
        owner[y0:y1, x0:x1] = np.where(occ[y0:y1, x0:x1], -3, owner[y0:y1, x0:x1])
    out = []
    for k, (kind, (y, x), idx) in enumerate(items):
        rr, cc = np.nonzero(owner == k); counts = Counter(zip((rr//C).tolist(), (cc//C).tolist()))
        n = sum(counts.values())
        if n < MIN_ACTOR_PX: continue
        engine = ((y+8)//C, (x+8)//C)
        best = max(counts.values()); cells = sorted(k2 for k2, v in counts.items() if v == best)
        r, c = engine if engine in cells else cells[0]
        box = sorted({(yy//C, xx//C) for yy in (max(0, y), min(H*C-1, y+C-1)) for xx in (max(0, x), min(W*C-1, x+C-1))})
        out.append(dict(symbol=ACTOR[kind], state=kind, rc=[r, c], pixel_xy=[x+8, y+8], visible_px=n, cell_px=counts[r, c],
                        draw_index=idx, box_cells=[list(b) for b in box]))
    ghosts = sorted([a for a in out if a['symbol'] != 'P'], key=lambda a: (a['rc'], a['symbol']))
    return [a for a in out if a['symbol'] == 'P'] + ghosts


def overlay(board, actors):
    """Board plus actor letters; in a shared cell the actor showing more pixels there wins."""
    g = [list(r) for r in board]; best = {}
    for a in actors:
        r, c = a['rc']
        if (r, c) not in best or a['cell_px'] > best[r, c]['cell_px']: best[r, c] = a
    for (r, c), a in best.items(): g[r][c] = a['symbol']
    return [''.join(r) for r in g]


def visible_board(state, frame, tiles):
    H, W = len(tiles), len(tiles[0])
    assert (state['height'], state['width']) == (H, W) and frame.shape[:2] == (H*C, W*C)
    occ = occluder_mask(frame, tiles)
    pel = np.all(frame[..., :3] == PELLET, -1)
    normal = {tuple(p) for p in state['normal_pellet_positions']}; power = {tuple(p) for p in state['power_pellet_positions']}
    grid = []
    for r in range(H):
        row = []
        for c in range(W):
            t = tiles[r][c]; cell = (slice(r*C, (r+1)*C), slice(c*C, (c+1)*C))
            if 100 <= t < 200 or t == 1:
                row.append('?' if occ[cell].mean() >= .5 else '#' if t != 1 else '-')
            elif pel[cell].any() and (r, c) in normal | power:
                row.append('o' if (r, c) in power else '.')
            elif occ[r*C+6:r*C+10, c*C+6:c*C+10].any():
                row.append('?')
            else:
                row.append('_')
        grid.append(''.join(row))
    return grid, actor_list(state, occ)


def questions(grid, actors, ident, heldout=False):
    """Ten questions per frame. map/crop/cells read the board with actor letters; count and
    adjacency read the board."""
    H, W = len(grid), len(grid[0])
    rng = random.Random(ident); shown = overlay(grid, actors)
    P = [a for a in actors if a['symbol'] == 'P']; ghosts = [a for a in actors if a['symbol'] != 'P']; pre = contract(H, W); out = []
    for prompt in ['Transcribe this entire visible board', 'Read every cell of this screenshot']:
        out.append(('map', pre + prompt + f'. Output exactly {{"rows":[{H} strings]}}, each string exactly {W} symbols separated by single spaces. No indices.',
                    {'rows': [sep(r) for r in shown]}))
    out.append(('entities', pre + 'Locate all visible characters, including any partly hidden behind another. Schema: {"pacman_rc":[row,col] or null,"ghosts":[{"rc":[row,col],"state":"normal|vulnerable|eyes"}]}; ghosts sorted by row then col.',
                {'pacman_rc': P[0]['rc'] if P else None, 'ghosts': [{'rc': g['rc'], 'state': g['state']} for g in ghosts]}))
    out.append(('pixel', pre + 'Locate Pacman in pixels. Schema: {"pixel_xy":[x,y]}, or {"pixel_xy":null} if Pacman is not visible.',
                {'pixel_xy': P[0]['pixel_xy'] if P else None}))
    pr, pc = P[0]['rc'] if P else (rng.randrange(H), rng.randrange(W))
    r0, c0, r1, c1 = max(0, pr-2), max(0, pc-2), min(H-1, pr+2), min(W-1, pc+2)
    region = [line[c0:c1+1] for line in grid[r0:r1+1]]
    out.append(('count', pre + f'Count visible small and power pellets in inclusive rows {r0}..{r1}, cols {c0}..{c1}. Hidden cells do not count. Schema: {{"small":integer,"power":integer}}.',
                {'small': sum(s.count('.') for s in region), 'power': sum(s.count('o') for s in region)}))
    out.append(('states', pre + 'Count visible ghosts by state. Schema: {"normal":integer,"vulnerable":integer,"eyes":integer}.',
                {s: sum(g['state'] == s for g in ghosts) for s in ['normal', 'vulnerable', 'eyes']}))
    nb = {}
    for d, (dr, dc) in {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}.items():
        r, c = pr+dr, pc+dc; v = grid[r][c] if 0 <= r < H and 0 <= c < W else '#'
        nb[d] = 'unknown' if v == '?' else 'wall' if v == '#' else 'door' if v == '-' else 'open'
    out.append(('adjacency', pre + ('Classify the four cells next to Pacman' if P else f'Classify the four cells next to cell [{pr},{pc}]')
                + ' visually as wall, door, open, or unknown (hidden). Open means no visible wall or door, not safe movement. Schema: {"U":string,"D":string,"L":string,"R":string}.', nb))
    if P and ghosts:
        g = ghosts[0]; dy, dx = g['rc'][0]-pr, g['rc'][1]-pc
        out.append(('relation', pre + 'For the visible ghost first in row-then-column order, report signed grid displacement from Pacman and Manhattan grid distance (abs(dr)+abs(dc)); not path distance. Schema: {"delta_rc":[dr,dc],"manhattan":integer}.',
                    {'delta_rc': [dy, dx], 'manhattan': abs(dy)+abs(dx)}))
    else:
        out.append(('relation', pre + 'For the visible ghost first in row-then-column order, report displacement from Pacman and Manhattan distance, or unknown if Pacman or no ghost is visible. Schema: {"delta_rc":value,"manhattan":value}.',
                    {'delta_rc': 'unknown', 'manhattan': 'unknown'}))
    r, c = rng.randrange(H-4), rng.randrange(W-4)
    out.append(('crop', pre + f'Read inclusive rows {r}..{r+4}, columns {c}..{c+4}. Output {{"rows":[5 strings]}}; each string exactly 5 symbols separated by single spaces.',
                {'rows': [sep(line[c:c+5]) for line in shown[r:r+5]]}))
    cells = rng.sample([(r, c) for r in range(H) for c in range(W)], 4)
    out.append(('cells', pre + f'Read these [row,col] cells in this order: {json.dumps(cells)}. Output {{"symbols":[4 one-character strings]}}.',
                {'symbols': [shown[r][c] for r, c in cells]}))
    rows = []
    for task, q, target in out:
        if heldout: q = 'Use this image alone. ' + q
        rows.append(dict(task=task, question=q, target=target, answer=json.dumps(target, separators=(',', ':'))))
    assert len(rows) == 10
    return rows


def decode_map(v, H, W):
    assert type(v) is dict and set(v) == {'rows'}
    rows = v['rows']; assert type(rows) is list and len(rows) == H
    grid = []
    for s in rows:
        assert type(s) is str and len(s) == 2*W-1 and all(s[i] == ' ' for i in range(1, len(s), 2))
        g = s[0::2]; assert set(g) <= SYMBOLS; grid.append(g)
    return grid


def same_typed(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return set(a) == set(b) and all(same_typed(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(same_typed(x, y) for x, y in zip(a, b))
    return a == b


def score(row, text):
    """Strict exact for every task; for maps also cell accuracy and the tolerant actor score."""
    task = row['task']; res = dict(valid=False, exact=False); H, W = len(row['grid']), len(row['grid'][0])
    try:
        v = json.loads(text)
        if task == 'map':
            pred = decode_map(v, H, W); truth = decode_map(row['target'], H, W)
            acts = row['actors_v2']; zone = {tuple(b) for a in acts for b in a['box_cells']}
            outside = all(pred[r][c] == truth[r][c] for r in range(H) for c in range(W) if (r, c) not in zone)
            shown = [a for a in acts if truth[a['rc'][0]][a['rc'][1]] == a['symbol']]
            found = [any(pred[r][c] == a['symbol'] for r, c in map(tuple, a['box_cells'])) for a in shown]
            n_letters = lambda g: sum(ch in 'PGVE' for x in g for ch in x)
            pac = next((a for a in acts if a['symbol'] == 'P'), None)
            res.update(valid=True, exact=same_typed(v, row['target']), cell_correct=sum(a == b for x, y in zip(pred, truth) for a, b in zip(x, y)),
                       cell_total=H*W, actors_found_tol=sum(found), actors_shown=len(shown),
                       exact_tol=outside and all(found) and n_letters(pred) == n_letters(truth),
                       pacman_tol=(any(pred[r][c] == 'P' for r, c in map(tuple, pac['box_cells'])) if pac
                                   else not any('P' in x for x in pred)))
        else:
            target = row['target']; ok = type(v) is dict and set(v) == set(target)
            res.update(valid=ok, exact=ok and same_typed(v, target))
    except (ValueError, TypeError, KeyError, AssertionError):
        pass
    return res
