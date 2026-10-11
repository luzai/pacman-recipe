"""Build the visual-grounding SFT dataset from scratch: layouts -> real game episodes -> labeled
screenshots -> ten VQA questions per frame -> frozen train / validation / test files.

Layouts (vg_sft.mazes), all with the level-1 palette:
- train: native levels 1, 4, 6, 7, 8, 10 (level 1 exactly as shipped, the others normalized to
  25x21) each with 4 wall variants; 30 procedural 25x21 skeletons; 2 procedural skeletons for
  every other odd size up to 25x21 (47 sizes).
- validation: 2 variants each of native 1 and 4; 8 procedural sizes.
- test-family: native levels 3 and 11 (base + 1 variant), never trained; 6 procedural 25x21.
- test-size: 18 procedural sizes from 11x11 to 25x21 with seeds never trained.
- test-stream: the shipped level 1, unfiltered frames of 6 three-life episodes (map task only).

Each layout gets 3 episodes (normal ghosts/single death 512 steps, normal/three lives 768,
ghosts disabled/single death 512) played by the Edward planner with 10% random legal moves;
every 3rd playing frame is a candidate. Per layout half the selected frames are hard (actor in
the bottom two rows, overlapping sprites, fruit overlap, actor at the image edge) and half
ordinary, at least 8 steps apart within an episode. Held-out frames are selected first; seeds,
layouts and frame hashes are disjoint across splits.

usage: python -m vg_sft.make_data --game /path/to/pacman-python --out /path/to/new-dataset [--workers 48] [--tiny]
"""
import argparse, hashlib, json, multiprocessing, random, shutil, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from vg_sft import labels, mazes

NATIVE_TRAIN, NATIVE_TEST = (1, 4, 6, 7, 8, 10), (3, 11)
VALIDATION_SIZES = [(11, 11), (13, 17), (15, 21), (17, 13), (19, 19), (21, 11), (23, 15), (25, 19)]
TEST_SIZES = [(11, 11), (11, 17), (11, 21), (13, 13), (13, 19), (15, 11), (15, 15), (15, 21), (17, 17), (19, 13),
              (19, 21), (21, 15), (21, 19), (23, 11), (23, 21), (25, 13), (25, 17), (25, 21)]
FRAMES = {'train': 40, 'train-sized': 12, 'validation': 4, 'test-family': 16, 'test-size': 12}
EPISODES = [('normal', 'single_death', 512), ('normal', 'original_three_lives', 768), ('disabled', 'single_death', 512)]
STREAM_EPISODES, STREAM_STEPS = 6, 600
STEP_GAP = 8


def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def lines(p, rows): Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wall_bytes(tiles): return bytes(int(mazes.is_wall(t)) for row in tiles for t in row)


def specs(tiny=False):
    """(name, kind, arg, seed, role) in selection order: held-out roles first."""
    out = []
    for n in NATIVE_TEST:
        out += [(f'n{n:02d}a', 'native', n, 0, 'test-family'), (f'n{n:02d}b', 'native', n, 600+n, 'test-family')]
    out += [(f'p25x21h{s}', 'procedural', (25, 21), 1000+s, 'test-family') for s in range(6)]
    out += [(f'z{R:02d}x{C:02d}h', 'procedural', (R, C), 3000, 'test-size') for R, C in TEST_SIZES]
    out += [(f'n{n:02d}v{k}', 'native', n, 200+10*n+k, 'validation') for n in (1, 4) for k in range(2)]
    out += [(f'z{R:02d}x{C:02d}v', 'procedural', (R, C), 2000, 'validation') for R, C in VALIDATION_SIZES]
    for n in NATIVE_TRAIN:
        out += [(f'n{n:02d}t0', 'native', n, 0, 'train')] + [(f'n{n:02d}t{k}', 'native', n, 500+10*n+k, 'train') for k in range(1, 5)]
    out += [(f'p25x21t{s}', 'procedural', (25, 21), s, 'train') for s in range(30)]
    out += [(f'z{R:02d}x{C:02d}t{s}', 'procedural', (R, C), s, 'train') for R, C in mazes.SIZES if (R, C) != (25, 21) for s in (0, 1)]
    if tiny:   # smoke test: a couple of layouts per role
        keep = {'test-family': 2, 'test-size': 2, 'validation': 2, 'train': 4}; seen = Counter(); small = []
        for s in out:
            if seen[s[4]] < keep[s[4]]: small.append(s); seen[s[4]] += 1
        out = small + [s for s in out if s[4] == 'train' and s[0].startswith('z')][:2]
    return out


def build_layout(levels, kind, arg, seed):
    if kind == 'procedural':
        R, C = arg; return mazes.procedural(seed, R, C, asymmetric=seed % 4 == 3)
    if arg == 1 and seed == 0:
        return mazes.raw_level(levels/'1.txt')      # level 1 exactly as shipped (the RL level)
    base = mazes.native(levels/f'{arg}.txt')
    return mazes.variant(base, seed) if seed else base


def make_layouts(game, out, tiny):
    from pacman_env.env.level import parse_level_text
    from pacman_env.planner import _topology
    levels = game/'pacman/res/levels'; root = out/'layouts'; root.mkdir(); layouts = {}; manifest = []
    for name, kind, arg, seed, role in specs(tiny):
        a = build_layout(levels, kind, arg, seed)
        raw = mazes.text(a).encode() if not (kind == 'native' and arg == 1 and seed == 0) else (levels/'1.txt').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        level = parse_level_text(raw.decode(), number=1, revision=digest)
        _topology(level).adjacency('pacman', True); _topology(level).adjacency('ghost', False)
        w = wall_bytes(level.tiles); assert w not in layouts, (name, layouts.get(w)); layouts[w] = name
        target = root/name; shutil.copytree(game, target, ignore=shutil.ignore_patterns('.git', 'pacman-large', '*.exe'))
        (target/'pacman/res/levels/1.txt').write_bytes(raw)
        manifest.append(dict(map_id=name, kind=kind, source=arg, seed=seed, role=role, size=[level.height, level.width],
                             level_sha256=digest, game=str(target), tiles=[list(r) for r in level.tiles]))
    (out/'layouts.json').write_text(json.dumps(manifest, indent=1)+'\n')
    return {m['map_id']: m for m in manifest}


def categories(state):
    R, C = state['height'], state['width']
    boxes = [tuple(state['pacman_pixel_position'])] + [tuple(g['pixel_position']) for g in state['ghosts'] if g['state'] != 'gone']
    cats = set()
    if any((y+15)//16 >= R-2 for y, _ in boxes): cats.add('bottom')
    if any(x < 0 or x > C*16-16 for _, x in boxes): cats.add('edge')
    if any(abs(a[0]-b[0]) < 16 and abs(a[1]-b[1]) < 16 for i, a in enumerate(boxes) for b in boxes[i+1:]): cats.add('overlap')
    fr = state['fruit']
    if fr['active']:
        fy, fx = fr['pixel_position']
        if any(abs(fy-y) < 24 and abs(fx-x) < 24 for y, x in boxes): cats.add('fruit_overlap')
    return sorted(cats)


def episode(task):
    """Edward + 10% random legal moves; every 3rd playing frame becomes a labeled candidate."""
    import pacman_env.env as envmod
    from PIL import Image
    from pacman_env.env.level import parse_level_text
    from pacman_env.planner import EdwardPlanner, EdwardSafetyRefusal
    out, game, map_id, seed, split, ghost_mode, life_mode, max_steps = task
    out = Path(out); path = out/'episodes'/f'{seed}.json'
    if path.exists(): return json.loads(path.read_text())
    raw = (Path(game)/'pacman/res/levels/1.txt').read_text()
    level = parse_level_text(raw, number=1, revision=hashlib.sha256(raw.encode()).hexdigest()); tiles = level.tiles
    rng = random.Random(seed); planner = EdwardPlanner(level=level); recs = []; info = {}; steps = 0
    cfg = envmod.PygamePacmanEnvConfig(pacman_python_root=game, max_steps=max_steps, ghost_mode=ghost_mode, episode_life_mode=life_mode,
                                       timeout_seconds=60, video_driver='dummy', audio_driver='dummy', worker_base_dir=str(out/'workers'))
    with envmod.PygamePacmanEnv(cfg) as env:
        frame, info = env.reset(seed=seed)
        for step in range(max_steps + 1):
            state = env.snapshot()
            if state['mode'] == 1 and step % 3 == 0:
                ident = f'{split}-{map_id}-seed{seed}-step{step:04d}'; image = f'images/{ident}.png'
                Image.fromarray(frame).save(out/image)
                grid, actors = labels.visible_board(state, frame, tiles)
                pac = next((a['rc'] for a in actors if a['symbol'] == 'P'), None)
                eng = [(state['pacman_pixel_position'][0]+8)//16, (state['pacman_pixel_position'][1]+8)//16]
                recs.append(dict(id=ident, split=split, seed=seed, step=step, image=image, map_id=map_id, categories=categories(state),
                                 frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(), engine_state=state, grid=grid,
                                 actors_v2=actors, pacman_rc=pac or eng, ghost_mode=ghost_mode, life_mode=life_mode))
            if step == max_steps: break
            try:
                act = planner.decide(state).action
                if rng.random() < 0.1:
                    legal = [a.value for a in env.legal_actions() if a.value in 'UDLR']; act = rng.choice(legal) if legal else act
            except EdwardSafetyRefusal:
                legal = [a.value for a in env.legal_actions() if a.value in 'UDLR']; act = rng.choice(legal) if legal else 'S'
            frame, _, term, trunc, info = env.step(act); steps += 1
            if term or trunc: break
    ep = dict(seed=seed, map_id=map_id, split=split, steps=steps, terminal_reason=info.get('terminal_reason'), records=recs)
    path.write_text(json.dumps(ep)); return ep


def pick(cands, cap, seed):
    """Half hard frames (round-robin over categories, rarest first), half ordinary; STEP_GAP per seed."""
    rng = random.Random(seed); by = defaultdict(list); plain = [f for f in cands if not f['categories']]
    for f in cands:
        for c in f['categories']: by[c].append(f)
    for v in list(by.values()) + [plain]: rng.shuffle(v)
    chosen, ids, steps = [], set(), defaultdict(list)

    def take(pool):
        while pool:
            f = pool.pop()
            if f['id'] in ids or any(abs(f['step']-s) < STEP_GAP for s in steps[f['seed']]): continue
            chosen.append(f); ids.add(f['id']); steps[f['seed']].append(f['step']); return True
        return False
    order = sorted(by, key=lambda c: len(by[c]))
    while len(chosen) < cap//2 and any(by.values()):
        for c in order:
            if len(chosen) < cap//2: take(by[c])
    while len(chosen) < cap and (take(plain) or any(take(by[c]) for c in order)):
        pass
    return sorted(chosen, key=lambda f: (f['seed'], f['step']))


def qa(frames, layouts, split, heldout, tasks=None):
    """Ten rows per frame; held-out sets keep one map phrasing and say 'Use this image alone.'"""
    rows = []
    for f in sorted(frames, key=lambda x: (x['map_id'], x['seed'], x['step'])):
        for j, q in enumerate(labels.questions(f['grid'], f['actors_v2'], f['id'], heldout=heldout)):
            if heldout and q['task'] == 'map' and j == 1: continue
            if tasks and q['task'] not in tasks: continue
            r = {k: f[k] for k in ['seed', 'step', 'image', 'frame_sha256', 'grid', 'map_id', 'actors_v2', 'categories']}
            r.update(frame_id=f['id'], id=f['id']+f'-{j:02d}-'+q['task'], split=split, size=list(layouts[f['map_id']]['size']))
            r.update(q); rows.append(r)
            if q['task'] == 'map':
                assert labels.decode_map(q['target'], len(f['grid']), len(f['grid'][0])) == labels.overlay(f['grid'], f['actors_v2'])
    return rows


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--game', required=True, type=Path); a.add_argument('--out', required=True, type=Path)
    a.add_argument('--workers', type=int, default=48); a.add_argument('--tiny', action='store_true')
    args = a.parse_args(); game, out = args.game.resolve(), args.out.resolve()
    out.mkdir(parents=True)
    for d in ('images', 'episodes', 'workers'): (out/d).mkdir()
    started = time.monotonic(); layouts = make_layouts(game, out, args.tiny)
    tasks = []
    for i, (name, m) in enumerate(layouts.items()):
        tasks += [(str(out), m['game'], name, 100000 + i*10 + e, m['role'], g, l, s) for e, (g, l, s) in enumerate(EPISODES)]
    stream_game = out/'layouts'/'stream-level1'; shutil.copytree(game, stream_game, ignore=shutil.ignore_patterns('.git', 'pacman-large', '*.exe'))
    layouts['level1-stream'] = dict(map_id='level1-stream', size=[25, 21], role='test-stream', tiles=mazes.raw_level(game/'pacman/res/levels/1.txt'))
    tasks += [(str(out), str(stream_game), 'level1-stream', 90000+e, 'test-stream', 'normal', 'original_three_lives', STREAM_STEPS)
              for e in range(1 if args.tiny else STREAM_EPISODES)]
    assert len({t[3] for t in tasks}) == len(tasks), 'episode seeds must be distinct (episodes are cached by seed)'
    eps = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for fut in as_completed([pool.submit(episode, t) for t in tasks]):
            eps.append(fut.result())
            if len(eps) % 30 == 0 or len(eps) == len(tasks):
                print(json.dumps(dict(episodes=len(eps), total=len(tasks), seconds=round(time.monotonic()-started))), flush=True)
    cands = [f for ep in eps for f in ep['records'] if ep['split'] != 'test-stream']
    stream = sorted([f for ep in eps if ep['split'] == 'test-stream' for f in ep['records']], key=lambda f: (f['seed'], f['step']))
    taken = set(); chosen = defaultdict(list)
    for role in ('test-family', 'test-size', 'validation', 'train'):          # held-out frames are reserved first
        for i, (name, m) in enumerate((n, m) for n, m in layouts.items() if m['role'] == role):
            cap = FRAMES['train-sized' if role == 'train' and m['size'] != [25, 21] else role]
            fs = [f for f in cands if f['map_id'] == name and f['frame_sha256'] not in taken]
            ch = pick(fs, cap, 20261011 + i + 1000*len(chosen))
            if len(ch) < cap//2: raise RuntimeError(f'{name}: only {len(ch)} of {cap} frames (episodes too short)')
            chosen[role] += ch; taken |= {f['frame_sha256'] for f in ch}
    train_sha = {f['frame_sha256'] for f in chosen['train']}
    stream = [f for f in stream if f['frame_sha256'] not in train_sha]   # level 1 is a training layout: drop identical frames
    held = [f for r in ('test-family', 'test-size', 'validation') for f in chosen[r]] + stream
    assert not {f['seed'] for f in chosen['train']} & {f['seed'] for f in held}
    assert not {f['frame_sha256'] for f in chosen['train']} & {f['frame_sha256'] for f in held}
    assert not {f['map_id'] for f in chosen['train']} & {f['map_id'] for f in held if f['map_id'] != 'level1-stream'}
    files = {'train.jsonl': qa(chosen['train'], layouts, 'train', False), 'validation.jsonl': qa(chosen['validation'], layouts, 'validation', False),
             'test-family.jsonl': qa(chosen['test-family'], layouts, 'test', True), 'test-size.jsonl': qa(chosen['test-size'], layouts, 'test', True),
             'test-stream.jsonl': qa(stream, layouts, 'test', True, tasks={'map'})}
    for name, rows in files.items(): lines(out/name, rows)
    lines(out/'train-frames.jsonl', [{k: v for k, v in f.items() if k != 'engine_state'} for f in chosen['train']])
    size = lambda fs: dict(sorted(Counter(f"{len(f['grid'])}x{len(f['grid'][0])}" for f in fs).items()))
    manifest = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), tiny=args.tiny, game=str(game),
                    sha256={n: sha(out/n) for n in list(files) + ['train-frames.jsonl', 'layouts.json']},
                    code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                    rows={n: len(r) for n, r in files.items()},
                    frames={r: len(chosen[r]) for r in chosen} | {'test-stream': len(stream)},
                    sizes={r: size(chosen[r]) for r in chosen}, categories={r: dict(Counter(c for f in chosen[r] for c in f['categories'])) for r in chosen},
                    layouts=dict(Counter(m['role'] for m in layouts.values())), episodes=len(eps),
                    terminal=dict(Counter(f"{ep['split']}/{ep['terminal_reason']}" for ep in eps)), seconds=round(time.monotonic()-started))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.rmtree(out/'workers', ignore_errors=True)
    print('DATASET_FROZEN ' + json.dumps({k: manifest[k] for k in ('rows', 'frames', 'layouts')}), flush=True)


if __name__ == '__main__':
    main()
