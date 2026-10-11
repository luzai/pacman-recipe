"""Freeze the training draw sequence: `updates` lists of `batch` QA ids.

Frames are split into two groups of equal draw mass: 25x21 boards and every other size. Inside a
group each layout gets equal mass, then each occupied Pac-Man row within a layout, then each
occupied position within that row (training-only importance weights; they cannot create missing
coverage). A drawn frame rotates through its own shuffled questions before any question repeats.

usage: python -m vg_sft.make_plan --dataset DIR --updates 6000 --out plan.json
"""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path


def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def position_weights(frames):
    """Equal mass per layout, per occupied Pac-Man row in it, per occupied position in that row."""
    counts = Counter(); rows = defaultdict(set); positions = defaultdict(set)
    for f in frames:
        r, c = f['pacman_rc']; m = f['map_id']
        counts[m, r, c] += 1; rows[m].add(r); positions[m, r].add(c)
    w = []
    for f in frames:
        (r, c), m = f['pacman_rc'], f['map_id']
        w.append(1/(len(rows)*len(rows[m])*len(positions[m, r])*counts[m, r, c]))
    assert abs(sum(w)-1) < 1e-9
    return w


def frame_weights(frames):
    full = [f for f in frames if (len(f['grid']), len(f['grid'][0])) == (25, 21)]
    other = [f for f in frames if (len(f['grid']), len(f['grid'][0])) != (25, 21)]
    groups = [g for g in (full, other) if g]
    weights = {}
    for g in groups:
        for f, w in zip(g, position_weights(g)): weights[f['id']] = w / len(groups)
    return weights


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--dataset', required=True, type=Path); a.add_argument('--updates', type=int, default=6000)
    a.add_argument('--batch', type=int, default=16); a.add_argument('--seed', type=int, default=20261008)
    a.add_argument('--out', required=True, type=Path)
    args = a.parse_args(); root = args.dataset
    frames = read(root/'train-frames.jsonl'); qa = read(root/'train.jsonl')
    fw = frame_weights(frames)
    pool = defaultdict(list)
    for r in qa: pool[r['frame_id']].append(r['id'])
    assert set(pool) == set(fw), 'QA/frame mismatch'
    rng = random.Random(args.seed); fids = [f['id'] for f in frames]; weights = [fw[i] for i in fids]; queues = {f: [] for f in fids}

    def draw(f):
        if not queues[f]: queues[f] = rng.sample(pool[f], len(pool[f]))
        return queues[f].pop()
    updates = [[draw(f) for f in rng.choices(fids, weights=weights, k=args.batch)] for _ in range(args.updates)]
    draws = [i for u in updates for i in u]; byid = {r['id']: r for r in qa}
    plan = dict(train_file='train.jsonl', seed=args.seed, batch=args.batch, n_updates=args.updates, draws=len(draws),
                distinct_qa=len(set(draws)), distinct_frames=len({byid[i]['frame_id'] for i in draws}), available_qa=len(qa),
                available_frames=len(frames), max_repeat_qa=max(Counter(draws).values()), tasks=dict(Counter(byid[i]['task'] for i in draws)),
                full_size_draw_fraction=sum(byid[i]['size'] == [25, 21] for i in draws)/len(draws),
                train_sha256=hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(), updates=updates)
    args.out.write_text(json.dumps(plan)+'\n')
    print(json.dumps({k: v for k, v in plan.items() if k != 'updates'}))


if __name__ == '__main__':
    main()
