"""Score vg_sft.evaluate outputs: strict exact per task, per board size and per hard-frame
category; for maps also exact-tolerant (actor letter anywhere in its sprite box), cell accuracy
over valid outputs, and Pac-Man found in the bottom two rows.

usage: python -m vg_sft.score --dataset DIR --evals EVALDIR --sets test-family test-size test-stream
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

from vg_sft.labels import score


def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def score_set(rows, outputs):
    agg = defaultdict(Counter)
    for o in outputs:
        r = rows[o['id']]; s = score(r, o['raw_text']); size = f"{len(r['grid'])}x{len(r['grid'][0])}"
        for key in ['all', r['task'], f"{r['task']}@{size}"] + [f"{r['task']}:{c}" for c in r.get('categories', [])]:
            a = agg[key]; a['n'] += 1; a['valid'] += s['valid']; a['exact'] += s['exact']
            if r['task'] == 'map':
                a['exact_tol'] += bool(s.get('exact_tol')); a['pacman_tol'] += bool(s.get('pacman_tol'))
                a['cell_correct'] += s.get('cell_correct', 0); a['cell_total'] += s.get('cell_total', 0)
                pac = next((x for x in r['actors_v2'] if x['symbol'] == 'P'), None)
                if pac and pac['rc'][0] >= len(r['grid'])-2: a['bottom_pacman_n'] += 1; a['bottom_pacman_tol'] += bool(s.get('pacman_tol'))
    return {k: dict(v, cell_acc=round(v['cell_correct']/v['cell_total'], 5) if v['cell_total'] else None) for k, v in sorted(agg.items())}


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--dataset', required=True, type=Path); a.add_argument('--evals', required=True, type=Path); a.add_argument('--sets', nargs='+', required=True)
    args = a.parse_args(); report = {}
    for name in args.sets:
        rows = {r['id']: r for r in read(args.dataset/f'{name}.jsonl')}; report[name] = {}
        for kind in ('maps-fixed', 'maps-raw', 'other'):
            p = args.evals/f'{name}-{kind}.jsonl'
            if p.exists(): report[name][kind] = score_set(rows, read(p))
    (args.evals/'score.json').write_text(json.dumps(report, indent=1))
    for name, out in report.items():
        for kind, d in out.items():
            m = d.get('map', {}); al = d.get('all', {})
            print(name, kind, 'n', al.get('n'), 'exact', al.get('exact'), '| map exact', m.get('exact'), 'tol', m.get('exact_tol'),
                  'cell', m.get('cell_acc'), 'valid', m.get('valid'), 'bottomP', f"{m.get('bottom_pacman_tol', 0)}/{m.get('bottom_pacman_n', 0)}")


if __name__ == '__main__':
    main()
