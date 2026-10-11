"""CPU checks for the visual-grounding SFT recipe (vg_sft): layouts, labels, plan, grammar, data specs."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vg_sft import labels, make_data, make_plan, mazes
from vg_sft.evaluate import GridGrammar

LEVELS = Path(__file__).resolve().parents[2] / 'pacman-python' / 'pacman' / 'res' / 'levels'


@pytest.mark.parametrize('rows,cols', mazes.SIZES)
def test_procedural_layouts_are_connected_and_sized(rows, cols):
    for seed in range(3):
        a = mazes.procedural(seed, rows, cols, asymmetric=seed % 4 == 3)
        assert mazes.dims(a) == (rows, cols)
        assert mazes.reachable(a) and mazes.ordinary_reachable(a)
        flat = [t for row in a for t in row]
        assert flat.count(4) == 1 and flat.count(20) == 2 and flat.count(21) == 2


def test_procedural_rejects_unsupported_sizes():
    with pytest.raises(ValueError):
        mazes.procedural(0, 27, 21)
    with pytest.raises(ValueError):
        mazes.procedural(0, 12, 11)


@pytest.mark.skipif(not LEVELS.exists(), reason='needs the sibling pacman-python checkout')
@pytest.mark.parametrize('level', [*make_data.NATIVE_TRAIN, *make_data.NATIVE_TEST])
def test_native_levels_normalize_and_vary(level):
    base = mazes.native(LEVELS / f'{level}.txt')
    assert mazes.dims(base) == (25, 21) and mazes.reachable(base) and mazes.ordinary_reachable(base)
    var = mazes.variant(base, 500 + level)
    changed = sum(mazes.is_wall(x) != mazes.is_wall(y) for r1, r2 in zip(base, var) for x, y in zip(r1, r2))
    assert 32 <= changed <= 40 and mazes.reachable(var)
    assert mazes.text(var).count('\n') > 25


def _synthetic(rows=11, cols=13, pac=(5, 3)):
    """Black board with a yellow 16x16 Pac-Man sprite; walls on the border."""
    tiles = [[120 if r in (0, rows-1) or c in (0, cols-1) else 2 for c in range(cols)] for r in range(rows)]
    frame = np.zeros((rows*16, cols*16, 3), np.uint8)
    y, x = pac[0]*16, pac[1]*16
    frame[y:y+16, x:x+16] = (255, 255, 0)
    pellets = [[r, c] for r in range(1, rows-1) for c in range(1, cols-1) if [r, c] != list(pac)]
    for r, c in pellets:
        frame[r*16+7:r*16+9, c*16+7:c*16+9] = labels.PELLET
    state = dict(height=rows, width=cols, pacman_pixel_position=[y, x], ghosts=[], fruit=dict(active=False),
                 score=0, lives=0, level=1, normal_pellet_positions=pellets, power_pellet_positions=[])
    return state, frame, tiles


def test_labels_on_a_synthetic_frame():
    state, frame, tiles = _synthetic()
    grid, actors = labels.visible_board(state, frame, tiles)
    assert len(grid) == 11 and all(len(row) == 13 for row in grid)
    assert actors[0]['symbol'] == 'P' and actors[0]['rc'] == [5, 3] and actors[0]['visible_px'] == 256
    assert grid[0] == '#' * 13 and grid[1][1] == '.'
    rows = labels.questions(grid, actors, 'frame-1')
    assert [r['task'] for r in rows] == ['map', 'map', 'entities', 'pixel', 'count', 'states', 'adjacency', 'relation', 'crop', 'cells']
    assert '11-row by 13-column board' in rows[0]['question'] and '[11 strings]' in rows[0]['question']
    for r in rows:
        row = dict(r, grid=grid, actors_v2=actors)
        result = labels.score(row, r['answer'])
        assert result['valid'] and result['exact'], r['task']
    shown = labels.decode_map(rows[0]['target'], 11, 13)
    assert shown[5][3] == 'P' and shown == labels.overlay(grid, actors)


def test_score_rejects_wrong_board_shape_and_counts_cells():
    state, frame, tiles = _synthetic()
    grid, actors = labels.visible_board(state, frame, tiles)
    q = labels.questions(grid, actors, 'frame-2')[0]; row = dict(q, grid=grid, actors_v2=actors)
    wide = {'rows': [r + ' #' for r in q['target']['rows']]}
    assert not labels.score(row, json.dumps(wide))['valid']
    wrong = {'rows': list(q['target']['rows'])}; wrong['rows'][2] = '# ' + wrong['rows'][2][2:].replace('.', '_', 1)
    s = labels.score(row, json.dumps(wrong))
    assert s['valid'] and not s['exact'] and s['cell_correct'] == 11*13 - 1 and s['cell_total'] == 11*13


def test_hud_boxes_follow_the_screen_bottom():
    small = labels.hud_boxes(dict(score=120, lives=2, level=1), 11)
    full = labels.hud_boxes(dict(score=120, lives=2, level=1), 25)
    assert small[0][:2] == (11*16-24, 11*16-12) and full[0][:2] == (376, 388)


def test_plan_gives_each_size_group_half_the_mass():
    frames = [dict(id=f'a{i}', map_id=f'm{i % 3}', pacman_rc=[i % 5, i % 7], grid=['_'*21]*25) for i in range(30)]
    frames += [dict(id=f'b{i}', map_id=f'z{i % 4}', pacman_rc=[i % 4, i % 3], grid=['_'*11]*11) for i in range(12)]
    w = make_plan.frame_weights(frames)
    assert abs(sum(w[f['id']] for f in frames[:30]) - 0.5) < 1e-9 and abs(sum(w.values()) - 1) < 1e-9
    per_map = Counter()
    for f in frames[:30]: per_map[f['map_id']] += w[f['id']]
    assert max(per_map.values()) - min(per_map.values()) < 1e-9


class _CharTokenizer:
    """One token per character, enough to exercise the grid grammar."""
    def __init__(self):
        chars = sorted(set('{}[]",: rows') | labels.SYMBOLS | set('xyz'))
        self.vocab = {ch: i for i, ch in enumerate(chars)}; self.inv = {i: ch for ch, i in self.vocab.items()}
        self.eos_token_id = len(chars); self.all_special_ids = [self.eos_token_id]
    def get_vocab(self): return dict(self.vocab, **{'<eos>': self.eos_token_id})
    def batch_decode(self, seqs, **kw): return [''.join(self.inv.get(i, '') for i in s) for s in seqs]
    def decode(self, ids, **kw): return ''.join(self.inv.get(i, '') for i in ids)


def test_grid_grammar_forces_the_board_shape():
    tok = _CharTokenizer(); grammar = GridGrammar(tok)
    assert 'x' not in grammar.fragments.values()
    c = grammar.constraint(prompt_length=0, rows=2, cols=3)
    good = json.dumps({'rows': ['# . _', 'P # ?']}, separators=(',', ':'))
    c.validate(good)
    with pytest.raises(ValueError):
        c.validate(json.dumps({'rows': ['# . _ #', 'P # ?']}, separators=(',', ':')))
    first = {tok.inv[i] for i in c.allowed_at(0)}
    assert first == {'{'}
    at_cell = {tok.inv[i] for i in c.allowed_at(len('{"rows":["'))}
    assert at_cell == set(labels.SYMBOLS)
    assert c.allowed_at(len(good)) == [tok.eos_token_id]


def test_data_specs_are_disjoint_and_complete():
    specs = make_data.specs()
    names = [s[0] for s in specs]; assert len(names) == len(set(names))
    roles = Counter(s[4] for s in specs)
    assert roles == {'test-family': 10, 'test-size': 18, 'validation': 12, 'train': 30 + 30 + 94}
    train_sizes = {s[2] for s in specs if s[4] == 'train' and s[1] == 'procedural'}
    assert train_sizes == set(mazes.SIZES)
    assert not {s[2] for s in specs if s[1] == 'native' and s[4] == 'train'} & set(make_data.NATIVE_TEST)
    tiny = make_data.specs(tiny=True)
    assert set(Counter(s[4] for s in tiny)) == {'test-family', 'test-size', 'validation', 'train'} and len(tiny) <= 12


def test_pick_takes_half_hard_frames_with_a_step_gap():
    cands = [dict(id=f'f{s}', seed=1, step=s, categories=['overlap'] if s % 2 else []) for s in range(0, 200, 3)]
    chosen = make_data.pick(cands, 10, 0)
    assert len(chosen) == 10 and sum(bool(f['categories']) for f in chosen) >= 5
    steps = sorted(f['step'] for f in chosen)
    assert all(b - a >= make_data.STEP_GAP for a, b in zip(steps, steps[1:]))
