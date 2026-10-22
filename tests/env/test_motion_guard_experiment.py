"""Focused CPU contract tests, no model/GPU dependencies.

Real package imports; only topology transition is mocked for envelope contracts.
No sys.modules injection is used. The native ghost test executes real game class methods,
with FindPathTo controlled to return a legal reverse route, not a full replay.
Set PACMAN_GHOST_SOURCE to the frozen experiment pacman.pyw for remote checks.
"""
import ast
from collections import namedtuple
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
Position = namedtuple('Position', 'row col')
DIRS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def move(level, pos, action, actor='pacman'):
    dr, dc = DIRS[action]
    target = Position(pos.row + dr, pos.col + dc)
    return target if target in level else None


from experiments.edward_route_options import motion_guard as N
bound = N.B
escape = N.E
planner = N.B.P

class MotionContract(unittest.TestCase):
    def setUp(self):
        self.move_patch = patch.object(bound, 'move', side_effect=move)
        self.move_patch.start()
        self.addCleanup(self.move_patch.stop)
        self.corridor = {Position(1, x) for x in range(8)}
        self.state = {'pacman_position': [1, 1], 'open': 'LR', 'ghosts': [
            dict(id=0, state='normal', pixel_position=[160, 160],
                 direction='R', speed=1, position=[10, 10])]}

    def test_future_corridor_includes_reversal(self):
        self.assertEqual(N.advance(self.corridor, {(16., 32., 'R')}, 1),
                         {(16., 31., 'L'), (16., 33., 'R')})

    def test_first_frame_obeys_observed_direction(self):
        self.assertEqual(N.advance(self.corridor, {(16., 32., 'R')}, 1, False),
                         {(16., 33., 'R')})

    def test_between_centers_no_early_turn(self):
        self.assertEqual(N.advance(self.corridor, {(16., 31., 'R')}, 1),
                         {(16., 32., 'R')})

    def test_native_path_expiry_reverse_is_in_envelope(self):
        default = HERE.parents[2] / 'pacman-python' / 'pacman' / 'pacman.pyw'
        source = Path(os.environ.get('PACMAN_GHOST_SOURCE', default))
        self.assertTrue(source.is_file(), str(source))
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ghost')
        ns = {'player': types.SimpleNamespace(nearestRow=1, nearestCol=0)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)
        g = ns['ghost'].__new__(ns['ghost'])
        g.__dict__.update(state=1, x=31, y=16, velX=1, velY=0, speed=1, currentPath='R')
        g.FindPathTo = lambda target: 'LL'
        g.CanTakePathDirection = lambda direction: direction in ('L', 'R')
        envelope = {(16., 31., 'R')}
        for i in range(3):
            g.Move()
            envelope = N.advance(self.corridor, envelope, 1, allow_turn=i != 0)
            self.assertIn((g.y, g.x), {(y, x) for y, x, d in envelope})
        self.assertEqual((g.x, g.velX), (30, -1))

    def test_turn_corner_and_dead_end_are_topological(self):
        corner = {Position(1, 1), Position(1, 2), Position(2, 2)}
        self.assertEqual({d for y, x, d in N.advance(corner, {(16., 32., 'R')}, 1)}, {'L', 'D'})
        self.assertEqual(N.advance({Position(1, 1), Position(1, 2)}, {(16., 32., 'R')}, 1),
                         {(16., 31., 'L')})

    def test_hidden_fields_never_read_and_removed(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key not in N.FIELDS:
                    raise AssertionError(key)
                return super().__getitem__(key)
        ghost = Poison(dict(self.state['ghosts'][0], path_remaining='LL',
                            currentPath='LL', path_target=[0, 0], rng='secret'))
        result = N.clean(dict(self.state, ghosts=[ghost]))
        self.assertEqual(set(result['ghosts'][0]), set(ghost) & set(N.FIELDS))

    def test_missing_or_nonfinite_kinematics_fail_closed(self):
        for key, value in [('speed', float('nan')), ('speed', -1), ('direction', 'X')]:
            ghost = dict(self.state['ghosts'][0]); ghost[key] = value
            self.assertFalse(N.short_safe(self.corridor, dict(self.state, ghosts=[ghost]), ('R',)))
        ghost = dict(self.state['ghosts'][0]); del ghost['speed']
        self.assertFalse(N.short_safe(self.corridor, dict(self.state, ghosts=[ghost]), ('R',)))

    def test_branch_cap_and_empty_envelope_fail_closed(self):
        for branches in [set(), {(10000., float(x), 'R') for x in range(513)}]:
            with patch.object(N, 'advance', return_value=branches):
                self.assertFalse(N.short_safe(self.corridor, self.state, ('R',)))

    def test_any_branch_collision_rejects(self):
        with patch.object(N, 'advance', return_value={(16., 32., 'L'), (1000., 1000., 'R')}):
            self.assertFalse(N.short_safe(self.corridor, self.state, ('R',)))

    def test_window_first_frame_only_suppresses_turn(self):
        flags = []
        def advance(level, branches, speed, allow_turn=True):
            flags.append(allow_turn)
            return branches
        with patch.object(N, 'advance', side_effect=advance):
            self.assertTrue(N.short_safe(self.corridor, self.state, ('R', 'R')))
        self.assertEqual(flags, [False] + [True] * 31)

    def test_patch_modes_restore_even_on_exception(self):
        originals = (bound.short_safe, escape.short_safe, planner._one_step_pixel_safe, planner._fallback_motion_risk)
        for mode in ('no_path', 'final_base', 'final_lure'):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                with N.guard_mode(mode):
                    self.assertIs(bound.short_safe, N.short_safe)
                    self.assertIs(escape.short_safe, N.short_safe)
                    self.assertIs(planner._one_step_pixel_safe, N.one_step)
                    self.assertIs(planner._fallback_motion_risk, N.fallback)
                    raise RuntimeError('probe')
            self.assertIs(bound.short_safe, originals[0])
            self.assertIs(escape.short_safe, originals[1])
            self.assertIs(planner._one_step_pixel_safe, originals[2])
            self.assertIs(planner._fallback_motion_risk, originals[3])
        with N.guard_mode('continuous'):
            self.assertIs(bound.short_safe, originals[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
