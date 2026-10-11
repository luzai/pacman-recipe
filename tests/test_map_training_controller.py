import unittest
from copy import deepcopy
from slime_pacman.map_training.controller import TrainingPolicy, TrainingWinController


class TrainingWinTests(unittest.TestCase):
    def setUp(self):
        # Synthetic values for tests only, not a formal training policy.
        self.c = TrainingWinController(TrainingPolicy(.75, 2, 24, 2, 1),
                                       catalog_identity='fixture', max_tier=2)

    def batch(self, update, won=True, review=False):
        s = self.c.state
        groups = [dict(group=0, start='initial', sampling_seeds=list(range(12)),
                       task=dict(task_id='current', split='train', tier=s['tier'], layout_sha256='a'*64))]
        if review:
            groups.append(dict(group=1, start='initial', sampling_seeds=list(range(12)),
                               task=dict(task_id='review', split='train', tier=0, layout_sha256='b'*64)))
        plan = dict(identity=self.c.identity, bank_version=s['bank_version'],
                    tier=s['tier'], update=update, groups=groups)
        rows = [dict(group=g['group'], replica=r, task_id=g['task']['task_id'],
                     won=won if g['group'] == 0 else True)
                for g in groups for r in range(12)]
        return dict(update=update, plan=plan, episodes=rows, optimizer_completed=True)

    def test_window_streak_and_next_bank(self):
        self.assertFalse(self.c.commit(**self.batch(0))['ready'])
        self.c.commit(**self.batch(1))
        self.assertEqual(self.c.state['tier'], 0)
        self.c.commit(**self.batch(2))
        self.assertEqual(self.c.state['tier'], 1)
        self.assertEqual(self.c.state['window'], [])

    def test_review_wins_cannot_upgrade_current_tier(self):
        for u in range(3):
            self.c.commit(**self.batch(u))
        for u in range(3, 6):
            e = self.c.commit(**self.batch(u, False, review=True))
            self.assertEqual(e['rate'], 0)
        self.assertEqual(self.c.state['tier'], 1)

    def test_invalid_batches_are_transactional(self):
        valid = self.batch(0)
        cases = []
        for field, value in [('optimizer_completed', False), ('update', 1)]:
            b = deepcopy(valid); b[field] = value; cases.append(b)
        b = deepcopy(valid); b['episodes'].pop(); cases.append(b)
        b = deepcopy(valid); b['episodes'][0]['won'] = 1; cases.append(b)
        b = deepcopy(valid); b['plan']['groups'][0]['task']['split'] = 'validation'; cases.append(b)
        b = deepcopy(valid); b['plan']['bank_version'] = 1; cases.append(b)
        before = self.c.snapshot()
        for b in cases:
            with self.assertRaises(ValueError):
                self.c.commit(**b)
            self.assertEqual(self.c.snapshot(), before)

    def test_resume_bound_to_optimizer_and_bank(self):
        self.c.commit(**self.batch(0))
        saved = self.c.snapshot()
        self.c.restore(saved, completed_update=0, bank_version=0)
        with self.assertRaises(ValueError):
            self.c.restore(saved, completed_update=1, bank_version=0)
        with self.assertRaises(ValueError):
            self.c.restore(saved, completed_update=0, bank_version=1)

    def test_failure_resets_streak(self):
        self.c.commit(**self.batch(0))
        self.c.commit(**self.batch(1))
        self.c.commit(**self.batch(2, False))
        self.assertEqual(self.c.state['streak'], 0)
        self.assertEqual(self.c.state['tier'], 0)

    def test_repeated_easy_map_cannot_satisfy_map_coverage(self):
        self.c = TrainingWinController(TrainingPolicy(.75, 2, 24, 2, 2),
                                       catalog_identity='fixture', max_tier=2)
        for u in range(2):
            e = self.c.commit(**self.batch(u))
            self.assertFalse(e['ready'])
        b = self.batch(2)
        b['plan']['groups'][0]['task'].update(task_id='other', layout_sha256='b'*64)
        for row in b['episodes']: row['task_id'] = 'other'
        self.assertTrue(self.c.commit(**b)['passed'])
        self.assertEqual(self.c.state['tier'], 0)
        self.c.commit(**self.batch(3))
        self.assertEqual(self.c.state['tier'], 1)


if __name__ == '__main__':
    unittest.main()
