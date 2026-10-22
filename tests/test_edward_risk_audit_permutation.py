"""Risk ranks remain bound to candidate IDs regardless of display order."""
import copy
import itertools
import unittest

from pacman_recipe.level1.trajectories import _audit_risk_fallback_step as NEW


def fixture(n=4):
    candidates = [dict(option_id=f"A{i}", strategy="RISK_FALLBACK", first_action=a,
                       target=[i, 1], commit_moves=1, route_distance=1,
                       safety_margin=None, future_safe_exits=None,
                       risk=dict(rank=i+1, motion="clear_estimate", ghost_clearance=2,
                                 route_margin=1, safe_next_cells=1, dead_end=False, reverse=False))
                  for i, a in enumerate("UDLR"[:n])]
    return (dict(decoding=dict(edward_fallback_mode="risk_ranked"), prompt_template_sha256="test"),
            dict(observation_context=dict(planner_candidates=candidates),
                 option_strategy="RISK_FALLBACK", model_called=True, option_step=1,
                 option_end=True, option_status="max_commit", open_action_mask=list("UDLR"[:n]),
                 option_id="A0", action="U", option_target=[0, 1]))


class AuditTests(unittest.TestCase):
    def test_all_33_permutations_one_to_four_candidates(self):
        count = 0
        for n in range(1, 5):
            payload, step = fixture(n)
            candidates = step["observation_context"]["planner_candidates"]
            for perm in itertools.permutations(candidates):
                current = copy.deepcopy(step)
                current["observation_context"]["planner_candidates"] = list(perm)
                NEW(payload, current)
                count += 1
        self.assertEqual(count, 33)

    def test_bad_rank_and_identity_evidence_rejected(self):
        for rank in [None, True, "1", 0, 5, 2]:
            with self.subTest(rank=rank):
                payload, step = fixture()
                step["observation_context"]["planner_candidates"][0]["risk"]["rank"] = rank
                with self.assertRaises(ValueError): NEW(payload, step)
        for oid in ["A1", "A9", "C0", None]:
            with self.subTest(oid=oid):
                payload, step = fixture()
                step["observation_context"]["planner_candidates"][0]["option_id"] = oid
                with self.assertRaises(ValueError): NEW(payload, step)

    def test_existing_safety_and_execution_checks_remain(self):
        mutations = [lambda p,s:s.update(action="R"),
                     lambda p,s:s.update(option_target=[99,99]),
                     lambda p,s:s.update(open_action_mask=["U"]),
                     lambda p,s:s.update(model_called=False),
                     lambda p,s:p["decoding"].update(edward_fallback_mode="refuse"),
                     lambda p,s:s["observation_context"]["planner_candidates"][0].update(commit_moves=2),
                     lambda p,s:s["observation_context"]["planner_candidates"][0]["risk"].update(motion="safe"),
                     lambda p,s:s["observation_context"]["planner_candidates"][0]["risk"].update(safe_next_cells=True)]
        for i, mutation in enumerate(mutations):
            with self.subTest(mutation=i):
                payload, step = fixture()
                mutation(payload, step)
                for audit in (NEW,):
                    with self.assertRaises(ValueError): audit(payload, step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
