from __future__ import annotations

import unittest
from dataclasses import dataclass

from pacman_env.env.state import Position
from pacman_env.planner import EdwardPlanner, GhostETA


@dataclass(frozen=True)
class GridLevel:
    """Small actor-aware topology for deterministic planner unit tests."""

    width: int
    height: int
    open_tiles: frozenset[Position]
    pellets: frozenset[Position]
    power_pellets: frozenset[Position] = frozenset()
    ghost_doors: frozenset[Position] = frozenset()
    pacman_start: Position = Position(0, 0)

    def is_wall(self, position: Position, *, actor: str) -> bool:
        if actor not in {"pacman", "ghost", "vulnerable", "eyes"}:
            raise ValueError(actor)
        if position not in self.open_tiles:
            return True
        return position in self.ghost_doors and actor == "pacman"

    def portal_exit(self, position: Position, _action: str) -> Position:
        return position


def positions(*items: tuple[int, int]) -> frozenset[Position]:
    return frozenset(Position(row, col) for row, col in items)


class EdwardPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = EdwardPlanner()

    def state(self, **overrides: object) -> dict[str, object]:
        state: dict[str, object] = {
            "pacman_position": [16, 10],
            "open": ["L", "R"],
            "ghosts": [
                {"id": 0, "position": [10, 10], "state": "normal", "direction": "L"},
                {"id": 1, "position": [12, 9], "state": "gone", "direction": "S"},
            ],
            "edible_ticks": 0,
        }
        state.update(overrides)
        return state

    def test_candidates_are_deterministic_and_use_option_namespaces(self) -> None:
        first = self.planner.candidates(self.state())
        second_planner = EdwardPlanner()
        second = second_planner.candidates(self.state())
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(item.option_id[0] in "CAE" for item in first))
        self.assertEqual(
            len({item.option_id for item in first}),
            len(first),
        )
        self.assertTrue(all(item.first_action in {"L", "R"} for item in first))

    def test_near_lethal_ghost_adds_avoid_candidate(self) -> None:
        candidates = self.planner.candidates(
            self.state(
                ghosts=[
                    {
                        "id": 0,
                        "position": [16, 12],
                        "state": "normal",
                        "direction": "L",
                    }
                ]
            )
        )
        self.assertTrue(any(item.strategy == "AVOID" for item in candidates))

    def test_reachable_vulnerable_ghost_adds_eliminate_candidate(self) -> None:
        candidates = self.planner.candidates(
            self.state(
                ghosts=[
                    {
                        "id": 0,
                        "position": [16, 8],
                        "state": "vulnerable",
                        "direction": "R",
                    }
                ],
                edible_ticks=160,
            )
        )
        eliminate = [item for item in candidates if item.strategy == "ELIMINATE"]
        self.assertTrue(eliminate)
        self.assertEqual(eliminate[0].option_id, "E0")
        self.assertEqual(eliminate[0].first_action, "L")

    def test_decision_is_always_one_of_advertised_candidates(self) -> None:
        decision = self.planner.decide(self.state())
        by_id = {item.option_id: item for item in decision.candidates}
        self.assertIn(decision.option_id, by_id)
        self.assertEqual(decision.action, by_id[decision.option_id].first_action)

    def test_option_is_invalidated_when_ghost_state_changes_materially(self) -> None:
        vulnerable_state = self.state(
            ghosts=[
                {
                    "id": 0,
                    "position": [16, 8],
                    "state": "vulnerable",
                    "direction": "R",
                }
            ],
            edible_ticks=160,
        )
        option = next(
            item
            for item in self.planner.candidates(vulnerable_state)
            if item.strategy == "ELIMINATE"
        )
        action, status = self.planner.continue_option(
            option,
            self.state(
                ghosts=[
                    {
                        "id": 0,
                        "position": [16, 8],
                        "state": "normal",
                        "direction": "R",
                    }
                ]
            ),
        )
        self.assertIsNone(action)
        self.assertEqual(status, "invalidated")

    def test_eliminate_option_tracks_same_moving_vulnerable_ghost(self) -> None:
        initial = self.state(
            ghosts=[
                {
                    "id": 2,
                    "position": [16, 8],
                    "state": "vulnerable",
                    "direction": "R",
                }
            ],
            edible_ticks=160,
        )
        option = next(
            item
            for item in self.planner.candidates(initial)
            if item.strategy == "ELIMINATE"
        )
        action, status = self.planner.continue_option(
            option,
            self.state(
                pacman_position=[16, 9],
                open=["L", "R"],
                ghosts=[
                    {
                        "id": 2,
                        "position": [16, 7],
                        "state": "vulnerable",
                        "direction": "L",
                    }
                ],
                edible_ticks=128,
            ),
        )
        self.assertEqual(status, "active")
        self.assertIn(action, {"L", "R"})


class EdwardPlannerSafetyTests(unittest.TestCase):
    def _dead_end_level(self) -> GridLevel:
        open_tiles = {
            *(Position(1, col) for col in range(1, 6)),
            *(Position(row, 2) for row in range(1, 12)),
        }
        return GridLevel(
            width=6,
            height=12,
            open_tiles=frozenset(open_tiles),
            pellets=positions((1, 5)),
            pacman_start=Position(1, 2),
        )

    @staticmethod
    def _state(
        player: tuple[int, int],
        *,
        open_actions: list[str],
        ghosts: list[dict[str, object]],
        edible_ticks: int = 0,
    ) -> dict[str, object]:
        return {
            "pacman_position": list(player),
            "open": open_actions,
            "ghosts": ghosts,
            "edible_ticks": edible_ticks,
        }

    def test_collect_audits_true_chokes_safe_return_and_each_ghost_eta(self) -> None:
        planner = EdwardPlanner(self._dead_end_level())
        candidates = planner.candidates(
            self._state(
                (1, 2),
                open_actions=["L", "R", "D"],
                ghosts=[
                    {"id": 1, "position": [10, 2], "state": "normal"},
                    {"id": 0, "position": [11, 2], "state": "normal"},
                ],
            )
        )
        collect = next(
            item
            for item in candidates
            if item.strategy == "COLLECT" and item.target == (1, 5)
        )
        self.assertEqual(collect.choke_points, ((1, 3), (1, 4)))
        self.assertEqual(collect.safe_return_distance, 3)
        self.assertEqual(
            collect.lethal_ghost_etas,
            (GhostETA(0, 13), GhostETA(1, 12)),
        )

    def test_collect_rejects_dead_end_when_ghost_can_seal_return_choke(self) -> None:
        planner = EdwardPlanner(self._dead_end_level())
        candidates = planner.candidates(
            self._state(
                (1, 2),
                open_actions=["L", "R", "D"],
                ghosts=[
                    {"id": 0, "position": [4, 2], "state": "normal"},
                ],
            )
        )
        self.assertFalse(
            any(
                item.strategy == "COLLECT" and item.target == (1, 5)
                for item in candidates
            )
        )

    def test_collect_never_routes_pacman_through_ghost_door(self) -> None:
        door = Position(1, 2)
        level = GridLevel(
            width=4,
            height=3,
            open_tiles=positions((1, 1), (1, 2), (1, 3)),
            pellets=positions((1, 3)),
            ghost_doors=frozenset({door}),
            pacman_start=Position(1, 1),
        )
        candidates = EdwardPlanner(level).candidates(
            self._state(
                (1, 1),
                open_actions=["R"],
                ghosts=[],
            )
        )
        self.assertFalse(any(item.strategy == "COLLECT" for item in candidates))

    def test_avoid_increases_clearance_keeps_two_exits_and_skips_reverse(self) -> None:
        open_tiles = frozenset(
            Position(row, col)
            for row in range(7)
            for col in range(7)
        )
        level = GridLevel(
            width=7,
            height=7,
            open_tiles=open_tiles,
            pellets=frozenset(),
            pacman_start=Position(3, 3),
        )
        planner = EdwardPlanner(level)
        planner.record_action("U")
        candidates = planner.candidates(
            self._state(
                (3, 3),
                open_actions=["U", "D", "L", "R"],
                ghosts=[
                    {"id": 0, "position": [3, 1], "state": "normal"},
                ],
            )
        )
        avoid = [item for item in candidates if item.strategy == "AVOID"]
        self.assertTrue(avoid)
        self.assertTrue(all(item.first_action != "D" for item in avoid))
        self.assertTrue(all((item.future_safe_exits or 0) >= 2 for item in avoid))
        self.assertTrue(
            all(
                item.lethal_ghost_etas[0].eta is not None
                and item.lethal_ghost_etas[0].eta > 2
                for item in avoid
            )
        )

    def test_decide_keeps_a_proven_collect_instead_of_coarse_avoid(self) -> None:
        planner = EdwardPlanner()
        state = self._state(
            (16, 10),
            open_actions=["L", "R"],
            ghosts=[{"id": 0, "position": [16, 12], "state": "normal"}],
        )
        decision = planner.decide(state)
        by_id = {item.option_id: item for item in decision.candidates}
        self.assertTrue(any(item.strategy == "AVOID" for item in by_id.values()))
        self.assertEqual(by_id[decision.option_id].strategy, "COLLECT")
        self.assertGreater(by_id[decision.option_id].safety_margin or 0, 1)

    def test_avoid_does_not_target_seed0_bottom_two_way_pocket(self) -> None:
        planner = EdwardPlanner()
        planner.remaining.clear()
        candidates = planner.candidates(
            self._state(
                (20, 3),
                open_actions=["U", "D", "L"],
                ghosts=[
                    {"id": 0, "position": [16, 8], "state": "normal"},
                    {"id": 1, "position": [18, 7], "state": "normal"},
                    {"id": 2, "position": [16, 8], "state": "normal"},
                    {"id": 3, "position": [16, 6], "state": "normal"},
                ],
            )
        )
        avoid = [item for item in candidates if item.strategy == "AVOID"]
        self.assertTrue(avoid)
        self.assertEqual({item.target for item in avoid}, {(21, 3), (22, 3)})
        self.assertTrue(all(item.safe_return_distance == 0 for item in avoid))

    def test_emergency_is_audited_and_fails_closed_when_no_action_is_safe(self) -> None:
        corridor = GridLevel(
            width=7,
            height=3,
            open_tiles=positions((1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
            pellets=frozenset(),
            pacman_start=Position(1, 2),
        )
        planner = EdwardPlanner(corridor)
        emergency = planner.advertised_candidates(
            self._state(
                (1, 2),
                open_actions=["L", "R"],
                ghosts=[{"id": 7, "position": [1, 5], "state": "normal"}],
            )
        )
        self.assertEqual(len(emergency), 1)
        self.assertEqual(emergency[0].first_action, "L")
        self.assertGreater(emergency[0].safety_margin or 0, 1)
        self.assertGreaterEqual(emergency[0].future_safe_exits or 0, 1)
        self.assertEqual(emergency[0].lethal_ghost_etas, (GhostETA(7, 4),))

        one_step_level = GridLevel(
            width=4,
            height=4,
            open_tiles=positions((1, 1), (1, 2), (1, 3), (2, 1)),
            pellets=frozenset(),
            pacman_start=Position(1, 2),
        )
        one_step = EdwardPlanner(one_step_level).advertised_candidates(
            self._state(
                (1, 2),
                open_actions=["L", "R"],
                ghosts=[
                    {
                        "id": 8,
                        "position": [1, 3],
                        "pixel_position": [16, 48],
                        "direction": "L",
                        "path_remaining": "L",
                        "speed": 1.0,
                        "state": "normal",
                    }
                ],
            )
        )[0]
        self.assertEqual(one_step.first_action, "L")
        self.assertEqual(one_step.safety_margin, 1)
        self.assertEqual(one_step.future_safe_exits, 1)

        pixel_trap = EdwardPlanner()
        pixel_trap.remaining.clear()
        with self.assertRaisesRegex(RuntimeError, "no ghost-safe action"):
            pixel_trap.advertised_candidates(
                self._state(
                    (22, 4),
                    open_actions=["L", "R"],
                    ghosts=[
                        {
                            "id": 2,
                            "position": [20, 3],
                            "pixel_position": [324, 48],
                            "direction": "D",
                            "path_remaining": "DDRR",
                            "speed": 1.0,
                            "state": "normal",
                        },
                        {
                            "id": 3,
                            "position": [22, 6],
                            "pixel_position": [352, 99],
                            "direction": "L",
                            "path_remaining": "LL",
                            "speed": 1.0,
                            "state": "normal",
                        },
                    ],
                )
            )

        trapped = EdwardPlanner()
        trapped.remaining.clear()
        with self.assertRaisesRegex(RuntimeError, "no ghost-safe action"):
            trapped.advertised_candidates(
                self._state(
                    (22, 1),
                    open_actions=["U", "D"],
                    ghosts=[
                        {"id": 0, "position": [21, 3], "state": "normal"},
                        {"id": 1, "position": [23, 2], "state": "normal"},
                        {"id": 2, "position": [21, 3], "state": "normal"},
                        {"id": 3, "position": [21, 1], "state": "normal"},
                    ],
                )
            )

    def test_eliminate_accepts_exact_32_tick_margin_and_rejects_recovery(self) -> None:
        level = GridLevel(
            width=5,
            height=3,
            open_tiles=positions((1, 1), (1, 2), (1, 3)),
            pellets=frozenset(),
            pacman_start=Position(1, 1),
        )
        planner = EdwardPlanner(level)
        vulnerable = [{"id": 3, "position": [1, 3], "state": "vulnerable"}]
        exact = planner.candidates(
            self._state(
                (1, 1),
                open_actions=["R"],
                ghosts=vulnerable,
                edible_ticks=64,
            )
        )
        self.assertTrue(any(item.strategy == "ELIMINATE" for item in exact))

        too_late = planner.candidates(
            self._state(
                (1, 1),
                open_actions=["R"],
                ghosts=vulnerable,
                edible_ticks=63,
            )
        )
        self.assertFalse(any(item.strategy == "ELIMINATE" for item in too_late))

        recovering = planner.candidates(
            self._state(
                (1, 1),
                open_actions=["R"],
                ghosts=[{"id": 3, "position": [1, 3], "state": "eyes"}],
                edible_ticks=64,
            )
        )
        self.assertFalse(any(item.strategy == "ELIMINATE" for item in recovering))

    def test_eliminate_rejects_unreachable_vulnerable_ghost(self) -> None:
        level = GridLevel(
            width=5,
            height=3,
            open_tiles=positions((1, 1), (1, 3)),
            pellets=frozenset(),
            pacman_start=Position(1, 1),
        )
        candidates = EdwardPlanner(level).candidates(
            self._state(
                (1, 1),
                open_actions=["R"],
                ghosts=[
                    {"id": 2, "position": [1, 3], "state": "vulnerable"},
                ],
                edible_ticks=160,
            )
        )
        self.assertFalse(any(item.strategy == "ELIMINATE" for item in candidates))


if __name__ == "__main__":
    unittest.main()
