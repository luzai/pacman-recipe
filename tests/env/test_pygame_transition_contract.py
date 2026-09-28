from __future__ import annotations

from collections import deque
import unittest

from pacman_env.env import PygamePacmanEnv, PygameWorkerError


class PygameTransitionContractTests(unittest.TestCase):
    @staticmethod
    def environment_with_mode(
        *, mode: int = 1, mode_name: str = "playing", mode_timer: int = 0
    ) -> PygamePacmanEnv:
        env = object.__new__(PygamePacmanEnv)
        env._stderr = deque()
        env._state = {
            "mode": mode,
            "mode_name": mode_name,
            "mode_timer": mode_timer,
        }
        return env

    def test_final_atomic_mode_matches_transition_state(self) -> None:
        env = self.environment_with_mode()
        env._audit_final_atomic_mode(
            {"mode": 1, "mode_name": "playing", "mode_timer": 0}
        )

    def test_final_atomic_mode_divergence_is_rejected(self) -> None:
        env = self.environment_with_mode()
        mismatches = (
            {"mode": 5, "mode_name": "playing", "mode_timer": 0},
            {"mode": 1, "mode_name": "ghost_eaten_pause", "mode_timer": 0},
            {"mode": 1, "mode_name": "playing", "mode_timer": 1},
        )
        for final in mismatches:
            with self.subTest(final=final), self.assertRaisesRegex(
                PygameWorkerError, "does not match final atomic frame"
            ):
                env._audit_final_atomic_mode(final)

    def test_event_position_must_match_atomic_state(self) -> None:
        env = self.environment_with_mode()
        event = {
            "event_type": "normal_pellet_eaten",
            "logic_frame": 17,
            "pacman_position": [12, 0],
        }
        substep = {
            "pacman_position": [12, 19],
            "height": 25,
            "width": 21,
        }
        with self.assertRaisesRegex(
            PygameWorkerError,
            "event_type='normal_pellet_eaten'.*event_position=\\[12, 0\\]",
        ):
            env._audit_logic_frame_event_position(event, substep)

    def test_matching_event_position_is_accepted(self) -> None:
        env = self.environment_with_mode()
        event = {
            "event_type": "normal_pellet_eaten",
            "logic_frame": 17,
            "pacman_position": [12, 19],
        }
        substep = {
            "pacman_position": [12, 19],
            "height": 25,
            "width": 21,
        }
        env._audit_logic_frame_event_position(event, substep)


if __name__ == "__main__":
    unittest.main()
