from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from pacman_env.env import (
    Action,
    EnvironmentClosedError,
    EpisodeFinishedError,
    EpisodeNotStartedError,
    InvalidActionError,
    InvalidConfigurationError,
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
    Position,
    load_bundled_level,
    nearest_reachable_distance,
    route_to_nearest,
    transition,
)
from pacman_env.env.level import GHOST_DOOR
from pacman_env.env._pygame_worker import _PygameBridge
from pacman_env.planner import EdwardPlanner


ROOT = Path(__file__).resolve().parents[2]
PACMAN_PYTHON_ROOT = Path(
    os.getenv("MAAPACMAN_PACMAN_ROOT")
    or os.getenv("MAAPACMAN_PACMAN_PYTHON_ROOT")
    or ROOT.parent / "pacman-python"
).resolve()
EXPECTED_LEVEL_REVISION = (
    "36116c17c6c0805fdb1a07216357ac64c88d2c3108a0e37dce2a01b4ea2a8b97"
)


class PygameWorkerEventTests(unittest.TestCase):
    @staticmethod
    def atomic_state(
        *, score: int, fruit_active: bool, ghost_state: str, mode: int = 1
    ) -> dict[str, object]:
        return {
            "score": score,
            "normal_pellets_remaining": 10,
            "power_pellets_remaining": 2,
            "fruit": {"active": fruit_active},
            "ghosts": [{"id": 0, "state": ghost_state}],
            "pacman_position": [20, 10],
            "edible_ticks": 31,
            "frame": 80,
            "mode": mode,
        }

    def test_worker_records_ghost_event_with_required_source_fields(self) -> None:
        previous = self.atomic_state(
            score=100, fruit_active=False, ghost_state="vulnerable"
        )
        current = self.atomic_state(
            score=300, fruit_active=False, ghost_state="eyes"
        )
        source_events = [
            {
                "frame_index": 80,
                "type": "ghost_eaten",
                "pacman_position": [20, 10],
                "ghost_id": 0,
                "score_delta": 200,
                "ghost_state": "eyes",
                "edible_ticks": 31,
                "frame_score_delta": 200,
            }
        ]
        components, events = _PygameBridge._consume_source_events(
            source_events, previous, current, logic_frame_index=7
        )
        self.assertEqual(components["ghost"], 200)
        self.assertEqual(
            events,
            [
                {
                    "logic_frame_index": 7,
                    "logic_frame": 80,
                    "event_type": "ghost_eaten",
                    "pacman_position": [20, 10],
                    "ghost_id": 0,
                    "score_delta": 200,
                    "post_ghost_state": "eyes",
                    "edible_ticks": 31,
                }
            ],
        )

    def test_worker_does_not_report_expired_fruit_as_eaten(self) -> None:
        previous = self.atomic_state(
            score=100, fruit_active=True, ghost_state="normal"
        )
        current = self.atomic_state(
            score=100, fruit_active=False, ghost_state="normal"
        )
        components, events = _PygameBridge._consume_source_events(
            [], previous, current, logic_frame_index=3
        )
        self.assertEqual(components["fruit"], 0)
        self.assertFalse(any(event["event_type"] == "fruit_eaten" for event in events))

    def test_worker_rejects_unledgered_score_change(self) -> None:
        previous = self.atomic_state(
            score=100, fruit_active=False, ghost_state="normal"
        )
        current = self.atomic_state(
            score=110, fruit_active=False, ghost_state="normal"
        )
        with self.assertRaisesRegex(RuntimeError, "do not reconcile"):
            _PygameBridge._consume_source_events(
                [], previous, current, logic_frame_index=1
            )

    def test_original_life_mode_waits_for_respawn_without_model_step(self) -> None:
        bridge = object.__new__(_PygameBridge)
        bridge._episode_life_mode = "original_three_lives"
        request = {"death_seen": True, "logic_frames": 180}
        self.assertFalse(bridge._step_is_complete(request, {"mode": 2}))
        self.assertFalse(bridge._step_is_complete(request, {"mode": 4}))
        self.assertTrue(bridge._step_is_complete(request, {"mode": 1}))
        self.assertTrue(bridge._step_is_complete(request, {"mode": 3}))


@unittest.skipUnless(
    importlib.util.find_spec("pygame") is not None and PACMAN_PYTHON_ROOT.is_dir(),
    "requires pygame and a configured or sibling pacman-python checkout",
)
class PygamePacmanEnvTests(unittest.TestCase):
    def make_env(self, **overrides: object) -> PygamePacmanEnv:
        return PygamePacmanEnv(
            PygamePacmanEnvConfig(
                pacman_python_root=PACMAN_PYTHON_ROOT,
                video_driver="dummy",
                audio_driver="dummy",
                **overrides,
            )
        )

    def test_reset_is_headless_and_exports_v3_state_and_versions(self) -> None:
        env = self.make_env()
        try:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DISPLAY", None)
                frame, info = env.reset(seed=7)
            self.assertEqual(frame.shape, (400, 336, 3))
            self.assertEqual(str(frame.dtype), "uint8")
            self.assertEqual(
                info["pacman_python_revision"], env.pacman_python_revision
            )
            self.assertEqual(env.spec.api_version, "3.0")
            self.assertEqual(
                env.spec.env_id, "pacman-python-level1-ghostdoor-v3"
            )
            self.assertEqual(len(env.spec.ruleset_revision), 64)
            self.assertEqual(env.spec.level_revision, EXPECTED_LEVEL_REVISION)
            self.assertEqual(info["video_driver"], "dummy")
            self.assertEqual(info["seed"], 7)
            self.assertEqual(info["env_api_version"], "3.0")
            self.assertEqual(info["env_id"], env.spec.env_id)
            self.assertEqual(info["ruleset_revision"], env.spec.ruleset_revision)
            self.assertEqual(
                env.provenance,
                {
                    key: value
                    for key, value in info["version_metadata"].items()
                    if key != "seed"
                },
            )
            self.assertEqual(
                info["version_metadata"],
                {
                    "pacman_python_commit": env.pacman_python_revision,
                    "pacman_python_source_sha256": info[
                        "pacman_python_source_sha256"
                    ],
                    "pacman_python_dirty": info["pacman_python_dirty"],
                    "maapacman_commit": info["maapacman_revision"],
                    "maapacman_env_source_sha256": info[
                        "maapacman_env_source_sha256"
                    ],
                    "maapacman_dirty": info["maapacman_dirty"],
                    "level": 1,
                    "ghost_mode": "normal",
                    "episode_life_mode": "single_death",
                    "level_revision": EXPECTED_LEVEL_REVISION,
                    "env_id": env.spec.env_id,
                    "api_version": "3.0",
                    "ruleset_revision": env.spec.ruleset_revision,
                    "seed": 7,
                },
            )
            self.assertEqual(len(info["pacman_python_source_sha256"]), 64)
            self.assertEqual(len(info["maapacman_env_source_sha256"]), 64)
            self.assertIsInstance(info["pacman_python_dirty"], bool)
            self.assertIsInstance(info["maapacman_dirty"], bool)
            self.assertEqual(info["state"], env.snapshot())
            json.dumps(info["state"])
            self.assertTrue(info["state"]["ghost_door"]["pacman_blocked"])
            self.assertFalse(info["state"]["ghost_door"]["ghost_blocked"])
            self.assertEqual(info["pellets_remaining"], 196)
            self.assertEqual(info["normal_pellets_remaining"], 192)
            self.assertEqual(info["power_pellets_remaining"], 4)
            self.assertEqual(len(info["ghosts"]), 4)
            self.assertEqual(info["ghost_value"], 0)
            self.assertEqual(info["fruit_timer"], 0)
            self.assertEqual(info["fruit_score_ticks"], 0)
            self.assertEqual(len(info["state"]["pacman_pixel_position"]), 2)
            self.assertEqual(len(info["state"]["pacman_velocity"]), 2)
            for ghost_id, ghost in enumerate(info["ghosts"]):
                self.assertEqual(ghost["id"], ghost_id)
                self.assertEqual(ghost["state"], "normal")
                self.assertEqual(len(ghost["position"]), 2)
                self.assertEqual(len(ghost["pixel_position"]), 2)
                self.assertEqual(len(ghost["velocity"]), 2)
                self.assertIn(ghost["direction"], {"U", "D", "L", "R", "S"})
                self.assertIsInstance(ghost["path_remaining"], str)
                self.assertTrue(ghost["path_found"])
                self.assertEqual(len(ghost["path_target"]), 2)
                self.assertEqual(len(ghost["home_position"]), 2)
            self.assertEqual(
                set(info["fruit"]),
                {
                    "active",
                    "position",
                    "pixel_position",
                    "velocity",
                    "speed",
                    "path_remaining",
                    "path_found",
                    "path_target",
                    "slow_timer",
                    "bounce_counter",
                    "bounce_offset",
                    "type",
                },
            )
            level = load_bundled_level(1)
            self.assertEqual(
                tuple(info["pacman_position"]),
                (level.pacman_start.row, level.pacman_start.col),
            )
        finally:
            env.close()

    def test_environment_variable_selects_pacman_checkout(self) -> None:
        with patch.dict(
            os.environ,
            {"MAAPACMAN_PACMAN_ROOT": str(PACMAN_PYTHON_ROOT)},
        ):
            env = PygamePacmanEnv()
        try:
            self.assertTrue(env.pacman_python_revision)
            self.assertIsNone(env.worker_runtime_dir)
            self.assertEqual(env.provenance["level"], 1)
            self.assertEqual(env.provenance["api_version"], "3.0")
        finally:
            env.close()

    def test_navigation_matches_original_ghost_door_collision_rule(self) -> None:
        level = load_bundled_level(1)
        ghost_door = Position(11, 10)
        ghost_box_center = Position(12, 10)
        self.assertEqual(level.tile_at(ghost_door), GHOST_DOOR)
        self.assertTrue(level.is_wall(ghost_door, actor="pacman"))
        for actor in ("ghost", "vulnerable", "eyes"):
            self.assertFalse(level.is_wall(ghost_door, actor=actor))
        with self.assertRaisesRegex(ValueError, "unknown maze actor"):
            level.is_wall(ghost_door, actor="fruit")
        self.assertIsNone(transition(level, ghost_box_center, Action.UP))
        route = route_to_nearest(level, level.pacman_start, level.pellets)
        self.assertEqual(
            len(route),
            nearest_reachable_distance(
                level, level.pacman_start, level.pellets
            ),
        )

    def test_real_worker_horizontal_portal_matches_level_topology(self) -> None:
        level = load_bundled_level(1)
        portal_approach = Position(12, 1)
        route = route_to_nearest(
            level, level.pacman_start, {portal_approach}
        )
        env = self.make_env(max_steps=len(route) + 1)
        try:
            env.reset(seed=0)
            for action in route:
                _, _, terminated, truncated, info = env.step(action.value)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            self.assertEqual(
                tuple(info["pacman_position"]),
                (portal_approach.row, portal_approach.col),
            )
            expected = level.portal_exit(Position(12, 0), "L")
            _, _, terminated, truncated, portal_info = env.step("L")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(
                tuple(portal_info["pacman_position"]),
                (expected.row, expected.col),
            )
        finally:
            env.close()

    def test_stay_and_blocked_actions_advance_exactly_sixteen_frames(self) -> None:
        env = self.make_env()
        try:
            _, _ = env.reset(seed=3)
            start = env.snapshot()
            _, stay_reward, terminated, truncated, stay_info = env.step("S")
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(stay_reward, 0.0)
            self.assertEqual(stay_info["pacman_position"], [start["row"], start["col"]])
            self.assertEqual(stay_info["logic_frames"], 16)
            self.assertEqual(
                stay_info["logic_frame"],
                stay_info["atomic_substeps"][-1]["frame"],
            )
            self.assertEqual(len(stay_info["atomic_substeps"]), 16)
            self.assertEqual(
                [item["logic_frame_index"] for item in stay_info["atomic_substeps"]],
                list(range(1, 17)),
            )
            for substep in stay_info["atomic_substeps"]:
                self.assertEqual(len(substep["ghosts"]), 4)
                self.assertIn("direction", substep["ghosts"][0])
                self.assertIn("path_remaining", substep["ghosts"][0])
                self.assertIn("path_target", substep["ghosts"][0])
                self.assertIn("inside_ghost_house", substep["ghosts"][0])
                self.assertIn("fruit", substep)
                self.assertIn("path_remaining", substep["fruit"])
                self.assertIn("fruit_timer", substep)
                self.assertIn("ghost_value", substep)
                self.assertIn("open", substep)
                self.assertIn("blocked", substep)
                self.assertEqual(
                    substep["score_components"]["total"],
                    substep["score_delta"],
                )
                self.assertEqual(
                    sum(event["score_delta"] for event in substep["events"]),
                    substep["score_delta"],
                )

            blocked = env.snapshot()["blocked"][0]
            before_blocked = env.snapshot()
            _, _, terminated, _, blocked_info = env.step(blocked)
            self.assertFalse(terminated)
            self.assertTrue(blocked_info["wall_collision"])
            self.assertEqual(
                blocked_info["pacman_position"],
                [before_blocked["row"], before_blocked["col"]],
            )
            self.assertEqual(blocked_info["logic_frames"], 16)
            self.assertEqual(len(blocked_info["atomic_substeps"]), 16)
        finally:
            env.close()

    def test_seed_controls_ghost_trajectory(self) -> None:
        def rollout(seed: int) -> tuple[str, list[list[dict[str, object]]]]:
            env = self.make_env()
            try:
                frame, _ = env.reset(seed=seed)
                traces: list[list[dict[str, object]]] = []
                for _ in range(4):
                    frame, _, terminated, truncated, info = env.step("S")
                    traces.append(info["ghosts"])
                    if terminated or truncated:
                        break
                return hashlib.sha256(frame.tobytes()).hexdigest(), traces
            finally:
                env.close()

        first = rollout(7)
        second = rollout(7)
        different = rollout(8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_real_worker_all_ghosts_exit_and_normal_ghosts_do_not_reenter(self) -> None:
        env = self.make_env(max_steps=64)
        first_outside: dict[int, int | None] = {}
        normal_reentries: list[tuple[int, int]] = []
        try:
            env.reset(seed=0)
            for ghost in env.snapshot()["ghosts"]:
                first_outside[int(ghost["id"])] = (
                    0 if not ghost["inside_ghost_house"] else None
                )
            for _ in range(64):
                _, _, terminated, truncated, info = env.step("S")
                for substep in info["atomic_substeps"]:
                    for ghost in substep["ghosts"]:
                        ghost_id = int(ghost["id"])
                        if not ghost["inside_ghost_house"]:
                            if first_outside[ghost_id] is None:
                                first_outside[ghost_id] = int(substep["frame"])
                        elif (
                            first_outside[ghost_id] is not None
                            and ghost["state"] == "normal"
                        ):
                            normal_reentries.append(
                                (int(substep["frame"]), ghost_id)
                            )
                if terminated or truncated:
                    break
            self.assertTrue(all(frame is not None for frame in first_outside.values()))
            self.assertEqual(normal_reentries, [])
        finally:
            env.close()

    def test_real_worker_eyes_return_revive_and_exit_again(self) -> None:
        env = self.make_env(max_steps=96)
        planner = EdwardPlanner()
        tracked_ghost: int | None = None
        saw_eyes = saw_revived_in_pen = saw_reexit = False
        try:
            env.reset(seed=0)
            for _ in range(96):
                decision = planner.decide(env.snapshot())
                _, _, terminated, truncated, info = env.step(decision.action)
                for substep in info["atomic_substeps"]:
                    for event in substep["events"]:
                        if (
                            tracked_ghost is None
                            and event["event_type"] == "ghost_eaten"
                        ):
                            tracked_ghost = int(event["ghost_id"])
                            self.assertEqual(event["post_ghost_state"], "eyes")
                    if tracked_ghost is None:
                        continue
                    ghost = next(
                        item
                        for item in substep["ghosts"]
                        if int(item["id"]) == tracked_ghost
                    )
                    if ghost["state"] == "eyes":
                        saw_eyes = True
                    elif (
                        saw_eyes
                        and ghost["state"] == "normal"
                        and ghost["inside_ghost_house"]
                    ):
                        saw_revived_in_pen = True
                        self.assertEqual(float(ghost["speed"]), 1.0)
                    elif (
                        saw_revived_in_pen
                        and ghost["state"] == "normal"
                        and not ghost["inside_ghost_house"]
                    ):
                        saw_reexit = True
                        break
                if saw_reexit or terminated or truncated:
                    break
            self.assertIsNotNone(tracked_ghost)
            self.assertTrue(saw_eyes)
            self.assertTrue(saw_revived_in_pen)
            self.assertTrue(saw_reexit)
        finally:
            env.close()

    def test_power_pellet_has_exactly_360_visible_vulnerable_ticks(self) -> None:
        env = self.make_env(max_steps=96)
        planner = EdwardPlanner()
        timer_trace: list[int] = []
        acquired = False
        terminal_before_expiry = False
        try:
            env.reset(seed=0)
            for _ in range(96):
                decision = planner.decide(env.snapshot())
                _, _, terminated, truncated, info = env.step(decision.action)
                for substep in info["atomic_substeps"]:
                    power_event = any(
                        event["event_type"] == "power_pellet_eaten"
                        for event in substep["events"]
                    )
                    if power_event and not acquired:
                        acquired = True
                        self.assertEqual(substep["edible_ticks"], 360)
                        self.assertTrue(
                            all(
                                ghost["state"] in {"vulnerable", "eyes"}
                                for ghost in substep["ghosts"]
                            )
                        )
                    if acquired:
                        timer_trace.append(int(substep["edible_ticks"]))
                        if substep["edible_ticks"] == 0:
                            self.assertFalse(
                                any(
                                    ghost["state"] == "vulnerable"
                                    for ghost in substep["ghosts"]
                                )
                            )
                            break
                if timer_trace and timer_trace[-1] == 0:
                    break
                if terminated or truncated:
                    terminal_before_expiry = True
                    break
            self.assertTrue(acquired)
            self.assertFalse(terminal_before_expiry)
            self.assertEqual(timer_trace, list(range(360, -1, -1)))
        finally:
            env.close()

    def test_real_worker_emits_source_pellet_event_and_exact_score(self) -> None:
        env = self.make_env()
        try:
            env.reset(seed=3)
            pellet_events: list[dict[str, object]] = []
            reward_total = 0.0
            for action in ("L", "L", "L"):
                _, reward, terminated, truncated, info = env.step(action)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                reward_total += reward
                pellet_events.extend(
                    event
                    for event in info["logic_frame_events"]
                    if event["event_type"] == "normal_pellet_eaten"
                )
            self.assertEqual(reward_total, 10.0)
            self.assertEqual(len(pellet_events), 1)
            self.assertEqual(pellet_events[0]["score_delta"], 10)
            self.assertEqual(
                set(pellet_events[0]),
                {
                    "logic_frame_index",
                    "logic_frame",
                    "event_type",
                    "pacman_position",
                    "ghost_id",
                    "score_delta",
                    "post_ghost_state",
                    "edible_ticks",
                },
            )
        finally:
            env.close()

    def test_first_lethal_contact_terminates_without_respawn(self) -> None:
        env = self.make_env(max_steps=128)
        try:
            env.reset(seed=11)
            final_info = None
            for _ in range(64):
                _, _, terminated, truncated, info = env.step("S")
                if terminated or truncated:
                    final_info = info
                    break
            self.assertIsNotNone(final_info)
            assert final_info is not None
            self.assertTrue(final_info["terminated"])
            self.assertFalse(final_info["truncated"])
            self.assertTrue(final_info["death"])
            self.assertEqual(final_info["terminal_reason"], "death")
            self.assertEqual(
                final_info["lives_after_step"], final_info["lives"] - 1
            )
            self.assertLessEqual(final_info["logic_frames"], 16)
            with self.assertRaises(EpisodeFinishedError):
                env.step("S")
        finally:
            env.close()

    def test_original_three_lives_respawns_then_fourth_death_is_game_over(self) -> None:
        env = self.make_env(
            max_steps=512, episode_life_mode="original_three_lives"
        )
        try:
            env.reset(seed=11)
            death_infos = []
            terminated = truncated = False
            for _ in range(512):
                _, _, terminated, truncated, info = env.step("S")
                if info["death"]:
                    death_infos.append(info)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(len(death_infos), 4)
            for index, death_info in enumerate(death_infos[:3], 1):
                self.assertEqual(death_info["death_count"], index)
                self.assertTrue(death_info["respawned"])
                self.assertFalse(death_info["terminated"])
                self.assertIsNone(death_info["terminal_reason"])
                self.assertEqual(
                    death_info["lives_after_step"], death_info["lives"] - 1
                )
            final = death_infos[-1]
            self.assertEqual(final["death_count"], 4)
            self.assertFalse(final["respawned"])
            self.assertEqual(final["lives_after_step"], 0)
            self.assertEqual(final["terminal_reason"], "game_over")
        finally:
            env.close()

    def test_adjacent_exchange_collision_is_detected_on_first_subframe(self) -> None:
        opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
        action_for_delta = {
            (-1, 0): "U",
            (1, 0): "D",
            (0, -1): "L",
            (0, 1): "R",
        }
        found_exchange = False
        for seed in range(16):
            env = self.make_env(max_steps=128)
            try:
                env.reset(seed=seed)
                for _ in range(48):
                    state = env.snapshot()
                    pacman = (int(state["row"]), int(state["col"]))
                    open_actions = set(state["open"])
                    exchange = None
                    for ghost in state["ghosts"]:
                        if ghost["state"] != "normal":
                            continue
                        ghost_position = tuple(int(value) for value in ghost["position"])
                        delta = (
                            ghost_position[0] - pacman[0],
                            ghost_position[1] - pacman[1],
                        )
                        action = action_for_delta.get(delta)
                        if (
                            action in open_actions
                            and ghost["direction"] == opposite[action]
                        ):
                            exchange = (action, ghost_position)
                            break
                    if exchange is not None:
                        action, ghost_position = exchange
                        _, _, terminated, truncated, info = env.step(action)
                        self.assertTrue(terminated)
                        self.assertFalse(truncated)
                        self.assertTrue(info["death"])
                        self.assertEqual(info["logic_frames"], 1)
                        self.assertEqual(
                            tuple(info["atomic_substeps"][0]["pacman_position"]),
                            ghost_position,
                        )
                        found_exchange = True
                        break
                    _, _, terminated, truncated, _ = env.step("S")
                    if terminated or truncated:
                        break
            finally:
                env.close()
            if found_exchange:
                break
        self.assertTrue(found_exchange, "no deterministic adjacent exchange found")

    def test_ghost_event_and_score_components_are_auditable(self) -> None:
        env = self.make_env(max_steps=160)
        planner = EdwardPlanner()
        ghost_event = None
        try:
            env.reset(seed=0)
            for _ in range(160):
                decision = planner.decide(env.snapshot())
                _, reward, terminated, truncated, info = env.step(decision.action)
                components = info["score_components"]
                self.assertEqual(components["total"], int(reward))
                self.assertEqual(
                    components["total"],
                    components["normal_pellet"]
                    + components["power_pellet"]
                    + components["ghost"]
                    + components["fruit"]
                    + components["other"],
                )
                self.assertFalse(info["option_invalidated"])
                self.assertEqual(
                    sum(
                        substep["score_components"]["total"]
                        for substep in info["atomic_substeps"]
                    ),
                    int(reward),
                )
                self.assertEqual(
                    info["pygame_mode"], info["atomic_substeps"][-1]["mode"]
                )
                for event in info["logic_frame_events"]:
                    self.assertTrue(
                        {
                            "logic_frame_index",
                            "event_type",
                            "pacman_position",
                            "ghost_id",
                            "score_delta",
                            "post_ghost_state",
                            "edible_ticks",
                        }.issubset(event)
                    )
                if info["ghost_eaten"]:
                    ghost_event = info
                    break
                if terminated or truncated:
                    break
        finally:
            env.close()
        self.assertIsNotNone(ghost_event)
        assert ghost_event is not None
        self.assertIn("ghost_eaten", ghost_event["events"])
        self.assertGreaterEqual(ghost_event["ghosts_eaten_step"], 1)
        self.assertGreater(ghost_event["score_components"]["ghost"], 0)

    def test_seed0_planner_clear_preserves_atomic_post_state_alignment(self) -> None:
        env = self.make_env(max_steps=512)
        planner = EdwardPlanner()
        final_info = None
        steps = 0
        try:
            env.reset(seed=0)
            for steps in range(1, 513):
                decision = planner.decide(env.snapshot())
                _, _, terminated, truncated, info = env.step(decision.action)
                final_atomic = info["atomic_substeps"][-1]
                self.assertEqual(info["pygame_mode"], final_atomic["mode"])
                self.assertEqual(info["edible_ticks"], final_atomic["edible_ticks"])
                self.assertEqual(info["ghosts"], final_atomic["ghosts"])
                self.assertEqual(
                    info["normal_pellets_remaining"],
                    final_atomic["normal_pellets_remaining"],
                )
                if terminated or truncated:
                    final_info = info
                    break
        finally:
            env.close()
        self.assertIsNotNone(final_info)
        assert final_info is not None
        self.assertEqual(steps, 389)
        self.assertTrue(final_info["terminated"])
        self.assertFalse(final_info["truncated"])
        self.assertEqual(final_info["terminal_reason"], "all_normal_pellets")
        self.assertEqual(final_info["normal_pellets_remaining"], 0)

    def test_invalid_action_configuration_and_lifecycle_errors(self) -> None:
        with self.assertRaises(InvalidConfigurationError):
            self.make_env(max_steps=0)
        with self.assertRaises(InvalidConfigurationError):
            PygamePacmanEnv(
                PygamePacmanEnvConfig(
                    pacman_python_root=ROOT / "does-not-exist"
                )
            )
        env = self.make_env(max_steps=1)
        with self.assertRaises(EpisodeNotStartedError):
            env.render()
        try:
            with self.assertRaises(InvalidConfigurationError):
                env.reset(seed=True)
            env.reset(seed=0)
            with self.assertRaises(InvalidActionError):
                env.step("LEFT")
            _, _, terminated, truncated, _ = env.step("S")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            with self.assertRaises(EpisodeFinishedError):
                env.step("S")
        finally:
            env.close()
        with self.assertRaises(EnvironmentClosedError):
            env.reset()

    def test_worker_runtime_is_isolated_and_removed_on_close(self) -> None:
        source_script = PACMAN_PYTHON_ROOT / "pacman" / "pacman.pyw"
        source_hash = hashlib.sha256(source_script.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="maapacman-worker-test-") as base:
            env = self.make_env(worker_base_dir=base)
            env.reset(seed=5)
            runtime_dir = env.worker_runtime_dir
            self.assertIsNotNone(runtime_dir)
            assert runtime_dir is not None
            self.assertEqual(runtime_dir.parent, Path(base).resolve())
            self.assertEqual(
                (runtime_dir / "pacman.pyw").read_bytes(),
                source_script.read_bytes(),
            )
            self.assertTrue((runtime_dir / "res").is_dir())
            env.step("S")
            env.close()
            self.assertFalse(runtime_dir.exists())
        self.assertEqual(
            hashlib.sha256(source_script.read_bytes()).hexdigest(), source_hash
        )

    def test_four_parallel_workers_have_independent_runtime_and_state(self) -> None:
        sequences = ("LLS", "RRS", "S", "LLL")
        with tempfile.TemporaryDirectory(prefix="maapacman-workers-") as base:

            def run_worker(item: tuple[int, str]) -> dict[str, object]:
                seed, actions = item
                env = self.make_env(worker_base_dir=base)
                try:
                    frame, _ = env.reset(seed=seed)
                    runtime_dir = env.worker_runtime_dir
                    info = {"pacman_position": [env.snapshot()["row"], env.snapshot()["col"]]}
                    for action in actions:
                        frame, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                    return {
                        "runtime": runtime_dir,
                        "position": tuple(info["pacman_position"]),
                        "hash": hashlib.sha256(frame.tobytes()).hexdigest(),
                    }
                finally:
                    env.close()

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(run_worker, enumerate(sequences)))

            runtime_dirs = [result["runtime"] for result in results]
            self.assertEqual(len(set(runtime_dirs)), 4)
            self.assertTrue(all(not path.exists() for path in runtime_dirs))
            self.assertGreater(len({result["position"] for result in results}), 1)
            self.assertGreater(len({result["hash"] for result in results}), 1)


if __name__ == "__main__":
    unittest.main()
