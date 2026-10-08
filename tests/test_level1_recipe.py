from __future__ import annotations

import asyncio
import base64
import copy
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from pacman_env.env import (
    Action,
    Position,
    PygamePacmanEnv,
    load_bundled_level,
    route_to_nearest,
)
from pacman_env.planner import (
    EdwardPlanner,
    EdwardSafetyRefusal,
    PlannerCandidate,
)

from pacman_recipe.actions import ActionParseError, parse_action
from pacman_recipe.level1_dataset import (
    ENV_BACKEND,
    LONG_HORIZON_MAX_STEPS,
    PRODUCTION_MAX_STEPS,
    REPOSITORY_NAMES,
    SHORT_HORIZON_MAX_STEPS,
    STRESS_MAX_STEPS,
    environment_metadata,
    generate_balanced_corridor_rows,
    generate_episode_rows,
    generate_single_step_rows,
    make_episode_row,
    repository_revisions,
    validate_episode_row,
    write_jsonl,
)
from pacman_recipe.prompts import (
    EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT,
    LIVE_STATE_V3_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
    build_image_messages,
    encode_png,
    image_count,
    png_sha256,
)
from pacman_recipe.rewards import RewardConfig, audit_reward, shape_reward
from pacman_recipe.trajectories import audit_trajectory, summarize_episodes
from pacman_recipe.level1.token_constraints import (
    ObjectiveParseError,
    ObjectiveTokenConstraint,
)
from pacman_recipe.level1.workflow import (
    _nearest_reachable_distance_with_diagnostics,
    _normal_pellet_event_position,
)
from pacman_recipe.workflow import (
    ModelTurn,
    PacmanImageOnlyWorkflow,
    PacmanNativeVisionWorkflow,
)
from train_areal import _build_workflow_kwargs


def fake_action_tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        encode=lambda token, **_: [
            {"U": 40, "D": 41, "L": 42, "R": 43}[token]
        ]
    )


def test_normal_pellet_tracker_uses_source_event_position_after_respawn():
    info = {
        "pellet_eaten": True,
        "pacman_position": [23, 13],
        "logic_frame_events": [
            {
                "event_type": "normal_pellet_eaten",
                "pacman_position": [8, 11],
            }
        ],
    }
    assert _normal_pellet_event_position(info) == Position(8, 11)


class FakeObjectiveTokenizer:
    def encode(self, text, **_):
        return [ord(character) for character in text]

    def decode(self, token_ids, **_):
        return "".join(chr(token_id) for token_id in token_ids)


def successful_planner_baseline_actions() -> list[str]:
    env = PygamePacmanEnv()
    planner = EdwardPlanner()
    try:
        _, info = env.reset(seed=0)
        actions: list[str] = []
        terminated = truncated = False
        while not (terminated or truncated):
            decision = planner.decide(env.snapshot())
            _, _, terminated, truncated, info = env.step(decision.action)
            actions.append(decision.action)
        if (
            not terminated
            or truncated
            or info["terminal_reason"] != "all_normal_pellets"
        ):
            raise AssertionError(
                "Edward planner baseline did not clear level 1: "
                f"{info['terminal_reason']!r}"
            )
        return actions
    finally:
        env.close()


class OneStepEnv(PygamePacmanEnv):
    """Real original-pygame reset/step with a test-only one-step terminal."""

    def step(self, action):
        image, reward, _, _, info = super().step(action)
        info = dict(info)
        info.update(
            terminated=True,
            truncated=False,
            terminal_reason="test_complete",
        )
        return image, reward, True, False, info


class TwoStepEnv(PygamePacmanEnv):
    """Real original-pygame behavior with a test-only two-step terminal."""

    def step(self, action):
        image, reward, _, _, info = super().step(action)
        done = int(info["step"]) >= 2
        info = dict(info)
        info.update(
            terminated=done,
            truncated=False,
            terminal_reason="test_complete" if done else None,
        )
        return image, reward, done, False, info


class DatasetContractTests(unittest.TestCase):
    def test_long_horizon_256_step_rows_are_supported(self) -> None:
        row = make_episode_row(
            1,
            split="train",
            max_steps=LONG_HORIZON_MAX_STEPS,
        )
        validate_episode_row(row)
        self.assertEqual(row["env"]["max_steps"], 256)

    def test_stress_512_step_rows_are_supported_without_changing_256(self) -> None:
        stress = make_episode_row(
            1,
            split="train",
            max_steps=STRESS_MAX_STEPS,
        )
        long_horizon = make_episode_row(
            2,
            split="train",
            max_steps=LONG_HORIZON_MAX_STEPS,
        )
        validate_episode_row(stress)
        validate_episode_row(long_horizon)
        self.assertEqual(stress["env"]["max_steps"], 512)
        self.assertEqual(long_horizon["env"]["max_steps"], 256)

    def test_balanced_corridor_splits_are_disjoint_and_defeat_constant_actions(
        self,
    ) -> None:
        train = list(
            generate_balanced_corridor_rows(32, split="train", seed=0)
        )
        validation = list(
            generate_balanced_corridor_rows(16, split="validation", seed=0)
        )
        self.assertFalse(
            {row["id"] for row in train} & {row["id"] for row in validation}
        )
        for rows in (train, validation):
            self.assertTrue(
                all(
                    row["state_prefix_audit"]["prefix_verified_nonterminal"]
                    for row in rows
                )
            )
            for action in ("U", "D", "L", "R"):
                self.assertEqual(
                    sum(
                        action in row["state_open_actions_for_audit"]
                        for row in rows
                    ),
                    len(rows) // 2,
                )
            for start in range(0, len(rows), 4):
                batch = rows[start : start + 4]
                self.assertEqual(
                    [row["state_open_actions_for_audit"] for row in batch],
                    [["L", "R"], ["U", "D"], ["L", "R"], ["U", "D"]],
                )

    def test_single_step_rows_use_distinct_collision_free_prefix_states(self) -> None:
        rows = list(
            generate_single_step_rows(4, split="train", seed=0, offset=0)
        )
        self.assertTrue(all(row["decision_steps"] == 1 for row in rows))
        self.assertTrue(
            all(
                row["state_prefix_audit"]["prefix_verified_nonterminal"]
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["state_prefix_audit"]["source_terminal_reason"]
                in {"death", "all_normal_pellets"}
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["state_prefix_audit"]["successful_baseline"]
                is row["state_prefix_audit"]["source_cleared_level"]
                for row in rows
            )
        )
        self.assertEqual(
            [len(row["state_prefix_actions"]) for row in rows],
            [0, 1, 2, 3],
        )
        self.assertEqual(len({row["id"] for row in rows}), 4)

    def test_rows_are_deterministic_and_validate_revision(self) -> None:
        first = list(generate_episode_rows(3, split="train", seed=0))
        second = list(generate_episode_rows(3, split="train", seed=0))
        self.assertEqual(first, second)
        self.assertEqual(first[0]["id"], "level1-normal-seed0-train-0001")
        self.assertEqual(
            first[0]["env"]["name"], "pacman-python-level1-ghostdoor-v3"
        )
        self.assertEqual(first[0]["env"]["backend"], ENV_BACKEND)
        self.assertEqual(first[0]["env"]["max_steps"], PRODUCTION_MAX_STEPS)
        self.assertEqual(
            first[0]["env"]["pacman_python_revision"],
            environment_metadata()["pacman_python_revision"],
        )
        self.assertEqual(
            set(first[0]["source_revisions"]), set(REPOSITORY_NAMES)
        )
        self.assertEqual(
            first[0]["env"]["maapacman_revision"],
            first[0]["source_revisions"]["areal-pacman"]["commit"],
        )
        self.assertEqual(
            first[0]["env"]["maapacman_dirty"],
            first[0]["source_revisions"]["areal-pacman"]["dirty"],
        )
        self.assertEqual(first[0]["source_revisions"], repository_revisions())
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.jsonl"
            two = Path(directory) / "two.jsonl"
            self.assertEqual(write_jsonl(first, one), write_jsonl(second, two))
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_short_horizon_rows_are_valid(self) -> None:
        rows = list(
            generate_episode_rows(
                2,
                split="train",
                seed=0,
                max_steps=SHORT_HORIZON_MAX_STEPS,
            )
        )
        self.assertEqual(
            [row["env"]["max_steps"] for row in rows],
            [SHORT_HORIZON_MAX_STEPS, SHORT_HORIZON_MAX_STEPS],
        )

    def test_invalid_rows_are_rejected(self) -> None:
        row = make_episode_row(1, split="train")
        for key, value in (
            ("api_version", "1.0"),
            ("backend", "synthetic"),
            ("pacman_python_revision", "wrong"),
            ("level", 2),
            ("seed", True),
            ("max_steps", PRODUCTION_MAX_STEPS + 1),
            ("observation_mode", "text"),
        ):
            broken = copy.deepcopy(row)
            broken["env"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_episode_row(broken)

        broken = copy.deepcopy(row)
        broken["source_revisions"]["Pacman"] = {
            "commit": "0" * 40,
            "dirty": False,
        }
        with self.assertRaisesRegex(ValueError, "all three repositories"):
            validate_episode_row(broken)


class PromptAndActionTests(unittest.TestCase):
    def test_prompt_contains_one_png_and_no_privileged_dynamic_state(self) -> None:
        env = PygamePacmanEnv()
        image, _ = env.reset()
        png = encode_png(image)
        messages = build_image_messages(png)
        self.assertEqual(image_count(messages), 1)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1]["content"][0]["text"], USER_INSTRUCTION)
        combined = json.dumps(messages)
        for forbidden in ("legal_actions", "pellets_remaining", "pacman_position", "route"):
            self.assertNotIn(forbidden, combined)
        url = messages[1]["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), png)
        env.close()

    def test_png_encoding_is_deterministic(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(encode_png(image), encode_png(image.copy()))

    def test_live_state_v3_contains_authoritative_navigation_context(self) -> None:
        png = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
        context = {
            "pacman_position": [21, 14],
            "facing": "R",
            "pellets_remaining": 93,
            "open_actions": ["L", "R"],
            "legal_actions": ["L", "R", "S"],
            "blocked_actions": ["U", "D"],
            "current_cell_exit_history": ["L"],
            "current_cell_exit_counts": {"L": 17, "R": 2},
            "last_action": "R",
            "immediate_reverse_action": "L",
            "recent_actions": ["L", "R"],
            "recent_positions": [[21, 13], [21, 14]],
            "preferred_open_actions": ["R"],
        }
        messages = build_image_messages(
            png,
            prompt_style="live_state_v3",
            state_context=context,
        )
        self.assertEqual(
            messages[0],
            {"role": "system", "content": LIVE_STATE_V3_SYSTEM_PROMPT},
        )
        text = messages[1]["content"][0]["text"]
        for expected in (
            "row=21, col=14",
            "Facing: R",
            "Remaining pellets: 93",
            "OPEN dirs here: L, R",
            "BLOCKED dirs here: U, D",
            "directions already taken before: L",
            "Last move: R.",
            "Choose ONE ACTION from [L, R]",
        ):
            self.assertIn(expected, text)
        self.assertEqual(image_count(messages), 1)
        with self.assertRaisesRegex(ValueError, "requires state_context"):
            build_image_messages(png, prompt_style="live_state_v3")

    def test_action_parser_is_strict(self) -> None:
        for token in ("U", "D", "L", "R", "S"):
            self.assertEqual(parse_action(f" \n{token}\t").value, token)
        for invalid in ("RIGHT", "Action: R", '{"action":"R"}', "R R", "r", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ActionParseError):
                parse_action(invalid)


class RewardAndTrajectoryTests(unittest.TestCase):
    @staticmethod
    def event(
        event_type: str,
        score_delta: int,
        *,
        logic_frame_index: int = 1,
        ghost_id: int | None = None,
        post_ghost_state: str | None = None,
    ) -> dict[str, object]:
        return {
            "logic_frame_index": logic_frame_index,
            "logic_frame": logic_frame_index,
            "event_type": event_type,
            "pacman_position": [20, 10],
            "ghost_id": ghost_id,
            "score_delta": score_delta,
            "post_ghost_state": post_ghost_state,
            "edible_ticks": 0,
        }

    @classmethod
    def reward_info(
        cls,
        *events: dict[str, object],
        wall_collision: bool = False,
        **extra: object,
    ) -> dict[str, object]:
        return {
            "logic_frame_events": list(events),
            "wall_collision": wall_collision,
            **extra,
        }

    def test_unreachable_bfs_logs_debug_state_and_fails(self) -> None:
        level = load_bundled_level(1)
        start = Position(12, 10)
        targets = {Position(16, 9), Position(16, 11)}
        with (
            patch(
                "pacman_recipe.level1.episode.nearest_reachable_distance",
                side_effect=ValueError(
                    "no target is reachable from the requested position"
                ),
            ),
            self.assertLogs(
                "pacman_recipe.level1.episode",
                level="ERROR",
            ) as captured,
            self.assertRaisesRegex(
                RuntimeError,
                '"phase": "after_step"',
            ),
        ):
            _nearest_reachable_distance_with_diagnostics(
                level,
                start,
                targets,
                phase="after_step",
                previous_position=(12, 9),
                action="R",
                live_legal_actions=["L", "R"],
            )
        message = captured.output[0]
        self.assertIn('"start_position": [12, 10]', message)
        self.assertIn('"remaining_normal_pellet_count": 2', message)
        self.assertIn('"previous_position": [12, 9]', message)
        self.assertIn('"action": "R"', message)
        self.assertIn('"live_legal_actions": ["L", "R"]', message)

    def test_unreachable_bfs_can_truncate_a_trapped_rollout(self) -> None:
        level = load_bundled_level(1)
        start = Position(12, 10)
        targets = {Position(16, 9), Position(16, 11)}
        with (
            patch(
                "pacman_recipe.level1.episode.nearest_reachable_distance",
                side_effect=ValueError(
                    "no target is reachable from the requested position"
                ),
            ),
            self.assertLogs(
                "pacman_recipe.level1.episode",
                level="WARNING",
            ) as captured,
        ):
            distance = _nearest_reachable_distance_with_diagnostics(
                level,
                start,
                targets,
                phase="after_step",
                previous_position=(11, 10),
                action="D",
                live_legal_actions=["U", "L", "R", "S"],
                allow_unreachable=True,
            )
        self.assertIsNone(distance)
        self.assertIn("truncating trapped rollout", captured.output[0])
        self.assertIn('"start_position": [12, 10]', captured.output[0])

    def test_reward_formula_and_audit(self) -> None:
        pellet_event = self.event("normal_pellet_eaten", 10)
        result = shape_reward(
            10.0,
            {"pellet_clear_rate": 0.0},
            self.reward_info(pellet_event, pellet_clear_rate=1 / 196),
            RewardConfig(step_penalty=1.0, wall_penalty=1.0),
        )
        self.assertAlmostEqual(result.step_penalty, 1.0)
        self.assertAlmostEqual(result.shaped_reward, 9.0)
        audit_reward({**result.as_dict(), "logic_frame_events": [pellet_event]})
        wall = shape_reward(
            0.0,
            {"pellet_clear_rate": 0.0},
            self.reward_info(wall_collision=True, pellet_clear_rate=0.0),
            RewardConfig(step_penalty=1.0, wall_penalty=1.0),
        )
        self.assertAlmostEqual(wall.shaped_reward, -2.0)
        progress = shape_reward(
            0.0,
            {"pellet_clear_rate": 0.0},
            self.reward_info(pellet_clear_rate=0.0),
            RewardConfig(
                step_penalty=1.0,
                wall_penalty=1.0,
                nearest_pellet_alpha=1.0,
            ),
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertEqual(progress.nearest_pellet_progress_reward, 1.0)
        self.assertEqual(progress.nearest_pellet_progress_weight, 1.0)
        self.assertEqual(progress.shaped_reward, 0.0)
        audit_reward({**progress.as_dict(), "logic_frame_events": []})
        broken = result.as_dict()
        broken["shaped_reward"] = 99.0
        with self.assertRaises(ValueError):
            audit_reward(broken)

    def test_nearest_pellet_progress_scales_by_cleared_ratio(self) -> None:
        config = RewardConfig(
            step_penalty=0.05,
            nearest_pellet_alpha=0.1,
            nearest_pellet_scale_by_cleared_ratio=True,
        )
        early = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.75,
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        late = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.10,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=3,
        )

        self.assertAlmostEqual(early.nearest_pellet_progress_weight, 0.025)
        self.assertAlmostEqual(early.nearest_pellet_progress_reward, 0.025)
        self.assertAlmostEqual(early.shaped_reward, -0.025)
        self.assertAlmostEqual(late.nearest_pellet_progress_weight, 0.09)
        self.assertAlmostEqual(late.nearest_pellet_progress_reward, -0.09)
        self.assertAlmostEqual(late.shaped_reward, -0.14)
        audit_reward({**early.as_dict(), "logic_frame_events": []})
        audit_reward({**late.as_dict(), "logic_frame_events": []})
        broken_progress = early.as_dict()
        broken_progress["logic_frame_events"] = []
        broken_progress["nearest_pellet_progress_reward"] = 0.5
        with self.assertRaisesRegex(ValueError, "nearest-pellet reward audit"):
            audit_reward(broken_progress)

    def test_step_penalty_scales_by_pre_action_cleared_ratio(self) -> None:
        config = RewardConfig(
            step_penalty=0.05,
            step_penalty_cleared_ratio_scale=0.45,
            wall_penalty=0.5,
        )
        early = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.99,
            normal_pellet_remaining_ratio_before=1.0,
        )
        middle = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.49,
            normal_pellet_remaining_ratio_before=0.5,
        )
        late = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.0,
            normal_pellet_remaining_ratio_before=0.0,
        )

        self.assertAlmostEqual(early.step_penalty, 0.05)
        self.assertAlmostEqual(middle.step_penalty, 0.275)
        self.assertAlmostEqual(late.step_penalty, 0.5)
        self.assertAlmostEqual(early.shaped_reward, -0.05)
        self.assertAlmostEqual(middle.shaped_reward, -0.275)
        self.assertAlmostEqual(late.shaped_reward, -0.5)
        audit_reward({**early.as_dict(), "logic_frame_events": []})
        audit_reward({**middle.as_dict(), "logic_frame_events": []})
        audit_reward({**late.as_dict(), "logic_frame_events": []})

        with self.assertRaisesRegex(
            ValueError,
            "step_penalty_cleared_ratio_scale must be non-negative",
        ):
            RewardConfig(step_penalty_cleared_ratio_scale=-0.1)

    def test_explicit_task_and_late_bfs_distance_reward_formula(self) -> None:
        config = RewardConfig(
            use_base_reward=False,
            normal_pellet_reward=1.0,
            power_pellet_reward=1.0,
            ghost_reward=5.0,
            fruit_reward=2.0,
            death_penalty=25.0,
            completion_reward=50.0,
            step_penalty=0.05,
            wall_penalty=0.5,
            nearest_pellet_alpha=0.1,
            nearest_pellet_remaining_ratio_threshold=0.25,
            nearest_pellet_skip_on_eat=True,
        )
        ordinary = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(ordinary.shaped_reward, -0.05)
        self.assertEqual(ordinary.base_reward_contribution, 0.0)
        self.assertFalse(ordinary.nearest_pellet_shaping_active)

        closer = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.25,
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(closer.shaped_reward, 0.05)
        self.assertTrue(closer.nearest_pellet_shaping_active)

        farther = shape_reward(
            0.0,
            {},
            self.reward_info(),
            config,
            normal_pellet_remaining_ratio=0.2,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=3,
        )
        self.assertAlmostEqual(farther.shaped_reward, -0.15)
        audit_reward({**farther.as_dict(), "logic_frame_events": []})

        wall = shape_reward(
            0.0,
            {},
            self.reward_info(wall_collision=True),
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(wall.shaped_reward, -0.55)

        pellet = shape_reward(
            10.0,
            {},
            self.reward_info(self.event("normal_pellet_eaten", 10)),
            config,
            normal_pellet_remaining_ratio=0.2,
            nearest_pellet_distance_before=1,
            nearest_pellet_distance_after=12,
        )
        self.assertAlmostEqual(pellet.shaped_reward, 0.95)
        self.assertEqual(pellet.normal_pellet_reward, 1.0)
        self.assertFalse(pellet.nearest_pellet_shaping_active)
        self.assertEqual(pellet.nearest_pellet_progress_reward, 0.0)

        power_pellet = shape_reward(
            100.0,
            {},
            self.reward_info(self.event("power_pellet_eaten", 100)),
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(power_pellet.shaped_reward, 0.95)
        self.assertTrue(power_pellet.power_pellet_eaten)
        self.assertEqual(power_pellet.power_pellet_reward, 1.0)
        audit_reward({**power_pellet.as_dict(), "logic_frame_events": [self.event("power_pellet_eaten", 100)]})

        ghost = shape_reward(
            600.0,
            {},
            self.reward_info(
                self.event("ghost_eaten", 200, ghost_id=0, post_ghost_state="eyes"),
                self.event("ghost_eaten", 400, ghost_id=1, post_ghost_state="eyes"),
            ),
            config,
        )
        self.assertAlmostEqual(ghost.shaped_reward, 9.95)
        self.assertTrue(ghost.ghost_eaten)
        self.assertEqual(ghost.ghost_reward, 10.0)
        audit_reward({**ghost.as_dict(), "logic_frame_events": [
            self.event("ghost_eaten", 200, ghost_id=0, post_ghost_state="eyes"),
            self.event("ghost_eaten", 400, ghost_id=1, post_ghost_state="eyes"),
        ]})

        death = shape_reward(
            0.0,
            {},
            self.reward_info(self.event("death", 0)),
            config,
        )
        self.assertAlmostEqual(death.shaped_reward, -25.05)
        self.assertTrue(death.death)
        self.assertEqual(death.death_penalty, 25.0)
        audit_reward({**death.as_dict(), "logic_frame_events": [self.event("death", 0)]})

        completion = shape_reward(
            10.0,
            {},
            self.reward_info(
                self.event("normal_pellet_eaten", 10),
                self.event("level_cleared", 0),
                terminal_reason="all_normal_pellets",
                normal_pellets_remaining=0,
            ),
            config,
            normal_pellet_remaining_ratio=0.0,
            nearest_pellet_distance_before=1,
            nearest_pellet_distance_after=0,
        )
        self.assertAlmostEqual(completion.shaped_reward, 50.95)
        self.assertTrue(completion.level_completed)
        audit_reward({**completion.as_dict(), "logic_frame_events": [
            self.event("normal_pellet_eaten", 10),
            self.event("level_cleared", 0),
        ]})

    def test_reward_config_rejects_invalid_shaping_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "normal_pellet_reward"):
            RewardConfig(normal_pellet_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "power_pellet_reward"):
            RewardConfig(power_pellet_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "ghost_reward"):
            RewardConfig(ghost_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "fruit_reward"):
            RewardConfig(fruit_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "death_penalty"):
            RewardConfig(death_penalty=-1.0)
        with self.assertRaisesRegex(ValueError, "completion_reward"):
            RewardConfig(completion_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "safety_refusal_penalty"):
            RewardConfig(safety_refusal_penalty=-1.0)
        with self.assertRaisesRegex(ValueError, "ratio_threshold"):
            RewardConfig(
                nearest_pellet_remaining_ratio_threshold=1.1
            )

    def test_event_reward_supports_fruit_and_rejects_score_inference(self) -> None:
        fruit_event = self.event("fruit_eaten", 2500)
        result = shape_reward(
            2500.0,
            {},
            self.reward_info(fruit_event),
            RewardConfig(
                use_base_reward=False,
                fruit_reward=2.0,
                step_penalty=0.0,
            ),
        )
        self.assertTrue(result.fruit_eaten)
        self.assertEqual(result.fruit_reward, 2.0)
        self.assertEqual(result.shaped_reward, 2.0)
        audit_reward({**result.as_dict(), "logic_frame_events": [fruit_event]})
        with self.assertRaisesRegex(ValueError, "event score sum"):
            shape_reward(
                2500.0,
                {},
                self.reward_info(),
                RewardConfig(step_penalty=0.0),
            )

    def test_curriculum1_reward_config_contract(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "train"
            / "curriculum1.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "recipe_version: maapacman-level1-ghostdoor-v3",
            "trial_name: curriculum1-qwen3p5-9b-step512-100update-direct",
            "total_train_epochs: 5",
            "total_train_steps: null",
            "path: Qwen/Qwen3.5-9B",
            "validation_contract: sampled12_uniform_shaped",
            "reward_objective_contract: step_local_raw_v1",
            "use_base_reward: false",
            "normal_pellet_reward: 1.0",
            "power_pellet_reward: 1.0",
            "ghost_reward: 5.0",
            "fruit_reward: 0.0",
            "reward_recipe_version: maapacman-level1-event-reward-v3",
            "death_penalty: 100.0",
            "completion_reward: 50.0",
            "safety_refusal_penalty: 100.0",
            "gdn_prefill_backend: triton",
            "kl_logprob_source: proximal",
            "prox_logp_method: recompute",
            "ppo_n_minibatches: 1",
            "level: sequence",
            "action: mask",
            "agg: sum",
            "lower: 0.8",
            "upper: 1.25",
            "step_penalty: 0.05",
            "step_penalty_cleared_ratio_scale: 0.0",
            "wall_penalty: 0.5",
            "nearest_pellet_alpha: 0.1",
            "nearest_pellet_remaining_ratio_threshold: 1.0",
            "nearest_pellet_scale_by_cleared_ratio: true",
            "nearest_pellet_skip_on_eat: true",
            "edward_options: false",
            "objective_encoding: direct-action-token-v1",
            "action_token_choice: true",
            "open_action_mask: true",
            "max_new_tokens: 1",
            "temperature: ${actor.temperature}",
            "keep_last: 2",
            "keep_best_metric: ppo_actor/task_reward/avg",
            "keep_best_mode: max",
            "enable_offload: true",
            "max_tokens_per_mb: 1024",
            "offload: true",
            "artifacts/datasets/curriculum1-step512-v1/train_hf",
            "artifacts/datasets/curriculum1-step512-v1/validation_hf",
        ):
            self.assertIn(expected, config)
        actor_section = config.split("\nref:\n", 1)[0].split("\nactor:\n", 1)[1]
        ref_section = config.split("\nref:\n", 1)[1].split("\nvllm:\n", 1)[0]
        self.assertIn("\n  offload: true", "\n" + actor_section)
        self.assertIn("\n  reward_norm: null", "\n" + actor_section)
        self.assertIn("\n  reward_clip: .inf", "\n" + actor_section)
        self.assertIn("\n  adv_norm: null", "\n" + actor_section)
        self.assertIn("\n  offload: true", "\n" + ref_section)
        self.assertNotIn("revisit_penalty:", config)

    def test_training_recipes_include_the_c2_overfit_recipe(self) -> None:
        train_dir = (
            Path(__file__).parents[1] / "configs" / "level1" / "train"
        )
        self.assertEqual(
            sorted(path.name for path in train_dir.glob("*.yaml")),
            [
                "curriculum1.yaml",
                "curriculum2.yaml",
                "curriculum2_binary_overfit.yaml",
                "curriculum2_overfit.yaml",
                "curriculum2_risk_fallback_open.yaml",
                "curriculum2_single_death.yaml",
                "curriculum2_three_lives.yaml",
            ],
        )

    def test_curriculum2_cold_starts_from_qwen_base(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "train"
            / "curriculum2.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "total_train_epochs: 5",
            "total_train_steps: null",
            "path: Qwen/Qwen3.5-9B",
            "tokenizer_path: ${actor.path}",
            "model: ${actor.path}",
            "reward_objective_contract: episode_return_group_v1",
            "artifacts/datasets/curriculum2-step512-40seed-v1/train_hf",
            "artifacts/datasets/curriculum2-step512-40seed-v1/validation_hf",
            "ghost_mode: normal",
            "episode_life_mode: original_three_lives",
            "mode: disabled",
        ):
            self.assertIn(expected, config)
        self.assertNotIn("CURRICULUM1_CHECKPOINT", config)

    def test_summary_counts_acceptance_metrics(self) -> None:
        summary = summarize_episodes(
            [
                {
                    "pellet_clear_rate": 0.5,
                    "won": False,
                    "parse_failures": 0,
                    "canonical_action_violations": 0,
                    "steps": 10,
                    "total_base_reward": 5,
                    "total_shaped_reward": 9,
                    "final_score": 50,
                    "wall_collisions": 2,
                    "oscillation_returns": 1,
                    "normal_pellets_eaten": 97,
                    "normal_pellet_clear_rate": 0.5,
                    "normal_pellets_remaining": 97,
                    "power_pellets_remaining": 2,
                    "trajectory": [
                        {"action": "R", "base_reward": 0.0},
                        {"action": "R", "base_reward": 10.0},
                        {"action": "S", "base_reward": 0.0},
                    ],
                }
            ]
        )
        self.assertEqual(summary["average_pellet_clear_rate"], 0.5)
        self.assertEqual(summary["parse_failures"], 0)
        self.assertEqual(summary["average_final_score"], 50)
        self.assertEqual(summary["average_normal_pellets_eaten"], 97)
        self.assertEqual(summary["average_normal_pellet_clear_rate"], 0.5)
        self.assertEqual(summary["average_wall_hit_rate"], 0.2)
        self.assertEqual(summary["average_oscillation_rate"], 0.1)
        self.assertEqual(summary["action_counts"], {"U": 0, "D": 0, "L": 0, "R": 2, "S": 1})
        self.assertEqual(summary["max_no_progress_streak"], 1)


class WorkflowContractTests(unittest.TestCase):
    def test_native_workflow_is_direct_areal_rollout_workflow(self) -> None:
        from areal.api import RolloutWorkflow

        self.assertTrue(issubclass(PacmanNativeVisionWorkflow, RolloutWorkflow))

    def test_single_step_row_replays_prefix_then_scores_one_action(self) -> None:
        row = list(
            generate_single_step_rows(1, split="train", seed=0, offset=3)
        )[0]
        workflow = PacmanImageOnlyWorkflow(
            env_factory=PygamePacmanEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
            scripted_actions=["U"],
            scripted_completion_ids=["single-step-id"],
        )
        result = asyncio.run(workflow.run(row))
        self.assertEqual(set(result), {"single-step-id"})
        self.assertEqual(workflow.last_episode["steps"], 1)
        self.assertEqual(
            workflow.last_episode["state_prefix_actions"],
            row["state_prefix_actions"],
        )
        payload = workflow.last_episode
        assert payload is not None
        self.assertEqual(
            payload["state_prefix_actions_executed"],
            len(row["state_prefix_actions"]),
        )
        self.assertEqual(
            len(payload["state_prefix_evidence"]),
            len(row["state_prefix_actions"]),
        )
        self.assertGreater(payload["prefix_end_score"], 0)
        self.assertGreater(payload["prefix_end_logic_frame"], 0)
        self.assertEqual(
            payload["prefix_end_score"],
            sum(item["score_delta"] for item in payload["state_prefix_evidence"]),
        )
        self.assertEqual(
            payload["prefix_end_logic_frame"],
            sum(item["logic_frames"] for item in payload["state_prefix_evidence"]),
        )
        audit_trajectory(payload)
        for field in ("prefix_end_score", "prefix_end_logic_frame"):
            corrupted = copy.deepcopy(payload)
            corrupted[field] = 0
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                audit_trajectory(corrupted)
        self.assertEqual(
            workflow.last_episode["terminal_reason"],
            "single_step_complete",
        )

    @staticmethod
    def _fake_native_processor():
        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return "processed prompt"

            def __call__(self, *, text, images, **kwargs):
                pixel_mean = float(np.asarray(images[0]).mean())
                return {
                    "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
                    "mm_token_type_ids": torch.tensor(
                        [[0, 1]], dtype=torch.long
                    ),
                    "pixel_values": torch.tensor(
                        [[pixel_mean]], dtype=torch.float32
                    ),
                    "image_grid_thw": torch.tensor(
                        [[1, 1, 1]], dtype=torch.long
                    ),
                }

        return FakeProcessor()

    def test_native_vision_processor_preserves_real_image_tensors(self) -> None:
        processor = self._fake_native_processor()
        workflow = PacmanNativeVisionWorkflow(
            gconfig=SimpleNamespace(),
            tokenizer=SimpleNamespace(
                encode=lambda token, **_: [
                    {"U": 40, "D": 41, "L": 42, "R": 43}[token]
                ]
            ),
            processor=processor,
            env_factory=OneStepEnv,
        )
        black = build_image_messages(
            encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
        )
        white = build_image_messages(
            encode_png(np.full((4, 4, 3), 255, dtype=np.uint8))
        )
        _, _, black_processed, black_ids = workflow._process_messages(black)
        _, _, white_processed, white_ids = workflow._process_messages(white)

        self.assertEqual(black_ids, [10, 11])
        self.assertEqual(white_ids, [10, 11])
        self.assertFalse(
            torch.equal(
                black_processed["pixel_values"],
                white_processed["pixel_values"],
            )
        )
        self.assertIs(processor.template_kwargs["enable_thinking"], False)

    def test_native_vision_episode_returns_official_tensor_contract(self) -> None:
        processor = self._fake_native_processor()

        class FakeTokenizer:
            def encode(self, token, **kwargs):
                return [{"U": 40, "D": 41, "L": 42, "R": 43}[token]]

            def decode(self, token_ids, **kwargs):
                return "L"

        class FakeGConfig:
            n_samples = 12

            def new(self, **kwargs):
                self.last_kwargs = kwargs
                return self

        class FakeModelRequest(SimpleNamespace):
            pass

        class FakeResponse:
            input_tokens = [10, 11]
            output_tokens = [42]
            output_logprobs = [-0.25]
            output_versions = [7]
            input_len = 2
            output_len = 1
            stop_reason = "stop"

        class FakeEngine:
            async def agenerate(self, request):
                self.request = request
                return FakeResponse()

        fake_areal_api = SimpleNamespace(ModelRequest=FakeModelRequest)
        fake_areal_image = SimpleNamespace(
            image2base64=lambda image: ["encoded-image"]
        )
        fake_areal_data = SimpleNamespace(
            concat_padded_tensors=lambda samples: samples[0]
        )
        workflow = PacmanNativeVisionWorkflow(
            gconfig=FakeGConfig(),
            tokenizer=FakeTokenizer(),
            processor=processor,
            env_factory=OneStepEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
            action_token_choice=True,
            reward_objective_contract="episode_return_group_v1",
        )
        engine = FakeEngine()
        with patch.dict(
            sys.modules,
            {
                "areal.api": fake_areal_api,
                "areal.utils.image": fake_areal_image,
                "areal.utils.data": fake_areal_data,
            },
        ):
            result = asyncio.run(
                workflow.arun_episode(
                    engine,
                    make_episode_row(1, split="train"),
                )
            )

        self.assertEqual(
            set(result),
            {
                "input_ids",
                "mm_token_type_ids",
                "loss_mask",
                "pacman_action_mask_bits",
                "logprobs",
                "versions",
                "attention_mask",
                "rewards",
                "rollout_episode_ids",
                "rollout_episode_returns",
                "rollout_episode_group_sizes",
                "multi_modal_input",
            },
        )
        self.assertEqual(result["input_ids"].tolist(), [[10, 11, 42]])
        self.assertEqual(
            result["mm_token_type_ids"].tolist(), [[0, 1, 0]]
        )
        self.assertEqual(result["loss_mask"].tolist(), [[0, 0, 1]])
        self.assertEqual(
            result["pacman_action_mask_bits"].tolist(), [[0, 0, 15]]
        )
        self.assertEqual(result["logprobs"].tolist(), [[0.0, 0.0, -0.25]])
        self.assertEqual(result["versions"].tolist(), [[-1, -1, 7]])
        self.assertEqual(
            result["rewards"].tolist(),
            result["rollout_episode_returns"].tolist(),
        )
        self.assertEqual(result["rollout_episode_group_sizes"].tolist(), [12])
        self.assertEqual(len(result["rollout_episode_ids"].tolist()), 1)
        sent_image = Image.open(BytesIO(base64.b64decode(engine.request.image_data[0])))
        self.assertEqual(sent_image.mode, "RGB")
        self.assertEqual(len(engine.request.image_data), 1)
        self.assertEqual(engine.request.input_ids, [10, 11])
        self.assertEqual(
            engine.request.metadata["allowed_token_ids"],
            [40, 41, 42, 43],
        )
        self.assertEqual(
            engine.request.metadata["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        json.dumps(engine.request.vision_msg_vllm)
        self.assertEqual(
            engine.request.vision_msg_vllm[0][1]["content"][1]["image_url"],
            {"url": "placeholder"},
        )
        self.assertIn(
            "pixel_values", result["multi_modal_input"][0]
        )

    def test_native_objective_ledger_uses_generic_token_support(self) -> None:
        response = SimpleNamespace(
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.5, 0.0],
            output_versions=[3, 3],
        )
        processed = {
            "mm_token_type_ids": torch.tensor([[0, 1]]),
            "pixel_values": torch.tensor([[1.0]]),
        }
        sample = PacmanNativeVisionWorkflow._tensor_sample(
            processed,
            response,
            2.5,
            [],
            [[20, 30], [21]],
            rollout_episode_id=1234,
            rollout_episode_return=2.5,
            rollout_episode_group_size=12,
        )
        self.assertNotIn("pacman_action_mask_bits", sample)
        self.assertEqual(
            sample["pacman_allowed_token_ids"].tolist(),
            [[[0, 0], [0, 0], [21, 31], [22, 0]]],
        )
        self.assertEqual(sample["rollout_episode_ids"].tolist(), [1234])
        self.assertEqual(sample["rollout_episode_returns"].tolist(), [2.5])
        self.assertEqual(
            sample["rollout_episode_group_sizes"].tolist(), [12]
        )

    def test_native_option_return_sample_carries_episode_id_only(self) -> None:
        response = SimpleNamespace(
            input_tokens=[10, 11],
            output_tokens=[20],
            output_logprobs=[-0.5],
            output_versions=[3],
        )
        processed = {
            "mm_token_type_ids": torch.tensor([[0, 1]]),
            "pixel_values": torch.tensor([[1.0]]),
        }

        sample = PacmanNativeVisionWorkflow._tensor_sample(
            processed,
            response,
            2.5,
            [],
            rollout_episode_id=1234,
        )

        self.assertEqual(sample["rewards"].tolist(), [2.5])
        self.assertEqual(sample["rollout_episode_ids"].tolist(), [1234])
        self.assertNotIn("rollout_episode_returns", sample)
        self.assertNotIn("rollout_episode_group_sizes", sample)

    @staticmethod
    def _fake_objective_call_model_fixture():
        """Shared scaffolding for the mask-leak retry tests below."""

        class FakeTokenizer:
            def encode(self, token, **kwargs):
                return [ord(token)]

            def decode(self, token_ids, **kwargs):
                return "".join(chr(int(item)) for item in token_ids)

        class FakeGConfig:
            n_samples = 12

            def new(self, **kwargs):
                self.last_kwargs = kwargs
                return self

        class FakeModelRequest(SimpleNamespace):
            pass

        tokenizer = FakeTokenizer()
        constraint = ObjectiveTokenConstraint.build(tokenizer, ["C0", "A0"])
        return tokenizer, constraint, FakeGConfig, FakeModelRequest

    def test_call_model_retries_transient_mask_leak_then_succeeds(self) -> None:
        (
            tokenizer,
            constraint,
            FakeGConfig,
            FakeModelRequest,
        ) = self._fake_objective_call_model_fixture()
        allowed = constraint.allowed_token_ids
        leaked_token = max(allowed) + 1000

        def make_response(token_id):
            return SimpleNamespace(
                input_tokens=[10, 11],
                output_tokens=[token_id],
                output_logprobs=[-0.1],
                output_versions=[1],
                input_len=2,
                output_len=1,
                stop_reason="stop",
            )

        class FakeEngine:
            def __init__(self, responses):
                self.responses = list(responses)
                self.calls = 0

            async def agenerate(self, request):
                response = self.responses[self.calls]
                self.calls += 1
                return response

        engine = FakeEngine([make_response(leaked_token), make_response(allowed[0])])

        processor = self._fake_native_processor()
        workflow = PacmanNativeVisionWorkflow(
            gconfig=FakeGConfig(),
            tokenizer=tokenizer,
            processor=processor,
            env_factory=OneStepEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
        )
        workflow._native_engine.set(engine)
        workflow._native_turns.set({})

        fake_areal_api = SimpleNamespace(ModelRequest=FakeModelRequest)
        fake_areal_image = SimpleNamespace(
            image2base64=lambda image: ["encoded-image"]
        )
        messages = build_image_messages(
            encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
        )

        with patch.dict(
            sys.modules,
            {
                "areal.api": fake_areal_api,
                "areal.utils.image": fake_areal_image,
            },
        ), self.assertLogs(
            "pacman_recipe.level1.episode", level="WARNING"
        ) as logs:
            turn = asyncio.run(
                workflow._call_model(
                    messages, objective_constraint=constraint
                )
            )

        self.assertEqual(engine.calls, 2)
        self.assertTrue(
            any("MASK_LEAK_RETRY" in message for message in logs.output)
        )
        self.assertEqual(turn.completion, chr(allowed[0]))

    def test_call_model_raises_after_exhausting_mask_leak_retries(self) -> None:
        (
            tokenizer,
            constraint,
            FakeGConfig,
            FakeModelRequest,
        ) = self._fake_objective_call_model_fixture()
        leaked_token = max(constraint.allowed_token_ids) + 1000

        def make_response(token_id):
            return SimpleNamespace(
                input_tokens=[10, 11],
                output_tokens=[token_id],
                output_logprobs=[-0.1],
                output_versions=[1],
                input_len=2,
                output_len=1,
                stop_reason="stop",
            )

        class FakeEngine:
            def __init__(self):
                self.calls = 0

            async def agenerate(self, request):
                self.calls += 1
                return make_response(leaked_token)

        engine = FakeEngine()

        processor = self._fake_native_processor()
        workflow = PacmanNativeVisionWorkflow(
            gconfig=FakeGConfig(),
            tokenizer=tokenizer,
            processor=processor,
            env_factory=OneStepEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
        )
        workflow._native_engine.set(engine)
        workflow._native_turns.set({})

        fake_areal_api = SimpleNamespace(ModelRequest=FakeModelRequest)
        fake_areal_image = SimpleNamespace(
            image2base64=lambda image: ["encoded-image"]
        )
        messages = build_image_messages(
            encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
        )

        with patch.dict(
            sys.modules,
            {
                "areal.api": fake_areal_api,
                "areal.utils.image": fake_areal_image,
            },
        ), self.assertLogs(
            "pacman_recipe.level1.episode", level="WARNING"
        ), self.assertRaises(ObjectiveParseError):
            asyncio.run(
                workflow._call_model(
                    messages, objective_constraint=constraint
                )
            )

        from pacman_recipe.level1.workflow import _MASK_LEAK_RETRY_ATTEMPTS

        self.assertEqual(engine.calls, _MASK_LEAK_RETRY_ATTEMPTS)

    def test_workflow_defaults_to_thinking_disabled_and_rejects_true(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured.update(options)
                return ModelTurn("S", "capture-id", messages)

        workflow = CapturingWorkflow(env_factory=OneStepEnv)
        asyncio.run(workflow.run(make_episode_row(1, split="test")))
        self.assertIs(captured["enable_thinking"], False)
        self.assertIs(workflow.last_episode["decoding"]["enable_thinking"], False)

        with self.assertRaisesRegex(ValueError, "enable_thinking=false"):
            asyncio.run(
                CapturingWorkflow(env_factory=OneStepEnv).run(
                    make_episode_row(1, split="test"),
                    enable_thinking=True,
                )
            )

    def test_model_request_explicitly_disables_thinking(self) -> None:
        captured = {}

        class FakeResponse:
            id = "response-1"
            choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content="S", reasoning_content=None)
                )
            ]

            def model_dump(self, mode):
                return {"id": self.id, "mode": mode}

        class FakeCompletions:
            async def create(self, **request):
                captured.update(request)
                return FakeResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                captured["closed"] = True

        fake_openai = SimpleNamespace(AsyncOpenAI=FakeClient)
        messages = [{"role": "user", "content": "test"}]
        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_action_tokenizer(),
            ),
        ):
            workflow = PacmanImageOnlyWorkflow(
                open_action_mask=True,
                tokenizer_path="test-tokenizer",
            )
            turn = asyncio.run(
                workflow._call_model(
                    messages,
                    model="test-model",
                    base_url="http://example.invalid/v1",
                    api_key="test",
                    temperature=0.7,
                    top_p=0.95,
                    max_completion_tokens=3,
                    enable_thinking=False,
                    open_action_mask=True,
                    current_open_actions=["U", "R"],
                )
            )

        self.assertEqual(turn.completion, "S")
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["top_p"], 0.95)
        self.assertEqual(
            captured["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            captured["extra_body"]["structured_outputs"],
            {"choice": ["U", "R"]},
        )
        self.assertEqual(
            captured["extra_body"]["allowed_token_ids"],
            [40, 43],
        )
        self.assertTrue(captured["closed"])

    def test_edward_request_uses_only_one_token_allowlist_without_xgrammar(self) -> None:
        captured = {}

        class FakeResponse:
            id = "response-option-1"
            choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content="B", reasoning_content=None)
                )
            ]

            def model_dump(self, mode):
                return {"id": self.id, "mode": mode}

        class FakeCompletions:
            async def create(self, **request):
                captured.update(request)
                return FakeResponse()

        class FakeClient:
            def __init__(self, **_):
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                return None

        constraint = ObjectiveTokenConstraint.build(
            FakeObjectiveTokenizer(), ["C0", "A0"]
        )
        workflow = PacmanImageOnlyWorkflow.__new__(PacmanImageOnlyWorkflow)
        with patch.dict(
            sys.modules,
            {"openai": SimpleNamespace(AsyncOpenAI=FakeClient)},
        ):
            turn = asyncio.run(
                workflow._call_model(
                    [{"role": "user", "content": "test"}],
                    objective_constraint=constraint,
                )
            )

        self.assertEqual(turn.completion, "B")
        self.assertEqual(captured["max_tokens"], 1)
        self.assertEqual(
            captured["extra_body"]["allowed_token_ids"],
            [ord("B"), ord("J")],
        )
        self.assertNotIn("structured_outputs", captured["extra_body"])

    def test_workflow_passes_current_open_actions_to_generation(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured["open_actions"] = options["current_open_actions"]
                return ModelTurn(
                    captured["open_actions"][0],
                    "capture-id",
                    messages,
                )

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_action_tokenizer(),
        ):
            workflow = CapturingWorkflow(
                env_factory=OneStepEnv,
                open_action_mask=True,
                tokenizer_path="test-tokenizer",
            )
        asyncio.run(workflow.run(make_episode_row(1, split="test")))

        self.assertTrue(captured["open_actions"])
        self.assertNotIn("S", captured["open_actions"])
        self.assertTrue(
            set(captured["open_actions"]).issubset({"U", "D", "L", "R"})
        )
        step = workflow.last_episode["trajectory"][0]
        self.assertEqual(step["open_action_mask"], captured["open_actions"])
        self.assertIn(step["action"], step["open_action_mask"])
        self.assertFalse(step["wall_collision"])

    def test_duplicate_dataset_row_writes_distinct_trajectory_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = make_episode_row(1, split="validation")
            for _ in range(2):
                workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
                asyncio.run(
                    workflow.run(
                        row,
                        scripted_actions=["S"],
                        trajectory_dir=directory,
                    )
                )
            files = sorted(Path(directory).glob("*.json"))
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]

        self.assertEqual(len(files), 2)
        self.assertEqual({payload["id"] for payload in payloads}, {row["id"]})
        self.assertEqual(
            len({payload["trajectory_sample_id"] for payload in payloads}),
            2,
        )

    def test_workflow_uses_real_maapacman_and_records_completion_reward(self) -> None:
        row = make_episode_row(1, split="train")
        workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
        result = asyncio.run(
            workflow.run(
                row,
                scripted_actions=["L"],
                scripted_completion_ids=["completion-1"],
            )
        )
        self.assertEqual(result, {"completion-1": workflow.last_episode["trajectory"][0]["shaped_reward"]})
        payload = workflow.last_episode
        assert payload is not None
        self.assertEqual(payload["env_api_version"], "3.0")
        self.assertEqual(payload["terminal_reason"], "test_complete")
        self.assertEqual(payload["trajectory"][0]["action"], "L")
        self.assertEqual(payload["backend"], "original-pygame")
        self.assertEqual(
            payload["pacman_python_revision"],
            row["env"]["pacman_python_revision"],
        )
        self.assertTrue(payload["renderer_revision"].startswith("pacman-python:"))
        self.assertIn("score", payload["trajectory"][0])
        self.assertIn("pygame_mode", payload["trajectory"][0])
        audit_trajectory(payload)

    def test_nearest_pellet_shaping_uses_hidden_maapacman_topology(self) -> None:
        row = make_episode_row(1, split="train")
        level = load_bundled_level(1)
        route = route_to_nearest(level, level.pacman_start, level.pellets)
        workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
        asyncio.run(
            workflow.run(
                row,
                scripted_actions=[route[0].value],
                nearest_pellet_alpha=1.0,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        step = payload["trajectory"][0]
        self.assertEqual(payload["nearest_pellet_alpha"], 1.0)
        self.assertEqual(
            payload["nearest_pellet_topology_revision"], level.revision
        )
        self.assertIsNotNone(step["nearest_pellet_distance_before"])
        self.assertIsNotNone(step["nearest_pellet_distance_after"])
        self.assertEqual(
            step["nearest_pellet_progress_reward"],
            step["nearest_pellet_distance_before"]
            - step["nearest_pellet_distance_after"],
        )
        combined_prompt = payload["system_prompt"] + payload["user_instruction"]
        self.assertNotIn("distance", combined_prompt.lower())
        audit_trajectory(payload)

    def test_workflow_skips_bfs_above_late_shaping_threshold(self) -> None:
        workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
        with patch(
            "pacman_recipe.level1.episode."
            "_nearest_reachable_distance_with_diagnostics"
        ) as distance:
            asyncio.run(
                workflow.run(
                    make_episode_row(1, split="train"),
                    scripted_actions=["L"],
                    nearest_pellet_alpha=0.1,
                    nearest_pellet_remaining_ratio_threshold=0.25,
                    nearest_pellet_skip_on_eat=True,
                )
            )

        distance.assert_not_called()
        payload = workflow.last_episode
        assert payload is not None
        step = payload["trajectory"][0]
        self.assertGreater(step["normal_pellet_remaining_ratio"], 0.25)
        self.assertFalse(step["nearest_pellet_shaping_active"])
        self.assertIsNone(step["nearest_pellet_distance_before"])
        self.assertIsNone(step["nearest_pellet_distance_after"])
        audit_trajectory(payload)

    def test_parse_failure_is_terminal_and_never_forwarded(self) -> None:
        class CountingEnv(PygamePacmanEnv):
            steps_called = 0

            def step(self, action):
                type(self).steps_called += 1
                return super().step(action)

        workflow = PacmanImageOnlyWorkflow(env_factory=CountingEnv)
        reward = asyncio.run(
            workflow.run(
                make_episode_row(1, split="train"),
                scripted_actions=["Action: R"],
            )
        )
        self.assertEqual(reward, -1.0)
        self.assertEqual(CountingEnv.steps_called, 0)
        payload = workflow.last_episode
        self.assertEqual(payload["terminal_reason"], "parse_failed")
        self.assertEqual(payload["total_shaped_reward"], -1.0)
        failure = payload["trajectory"][-1]
        self.assertIs(failure["contract_violation"], True)
        self.assertEqual(failure["contract_violation_type"], "parse_failure")
        self.assertEqual(failure["contract_violation_target_return"], -1.0)
        self.assertEqual(failure["reward_accumulated_before_violation"], 0.0)
        self.assertEqual(failure["contract_violation_adjustment"], -1.0)
        audit_trajectory(payload)
        tampered = copy.deepcopy(payload)
        tampered["trajectory"][-1]["ghosts"] = []
        with self.assertRaisesRegex(ValueError, "ghost count"):
            audit_trajectory(tampered)

    def test_parse_failure_overrides_accumulated_episode_return_to_minus_one(
        self,
    ) -> None:
        workflow = PacmanImageOnlyWorkflow(env_factory=TwoStepEnv)
        reward = asyncio.run(
            workflow.run(
                make_episode_row(1, split="train"),
                scripted_actions=["L", "Action: R"],
                use_base_reward=False,
                step_penalty=0.5,
            )
        )

        payload = workflow.last_episode
        self.assertEqual(reward, -1.0)
        self.assertEqual(payload["total_shaped_reward"], -1.0)
        self.assertEqual(len(payload["trajectory"]), 2)
        failure = payload["trajectory"][-1]
        accumulated = payload["trajectory"][0]["shaped_reward"]
        self.assertNotEqual(accumulated, 0.0)
        self.assertEqual(
            failure["reward_accumulated_before_violation"], accumulated
        )
        self.assertEqual(
            failure["contract_violation_adjustment"], -1.0 - accumulated
        )
        self.assertEqual(
            failure["shaped_reward"],
            failure["contract_violation_adjustment"],
        )
        audit_trajectory(payload)

    def test_environment_closes_on_success_and_exception(self) -> None:
        created = []

        class TrackingEnv(PygamePacmanEnv):
            def __init__(self, config):
                super().__init__(config)
                self.close_called = False
                created.append(self)

            def close(self):
                self.close_called = True
                super().close()

        class TrackingOneStepEnv(TrackingEnv):
            def step(self, action):
                image, reward, _, _, info = super().step(action)
                info = dict(info)
                info.update(
                    terminated=True,
                    truncated=False,
                    terminal_reason="test_complete",
                )
                return image, reward, True, False, info

        workflow = PacmanImageOnlyWorkflow(env_factory=TrackingOneStepEnv)
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="train"),
                scripted_actions=["S"],
            )
        )
        self.assertTrue(created[-1].close_called)
        workflow = PacmanImageOnlyWorkflow(env_factory=TrackingEnv)
        with self.assertRaises(RuntimeError):
            asyncio.run(
                workflow.run(
                    make_episode_row(2, split="train"),
                    scripted_actions=["S"],
                )
            )
        self.assertTrue(created[-1].close_called)

    def test_model_receives_same_png_hash_recorded_in_trajectory(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured["messages"] = messages
                return ModelTurn("S", "capture-id", messages)

        workflow = CapturingWorkflow(env_factory=OneStepEnv)
        asyncio.run(workflow.run(make_episode_row(1, split="test")))
        url = captured["messages"][1]["content"][1]["image_url"]["url"]
        sent_png = base64.b64decode(url.split(",", 1)[1])
        recorded = workflow.last_episode["trajectory"][0]["observation_png_sha256"]
        self.assertEqual(png_sha256(sent_png), recorded)

    def test_edward_option_accumulates_reward_across_bounded_actions(self) -> None:
        captured = {}
        candidate = PlannerCandidate(
            option_id="C0",
            strategy="COLLECT",
            target=(1, 1),
            first_action="L",
            route_distance=2,
            commit_moves=2,
        )

        class FakePlanner:
            def observe(self, state):
                return None

            def advertised_candidates(self, state):
                return (candidate,)

            def record_action(self, action):
                self.last_action = action

            def continue_option(self, option, state):
                return "L", "active"

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured["system_prompt"] = messages[0]["content"]
                return ModelTurn(
                    "B",
                    "objective-1",
                    messages,
                )

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=FakeObjectiveTokenizer(),
            ),
            patch(
                "pacman_recipe.level1.episode.EdwardPlanner",
                FakePlanner,
            ),
        ):
            workflow = CapturingWorkflow(
                env_factory=TwoStepEnv,
                tokenizer_path="test-tokenizer",
                edward_options=True,
                image_prompt_style="live_state_v3",
            )
            result = asyncio.run(
                workflow.run(make_episode_row(1, split="test"))
            )

        payload = workflow.last_episode
        assert payload is not None
        first, second = payload["trajectory"]
        self.assertEqual(set(result), {"objective-1"})
        self.assertAlmostEqual(
            result["objective-1"],
            first["shaped_reward"] + second["shaped_reward"],
        )
        self.assertEqual(
            [first["option_step"], second["option_step"]], [1, 2]
        )
        self.assertEqual(first["option_status"], "active")
        self.assertFalse(first["option_end"])
        self.assertEqual(second["option_status"], "terminal")
        self.assertTrue(second["option_end"])
        self.assertEqual(
            payload["action_constraint"], "edward-option-code-v1"
        )
        self.assertEqual(payload["decoding"]["max_completion_tokens"], 1)
        self.assertEqual(
            captured["system_prompt"], EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT
        )
        self.assertEqual(
            payload["system_prompt"], EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT
        )
        self.assertIn("uppercase option code", payload["system_prompt"])
        self.assertIn("not a movement action", payload["system_prompt"])
        for expected in (
            "this Pacman simulator",
            "trust the structured state",
            "level/tunnel door",
            "Eyes and gone ghosts are nonlethal",
            "The episode ends on the first death.",
            "COLLECT is the default",
            "Use AVOID for a threatened route",
            "Use ELIMINATE only for",
            "rechecks safety after every move",
            "Codes map to fixed objective ids",
            "only advertised candidates and their targets are valid",
            "no other text",
        ):
            self.assertIn(expected, payload["system_prompt"])
        self.assertNotIn("fruit", payload["system_prompt"].lower())
        self.assertIn(
            '["B","C0","COLLECT",[1,1],"L",2,2,null,null,null]',
            first["model_user_instruction"],
        )
        self.assertIn('"ghosts":', first["model_user_instruction"])
        self.assertIn('"edible_ticks":', first["model_user_instruction"])
        self.assertIn('"maze":', first["model_user_instruction"])
        for expected in (
            "p=Pac-Man [row,column]",
            "pellets=normal+power pellets remaining",
            "Use only the candidates shown for this turn",
            "distance=route steps, commit=max executed moves",
            "larger safety/exits are better",
            "entity=ELIMINATE ghost id",
            "nothing else",
        ):
            self.assertIn(expected, first["model_user_instruction"])
        self.assertIn(
            "Return exactly one code from [B]",
            first["model_user_instruction"],
        )
        self.assertNotIn(
            "recent_positions", first["model_user_instruction"]
        )
        audit_trajectory(payload)

        corrupted = copy.deepcopy(payload)
        corrupted["decoding"]["max_completion_tokens"] = 3
        with self.assertRaisesRegex(
            ValueError, "decoding.max_completion_tokens=1"
        ):
            audit_trajectory(corrupted)

        corrupted = copy.deepcopy(payload)
        del corrupted["decoding"]
        with self.assertRaisesRegex(
            ValueError, "decoding.max_completion_tokens=1"
        ):
            audit_trajectory(corrupted)

        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][0]["observation_context"][
            "option_code_map"
        ] = {}
        with self.assertRaisesRegex(ValueError, "invalid Edward option code"):
            audit_trajectory(corrupted)

        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][0]["option_code_map"]["F"] = "C1"
        corrupted["trajectory"][0]["observation_context"][
            "option_code_map"
        ]["F"] = "C1"
        with self.assertRaisesRegex(ValueError, "match planner candidates"):
            audit_trajectory(corrupted)

        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][1]["option_code_map"]["F"] = "C1"
        with self.assertRaisesRegex(ValueError, "breaks option continuity"):
            audit_trajectory(corrupted)

    def test_edward_safety_refusal_truncates_after_last_completed_option(self) -> None:
        candidate = PlannerCandidate(
            option_id="C0",
            strategy="COLLECT",
            target=(1, 1),
            first_action="L",
            route_distance=1,
            commit_moves=1,
        )

        class RefusingPlanner:
            decisions = 0

            def observe(self, state):
                return None

            def advertised_candidates(self, state):
                self.decisions += 1
                if self.decisions == 1:
                    return (candidate,)
                raise EdwardSafetyRefusal("no provably safe action")

            def record_action(self, action):
                return None

            def continue_option(self, option, state):
                return None, "completed"

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            calls = 0

            async def _call_model(self, messages, **options):
                self.calls += 1
                return ModelTurn(
                    "B",
                    "safe-before-refusal",
                    messages,
                )

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=FakeObjectiveTokenizer(),
            ),
            patch(
                "pacman_recipe.level1.episode.EdwardPlanner",
                RefusingPlanner,
            ),
        ):
            workflow = CapturingWorkflow(
                env_factory=TwoStepEnv,
                tokenizer_path="test-tokenizer",
                edward_options=True,
                image_prompt_style="live_state_v3",
            )
            rewards = asyncio.run(
                workflow.run(
                    make_episode_row(1, split="test"),
                    safety_refusal_penalty=25.0,
                )
            )

        payload = workflow.last_episode
        assert payload is not None
        self.assertEqual(workflow.calls, 1)
        self.assertEqual(set(rewards), {"safe-before-refusal"})
        self.assertEqual(len(payload["trajectory"]), 1)
        final = payload["trajectory"][-1]
        self.assertEqual(final["option_status"], "completed")
        self.assertTrue(final["option_end"])
        self.assertFalse(final["terminated"])
        self.assertTrue(final["truncated"])
        self.assertEqual(final["terminal_reason"], "safety_refusal")
        self.assertTrue(final["safety_refusal"])
        self.assertEqual(final["safety_refusal_penalty"], 25.0)
        self.assertAlmostEqual(
            rewards["safe-before-refusal"], final["option_return"]
        )
        self.assertAlmostEqual(
            payload["total_shaped_reward"], final["shaped_reward"]
        )
        self.assertEqual(
            payload["safety_refusal_penalty_coefficient"], 25.0
        )
        audit_trajectory(payload)

        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][-1]["safety_refusal_penalty"] = 0.0
        with self.assertRaisesRegex(
            ValueError, "inconsistent safety-refusal reward evidence"
        ):
            audit_trajectory(corrupted)

        corrupted = copy.deepcopy(payload)
        corrupted["trajectory"][-1]["terminated"] = True
        corrupted["trajectory"][-1]["truncated"] = False
        corrupted["terminated"] = True
        corrupted["truncated"] = False
        with self.assertRaisesRegex(ValueError, "invalid Edward safety refusal"):
            audit_trajectory(corrupted)

    def test_initial_edward_safety_refusal_returns_no_native_sample(self) -> None:
        class RefusingPlanner:
            def observe(self, state):
                return None

            def advertised_candidates(self, state):
                raise EdwardSafetyRefusal("no provably safe initial action")

        class FakeGConfig:
            n_samples = 12

            def new(self, **kwargs):
                return self

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=FakeObjectiveTokenizer(),
            ),
            patch(
                "pacman_recipe.level1.episode.EdwardPlanner",
                RefusingPlanner,
            ),
        ):
            workflow = PacmanNativeVisionWorkflow(
                gconfig=FakeGConfig(),
                tokenizer=FakeObjectiveTokenizer(),
                processor=self._fake_native_processor(),
                env_factory=TwoStepEnv,
                tokenizer_path="test-tokenizer",
                edward_options=True,
                image_prompt_style="live_state_v3",
            )
            result = asyncio.run(
                workflow.arun_episode(
                    SimpleNamespace(), make_episode_row(1, split="train")
                )
            )

        self.assertIsNone(result)
        self.assertIsNone(workflow.last_episode)

    def test_unrelated_planner_runtime_error_is_not_swallowed(self) -> None:
        class BrokenPlanner:
            def observe(self, state):
                return None

            def advertised_candidates(self, state):
                raise RuntimeError("planner implementation bug")

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=FakeObjectiveTokenizer(),
            ),
            patch(
                "pacman_recipe.level1.episode.EdwardPlanner",
                BrokenPlanner,
            ),
        ):
            workflow = PacmanImageOnlyWorkflow(
                env_factory=TwoStepEnv,
                tokenizer_path="test-tokenizer",
                edward_options=True,
                image_prompt_style="live_state_v3",
            )
            with self.assertRaisesRegex(RuntimeError, "planner implementation bug"):
                asyncio.run(workflow.run(make_episode_row(1, split="test")))

    def test_live_state_workflow_records_prompt_context_and_history(self) -> None:
        workflow = PacmanImageOnlyWorkflow(env_factory=TwoStepEnv)
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=["U", "S"],
                image_prompt_style="live_state_v3",
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        first, second = payload["trajectory"]
        self.assertEqual(
            payload["observation_contract"],
            "screenshot_plus_live_state_and_navigation_history",
        )
        initial_position = first["observation_context"]["pacman_position"]
        self.assertEqual(len(initial_position), 2)
        self.assertIn("U", first["observation_context"]["blocked_actions"])
        self.assertEqual(
            second["observation_context"]["current_cell_exit_history"],
            ["U"],
        )
        self.assertEqual(
            second["observation_context"]["current_cell_exit_counts"],
            {"U": 1},
        )
        self.assertEqual(second["observation_context"]["recent_actions"], ["U"])
        self.assertEqual(
            second["observation_context"]["immediate_reverse_action"],
            "D",
        )
        self.assertIn(
            (
                f"At this cell ({initial_position[0]},{initial_position[1]}), "
                "directions already taken before: U."
            ),
            second["model_user_instruction"],
        )

    def test_live_state_cell_history_remains_bounded_for_full_horizon(self) -> None:
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=["U"] * PRODUCTION_MAX_STEPS,
                image_prompt_style="live_state_v3",
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        instructions = [
            step["model_user_instruction"] for step in payload["trajectory"]
        ]
        self.assertLess(max(map(len, instructions)), 2200)
        last_context = payload["trajectory"][-1]["observation_context"]
        self.assertEqual(last_context["current_cell_exit_history"], ["U"])
        self.assertEqual(
            last_context["current_cell_exit_counts"],
            {"U": len(payload["trajectory"]) - 1},
        )

    def test_trajectory_preserves_verbatim_model_response(self) -> None:
        raw_response = {
            "id": "response-raw-1",
            "choices": [
                {
                    "message": {
                        "content": "S",
                        "reasoning_content": None,
                    }
                }
            ],
        }
        request_extra_body = {
            "structured_outputs": {"choice": ["U", "D", "L", "R", "S"]},
            "chat_template_kwargs": {"enable_thinking": False},
        }

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                return ModelTurn(
                    "S",
                    "response-raw-1",
                    messages,
                    reasoning_content=None,
                    raw_response=raw_response,
                    request_extra_body=request_extra_body,
                )

        with tempfile.TemporaryDirectory() as directory:
            workflow = CapturingWorkflow(env_factory=OneStepEnv)
            asyncio.run(
                workflow.run(
                    make_episode_row(1, split="test"),
                    trajectory_dir=directory,
                    enable_thinking=False,
                )
            )
            payload = json.loads(next(Path(directory).glob("*.json")).read_text())

        step = payload["trajectory"][0]
        self.assertEqual(step["completion"], "S")
        self.assertIsNone(step["reasoning_content"])
        self.assertEqual(step["raw_model_response"], raw_response)
        self.assertEqual(step["request_extra_body"], request_extra_body)

    def test_edward_baseline_replays_through_production_workflow(self) -> None:
        actions = successful_planner_baseline_actions()
        row = make_episode_row(
            1,
            split="test",
            max_steps=STRESS_MAX_STEPS,
        )
        self.assertGreater(len(actions), 32)
        self.assertLessEqual(len(actions), row["env"]["max_steps"])
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                row,
                scripted_actions=actions,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        self.assertEqual(payload["steps"], len(actions))
        self.assertEqual(payload["terminal_reason"], "all_normal_pellets")
        self.assertTrue(payload["won"])
        self.assertEqual(payload["normal_pellets_remaining"], 0)
        audit_trajectory(payload)

    def test_nearest_pellet_shaping_audits_over_edward_baseline(self) -> None:
        actions = successful_planner_baseline_actions()
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                make_episode_row(
                    1,
                    split="test",
                    max_steps=STRESS_MAX_STEPS,
                ),
                scripted_actions=actions,
                nearest_pellet_alpha=1.0,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        trajectory = payload["trajectory"]
        self.assertEqual(len(trajectory), len(actions))
        self.assertEqual(payload["terminal_reason"], "all_normal_pellets")
        self.assertTrue(payload["won"])
        self.assertTrue(
            all(
                step["nearest_pellet_distance_before"] is not None
                and step["nearest_pellet_distance_after"] is not None
                for step in trajectory
            )
        )
        for step in trajectory:
            self.assertEqual(
                step["nearest_pellet_progress_reward"],
                step["nearest_pellet_distance_before"]
                - step["nearest_pellet_distance_after"],
            )
        audit_trajectory(payload)

    def test_cancellation_closes_worker_and_removes_runtime(self) -> None:
        created = []
        model_started = asyncio.Event()

        class TrackingEnv(PygamePacmanEnv):
            def __init__(self, config):
                super().__init__(config)
                self.close_called = False
                created.append(self)

            def close(self):
                self.close_called = True
                super().close()

        class BlockingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                model_started.set()
                await asyncio.Future()

        async def scenario() -> None:
            workflow = BlockingWorkflow(env_factory=TrackingEnv)
            task = asyncio.create_task(
                workflow.run(make_episode_row(1, split="test"))
            )
            await asyncio.wait_for(model_started.wait(), timeout=15)
            runtime = created[-1].worker_runtime_dir
            self.assertIsNotNone(runtime)
            self.assertTrue(runtime.is_dir())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(created[-1].close_called)
            self.assertFalse(runtime.exists())

        asyncio.run(scenario())

    def test_production_workflow_does_not_import_private_environment(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "pacman_recipe"
            / "level1"
            / "workflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from pacman_env.env import", source)
        self.assertNotIn("from .env import", source)
        self.assertNotIn("from pacman_recipe.env import", source)

    def test_workflow_import_shim_preserves_public_classes(self) -> None:
        from pacman_recipe.level1.workflow import (
            PacmanImageOnlyWorkflow as CanonicalImageOnlyWorkflow,
        )
        from pacman_recipe.level1.workflow import (
            PacmanNativeVisionWorkflow as CanonicalNativeVisionWorkflow,
        )

        self.assertIs(PacmanImageOnlyWorkflow, CanonicalImageOnlyWorkflow)
        self.assertIs(PacmanNativeVisionWorkflow, CanonicalNativeVisionWorkflow)


class TrainerGenerationContractTests(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            environment=SimpleNamespace(ghost_mode="normal", max_steps=256),
            enable_thinking=False,
            reward_objective_contract="episode_return_group_v1",
            image_prompt_style="minimal_v1",
            tokenizer_path="test-tokenizer",
            legal_action_mask=False,
            open_action_mask=True,
            guided_action_choice=False,
            legal_action_choice=False,
            non_stay_legal_action_choice=False,
            non_backtracking_legal_action_choice=False,
            action_token_choice=True,
            completion_api="chat",
            parse_failure_penalty=-50,
            contract_violation_return=-1.0,
            safety_refusal_penalty=25.0,
            reward_mode="sparse",
            route_shaping_scale=1.0,
            safe_progress_alpha=1.0,
            step_penalty=1.0,
            wall_penalty=1.0,
            nearest_pellet_alpha=0.0,
            nearest_pellet_scale_by_cleared_ratio=True,
            observation_mode="rgb",
            vision_tile_size=32,
            store_observation_images=False,
            trajectory_dir="run_artifacts/trajectories",
        )

    def test_dynamic_filter_is_an_rpc_safe_import_path(self) -> None:
        """AReaL ships dynamic_filter_fn to rollout workers over JSON RPC.

        A function object is not JSON serializable, so passing a closure makes
        every rollout submit fail with "Object of type function is not JSON
        serializable" and the run never completes a single rollout. Only the
        import-path form survives the hop.
        """
        from train_areal import _build_group_reward_degeneracy_filter

        config = self._config()
        config.gconfig = SimpleNamespace(n_samples=12)

        spec = _build_group_reward_degeneracy_filter(config)

        self.assertIsInstance(
            spec, str, "dynamic_filter_fn must be an import path, not a callable"
        )
        json.dumps(spec)  # must survive the RPC hop

        # The rollout worker resolves it with exactly this helper
        # (areal/infra/remote_inf_engine.py::_resolve_should_accept_fn).
        from areal.utils.dynamic_import import import_from_string

        resolved = import_from_string(spec)
        self.assertTrue(callable(resolved))

    def test_dynamic_filter_disabled_outside_episode_group_contract(self) -> None:
        from train_areal import _build_group_reward_degeneracy_filter

        config = self._config()
        config.gconfig = SimpleNamespace(n_samples=12)
        config.reward_objective_contract = "option_return_raw_v1"
        self.assertIsNone(_build_group_reward_degeneracy_filter(config))

        config.reward_objective_contract = "episode_return_group_v1"
        config.gconfig = SimpleNamespace(n_samples=1)
        self.assertIsNone(_build_group_reward_degeneracy_filter(config))

    def test_degenerate_reward_group_is_rejected(self) -> None:
        from pacman_recipe.level1.dynamic_filter import (
            accept_non_degenerate_reward_group,
        )

        # 12 episodes with differing decision-row counts, all the same return,
        # concatenated the way GroupedRolloutWorkflow does.
        degenerate = torch.cat(
            [
                torch.full((5,), 293.9859375),
                torch.full((3,), 293.9859375),
                torch.full((7,), 293.9859375),
            ]
        )
        self.assertFalse(accept_non_degenerate_reward_group({"rewards": degenerate}))

        mixed = torch.cat(
            [
                torch.full((5,), 293.9859375),
                torch.full((3,), 80.358),
                torch.full((7,), 212.085),
            ]
        )
        self.assertTrue(accept_non_degenerate_reward_group({"rewards": mixed}))

        # Fail open when the field is absent or too small to judge.
        self.assertTrue(accept_non_degenerate_reward_group({}))
        self.assertTrue(
            accept_non_degenerate_reward_group({"rewards": torch.tensor([42.0])})
        )

    def test_train_and_eval_generation_kwargs_are_independent(self) -> None:
        config = self._config()
        train_generation = SimpleNamespace(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            max_new_tokens=3,
        )
        eval_generation = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            max_new_tokens=3,
        )

        training = _build_workflow_kwargs(config, train_generation)
        evaluation = _build_workflow_kwargs(
            config, eval_generation, training=False
        )

        self.assertEqual(training["temperature"], 0.7)
        self.assertEqual(evaluation["temperature"], 0.0)
        self.assertEqual(training["top_p"], 0.95)
        self.assertEqual(evaluation["top_p"], 1.0)
        self.assertIs(training["enable_thinking"], False)
        self.assertIs(evaluation["enable_thinking"], False)
        self.assertEqual(training["image_prompt_style"], "minimal_v1")
        self.assertEqual(evaluation["image_prompt_style"], "minimal_v1")
        self.assertIs(training["open_action_mask"], True)
        self.assertIs(evaluation["open_action_mask"], True)
        self.assertEqual(
            training["reward_objective_contract"],
            "episode_return_group_v1",
        )
        self.assertEqual(
            evaluation["reward_objective_contract"], "evaluation_only_v1"
        )
        self.assertEqual(training["safety_refusal_penalty"], 25.0)
        self.assertEqual(evaluation["safety_refusal_penalty"], 25.0)
        self.assertEqual(training["contract_violation_return"], -1.0)
        self.assertEqual(evaluation["contract_violation_return"], -1.0)
        self.assertIs(
            training["nearest_pellet_scale_by_cleared_ratio"],
            True,
        )
        self.assertIs(
            evaluation["nearest_pellet_scale_by_cleared_ratio"],
            True,
        )

    def test_group12_config_declares_true_greedy_validation(self) -> None:
        config = (
        config = (
        config = (
        config = (
        root = Path(__file__).parents[1]
        patch = (
            root / "patches" / "areal_pacman_action_logprobs.patch"
        ).read_text(encoding="utf-8")
        launcher = (
            root / "scripts" / "level1" / "train" / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("_apply_pacman_action_mask", patch)
        self.assertIn('logprobs_mode: str = "raw_logprobs"', patch)
        self.assertIn("pacman_recipe_action_logprobs_patch=ok", launcher)

    def test_fsdp_cpu_offload_patch_is_reproducible_and_preflighted(self) -> None:
        root = Path(__file__).parents[1]
        patch = (
            root / "patches" / "areal_fsdp_cpu_offload_empty_cache.patch"
        ).read_text(encoding="utf-8")
        launcher = (
            root / "scripts" / "level1" / "train" / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        marker = "CPUOffloadPolicy has already moved the persistent FSDP parameter"
        self.assertIn(marker, patch)
        self.assertIn("areal_fsdp_cpu_offload_empty_cache_patch=ok", launcher)
        self.assertIn(marker, launcher)

    def test_training_launcher_uses_ephemeral_nondefault_admin_key(self) -> None:
        launcher = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "train"
            / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("secrets.token_urlsafe(32)", launcher)
        self.assertIn('export AREAL_ADMIN_API_KEY', launcher)
        self.assertIn(
            '[[ "${AREAL_ADMIN_API_KEY}" == "areal-admin-key" ]]', launcher
        )
        self.assertNotIn("echo \"${AREAL_ADMIN_API_KEY}\"", launcher)

    def test_training_launcher_prefers_official_areal_worktree(self) -> None:
        launcher = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "train"
            / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'AREAL_ROOT="${AREAL_ROOT:-${WORKSPACE_ROOT}/AReaL}"', launcher
        )
        self.assertIn(
            'MAAPACMAN_PACMAN_PYTHON_ROOT:-${WORKSPACE_ROOT}/pacman-python',
            launcher,
        )
        self.assertIn(
            'export PYTHONPATH="${AREAL_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            launcher,
        )
        self.assertIn("AReaL import escaped selected checkout", launcher)

    def test_training_launcher_defaults_to_curriculum1_and_chunks_logps(self) -> None:
        launcher = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "train"
            / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "configs/level1/train/curriculum1.yaml",
            launcher,
        )
        self.assertIn('DATASET_ARGS=()', launcher)
        self.assertNotIn('DATASET_MAX_STEPS="${DATASET_MAX_STEPS:-256}"', launcher)
        self.assertIn('SMOKE_ARGS=(--smoke-updates "${SMOKE_UPDATES}")', launcher)
        self.assertNotIn("CURRICULUM1_CHECKPOINT", launcher)
        self.assertIn('validate_model_checkpoint.py', launcher)
        self.assertIn(
            'DATASET_OUTPUT_ROOT="${DATASET_OUTPUT_ROOT:-${ARTIFACT_ROOT}/dataset}"',
            launcher,
        )
        self.assertIn(
            'MAAPACMAN_LOGP_RPC_CHUNK_SIZE="${MAAPACMAN_LOGP_RPC_CHUNK_SIZE:-${ACTOR_DP_SIZE}}"',
            launcher,
        )
        self.assertIn("export MAAPACMAN_LOGP_RPC_CHUNK_SIZE", launcher)
        self.assertIn(
            'echo "logp_rpc_chunk_size=${MAAPACMAN_LOGP_RPC_CHUNK_SIZE}"',
            launcher,
        )

    def test_official_areal_single_step_smoke_contract(self) -> None:
        config = (
        evaluator = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "evaluate"
            / "evaluate_level1_run.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'if [[ -f "${EVAL_ROOT}/${label}/greedy.json" ]]', evaluator
        )
        self.assertIn('local served_model_name="${4:-${label}}"', evaluator)
        self.assertIn('--served-model-name "${served_model_name}"', evaluator)
        self.assertIn(
            '"${sampled_port}" \\\n    "${sampled_label}"',
            evaluator,
        )
        self.assertIn('SAMPLED_LABELS=("${LABELS[@]}")', evaluator)
        self.assertNotIn("comparison.partial.json", evaluator)
        self.assertIn(
            'sampled12_already_complete=${sampled_label}', evaluator
        )
        self.assertIn("best_sampled_label=${BEST_SAMPLED_LABEL}", evaluator)
        self.assertIn("best_greedy_label=${BEST_GREEDY_LABEL}", evaluator)
        self.assertIn("--require-complete-dual", evaluator)


if __name__ == "__main__":
    unittest.main()
