"""Run the API-v3 pacman-python ruleset behind a line-oriented IPC bridge.

This module belongs to Pacman. It drives pygame at runtime, drains the
ruleset's source event ledger, and never writes to the sibling checkout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import random
import runpy
import sys
import threading
import zlib
from pathlib import Path
from typing import Any

from pacman_env.env._saved_state import capture_game, restore_game


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--ghost-mode", required=True, choices=("disabled", "normal"))
    parser.add_argument(
        "--episode-life-mode",
        required=True,
        choices=("single_death", "original_three_lives", "three_lives"),
    )
    return parser.parse_args()


def _normalize_episode_life_mode(mode: str) -> str:
    return "original_three_lives" if mode == "three_lives" else mode


class _PygameBridge:
    LOGIC_FRAMES_PER_STEP = 16

    def __init__(
        self,
        pygame: Any,
        protocol_output: Any,
        ghost_mode: str = "normal",
        episode_life_mode: str = "single_death",
    ) -> None:
        self._ghost_mode = ghost_mode
        self._episode_life_mode = episode_life_mode
        self._pygame = pygame
        self._protocol_output = protocol_output
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._original_event_get = pygame.event.get
        self._original_flip = pygame.display.flip
        self._next_action: str | None = None
        self._pending_request: dict[str, Any] | None = None
        self._boot_started = False
        self._ready_sent = False

    def start(self) -> None:
        self._pygame.event.get = self._event_get
        self._pygame.display.flip = self._flip
        threading.Thread(target=self._read_commands, daemon=True).start()

    def _read_commands(self) -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                self._emit({"type": "error", "error": f"invalid command: {exc}"})
                continue
            self._commands.put(command)
        self._commands.put({"op": "close"})

    def _event_get(self, *args: Any, **kwargs: Any) -> list[Any]:
        events = list(self._original_event_get(*args, **kwargs))
        action = self._next_action
        self._next_action = None
        key_by_action = {
            "U": self._pygame.K_UP,
            "D": self._pygame.K_DOWN,
            "L": self._pygame.K_LEFT,
            "R": self._pygame.K_RIGHT,
        }
        if action in key_by_action:
            events.append(
                self._pygame.event.Event(
                    self._pygame.KEYDOWN,
                    key=key_by_action[action],
                    repeat=False,
                )
            )
        return events

    def _flip(self) -> None:
        self._original_flip()
        globals_dict = sys._getframe(1).f_globals
        game = globals_dict.get("thisGame")
        player = globals_dict.get("player")
        level = globals_dict.get("thisLevel")
        if game is None or player is None or level is None:
            return

        if not self._boot_started:
            # The original game writes res/hiscore.txt at game-over. Resources
            # are shared via a symlink: keep that display state in the episode
            # object so replay cannot modify a different worker's future frame.
            game._headless_hiscores = game.gethiscores()

            def get_hiscores(instance: Any) -> list[Any]:
                return list(instance._headless_hiscores)

            def write_hiscores(instance: Any, scores: list[Any]) -> None:
                instance._headless_hiscores = list(scores)

            type(game).gethiscores = get_hiscores
            type(game).writehiscores = write_hiscores
            type(game).getplayername = lambda instance: "User"
            game.StartNewGame()
            game.SetMode(1)
            player.SnapToGrid()
            globals_dict["agentPaused"] = True
            self._boot_started = True
            return

        payload = self._capture(globals_dict)
        state_writer = globals_dict.get("WriteAgentState")
        if callable(state_writer):
            # The original loop normally writes immediately after flip().
            # Because the bridge pauses inside flip(), invoke the unchanged
            # original writer here so the worker-local file matches this frame.
            state_writer()
        if not self._ready_sent:
            payload.update({"type": "ready", "request_id": 0})
            self._emit(payload)
            self._ready_sent = True
        elif self._pending_request is not None:
            self._pending_request["logic_frames"] += 1
            atomic_state = self._atomic_state(payload["state"])
            drain_events = globals_dict.get("DrainGameEvents")
            if not callable(drain_events):
                raise RuntimeError(
                    "pacman-python does not expose the API v3 source event ledger"
                )
            source_events = drain_events()
            score_components, events = self._consume_source_events(
                source_events,
                self._pending_request["previous_atomic_state"],
                atomic_state,
                logic_frame_index=self._pending_request["logic_frames"],
            )
            if any(event["event_type"] == "death" for event in events):
                self._pending_request["death_seen"] = True
            atomic_state.update(
                {
                    "logic_frame_index": self._pending_request["logic_frames"],
                    "score_delta": score_components["total"],
                    "score_components": score_components,
                    "events": events,
                }
            )
            if int(payload["state"]["mode"]) == 5:
                # Eating a vulnerable ghost triggers a presentation-only pause
                # in the original game. Preserve the event, but do not stall a
                # headless RL step for one 16-frame cell movement.  This is an
                # environment transition, so the atomic post-state must expose
                # the normalized mode too; otherwise the step state and its
                # final atomic substep describe different Markov states.
                game.SetMode(1)
                payload["state"]["mode"] = 1
                payload["state"]["mode_name"] = "playing"
                payload["state"]["mode_timer"] = 0
                atomic_state["mode"] = 1
                atomic_state["mode_name"] = "playing"
                atomic_state["mode_timer"] = 0
            self._pending_request["atomic_substeps"].append(atomic_state)
            self._pending_request["previous_atomic_state"] = atomic_state
            if self._step_is_complete(self._pending_request, payload["state"]):
                globals_dict["agentPaused"] = True
                payload.update(
                    {
                        "type": "step",
                        "request_id": self._pending_request["request_id"],
                        "action": self._pending_request["action"],
                        "logic_frames": self._pending_request["logic_frames"],
                        "atomic_substeps": self._pending_request[
                            "atomic_substeps"
                        ],
                    }
                )
                self._emit(payload)
                self._pending_request = None
            else:
                # Let the original loop run another logic/render frame.  No
                # wall-clock sleep is used to decide when the step is done.
                return

        while True:
            command = self._commands.get()
            operation = command.get("op")
            if operation == "close":
                self._pygame.quit()
                raise SystemExit(0)
            if operation == "save_state":
                self._emit({"type": "saved_state", "request_id": command["request_id"],
                            "saved_state": capture_game(self._pygame, globals_dict)})
                continue
            if operation == "restore_state":
                previous_game = capture_game(self._pygame, globals_dict)
                try:
                    restore_game(self._pygame, globals_dict, command["saved_state"])
                    # _flip's locals must follow the newly restored object graph.
                    game = globals_dict["thisGame"]
                    player = globals_dict["player"]
                    level = globals_dict["thisLevel"]
                    self._pending_request = None
                    self._next_action = None
                    payload = self._capture(globals_dict)
                    if callable(state_writer):
                        state_writer()
                    self._emit({**payload, "type": "restored_state",
                                "request_id": command["request_id"]})
                except Exception as exc:
                    # A checksum verifies transport integrity, not graph validity.
                    # A failed load must leave this worker ready at its old state.
                    restore_game(self._pygame, globals_dict, previous_game)
                    game = globals_dict["thisGame"]
                    player = globals_dict["player"]
                    level = globals_dict["thisLevel"]
                    payload = self._capture(globals_dict)
                    self._emit({"type": "error", "request_id": command["request_id"],
                                "error": f"invalid saved state: {exc}"})
                continue
            if operation == "step" and command.get("action") in {
                "U",
                "D",
                "L",
                "R",
                "S",
            }:
                command["start_row"] = payload["state"]["row"]
                command["start_col"] = payload["state"]["col"]
                command["blocked_at_start"] = (
                    command["action"] in payload["state"]["blocked"]
                )
                command["logic_frames"] = 0
                command["atomic_substeps"] = []
                command["death_seen"] = False
                command["previous_atomic_state"] = self._atomic_state(
                    payload["state"]
                )
                self._pending_request = command
                self._next_action = command["action"]
                globals_dict["agentPaused"] = False
                return
            self._emit(
                {
                    "type": "error",
                    "request_id": command.get("request_id"),
                    "error": f"unsupported command: {command!r}",
                }
            )

    def _step_is_complete(
        self, request: dict[str, Any], state: dict[str, Any]
    ) -> bool:
        mode = int(state["mode"])
        if mode in {3, 6, 9}:
            return True
        if self._episode_life_mode == "single_death" and mode == 2:
            return True
        if request["death_seen"]:
            return mode == 1
        return request["logic_frames"] >= self.LOGIC_FRAMES_PER_STEP

    @staticmethod
    def _atomic_state(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "frame": int(state["logic_frame"]),
            "pacman_position": [int(state["row"]), int(state["col"])],
            "pacman_pixel_position": list(state["pacman_pixel_position"]),
            "pacman_velocity": list(state["pacman_velocity"]),
            "pacman_speed": float(state["pacman_speed"]),
            "pacman_facing": str(state["facing"]),
            "level": int(state["level"]),
            "ghost_mode": state["ghost_mode"],
            "mode": int(state["mode"]),
            "mode_name": str(state["mode_name"]),
            "mode_timer": int(state["mode_timer"]),
            "score": int(state["score"]),
            "lives": int(state["lives"]),
            "width": int(state["width"]),
            "height": int(state["height"]),
            "normal_pellets_remaining": int(state["normal_pellets"]),
            "power_pellets_remaining": int(state["power_pellets"]),
            "collectibles_remaining": int(state["collectibles_remaining"]),
            "edible_ticks": int(state["edible_ticks"]),
            "edible_timer_started_frame": int(state["edible_timer_started_frame"]),
            "ghost_value": int(state["ghost_value"]),
            "fruit_timer": int(state["fruit_timer"]),
            "fruit_score_ticks": int(state["fruit_score_ticks"]),
            "fruit_score_position": list(state["fruit_score_position"]),
            "ghosts": [
                {
                    "id": int(ghost["id"]),
                    "position": list(ghost["position"]),
                    "pixel_position": list(ghost["pixel_position"]),
                    "velocity": list(ghost["velocity"]),
                    "speed": float(ghost["speed"]),
                    "direction": str(ghost["direction"]),
                    "state": str(ghost["state"]),
                    "state_code": int(ghost["state_code"]),
                    "path_remaining": ghost["path_remaining"],
                    "path_found": bool(ghost["path_found"]),
                    "path_target": (
                        None
                        if ghost["path_target"] is None
                        else list(ghost["path_target"])
                    ),
                    "inside_ghost_house": bool(ghost["inside_ghost_house"]),
                    "home_position": list(ghost["home_position"]),
                }
                for ghost in state["ghosts"]
            ],
            "fruit": dict(state["fruit"]),
            "ghost_door": dict(state["ghost_door"]),
            "blocked": list(state["blocked"]),
            "open": list(state["open"]),
        }

    @staticmethod
    def _consume_source_events(
        source_events: Any,
        previous: dict[str, Any],
        current: dict[str, Any],
        *,
        logic_frame_index: int,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        if not isinstance(source_events, list):
            raise RuntimeError("source event ledger did not return a list")
        score_delta = int(current["score"]) - int(previous["score"])
        event_components = {
            "normal_pellet_eaten": "normal_pellet",
            "power_pellet_eaten": "power_pellet",
            "ghost_eaten": "ghost",
            "fruit_eaten": "fruit",
            "death": None,
            "level_cleared": None,
        }
        components = {
            "normal_pellet": 0,
            "power_pellet": 0,
            "ghost": 0,
            "fruit": 0,
            "other": 0,
            "total": score_delta,
        }
        events: list[dict[str, Any]] = []
        required = {
            "frame_index",
            "type",
            "pacman_position",
            "ghost_id",
            "score_delta",
            "ghost_state",
            "edible_ticks",
            "frame_score_delta",
        }
        for source in source_events:
            if not isinstance(source, dict) or not required.issubset(source):
                raise RuntimeError("source event ledger entry is incomplete")
            event_type = str(source["type"])
            if event_type not in event_components:
                raise RuntimeError(f"unknown source event type: {event_type!r}")
            source_frame = int(source["frame_index"])
            if source_frame != int(current["frame"]):
                raise RuntimeError(
                    "source event frame does not match captured logic frame"
                )
            event_score = int(source["score_delta"])
            if int(source["frame_score_delta"]) != score_delta:
                raise RuntimeError(
                    "source frame score delta does not match captured score delta"
                )
            position = source["pacman_position"]
            if not isinstance(position, list) or len(position) != 2:
                raise RuntimeError("source event Pacman position is invalid")
            component = event_components[event_type]
            if component is not None:
                components[component] += event_score
            events.append({
                "logic_frame_index": logic_frame_index,
                "logic_frame": source_frame,
                "event_type": event_type,
                "pacman_position": [int(position[0]), int(position[1])],
                "ghost_id": (
                    None
                    if source["ghost_id"] is None
                    else int(source["ghost_id"])
                ),
                "score_delta": event_score,
                "post_ghost_state": (
                    None
                    if source["ghost_state"] is None
                    else str(source["ghost_state"])
                ),
                "edible_ticks": int(source["edible_ticks"]),
            })
        if sum(event["score_delta"] for event in events) != score_delta:
            raise RuntimeError(
                "source event scores do not reconcile to captured score delta"
            )
        return components, events

    @staticmethod
    def _path_state(entity: Any) -> dict[str, Any]:
        """Return the exact remaining route and its endpoint as JSON state."""

        raw_path = entity.currentPath
        path_found = raw_path is not False and raw_path is not None
        path_remaining = str(raw_path) if path_found else None
        if not path_found:
            return {
                "path_remaining": None,
                "path_found": False,
                "path_target": None,
            }

        pixel_x = int(entity.x)
        pixel_y = int(entity.y)
        velocity_x = float(entity.velX)
        velocity_y = float(entity.velY)
        if pixel_x % 16:
            col = pixel_x // 16 if velocity_x > 0 else (pixel_x + 15) // 16
        else:
            col = pixel_x // 16
        if pixel_y % 16:
            row = pixel_y // 16 if velocity_y > 0 else (pixel_y + 15) // 16
        else:
            row = pixel_y // 16
        offsets = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        for direction in path_remaining:
            if direction not in offsets:
                raise RuntimeError(f"invalid entity path direction: {direction!r}")
            row_delta, col_delta = offsets[direction]
            row += row_delta
            col += col_delta
        return {
            "path_remaining": path_remaining,
            "path_found": True,
            "path_target": [row, col],
        }

    def _capture(self, globals_dict: dict[str, Any]) -> dict[str, Any]:
        if (
            globals_dict.get("GHOST_MODE") != self._ghost_mode
            or globals_dict.get("CURRICULUM_ID") != 2
        ):
            raise RuntimeError(
                "pacman-python must support the explicit ghost-mode contract"
            )
        if self._ghost_mode == "disabled":
            # Validate the engine, not just the filtered observation. State 4
            # is natively non-rendering, non-moving and non-colliding.
            for index in range(4):
                ghost = globals_dict["ghosts"][index]
                if (
                    ghost.state != 4 or ghost.velX != 0 or ghost.velY != 0
                    or ghost.x != -64 or ghost.y != -64
                ):
                    raise RuntimeError("disabled ghost is active in the game engine")
        pygame = self._pygame
        game = globals_dict["thisGame"]
        player = globals_dict["player"]
        level = globals_dict["thisLevel"]
        ghosts = globals_dict["ghosts"]
        fruit = globals_dict["thisFruit"]
        tile_ids = globals_dict.get("tileID", {})
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1).copy()
        raw = frame.tobytes(order="C")

        pellet_id = tile_ids.get("pellet")
        power_id = tile_ids.get("pellet-power")
        normal_pellets = 0
        power_pellets = 0
        normal_pellet_positions = []
        power_pellet_positions = []
        for row in range(int(level.lvlHeight)):
            for col in range(int(level.lvlWidth)):
                tile = level.GetMapTile(row, col)
                normal_pellets += int(tile == pellet_id)
                power_pellets += int(tile == power_id)
                if tile == pellet_id:
                    normal_pellet_positions.append([row, col])
                if tile == power_id:
                    power_pellet_positions.append([row, col])

        blocked: list[str] = []
        open_actions: list[str] = []
        deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        row = int(player.nearestRow)
        col = int(player.nearestCol)
        for action, (row_delta, col_delta) in deltas.items():
            target_row = row + row_delta
            target_col = col + col_delta
            try:
                is_wall = level.IsWall(target_row, target_col, actor="pacman")
            except TypeError:
                is_wall = level.IsWall(target_row, target_col)
            (blocked if is_wall else open_actions).append(action)

        ghost_state_names = {
            1: "normal",
            2: "vulnerable",
            3: "eyes",
            4: "gone",
        }
        def direction_for(ghost: Any) -> str:
            velocity_x = float(ghost.velX)
            velocity_y = float(ghost.velY)
            if abs(velocity_x) > abs(velocity_y):
                return "R" if velocity_x > 0 else "L"
            if velocity_y:
                return "D" if velocity_y > 0 else "U"
            return "S"
        mode_names = {
            1: "playing",
            2: "death",
            3: "game_over",
            4: "ready",
            5: "ghost_eaten_pause",
            6: "level_cleared",
            7: "level_flash",
            8: "level_transition",
            9: "all_levels_cleared",
        }
        ghost_door_position = level.GetGhostBoxPos()
        if not ghost_door_position:
            raise RuntimeError("level does not define a ghost door")
        ghost_door_row, ghost_door_col = map(int, ghost_door_position)
        ghost_paths = {
            index: self._path_state(ghosts[index]) for index in range(4)
        }
        fruit_path = self._path_state(fruit)
        return {
            "frame": {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "encoding": "zlib+base64",
                "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "state": {
                "row": row,
                "col": col,
                "pacman_pixel_position": [int(player.y), int(player.x)],
                "pacman_velocity": [float(player.velY), float(player.velX)],
                "pacman_speed": float(player.speed),
                "facing": player.lastMoveDir if player.lastMoveDir in "UDLRS" else "S",
                "level": int(game.GetLevelNum()),
                "ghost_mode": self._ghost_mode,
                "mode": int(game.mode),
                "mode_name": mode_names.get(int(game.mode), "unknown"),
                "mode_timer": int(game.modeTimer),
                "logic_frame": int(globals_dict.get("GAME_LOGIC_FRAME", 0)),
                "score": int(game.score),
                "lives": int(game.lives),
                "width": int(level.lvlWidth),
                "height": int(level.lvlHeight),
                "normal_pellets": normal_pellets,
                "power_pellets": power_pellets,
                "normal_pellet_positions": normal_pellet_positions,
                "power_pellet_positions": power_pellet_positions,
                "collectibles_remaining": normal_pellets + power_pellets,
                "edible_ticks": int(game.ghostTimer),
                "edible_timer_started_frame": int(game.ghostTimerStartedFrame),
                "ghost_value": int(game.ghostValue),
                "fruit_timer": int(game.fruitTimer),
                "fruit_score_ticks": int(game.fruitScoreTimer),
                "fruit_score_position": [
                    int(game.fruitScorePos[1]),
                    int(game.fruitScorePos[0]),
                ],
                "ghosts": [
                    {
                        "id": index,
                        "position": [
                            int(ghosts[index].nearestRow),
                            int(ghosts[index].nearestCol),
                        ],
                        "pixel_position": [
                            int(ghosts[index].y),
                            int(ghosts[index].x),
                        ],
                        "velocity": [
                            float(ghosts[index].velY),
                            float(ghosts[index].velX),
                        ],
                        "speed": float(ghosts[index].speed),
                        "direction": direction_for(ghosts[index]),
                        "state": ghost_state_names.get(
                            int(ghosts[index].state), "unknown"
                        ),
                        "state_code": int(ghosts[index].state),
                        **ghost_paths[index],
                        "inside_ghost_house": bool(
                            ghosts[index].TileInGhostHouse(
                                int(ghosts[index].nearestRow),
                                int(ghosts[index].nearestCol),
                            )
                        ),
                        "home_position": [
                            int(ghosts[index].homeY // 16),
                            int(ghosts[index].homeX // 16),
                        ],
                    }
                    for index in range(4)
                    if self._ghost_mode == "normal"
                ],
                "fruit": {
                    "active": bool(fruit.active),
                    "position": [
                        int(fruit.nearestRow)
                        if isinstance(fruit.nearestRow, int)
                        else -1,
                        int(fruit.nearestCol)
                        if isinstance(fruit.nearestCol, int)
                        else -1,
                    ],
                    "pixel_position": [int(fruit.y), int(fruit.x)],
                    "velocity": [float(fruit.velY), float(fruit.velX)],
                    "speed": float(fruit.speed),
                    **fruit_path,
                    "slow_timer": int(fruit.slowTimer),
                    "bounce_counter": int(fruit.bouncei),
                    "bounce_offset": int(fruit.bounceY),
                    "type": int(fruit.fruitType),
                },
                "ghost_door": {
                    "position": [ghost_door_row, ghost_door_col],
                    "pacman_blocked": bool(
                        level.IsWall(
                            ghost_door_row,
                            ghost_door_col,
                            actor="pacman",
                        )
                    ),
                    "ghost_blocked": bool(
                        level.IsWall(
                            ghost_door_row,
                            ghost_door_col,
                            actor="ghost",
                        )
                    ),
                },
                "blocked": blocked,
                "open": open_actions,
            },
        }

    def _emit(self, payload: dict[str, Any]) -> None:
        self._protocol_output.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._protocol_output.flush()


def main() -> int:
    args = _parse_args()
    script = Path(args.script).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    sys.path[0] = str(script.parent)
    sys.argv = [
        str(script), "--start-level", "1", "--curriculum", "2",
        "--ghost-mode", args.ghost_mode,
    ]
    random.seed(args.seed)
    os.environ["MAAPACMAN_PAUSE_ON_START"] = "1"

    import pygame

    class _FastClock:
        def tick(self, *_args: Any, **_kwargs: Any) -> int:
            return 0

    pygame.time.Clock = _FastClock

    bridge = _PygameBridge(
        pygame,
        protocol_output,
        args.ghost_mode,
        _normalize_episode_life_mode(args.episode_life_mode),
    )
    bridge.start()
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
