"""External environment wrapper for the API-v3 pacman-python ruleset."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from maapacman.actions import ACTION_ORDER, Action, coerce_action
from maapacman.errors import (
    EnvironmentClosedError,
    EpisodeFinishedError,
    EpisodeNotStartedError,
    InvalidConfigurationError,
    PacmanEnvError,
)

from .config import PacmanEnvSpec
from .ghost_modes import validate_ghost_mode, validate_ghost_state
from ._saved_state import SCHEMA as SAVED_STATE_SCHEMA, checksum


def normalize_episode_life_mode(mode: str) -> str:
    """Normalize legacy alias names to canonical episode life modes."""
    return "original_three_lives" if mode == "three_lives" else mode


RULESET_CONTRACT = {
    "action_logic_frames": 16,
    "actor_roles": ["pacman", "ghost", "vulnerable", "eyes"],
    "event_score_contract": "per-logic-frame-v1",
    "source_event_ledger": "DrainGameEvents-v1",
    "ghost_door": "blocked-for-pacman-open-for-ghosts",
    "state_schema": "complete-transition-v1",
    "terminal_modes": [2, 3, 6, 9],
    "vulnerable_logic_frames": 360,
}
def ruleset_revision(ghost_mode: str = "normal") -> str:
    contract = {**RULESET_CONTRACT, "ghost_mode": validate_ghost_mode(ghost_mode)}
    return hashlib.sha256(
        (json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


RULESET_REVISION = ruleset_revision()

TRANSITION_STATE_FIELDS = {
    "row",
    "col",
    "pacman_pixel_position",
    "pacman_velocity",
    "pacman_speed",
    "facing",
    "level",
    "mode",
    "mode_name",
    "mode_timer",
    "logic_frame",
    "score",
    "lives",
    "width",
    "height",
    "normal_pellets",
    "power_pellets",
    "collectibles_remaining",
    "edible_ticks",
    "edible_timer_started_frame",
    "ghost_value",
    "fruit_timer",
    "fruit_score_ticks",
    "fruit_score_position",
    "ghosts",
    "fruit",
    "ghost_door",
    "blocked",
    "open",
}
ATOMIC_GHOST_FIELDS = {
    "id",
    "position",
    "pixel_position",
    "velocity",
    "speed",
    "direction",
    "state",
    "state_code",
    "path_remaining",
    "path_found",
    "path_target",
    "inside_ghost_house",
    "home_position",
}
ATOMIC_FRUIT_FIELDS = {
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
}


def _canonical_text_sha256(path: Path) -> str:
    """Hash text content with platform line endings normalized to LF."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _canonical_files_sha256(root: Path, paths: list[Path]) -> str:
    """Hash a named source bundle without depending on checkout location."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        )
        digest.update(b"\0")
    return digest.hexdigest()


class PygameWorkerError(PacmanEnvError):
    """The external original-pygame worker failed or violated its protocol."""


@dataclass(frozen=True)
class PygamePacmanEnvConfig:
    pacman_python_root: str | os.PathLike[str] | None = None
    level: int = 1
    max_steps: int = 512
    ghost_mode: str = "normal"
    episode_life_mode: str = "single_death"
    timeout_seconds: float = 15.0
    video_driver: str | None = "dummy"
    audio_driver: str | None = "dummy"
    python_executable: str | os.PathLike[str] = sys.executable
    worker_base_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_life_mode", normalize_episode_life_mode(self.episode_life_mode))
        try:
            validate_ghost_mode(self.ghost_mode)
        except ValueError as exc:
            raise InvalidConfigurationError(str(exc)) from exc
        if self.episode_life_mode not in {
            "single_death",
            "original_three_lives",
        }:
            raise InvalidConfigurationError(
                "episode_life_mode must be single_death or original_three_lives"
            )
        if self.level != 1:
            raise InvalidConfigurationError(
                "PygamePacmanEnv currently supports only pacman-python level 1"
            )
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise InvalidConfigurationError("max_steps must be an integer")
        if self.max_steps <= 0:
            raise InvalidConfigurationError("max_steps must be positive")
        if self.timeout_seconds <= 0:
            raise InvalidConfigurationError("timeout_seconds must be positive")


class PygamePacmanEnv:
    """Gym-style wrapper around the role-aware pygame game process."""

    API_VERSION = "3.0"
    ENV_ID = "pacman-python-level1-ghostdoor-v3"
    RULESET_REVISION = RULESET_REVISION

    def __init__(self, config: PygamePacmanEnvConfig | None = None) -> None:
        self.config = config or PygamePacmanEnvConfig()
        default_root = Path(__file__).resolve().parents[3] / "pacman-python"
        configured_root = (
            self.config.pacman_python_root
            or os.getenv("MAAPACMAN_PACMAN_ROOT")
            or os.getenv("MAAPACMAN_PACMAN_PYTHON_ROOT")
        )
        self._root = Path(configured_root or default_root).resolve()
        self._script = self._root / "pacman" / "pacman.pyw"
        self._level_path = self._root / "pacman" / "res" / "levels" / "1.txt"
        if not self._script.is_file() or not self._level_path.is_file():
            raise InvalidConfigurationError(
                f"invalid pacman-python checkout: {self._root}"
            )
        maapacman_root = Path(__file__).resolve().parents[2]
        self._revision = self._git_revision(self._root)
        self._revision_dirty = self._git_dirty(self._root)
        self._pacman_source_revision = _canonical_files_sha256(
            self._root, [self._script, self._level_path]
        )
        self._maapacman_revision = self._git_revision(maapacman_root)
        self._maapacman_revision_dirty = self._git_dirty(maapacman_root)
        maapacman_source_paths = [
            maapacman_root / "maapacman" / "actions.py",
            maapacman_root / "maapacman" / "errors.py",
            *(
                path
                for path in (maapacman_root / "maapacman" / "env").rglob("*")
                if path.is_file() and path.suffix in {".py", ".json"}
            ),
        ]
        self._maapacman_source_revision = _canonical_files_sha256(
            maapacman_root, maapacman_source_paths
        )
        self._level_revision = _canonical_text_sha256(self._level_path)
        self._spec = PacmanEnvSpec(
            api_version=self.API_VERSION,
            env_id=self.ENV_ID,
            ruleset_revision=ruleset_revision(self.config.ghost_mode),
            level_revision=self._level_revision,
            renderer_revision=f"pacman-python:{self._revision}",
            observation_shape=(400, 336, 3),
            observation_dtype="uint8",
            action_tokens=tuple(action.value for action in ACTION_ORDER),
            deterministic=True,
        )
        self._process: subprocess.Popen[str] | None = None
        self._runtime_dir: Path | None = None
        self._runtime_script: Path | None = None
        self._resource_mode: str | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=80)
        self._frame: np.ndarray | None = None
        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._request_id = 0
        self._seed = 0
        self._started = False
        self._finished = False
        self._closed = False
        self._initial_collectibles = 0
        self._last_logic_frames = 0
        self._death_count = 0

    @property
    def spec(self) -> PacmanEnvSpec:
        self._ensure_open()
        return self._spec

    @property
    def pacman_python_revision(self) -> str:
        return self._revision

    @property
    def provenance(self) -> dict[str, Any]:
        """Return immutable environment identity without starting a worker."""

        return {
            "pacman_python_commit": self._revision,
            "pacman_python_source_sha256": self._pacman_source_revision,
            "pacman_python_dirty": self._revision_dirty,
            "maapacman_commit": self._maapacman_revision,
            "maapacman_env_source_sha256": self._maapacman_source_revision,
            "maapacman_dirty": self._maapacman_revision_dirty,
            "level": self.config.level,
            "ghost_mode": self.config.ghost_mode,
            "episode_life_mode": self.config.episode_life_mode,
            "level_revision": self._level_revision,
            "env_id": self._spec.env_id,
            "api_version": self._spec.api_version,
            "ruleset_revision": self._spec.ruleset_revision,
        }

    @property
    def worker_runtime_dir(self) -> Path | None:
        return self._runtime_dir

    def reset(
        self, *, seed: int | None = None, saved_state: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._ensure_open()
        if saved_state is not None:
            if seed is not None:
                raise InvalidConfigurationError("seed and saved_state are mutually exclusive")
            return self.restore_state(saved_state)
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise InvalidConfigurationError("reset seed must be an integer")
        self._stop_worker()
        self._started = False
        self._frame = None
        self._state = None
        self._seed = seed if seed is not None else 0
        self._start_worker()
        message = self._receive(expected_type="ready")
        self._accept_message(message)
        self._steps = 0
        self._finished = False
        self._started = True
        self._initial_collectibles = int(self._state["collectibles_remaining"])
        self._death_count = 0
        return self.render(), self._build_info(score_delta=0, seed=self._seed)

    def step(
        self, action: Action | str
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._ensure_ready()
        if self._finished:
            raise EpisodeFinishedError("episode is finished; call reset()")
        canonical = coerce_action(action)
        previous = dict(self._state)
        self._request_id += 1
        self._send(
            {"op": "step", "request_id": self._request_id, "action": canonical.value}
        )
        message = self._receive(expected_type="step", request_id=self._request_id)
        self._accept_message(message)
        self._steps += 1

        score_delta = int(self._state["score"]) - int(previous["score"])
        mode = int(self._state["mode"])
        atomic_substeps = list(message.get("atomic_substeps", []))
        score_components, logic_frame_events = self._audit_atomic_substeps(
            previous,
            atomic_substeps,
            expected_score_delta=score_delta,
        )
        normal_pellets_eaten = score_components["normal_pellet"] // 10
        power_pellets_eaten = score_components["power_pellet"] // 100
        ghost_eaten_count = sum(
            event["event_type"] == "ghost_eaten" for event in logic_frame_events
        )
        fruit_eaten = any(
            event["event_type"] == "fruit_eaten" for event in logic_frame_events
        )
        death = any(
            event["event_type"] == "death" for event in logic_frame_events
        )
        if death:
            self._death_count += 1
        game_over = mode == 3
        terminated = mode in {6, 9} or game_over or (
            self.config.episode_life_mode == "single_death" and death
        )
        truncated = self._steps >= self.config.max_steps and not terminated
        self._finished = terminated or truncated
        level_cleared = mode in {6, 9}
        events = list(dict.fromkeys(event["event_type"] for event in logic_frame_events))
        info = self._build_info(score_delta=score_delta, seed=self._seed)
        info.update(
            {
                "action": canonical.value,
                "action_applied": canonical is Action.STAY
                or [previous["row"], previous["col"]]
                != [self._state["row"], self._state["col"]],
                "previous_position": [previous["row"], previous["col"]],
                "pacman_position": [self._state["row"], self._state["col"]],
                "pellet_eaten": bool(normal_pellets_eaten),
                "normal_pellets_eaten_step": normal_pellets_eaten,
                "power_pellet_eaten": bool(power_pellets_eaten),
                "power_pellets_eaten_step": power_pellets_eaten,
                "ghost_eaten": bool(ghost_eaten_count),
                "ghosts_eaten_step": ghost_eaten_count,
                "fruit_eaten": fruit_eaten,
                "death": death,
                "death_count": self._death_count,
                "respawned": bool(death and not terminated),
                "lives": max(0, int(previous["lives"])),
                "lives_after_step": max(
                    0,
                    int(self._state["lives"])
                    - int(
                        self.config.episode_life_mode == "single_death"
                        and mode in {2, 3}
                    ),
                ),
                "level_cleared": level_cleared,
                "option_invalidated": False,
                "events": events,
                "logic_frame_events": logic_frame_events,
                "score_components": score_components,
                "wall_collision": canonical is not Action.STAY
                and [previous["row"], previous["col"]]
                == [self._state["row"], self._state["col"]],
                "logic_frames": self._last_logic_frames,
                "atomic_substeps": atomic_substeps,
            }
        )
        return self.render(), float(score_delta), terminated, truncated, info

    def _audit_atomic_substeps(
        self,
        previous: dict[str, Any],
        atomic_substeps: list[dict[str, Any]],
        *,
        expected_score_delta: int,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        if len(atomic_substeps) != self._last_logic_frames:
            raise self._worker_failure(
                "worker atomic_substeps length does not match logic_frames"
            )
        contains_death = any(
            event.get("event_type") == "death"
            for substep in atomic_substeps
            for event in substep.get("events", [])
        )
        max_logic_frames = (
            256
            if self.config.episode_life_mode == "original_three_lives"
            and contains_death
            else 16
        )
        if not 1 <= len(atomic_substeps) <= max_logic_frames:
            raise self._worker_failure(
                f"worker step must contain 1 to {max_logic_frames} logic frames"
            )

        component_names = (
            "normal_pellet",
            "power_pellet",
            "ghost",
            "fruit",
            "other",
            "total",
        )
        required_state = {
            "frame",
            "logic_frame_index",
            "pacman_position",
            "pacman_pixel_position",
            "pacman_velocity",
            "pacman_speed",
            "pacman_facing",
            "level",
            "mode",
            "mode_name",
            "mode_timer",
            "score",
            "score_delta",
            "score_components",
            "lives",
            "width",
            "height",
            "normal_pellets_remaining",
            "power_pellets_remaining",
            "collectibles_remaining",
            "edible_ticks",
            "edible_timer_started_frame",
            "ghost_value",
            "fruit_timer",
            "fruit_score_ticks",
            "fruit_score_position",
            "ghosts",
            "fruit",
            "ghost_door",
            "blocked",
            "open",
            "events",
        }
        required_event = {
            "logic_frame_index",
            "logic_frame",
            "event_type",
            "pacman_position",
            "ghost_id",
            "score_delta",
            "post_ghost_state",
            "edible_ticks",
        }
        allowed_event_types = {
            "normal_pellet_eaten",
            "power_pellet_eaten",
            "ghost_eaten",
            "fruit_eaten",
            "death",
            "level_cleared",
        }
        totals = {name: 0 for name in component_names}
        logic_frame_events: list[dict[str, Any]] = []
        prior_score = int(previous["score"])
        for index, substep in enumerate(atomic_substeps, start=1):
            if not required_state.issubset(substep):
                missing = sorted(required_state - substep.keys())
                raise self._worker_failure(
                    f"worker atomic substep is missing state fields: {missing}"
                )
            if int(substep["logic_frame_index"]) != index:
                raise self._worker_failure("worker logic-frame indexes are not contiguous")
            try:
                validate_ghost_state(substep, self.config.ghost_mode)
            except ValueError as exc:
                raise self._worker_failure(str(exc)) from exc
            if any(
                not ATOMIC_GHOST_FIELDS.issubset(ghost)
                for ghost in substep["ghosts"]
            ):
                raise self._worker_failure("worker atomic ghost state is incomplete")
            if not ATOMIC_FRUIT_FIELDS.issubset(substep["fruit"]):
                raise self._worker_failure("worker atomic fruit state is incomplete")
            frame_delta = int(substep["score"]) - prior_score
            if int(substep["score_delta"]) != frame_delta:
                raise self._worker_failure("worker per-frame score delta mismatch")
            components = substep["score_components"]
            if set(components) != set(component_names):
                raise self._worker_failure("worker per-frame score components mismatch")
            if int(components["total"]) != frame_delta or int(components["total"]) != sum(
                int(components[name]) for name in component_names[:-1]
            ):
                raise self._worker_failure("worker per-frame score components do not add up")
            for name in component_names:
                totals[name] += int(components[name])
            event_score = 0
            for event in substep["events"]:
                if not required_event.issubset(event):
                    raise self._worker_failure("worker logic-frame event schema mismatch")
                if int(event["logic_frame_index"]) != index:
                    raise self._worker_failure("worker event logic-frame index mismatch")
                if str(event["event_type"]) not in allowed_event_types:
                    raise self._worker_failure("worker event type is not part of API v3")
                self._audit_logic_frame_event_position(event, substep)
                if int(event["edible_ticks"]) < 0:
                    raise self._worker_failure("worker event edible timer is invalid")
                event_score += int(event["score_delta"])
                logic_frame_events.append(dict(event))
            if event_score != frame_delta:
                raise self._worker_failure("worker per-frame events do not reconcile score")
            prior_score = int(substep["score"])

        if totals["total"] != expected_score_delta:
            raise self._worker_failure("worker step score does not reconcile atomic frames")
        if prior_score != int(self._state["score"]):
            raise self._worker_failure("worker final score does not match final atomic frame")
        self._audit_final_atomic_mode(atomic_substeps[-1])
        normal_delta = int(previous["normal_pellets"]) - int(
            self._state["normal_pellets"]
        )
        power_delta = int(previous["power_pellets"]) - int(
            self._state["power_pellets"]
        )
        if totals["normal_pellet"] != normal_delta * 10:
            raise self._worker_failure("normal-pellet event/score reconciliation failed")
        if totals["power_pellet"] != power_delta * 100:
            raise self._worker_failure("power-pellet event/score reconciliation failed")
        event_types = [str(event["event_type"]) for event in logic_frame_events]
        final_mode = int(self._state["mode"])
        if final_mode in {2, 3} and "death" not in event_types:
            raise self._worker_failure("terminal death is missing its source event")
        if final_mode in {6, 9} and "level_cleared" not in event_types:
            raise self._worker_failure("level completion is missing its source event")
        return totals, logic_frame_events

    def _audit_logic_frame_event_position(
        self, event: dict[str, Any], substep: dict[str, Any]
    ) -> None:
        """Require source-event coordinates to match the atomic game state."""

        source_position = event["pacman_position"]
        if (
            not isinstance(source_position, list)
            or len(source_position) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in source_position
            )
        ):
            raise self._worker_failure(
                "worker event source Pacman position is invalid"
            )
        source_row, source_col = source_position
        if not (
            0 <= source_row < int(substep["height"])
            and 0 <= source_col < int(substep["width"])
        ):
            raise self._worker_failure(
                "worker event source Pacman position is outside the level"
            )

        atomic_position = list(substep["pacman_position"])
        if source_position != atomic_position:
            raise self._worker_failure(
                "worker event Pacman position mismatch: "
                f"event_type={event['event_type']!r}, "
                f"logic_frame={event['logic_frame']}, "
                f"event_position={source_position!r}, "
                f"atomic_position={atomic_position!r}"
            )

    def _audit_final_atomic_mode(self, final: dict[str, Any]) -> None:
        """Reject a transition whose top-level and final atomic modes diverge."""

        comparisons = (
            ("mode", int, "mode"),
            ("mode_name", str, "mode name"),
            ("mode_timer", int, "mode timer"),
        )
        for field, convert, label in comparisons:
            if convert(final[field]) != convert(self._state[field]):
                raise self._worker_failure(
                    f"worker final {label} does not match final atomic frame"
                )

    def render(self) -> np.ndarray:
        self._ensure_ready()
        return self._frame.copy()

    def legal_actions(self) -> tuple[Action, ...]:
        self._ensure_ready()
        legal = set(self._state["open"])
        legal.add("S")
        return tuple(action for action in ACTION_ORDER if action.value in legal)

    def snapshot(self) -> dict[str, Any]:
        self._ensure_ready()
        return dict(self._state)

    def _saved_state_identity(self) -> dict[str, Any]:
        resources = self._script.parent / "res"
        assets = hashlib.sha256()
        for path in sorted(resources.rglob("*")):
            # High scores are now in the saved game object. Fonts are still
            # read at game-over, so include binary resources in compatibility.
            if path.is_file() and path != resources / "hiscore.txt":
                assets.update(path.relative_to(resources).as_posix().encode("utf-8"))
                assets.update(b"\0")
                assets.update(hashlib.sha256(path.read_bytes()).digest())
        return {
            "schema": SAVED_STATE_SCHEMA,
            "pacman_source": self._pacman_source_revision,
            "env_source": self._maapacman_source_revision,
            "ruleset": self._spec.ruleset_revision,
            "level": self._level_revision,
            "assets": assets.hexdigest(),
            "max_steps": self.config.max_steps,
            "episode_life_mode": self.config.episode_life_mode,
        }

    def save_state(self) -> dict[str, Any]:
        """Return an independent JSON-safe checkpoint at a reset/step boundary.

        Includes exact RNG, game objects, sprites, framebuffer and episode
        accounting. Unlike snapshot(), this is sufficient for continuation.
        """
        self._ensure_ready()
        self._request_id += 1
        self._send({"op": "save_state", "request_id": self._request_id})
        message = self._receive(expected_type="saved_state", request_id=self._request_id)
        payload = {
            "identity": self._saved_state_identity(),
            "worker": message["saved_state"],
            "episode": {
                "seed": self._seed, "steps": self._steps, "finished": self._finished,
                "initial_collectibles": self._initial_collectibles,
                "death_count": self._death_count, "last_logic_frames": self._last_logic_frames,
            },
        }
        return {"payload": payload, "sha256": checksum(payload)}

    def restore_state(self, saved_state: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Resume a matching checkpoint, also in a fresh environment/worker.

        Original step budget and terminal status are preserved. Finished saves
        remain finished; restoration does not silently grant another episode.
        """
        self._ensure_open()
        try:
            payload = saved_state["payload"]
            if checksum(payload) != saved_state["sha256"]:
                raise ValueError("checksum mismatch")
            if payload["identity"] != self._saved_state_identity():
                raise ValueError("source/config/schema mismatch")
            episode = payload["episode"]
            if set(episode) != {"seed", "steps", "finished", "initial_collectibles",
                                "death_count", "last_logic_frames"}:
                raise ValueError("episode fields mismatch")
            for name in ("seed", "steps", "initial_collectibles", "death_count", "last_logic_frames"):
                if type(episode[name]) is not int or episode[name] < 0:
                    raise ValueError(f"invalid episode {name}")
            if (type(episode["finished"]) is not bool or episode["initial_collectibles"] <= 0
                    or episode["steps"] > self.config.max_steps):
                raise ValueError("invalid episode counters")
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidConfigurationError(f"invalid saved state: {exc}") from exc
        if not self._started:
            self.reset(seed=episode["seed"])
        self._request_id += 1
        self._send({"op": "restore_state", "request_id": self._request_id,
                    "saved_state": payload["worker"]})
        message = self._receive(expected_type="restored_state", request_id=self._request_id)
        self._accept_message(message)
        self._seed = episode["seed"]
        self._steps = episode["steps"]
        self._finished = episode["finished"]
        self._initial_collectibles = episode["initial_collectibles"]
        self._death_count = episode["death_count"]
        self._last_logic_frames = episode["last_logic_frames"]
        return self.render(), self._build_info(score_delta=0, seed=self._seed)

    def close(self) -> None:
        if not self._closed:
            self._stop_worker()
            self._closed = True

    def __enter__(self) -> "PygamePacmanEnv":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _git_revision(root: Path) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    @staticmethod
    def _git_dirty(root: Path) -> bool | None:
        try:
            return bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def _start_worker(self) -> None:
        self._prepare_runtime_dir()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONHASHSEED"] = str(self._seed)
        package_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (package_root, environment.get("PYTHONPATH"))
            if part
        )
        if self.config.video_driver is not None:
            environment["SDL_VIDEODRIVER"] = self.config.video_driver
        else:
            environment.pop("SDL_VIDEODRIVER", None)
        if self.config.audio_driver is not None:
            environment["SDL_AUDIODRIVER"] = self.config.audio_driver
        try:
            process = subprocess.Popen(
                [
                    str(self.config.python_executable),
                    "-m",
                    "maapacman.env._pygame_worker",
                    "--script",
                    str(self._runtime_script),
                    "--seed",
                    str(self._seed),
                    "--ghost-mode",
                    self.config.ghost_mode,
                    "--episode-life-mode",
                    self.config.episode_life_mode,
                ],
                cwd=self._runtime_dir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except BaseException:
            self._cleanup_runtime_dir()
            raise
        self._process = process
        self._messages = queue.Queue()
        self._stderr.clear()
        threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            daemon=True,
        ).start()

    def _prepare_runtime_dir(self) -> None:
        base_dir = None
        if self.config.worker_base_dir is not None:
            base_dir = Path(self.config.worker_base_dir).resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(
            tempfile.mkdtemp(prefix="maapacman-worker-", dir=base_dir)
        ).resolve()
        runtime_script = runtime_dir / "pacman.pyw"
        source_resources = self._script.parent / "res"
        runtime_resources = runtime_dir / "res"
        try:
            shutil.copy2(self._script, runtime_script)
            if runtime_script.read_bytes() != self._script.read_bytes():
                raise PygameWorkerError("isolated pacman.pyw copy does not match source")
            try:
                os.symlink(
                    source_resources,
                    runtime_resources,
                    target_is_directory=True,
                )
                resource_mode = "source-symlink"
            except OSError:
                # Windows may deny directory symlink creation when Developer
                # Mode is disabled.  A private resource copy preserves worker
                # isolation; Linux/H100 is expected to use the read-only link.
                shutil.copytree(source_resources, runtime_resources)
                resource_mode = "private-copy"
        except BaseException:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            raise
        self._runtime_dir = runtime_dir
        self._runtime_script = runtime_script
        self._resource_mode = resource_mode

    def _cleanup_runtime_dir(self) -> None:
        runtime_dir = self._runtime_dir
        self._runtime_dir = None
        self._runtime_script = None
        self._resource_mode = None
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=False)

    def _read_stdout(self, stream: TextIO | None) -> None:
        if stream is None:
            return
        for line in stream:
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                # pygame/SDL or site customizations may print a banner before
                # the worker redirects ordinary stdout.  Keep it as diagnostic
                # context without confusing it for an IPC response.
                self._stderr.append(f"worker stdout: {line.rstrip()}")

    def _read_stderr(self, stream: TextIO | None) -> None:
        if stream is None:
            return
        for line in stream:
            self._stderr.append(line.rstrip())

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise self._worker_failure("worker is not running")
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _receive(
        self, *, expected_type: str, request_id: int | None = None
    ) -> dict[str, Any]:
        try:
            message = self._messages.get(timeout=self.config.timeout_seconds)
        except queue.Empty as exc:
            raise self._worker_failure(
                f"timed out waiting for worker {expected_type!r} response"
            ) from exc
        if message.get("type") == "error":
            raise self._worker_failure(str(message.get("error")))
        if message.get("type") != expected_type:
            raise self._worker_failure(
                f"expected {expected_type!r}, received {message.get('type')!r}"
            )
        if request_id is not None and message.get("request_id") != request_id:
            raise self._worker_failure(
                f"response id {message.get('request_id')!r} != {request_id!r}"
            )
        return message

    def _accept_message(self, message: dict[str, Any]) -> None:
        descriptor = message["frame"]
        raw = zlib.decompress(base64.b64decode(descriptor["data"]))
        if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise self._worker_failure("worker frame checksum mismatch")
        frame = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"]))
        self._frame = frame.reshape(tuple(descriptor["shape"])).copy()
        self._state = dict(message["state"])
        if not TRANSITION_STATE_FIELDS.issubset(self._state):
            missing = sorted(TRANSITION_STATE_FIELDS - self._state.keys())
            raise self._worker_failure(
                f"worker API v3 transition state is incomplete: {missing}"
            )
        ghosts = self._state.get("ghosts")
        try:
            validate_ghost_state(self._state, self.config.ghost_mode)
        except ValueError as exc:
            raise self._worker_failure(str(exc)) from exc
        if (
            not isinstance(ghosts, list)
            or any(not ATOMIC_GHOST_FIELDS.issubset(ghost) for ghost in ghosts)
        ):
            raise self._worker_failure("worker API v3 ghost state is incomplete")
        fruit = self._state.get("fruit")
        if not isinstance(fruit, dict) or not ATOMIC_FRUIT_FIELDS.issubset(fruit):
            raise self._worker_failure("worker API v3 fruit state is incomplete")
        ghost_door = self._state.get("ghost_door")
        if not isinstance(ghost_door, dict) or not ghost_door.get(
            "pacman_blocked"
        ) or ghost_door.get("ghost_blocked"):
            raise self._worker_failure(
                "worker does not implement the API v3 ghost-door ruleset"
            )
        self._last_logic_frames = int(message.get("logic_frames", 0))

    def _build_info(self, *, score_delta: int, seed: int) -> dict[str, Any]:
        remaining = int(self._state["collectibles_remaining"])
        eaten = self._initial_collectibles - remaining
        mode = int(self._state["mode"])
        terminated = mode in {2, 3, 6, 9}
        if self.config.episode_life_mode == "original_three_lives" and mode == 2:
            terminated = False
        truncated = self._steps >= self.config.max_steps and not terminated
        return {
            "env_api_version": self._spec.api_version,
            "env_id": self._spec.env_id,
            "ruleset_revision": self._spec.ruleset_revision,
            "backend": "original-pygame",
            "ghost_mode": self.config.ghost_mode,
            "episode_life_mode": self.config.episode_life_mode,
            "video_driver": self.config.video_driver or "platform-default",
            "worker_runtime_id": self._runtime_dir.name,
            "resource_mode": self._resource_mode,
            "pacman_python_revision": self._revision,
            "pacman_python_source_sha256": self._pacman_source_revision,
            "pacman_python_dirty": self._revision_dirty,
            "maapacman_revision": self._maapacman_revision,
            "maapacman_env_source_sha256": self._maapacman_source_revision,
            "maapacman_dirty": self._maapacman_revision_dirty,
            "level_revision": self._level_revision,
            "renderer_revision": self._spec.renderer_revision,
            "level": int(self._state["level"]),
            "seed": seed,
            "version_metadata": {**self.provenance, "seed": seed},
            "step": self._steps,
            "logic_frame": int(self._state["logic_frame"]),
            "pacman_position": [
                int(self._state["row"]),
                int(self._state["col"]),
            ],
            "score": int(self._state["score"]),
            "lives": int(self._state["lives"]),
            "lives_after_step": max(
                0,
                int(self._state["lives"])
                - int(
                    self.config.episode_life_mode == "single_death"
                    and mode in {2, 3}
                ),
            ),
            "death_count": self._death_count,
            "respawned": False,
            "score_delta": score_delta,
            "pellets_initial": self._initial_collectibles,
            "pellets_eaten": eaten,
            "pellets_remaining": remaining,
            "normal_pellets_remaining": int(self._state["normal_pellets"]),
            "power_pellets_remaining": int(self._state["power_pellets"]),
            "pellet_clear_rate": eaten / self._initial_collectibles,
            "legal_actions": [action.value for action in self.legal_actions()],
            "pygame_mode": int(self._state["mode"]),
            "pygame_mode_name": str(self._state.get("mode_name", "unknown")),
            "ghosts": list(self._state.get("ghosts", [])),
            "edible_ticks": int(self._state.get("edible_ticks", 0)),
            "edible_timer_started_frame": int(
                self._state.get("edible_timer_started_frame", -1)
            ),
            "ghost_value": int(self._state.get("ghost_value", 0)),
            "fruit": dict(self._state.get("fruit", {})),
            "fruit_timer": int(self._state.get("fruit_timer", 0)),
            "fruit_score_ticks": int(self._state.get("fruit_score_ticks", 0)),
            "fruit_score_position": list(
                self._state.get("fruit_score_position", [-1, -1])
            ),
            "state": json.loads(json.dumps(self._state)),
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": (
                "game_over"
                if mode == 3
                else "death"
                if mode == 2
                else "all_normal_pellets"
                if mode in {6, 9}
                else "max_steps"
                if truncated
                else None
            ),
        }

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        if process.stdin is not None:
                            process.stdin.write('{"op":"close"}\n')
                            process.stdin.flush()
                        process.wait(timeout=3)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        finally:
            self._cleanup_runtime_dir()

    def _worker_failure(self, message: str) -> PygameWorkerError:
        details = "\n".join(self._stderr)
        suffix = f"\nworker stderr:\n{details}" if details else ""
        return PygameWorkerError(message + suffix)

    def _ensure_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("environment is closed")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if not self._started or self._frame is None or self._state is None:
            raise EpisodeNotStartedError("call reset() before using the environment")
