"""Decision-boundary capture and exact restoration on the real game (dynamic-bank prerequisite)."""

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pacman_recipe.level1.contracts import make_episode_record
from slime_pacman.config import PacmanConfig
from slime_pacman.generation import Decision
from slime_pacman.rollout import EpisodeRunner
from test_slime_pacman import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _record(max_steps):
    return make_episode_record(28, split="test", recipe_root=ROOT, game_root=ROOT.parent / "pacman-python",
                               backend_root=ROOT.parent / "slime", max_steps=max_steps)


def _play(record, config, boundaries=None):
    decisions = []

    async def generate(messages, constraint):
        image = next(part["image_url"]["url"] for m in messages if isinstance(m["content"], list)
                     for part in m["content"] if part["type"] == "image_url")
        token = constraint.allowed_token_ids[0]
        decisions.append((hashlib.sha256(image.encode()).hexdigest(), messages[1]["content"][0]["text"],
                          tuple(constraint.allowed_token_ids), token))
        return Decision("prompt", [1, 2], token, chr(token), constraint.allowed_token_ids,
                        -float(np.log(len(constraint.allowed_token_ids))), "v0", {}, "0" * 64, [])

    runner = EpisodeRunner(record, tokenizer=Tokenizer(), generate=generate, config=config)
    if boundaries is not None:
        runner.decision_boundary_sink = boundaries.append
    result = asyncio.run(runner.collect(empty_weight_version="v0"))
    return result, decisions


def _restart_record(record, boundary, tmp_path, name="boundary"):
    saved = dict(boundary["env_state"], boundary_context=boundary["context"])
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(saved))
    restarted = deepcopy(record)
    restarted["restart"] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                id=f"{name}-test")
    return restarted


@pytest.fixture(scope="module")
def original():
    config = PacmanConfig(max_steps=32)
    record = _record(32)
    boundaries = []
    result, decisions = _play(record, config, boundaries)
    assert len(boundaries) == len(decisions) >= 4
    return config, record, boundaries, result, decisions


def test_boundaries_capture_env_planner_and_runner_context(original):
    _, _, boundaries, _, decisions = original
    first, later = boundaries[0], boundaries[-1]
    assert first["schema"] == "pacman-decision-boundary-v1"
    assert set(first["context"]) == {"planner", "runner", "verify"}
    assert first["env_state"]["payload"]["episode"]["steps"] <= later["env_state"]["payload"]["episode"]["steps"]
    assert later["context"]["runner"]["recent_actions"] and later["context"]["planner"]["remaining"]
    assert all(b["context"]["verify"]["allowed_token_ids"] == list(d[2]) for b, d in zip(boundaries, decisions))
    json.dumps(boundaries)  # JSON-safe for bank files
    features = later["features"]
    assert set(features) == {"pacman_position", "normal_pellets_remaining", "nearest_lethal_ghost_distance", "env_step"}
    assert features["env_step"] == later["env_state"]["payload"]["episode"]["steps"]


def test_restored_boundary_reproduces_the_original_suffix(original, tmp_path):
    config, record, boundaries, result, decisions = original
    k = len(boundaries) // 2
    resumed, resumed_decisions = _play(_restart_record(record, boundaries[k], tmp_path), config)
    assert resumed_decisions == decisions[k:]
    assert resumed.reward == result.reward and resumed.terminal_reason == result.terminal_reason


def test_tampered_boundary_is_rejected(original, tmp_path):
    config, record, boundaries, _, _ = original
    tampered = deepcopy(boundaries[1])
    tampered["context"]["verify"]["png_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="restored decision boundary differs"):
        _play(_restart_record(record, tampered, tmp_path, "tampered"), config)


def test_death_candidates_file_holds_restorable_boundaries(original, tmp_path):
    from slime_pacman import dynamic_bank as db
    from slime_pacman.rollout import EpisodeResult, write_death_candidates

    config, record, boundaries, result, decisions = original
    died = EpisodeResult(0.0, [], "v0", None, "death")
    path = write_death_candidates(tmp_path / "c" / "ep.json", boundaries, died, dict(artifact_id="ep"))
    payload = json.loads(path.read_text())
    assert payload["schema"] == db.CANDIDATE_SCHEMA and payload["source"]["terminal_reason"] == "death"
    expected = [d for d in db.DISTANCES if len(boundaries) > d]
    assert [c["distance"] for c in payload["candidates"]] == expected
    assert payload["last_choice_position"] == boundaries[-1]["features"]["pacman_position"]
    first = payload["candidates"][0]
    k = len(boundaries) - 1 - first["distance"]
    resumed, resumed_decisions = _play(_restart_record(record, first["boundary"], tmp_path, "cand"), config)
    assert resumed_decisions == decisions[k:]
    won = EpisodeResult(1.0, [], "v0", None, "all_normal_pellets")
    assert write_death_candidates(tmp_path / "c" / "won.json", boundaries, won, {}) is None
