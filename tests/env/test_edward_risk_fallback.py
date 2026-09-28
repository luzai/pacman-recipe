from dataclasses import replace
from unittest.mock import patch

import pytest

from pacman_env.planner import EdwardPlanner, EdwardSafetyRefusal
from pacman_env.env.state import Position
from .test_edward_planner import GridLevel, positions


def corridor():
    return GridLevel(7, 3, positions(*((1, col) for col in range(1, 6))), frozenset())


def state(**overrides):
    return {
        "pacman_position": [1, 3], "open": ["L", "R"],
        "ghosts": [{"id": 0, "state": "normal", "position": [1, 5]}],
        **overrides,
    }


def test_normal_options_take_priority_and_keep_legacy_serialization():
    level = replace(corridor(), pellets=positions((1, 1)))
    normal = EdwardPlanner(level).advertised_candidates(state(ghosts=[]))
    opted_in = EdwardPlanner(level, fallback_mode="risk_ranked").advertised_candidates(state(ghosts=[]))
    assert normal == opted_in
    assert normal and all("risk" not in candidate.as_dict() for candidate in normal)


def test_legacy_emergency_and_refusal_remain_default():
    legacy = EdwardPlanner(corridor())
    candidates = legacy.advertised_candidates(state(pacman_position=[1, 2]))
    assert len(candidates) == 1 and candidates[0].strategy == "AVOID"
    with pytest.raises(EdwardSafetyRefusal):
        legacy.advertised_candidates(state(ghosts=[
            {"state": "normal", "position": [1, 2]},
            {"state": "normal", "position": [1, 4]},
        ]))


def test_risk_mode_keeps_dangerous_directions_and_ends_after_one_step():
    planner = EdwardPlanner(corridor(), fallback_mode="risk_ranked")
    candidates = planner.advertised_candidates(state())
    assert [c.first_action for c in candidates] == ["L", "R"]
    assert [c.option_id for c in candidates] == ["A0", "A1"]
    assert [c.risk["rank"] for c in candidates] == [1, 2]
    for candidate in candidates:
        assert candidate.strategy == "RISK_FALLBACK"
        assert candidate.commit_moves == candidate.route_distance == 1
        assert candidate.safety_margin is candidate.future_safe_exits is None
        assert candidate.risk["motion"] == "unknown"
        assert planner.continue_option(candidate, state()) == (None, "max_commit")


def test_pixel_collision_ranked_last_but_not_removed():
    planner = EdwardPlanner(corridor(), fallback_mode="risk_ranked")
    candidates = planner.advertised_candidates(state(ghosts=[{
        "id": 0, "state": "normal", "position": [1, 5],
        "pixel_position": [16, 80], "direction": "L", "path_remaining": "LLLL", "speed": 1,
    }]))
    assert [c.first_action for c in candidates] == ["L", "R"]
    assert [c.risk["motion"] for c in candidates] == ["clear_estimate", "collision_predicted"]


@pytest.mark.parametrize("missing", ["pixel_position", "direction", "path_remaining", "speed"])
def test_missing_motion_is_unknown(missing):
    ghost = dict(state="normal", position=[1, 5], pixel_position=[16, 80],
                 direction="L", path_remaining="LLLL", speed=1)
    del ghost[missing]
    candidates = EdwardPlanner(corridor(), fallback_mode="risk_ranked").advertised_candidates(state(ghosts=[ghost]))
    assert all(c.risk["motion"] == "unknown" for c in candidates)


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -1, "bad"])
def test_invalid_motion_is_unknown(speed):
    ghost = dict(state="normal", position=[1, 5], pixel_position=[16, 80],
                 direction="L", path_remaining="LLLL", speed=speed)
    candidates = EdwardPlanner(corridor(), fallback_mode="risk_ranked").advertised_candidates(state(ghosts=[ghost]))
    assert all(c.risk["motion"] == "unknown" for c in candidates)


def test_all_four_directions_and_deterministic_ties():
    level = GridLevel(5, 5, positions((2, 2), (1, 2), (3, 2), (2, 1), (2, 3)), frozenset())
    planner = EdwardPlanner(level, fallback_mode="risk_ranked")
    snapshot = state(pacman_position=[2, 2], open=["R", "D", "U", "L"], ghosts=[])
    first = planner.advertised_candidates(snapshot)
    assert [c.first_action for c in first] == ["U", "D", "L", "R"]
    assert planner.advertised_candidates(snapshot) == first
    planner.record_action("D")
    assert planner.advertised_candidates(snapshot)[-1].first_action == "U"


@pytest.mark.parametrize("opened", [[], ["S"], ["U"]])
def test_no_physically_open_cardinal_move_still_refuses(opened):
    planner = EdwardPlanner(corridor(), fallback_mode="risk_ranked")
    with pytest.raises(EdwardSafetyRefusal):
        planner.advertised_candidates(state(open=opened, legal_actions=["L", "R"]))


def test_every_direction_survives_even_when_all_predict_collision():
    planner = EdwardPlanner(corridor(), fallback_mode="risk_ranked")
    with patch("pacman_env.planner._fallback_motion_risk", return_value="collision_predicted"):
        candidates = planner.advertised_candidates(state())
    assert {c.first_action for c in candidates} == {"L", "R"}


def test_invalid_mode_fails_closed():
    with pytest.raises(ValueError, match="edward_fallback_mode"):
        EdwardPlanner(fallback_mode="risk_rnaked")


def test_fallback_uses_real_tunnel_destination():
    planner = EdwardPlanner(fallback_mode="risk_ranked")
    planner.remaining.clear()
    candidates = planner.advertised_candidates(state(pacman_position=[12, 1], ghosts=[]))
    left = next(candidate for candidate in candidates if candidate.first_action == "L")
    target = planner.level.portal_exit(Position(12, 0), "L")
    assert left.target == (target.row, target.col)
    assert left.commit_moves == 1


def test_fallback_cannot_cross_ghost_door():
    level = replace(corridor(), ghost_doors=positions((1, 4)))
    planner = EdwardPlanner(level, fallback_mode="risk_ranked")
    candidates = planner.advertised_candidates(state(ghosts=[]))
    assert [candidate.first_action for candidate in candidates] == ["L"]


def test_known_collision_not_hidden_by_another_ghost_with_unknown_motion():
    planner = EdwardPlanner(corridor(), fallback_mode="risk_ranked")
    known = dict(state="normal", position=[1, 5], pixel_position=[16, 80],
                 direction="L", path_remaining="LLLL", speed=1)
    for ghosts in ([state()["ghosts"][0], known], [known, state()["ghosts"][0]]):
        candidates = planner.advertised_candidates(state(ghosts=ghosts))
        right = next(candidate for candidate in candidates if candidate.first_action == "R")
        assert right.risk["motion"] == "collision_predicted"
