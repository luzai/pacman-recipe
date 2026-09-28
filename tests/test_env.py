from pathlib import Path

from pacman_recipe.baselines import GreedyPelletAgent
from pacman_recipe.env import PacmanEnv, layout_by_name
from pacman_recipe.evaluate import run_episode
from pacman_recipe.maze_suite import SPLIT_SIZES, maze_records, suite_summary
from pacman_recipe.vision import env_png_bytes, image_sha256, render_env_image


def test_reset_and_observation_contains_actions():
    env = PacmanEnv()
    obs = env.observation_text()
    assert "Legal actions" in obs
    assert "Grid" in obs
    assert env.state.steps == 0


def test_invalid_action_rejected():
    env = PacmanEnv()
    try:
        env.step("jump")
    except ValueError as exc:
        assert "invalid action" in str(exc)
    else:
        raise AssertionError("invalid action should fail")


def test_greedy_episode_finishes():
    result = run_episode(GreedyPelletAgent(), max_steps=80)
    assert result["steps"] <= 80
    assert result["reason"] in {"all_pellets", "caught", "max_steps"}


def test_small_default_layout_available():
    layout = layout_by_name("small_default")
    env = PacmanEnv(layout_name="small_default", max_steps=30)
    assert layout[1].startswith("#P..")
    assert env.layout_name == "small_default"


def test_medium_default_layout_available():
    layout = layout_by_name("medium_default")
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    assert sum(row.count(".") for row in layout) == 5
    assert env.layout_name == "medium_default"


def test_multi_maze_suite_has_disjoint_expected_splits():
    summary = suite_summary()
    assert summary["layout_count"] == sum(SPLIT_SIZES.values()) == 112
    assert summary["unique_layout_hashes"] == 112
    assert summary["unique_topology_hashes"] == 112
    assert dict(summary["split_sizes"]) == SPLIT_SIZES
    hashes_by_split = {
        split: {record["layout_hash"] for record in maze_records(split)}
        for split in SPLIT_SIZES
    }
    topologies_by_split = {
        split: {record["topology_hash"] for record in maze_records(split)}
        for split in SPLIT_SIZES
    }
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["validation"])
    assert hashes_by_split["train"].isdisjoint(hashes_by_split["test"])
    assert hashes_by_split["validation"].isdisjoint(hashes_by_split["test"])
    assert topologies_by_split["train"].isdisjoint(topologies_by_split["validation"])
    assert topologies_by_split["train"].isdisjoint(topologies_by_split["test"])
    assert topologies_by_split["validation"].isdisjoint(topologies_by_split["test"])


def test_multi_maze_suite_is_declared_as_wheel_package_data():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"pacman_recipe.synthetic" = ["maze_suites/*.json"]' in pyproject


def test_every_multi_maze_oracle_solution_replays_to_win():
    for record in maze_records():
        env = PacmanEnv(layout_name=record["name"], max_steps=30)
        for action in record["oracle_actions"]:
            _, _, done, _ = env.step(action)
            if done:
                break
        assert env.state.won, record["name"]
        assert env.state.steps == record["oracle_steps"]


def test_image_observation_is_deterministic_and_sized():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    image = render_env_image(env, tile_size=16)
    assert image.size == (9 * 16, 6 * 16)
    assert image.mode == "RGB"
    first = env_png_bytes(env, tile_size=16)
    second = env_png_bytes(env, tile_size=16)
    assert first == second
    assert image_sha256(first) == image_sha256(second)


def test_image_observation_hash_changes_after_move():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    before = image_sha256(env_png_bytes(env, tile_size=16))
    env.step("right")
    after = image_sha256(env_png_bytes(env, tile_size=16))
    assert before != after


def test_ghost_legal_strict_observation_lists_forbidden_tokens():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_strict")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "Forbidden output tokens now: up, left" in obs


def test_route_prefix_stay_penalty_is_hidden_reward_only():
    env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="route_prefix_stay_penalty")
    obs = env.observation_text(prompt_style="ghost_legal_strict")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "route" not in obs.lower()
    _, stay_reward, _, _ = env.step("stay")

    env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="route_prefix_stay_penalty")
    _, route_reward, _, info = env.step("right")
    assert info["route_action"] is True
    assert stay_reward < route_reward


def test_route_prefix_progress_penalty_strongly_discourages_stay_without_prompt_hint():
    env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="route_prefix_progress_penalty")
    obs = env.observation_text(prompt_style="ghost_legal_strict")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "route" not in obs.lower()
    assert "distance" not in obs.lower()
    _, stay_reward, _, _ = env.step("stay")

    env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="route_prefix_progress_penalty")
    _, route_reward, _, info = env.step("right")
    assert info["route_action"] is True
    assert stay_reward <= route_reward - 20


def test_route_shaping_scale_zero_matches_sparse_for_non_terminal_move():
    sparse_env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="sparse")
    shaped_env = PacmanEnv(
        layout_name="medium_default",
        max_steps=30,
        reward_mode="route_prefix_progress_penalty",
        route_shaping_scale=0.0,
    )

    _, sparse_reward, _, _ = sparse_env.step("right")
    _, shaped_reward, _, info = shaped_env.step("right")

    assert info["route_action"] is True
    assert shaped_reward == sparse_reward


def test_route_shaping_scale_half_interpolates_hidden_terms():
    full_env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="route_prefix_progress_penalty")
    half_env = PacmanEnv(
        layout_name="medium_default",
        max_steps=30,
        reward_mode="route_prefix_progress_penalty",
        route_shaping_scale=0.5,
    )
    sparse_env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="sparse")

    _, full_reward, _, _ = full_env.step("stay")
    _, half_reward, _, _ = half_env.step("stay")
    _, sparse_reward, _, _ = sparse_env.step("stay")

    assert half_reward == sparse_reward + 0.5 * (full_reward - sparse_reward)


def test_safe_distance_uses_ghost_phase_and_collision_free_transitions():
    env = PacmanEnv(
        layout=(
            "#######",
            "#P G. #",
            "#######",
        ),
        max_steps=10,
    )

    assert env._nearest_pellet_distance(env.state.pacman) == 3
    assert env.safe_distance() == 11


def test_safe_progress_reward_matches_potential_difference():
    layout = (
        "########",
        "#P  .###",
        "#####G##",
        "########",
    )
    sparse_env = PacmanEnv(layout=layout, max_steps=10, reward_mode="sparse")
    shaped_env = PacmanEnv(layout=layout, max_steps=10, reward_mode="safe_progress", safe_progress_alpha=2.5)

    _, sparse_reward, _, _ = sparse_env.step("right")
    state, shaped_reward, _, info = shaped_env.step("right")

    assert info["safe_distance_before"] == 3
    assert info["safe_distance_after"] == 2
    assert info["safe_progress_reward"] == 2.5
    assert shaped_reward == sparse_reward + 2.5
    assert state.score == shaped_reward


def test_safe_progress_alpha_zero_matches_sparse_reward():
    sparse_env = PacmanEnv(layout_name="medium_default", max_steps=30, reward_mode="sparse")
    shaped_env = PacmanEnv(
        layout_name="medium_default",
        max_steps=30,
        reward_mode="safe_progress",
        safe_progress_alpha=0.0,
    )

    _, sparse_reward, _, _ = sparse_env.step("down")
    _, shaped_reward, _, info = shaped_env.step("down")

    assert info["safe_distance_before"] is not None
    assert info["safe_distance_after"] is not None
    assert info["safe_progress_reward"] == 0.0
    assert shaped_reward == sparse_reward


def test_safe_progress_telescopes_on_every_multi_maze_oracle_solution():
    alpha = 4.0
    for record in maze_records():
        sparse_env = PacmanEnv(layout_name=record["name"], max_steps=30, reward_mode="sparse")
        shaped_env = PacmanEnv(
            layout_name=record["name"],
            max_steps=30,
            reward_mode="safe_progress",
            safe_progress_alpha=alpha,
        )
        initial_safe_distance = shaped_env.safe_distance()
        sparse_total = 0.0
        shaped_total = 0.0
        for action in record["oracle_actions"]:
            _, sparse_reward, _, _ = sparse_env.step(action)
            _, shaped_reward, done, _ = shaped_env.step(action)
            sparse_total += sparse_reward
            shaped_total += shaped_reward
            if done:
                break

        assert shaped_env.state.won, record["name"]
        assert shaped_total == sparse_total + alpha * initial_safe_distance, record["name"]


def test_ghost_legal_reason_observation_requests_action_line():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_reason")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "Briefly explain" in obs
    assert "Action: <token>" in obs


def test_ghost_legal_json_observation_requests_json_object():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_json")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "JSON object" in obs
    assert '"action":"one_allowed_token"' in obs
    assert "makes progress through the maze toward remaining pellets" in obs


def test_ghost_legal_json_fast_think_observation_requests_brief_thinking():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_json_fast_think")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "Think briefly and quickly" in obs
    assert "wall-routing" in obs
    assert "JSON object" in obs


def test_ghost_legal_json_first_observation_requests_json_first():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_json_first")
    assert "Allowed output tokens now: down, right, stay" in obs
    assert "Start your answer with `{`" in obs
    assert "Do not choose stay unless it is the only allowed token" in obs
    assert "JSON object" in obs


def test_ghost_legal_route_json_observation_lists_shortest_route_actions():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_route_json")
    assert "Shortest-route pellet actions now: right" in obs
    env.step("right")
    env.step("right")
    obs = env.observation_text(prompt_style="ghost_legal_route_json")
    assert "Shortest-route pellet actions now: down" in obs


def test_ghost_legal_distance_json_observation_lists_pellet_distances():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = env.observation_text(prompt_style="ghost_legal_distance_json")
    assert "Pellet distance after action:" in obs
    assert "right=0" in obs
    env.step("right")
    env.step("right")
    obs = env.observation_text(prompt_style="ghost_legal_distance_json")
    assert "down=5" in obs
    assert "left=7" in obs
