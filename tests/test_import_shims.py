from __future__ import annotations

from pacman_recipe import baselines as old_baselines
from pacman_recipe import dataset as old_dataset
from pacman_recipe import env as old_env
from pacman_recipe import evaluate as old_evaluate
from pacman_recipe import level1_dataset as old_level1_dataset
from pacman_recipe import maze_suite as old_maze_suite
from pacman_recipe import prompts as old_prompts
from pacman_recipe import rewards as old_rewards
from pacman_recipe import trajectories as old_trajectories
from pacman_recipe import vision as old_vision
from pacman_recipe.level1 import level1_dataset, prompts, rewards, trajectories
from pacman_recipe.synthetic import (
    baselines,
    dataset,
    env,
    evaluate,
    maze_suite,
    vision,
)


def test_synthetic_import_shims_preserve_public_objects() -> None:
    assert old_env.PacmanEnv is env.PacmanEnv
    assert old_env.PacmanState is env.PacmanState
    assert old_baselines.GreedyPelletAgent is baselines.GreedyPelletAgent
    assert old_evaluate.run_episode is evaluate.run_episode
    assert old_maze_suite.maze_records is maze_suite.maze_records
    assert old_dataset.generate_episode_specs is dataset.generate_episode_specs
    assert old_vision.render_env_image is vision.render_env_image


def test_level1_import_shims_preserve_public_objects_without_areal() -> None:
    assert (
        old_level1_dataset.generate_episode_rows
        is level1_dataset.generate_episode_rows
    )
    assert old_prompts.build_image_messages is prompts.build_image_messages
    assert old_rewards.shape_reward is rewards.shape_reward
    assert old_trajectories.audit_trajectory is trajectories.audit_trajectory
