import asyncio
import json
import sys
import types
from pathlib import Path

from pacman_recipe.areal_workflow import (
    IMAGE_ONLY_SYSTEM_PROMPT,
    PARSE_FAILED_ACTION,
    PacmanEpisodeWorkflow,
    PacmanImageOnlyVLMWorkflow,
    PacmanWorkflow,
    build_observation_turn,
    guided_choices_for_prompt,
    parse_action,
    parse_action_for_prompt,
    raw_completion_prompt,
)
from pacman_recipe.analyze_trajectories import format_trajectory, step_legal_action, summarize
from pacman_recipe.render_trajectory_video import (
    compose_frames,
    compose_single_frames,
    final_grid_after_action,
    grid_from_obs,
)
from pacman_recipe.run_vision_baselines import (
    compose_vision_video_frames,
    run_vision_episode,
)
from pacman_recipe.dataset import generate_episode_specs, generate_examples, generate_multi_maze_episode_specs
from pacman_recipe.env import ACTIONS, PacmanEnv
from pacman_recipe.prepare_vision_sft_dataset import generate_shortest_route_sft
from pacman_recipe.train_vision_sft_smoke import iter_batches, prepare_rows


def test_dataset_examples_have_answer():
    example = next(generate_examples(episodes=1, max_steps=10))
    assert example["answer"] in {"up", "down", "left", "right", "stay"}
    assert example["messages"][0]["role"] == "system"
    assert "Game rules:" in example["messages"][0]["content"]
    assert "`#` is a wall" in example["messages"][0]["content"]
    assert "Touching the ghost" in example["messages"][0]["content"]


def test_parse_action_from_sentence():
    assert parse_action("I choose RIGHT.") == "right"


def test_parse_action_prefers_explicit_action_line():
    text = "Reason: left is blocked, down loops, right eats a pellet.\nAction: right"
    assert parse_action(text) == "right"


def test_parse_action_reads_json_action():
    assert parse_action('{"reason":"right eats a pellet","action":"right"}') == "right"


def test_guided_choices_for_json_prompt_are_legal_json_actions():
    choices = guided_choices_for_prompt("ghost_legal_json", ["right", "stay"])
    assert choices == [
        '{"reason":"choose right","action":"right"}',
        '{"reason":"choose stay","action":"stay"}',
    ]
    assert [parse_action(choice, allowed_actions=["right", "stay"]) for choice in choices] == ["right", "stay"]


def test_guided_choices_for_action_prompt_are_legal_action_tokens():
    assert guided_choices_for_prompt("ghost_legal", ["down", "stay"]) == ["down", "stay"]


def test_renderer_can_synthesize_terminal_win_frame():
    step = {
        "obs": "Grid:\n#####\n#P. #\n#####\n",
        "action": "right",
        "done": True,
    }
    assert final_grid_after_action(step) == ["#####", "# P #", "#####"]


def test_single_renderer_adds_terminal_win_frame():
    episode = {
        "won": True,
        "total_reward": 10,
        "steps": 1,
        "trajectory": [
            {
                "step": 1,
                "obs": "Grid:\n#####\n#P. #\n#####\n",
                "action": "right",
                "reward": 10,
                "done": True,
            }
        ],
    }
    assert len(compose_single_frames(episode, "success", fps=1, title="t", subtitle="s")) == 4


def test_legal_action_choices_for_strict_prompt_are_current_legal_tokens_only():
    assert guided_choices_for_prompt("ghost_legal_strict", ["right", "stay"]) == ["right", "stay"]


def test_non_stay_legal_action_choices_drop_stay_when_movement_exists():
    assert guided_choices_for_prompt(
        "ghost_legal_strict",
        ["right", "stay"],
        exclude_stay_when_possible=True,
    ) == ["right"]
    assert guided_choices_for_prompt(
        "ghost_legal_strict",
        ["stay"],
        exclude_stay_when_possible=True,
    ) == ["stay"]


def test_non_backtracking_choices_drop_reverse_when_other_movement_exists():
    assert guided_choices_for_prompt(
        "ghost_legal_strict",
        ["up", "down", "stay"],
        exclude_stay_when_possible=True,
        exclude_reverse_action="up",
    ) == ["up"]
    assert guided_choices_for_prompt(
        "ghost_legal_strict",
        ["down", "stay"],
        exclude_stay_when_possible=True,
        exclude_reverse_action="up",
    ) == ["down"]


def test_parse_action_reads_trailing_json_after_thinking():
    text = '<think>right eats a pellet.</think>\n\n{"reason":"right eats a pellet","action":"right"}'
    assert parse_action(text, allow_word_fallback=False) == "right"


def test_parse_action_strict_mode_ignores_loose_action_words():
    text = "I considered left and right but forgot the final action line."
    assert parse_action(text, allow_word_fallback=False) == "stay"


def test_parse_action_can_signal_failure_without_stay_fallback():
    assert parse_action("```", fail_on_no_action=True) == PARSE_FAILED_ACTION


def test_distance_json_prompt_allows_explanatory_fallback():
    text = "Right has distance 0, so move right to eat the pellet."
    assert parse_action_for_prompt(text, "ghost_legal_distance_json") == "right"


def test_ghost_legal_json_prompt_allows_explanatory_fallback():
    text = "I checked the legal moves. Right eats the nearest pellet, so move right."
    assert parse_action_for_prompt(text, "ghost_legal_json") == "right"


def test_ghost_legal_strict_rejects_chat_template_preamble_as_parse_failure():
    text = "The user wants me to play Pac-Man and stay safe."
    assert (
        parse_action_for_prompt(text, "ghost_legal_strict", fail_on_no_action=True)
        == PARSE_FAILED_ACTION
    )


def test_ghost_legal_strict_accepts_exact_action_token():
    assert parse_action_for_prompt("right", "ghost_legal_strict", fail_on_no_action=True) == "right"


def test_guided_choices_all_actions_can_ignore_current_legal_actions():
    assert guided_choices_for_prompt("ghost_legal_strict", allowed_actions=["right", "stay"]) == ["right", "stay"]
    assert list(ACTIONS) == ["up", "down", "left", "right", "stay"]


def test_raw_completion_prompt_uses_action_suffix():
    prompt = raw_completion_prompt("ghost_legal_strict", "Allowed output tokens now: right, stay.")
    assert "Game rules:" in prompt
    assert "Allowed output tokens now: right, stay." in prompt
    assert prompt.endswith("Action:")
    assert '"role"' not in prompt


def test_image_text_observation_builds_multimodal_message_content():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = build_observation_turn(
        env,
        prompt_style="ghost_legal_strict",
        observation_mode="image_text",
        vision_tile_size=16,
    )
    assert isinstance(obs.model_observation, list)
    assert obs.model_observation[0]["type"] == "text"
    assert "Allowed output tokens now: down, right, stay." in obs.model_observation[0]["text"]
    assert "Grid:" not in obs.model_observation[0]["text"]
    assert obs.model_observation[1]["type"] == "image_url"
    assert obs.model_observation[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert obs.image_sha256 is not None
    assert len(obs.image_sha256) == 64


def test_shortest_route_vision_sft_export_contains_image_and_answer(tmp_path):
    jsonl = tmp_path / "vision_sft.jsonl"
    images_dir = tmp_path / "images"
    count, summary = generate_shortest_route_sft(
        output_jsonl=jsonl,
        images_dir=images_dir,
        episodes=1,
        max_steps=30,
        layout_name="medium_default",
        vision_tile_size=16,
    )
    assert count == 10
    assert summary["wins"] == 1
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    first = rows[0]
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2] == {"role": "assistant", "content": "right"}
    assert first["answer"] == "right"
    assert first["obs_image_sha256"]
    assert Path(first["image_path"]).exists()
    user_content = first["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "Allowed output tokens now" in user_content[0]["text"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user_content[2]["type"] == "image_path"


def test_vision_sft_smoke_prepares_qwen_messages(tmp_path):
    jsonl = tmp_path / "vision_sft.jsonl"
    images_dir = tmp_path / "images"
    generate_shortest_route_sft(
        output_jsonl=jsonl,
        images_dir=images_dir,
        episodes=1,
        max_steps=30,
        layout_name="medium_default",
        vision_tile_size=16,
    )
    rows = prepare_rows(jsonl, limit=2)
    assert len(rows) == 2
    first = rows[0]
    assert first.answer == "right"
    assert first.image_path.exists()
    assert first.prompt_messages[1]["content"][0]["type"] == "image"
    assert first.prompt_messages[1]["content"][1]["type"] == "text"
    assert first.full_messages[-1] == {"role": "assistant", "content": "right"}


def test_vision_sft_batching_is_seeded_and_complete(tmp_path):
    jsonl = tmp_path / "vision_sft.jsonl"
    images_dir = tmp_path / "images"
    generate_shortest_route_sft(
        output_jsonl=jsonl,
        images_dir=images_dir,
        episodes=1,
        max_steps=30,
        layout_name="medium_default",
        vision_tile_size=16,
    )
    rows = prepare_rows(jsonl, limit=5)
    plain_batches = iter_batches(rows, batch_size=2, shuffle=False, seed=0)
    assert [[row.row_id for row in batch] for batch in plain_batches] == [
        ["vision-sft-0-0", "vision-sft-0-1"],
        ["vision-sft-0-2", "vision-sft-0-3"],
        ["vision-sft-0-4"],
    ]
    shuffled_a = iter_batches(rows, batch_size=2, shuffle=True, seed=7)
    shuffled_b = iter_batches(rows, batch_size=2, shuffle=True, seed=7)
    assert [[row.row_id for row in batch] for batch in shuffled_a] == [
        [row.row_id for row in batch] for batch in shuffled_b
    ]
    assert sorted(row.row_id for batch in shuffled_a for row in batch) == [row.row_id for row in rows]


def test_image_observation_omits_text_metadata_except_rules():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = build_observation_turn(
        env,
        prompt_style="ghost_legal_strict",
        observation_mode="image",
        vision_tile_size=16,
    )
    assert isinstance(obs.model_observation, list)
    assert "Allowed output tokens now" not in obs.model_observation[0]["text"]
    assert "Grid:" not in obs.model_observation[0]["text"]


def test_image_only_observation_has_no_maze_or_state_text_metadata():
    env = PacmanEnv(layout_name="medium_default", max_steps=30)
    obs = build_observation_turn(
        env,
        prompt_style="ghost_legal_strict",
        observation_mode="image_only",
        vision_tile_size=16,
    )
    assert isinstance(obs.model_observation, list)
    text = obs.model_observation[0]["text"]
    assert text == "Choose exactly one action token: up, down, left, right, or stay."
    assert "Blue tiles" not in text
    assert "Allowed output tokens now" not in text
    assert "Forbidden output tokens now" not in text
    assert "Pellets left" not in text
    assert "Grid:" not in text
    assert obs.model_observation[1]["type"] == "image_url"
    assert obs.model_observation[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_parser_masks_to_allowed_actions():
    text = "Up would hit a wall. Down has distance 0, so move down."
    assert parse_action_for_prompt(text, "ghost_legal_distance_json", allowed_actions=["down", "stay"]) == "down"


def test_parser_falls_back_to_legal_stay_when_no_allowed_action_is_named():
    text = "Up and left are tempting but both are blocked."
    assert parse_action_for_prompt(text, "ghost_legal_distance_json", allowed_actions=["down", "stay"]) == "stay"


def test_workflow_dry_run_reward():
    example = next(generate_examples(episodes=1, max_steps=10))
    reward = asyncio.run(PacmanWorkflow().run(example, completion=example["answer"]))
    assert reward == 1.0


def test_episode_specs_have_no_ground_truth_answer():
    example = next(generate_episode_specs(episodes=1, max_steps=10))
    assert "answer" not in example
    assert example["seed"] == 0
    assert example["messages"][0]["role"] == "system"


def test_multi_maze_episode_specs_cover_split_with_auditable_metadata():
    rows = list(
        generate_multi_maze_episode_specs(
            split="validation",
            episodes_per_layout=2,
            max_steps=30,
            prompt_style="ghost_legal_strict",
        )
    )
    assert len(rows) == 32
    assert len({row["layout_name"] for row in rows}) == 16
    assert len({row["layout_hash"] for row in rows}) == 16
    assert len({row["topology_hash"] for row in rows}) == 16
    assert all(row["maze_suite"] == "multi_maze_v1" for row in rows)
    assert all(row["maze_split"] == "validation" for row in rows)
    assert {row["rollout_index"] for row in rows} == {0, 1}
    assert len({row["id"] for row in rows}) == len(rows)


def test_legal_prompt_episode_specs_include_ablation_fields():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="legal",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "legal"
    assert example["illegal_action_penalty"] == -4
    assert "Legal actions list" in example["messages"][0]["content"]
    assert "Game rules:" in example["messages"][0]["content"]
    assert "extra penalty" in example["messages"][0]["content"]
    assert "LEGAL action" in example["messages"][1]["content"]


def test_ghost_legal_prompt_episode_specs_include_unsafe_actions():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "ghost_legal"
    assert "Unsafe immediate ghost actions" in example["messages"][0]["content"]
    assert "Unsafe immediate ghost actions" in example["messages"][1]["content"]
    assert "Do not stay unless" in example["messages"][0]["content"]


def test_ghost_legal_reason_prompt_allows_debug_explanation():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_reason",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "ghost_legal_reason"
    assert "debug run" in example["messages"][0]["content"]
    assert "Action: <token>" in example["messages"][0]["content"]
    assert "Allowed output tokens now" in example["messages"][1]["content"]
    assert "Briefly explain" in example["messages"][1]["content"]


def test_ghost_legal_json_prompt_requests_strict_json():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_json",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "ghost_legal_json"
    assert "exactly one JSON object" in example["messages"][0]["content"]
    assert "`action` value" in example["messages"][0]["content"]
    assert "shortest route to remaining pellets" in example["messages"][0]["content"]
    assert "Allowed output tokens now" in example["messages"][1]["content"]
    assert "No markdown, no extra text" in example["messages"][1]["content"]


def test_ghost_legal_json_concise_prompt_blocks_prose_analysis():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_json_concise",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "ghost_legal_json_concise"
    assert "concise JSON-only" in example["messages"][0]["content"]
    assert "Do not analyze the maze in prose" in example["messages"][1]["content"]
    assert "Start immediately with `{`" in example["messages"][1]["content"]
    assert (
        parse_action_for_prompt(
            "Right is good because it eats a pellet.",
            "ghost_legal_json_concise",
            fail_on_no_action=True,
        )
        == PARSE_FAILED_ACTION
    )


def test_ghost_legal_json_fast_think_prompt_limits_reasoning():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_json_fast_think",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "ghost_legal_json_fast_think"
    assert "concise thinking debug run" in example["messages"][0]["content"]
    assert "at most three short internal reasoning bullets" in example["messages"][0]["content"]
    assert "Think briefly and quickly" in example["messages"][1]["content"]
    assert "No markdown, no extra text" in example["messages"][1]["content"]


def test_ghost_legal_route_json_prompt_includes_route_feature():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_route_json",
            illegal_action_penalty=-4,
            layout_name="medium_default",
        )
    )
    assert example["prompt_style"] == "ghost_legal_route_json"
    assert "Shortest-route pellet actions" in example["messages"][0]["content"]
    assert "Shortest-route pellet actions now: right" in example["messages"][1]["content"]


def test_ghost_legal_distance_json_prompt_includes_distance_feature():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=10,
            prompt_style="ghost_legal_distance_json",
            illegal_action_penalty=-4,
            layout_name="medium_default",
        )
    )
    assert example["prompt_style"] == "ghost_legal_distance_json"
    assert "Pellet distance after action" in example["messages"][0]["content"]
    assert "Pellet distance after action:" in example["messages"][1]["content"]


def test_teacher_hint_prompt_episode_specs_include_teacher_action():
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=30,
            prompt_style="teacher_hint",
            illegal_action_penalty=-4,
        )
    )
    assert example["prompt_style"] == "teacher_hint"
    assert "Teacher action hint" in example["messages"][0]["content"]
    assert "Teacher action hint: down" in example["messages"][1]["content"]


def test_teacher_hint_pure_prompt_does_not_enforce_teacher_action(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=30,
            prompt_style="teacher_hint_pure",
            illegal_action_penalty=-4,
        )
    )
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["stay"] * 30,
            trajectory_dir=tmp_path,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert reward < 264.0
    assert payload["won"] is False
    assert payload["trajectory"][0]["model_output"] == "stay"
    assert payload["trajectory"][0]["action"] == "stay"
    assert "teacher_action" not in payload["trajectory"][0]


def test_tiny_corridor_episode_can_win_without_teacher_hint(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=4,
            prompt_style="ghost_legal",
            illegal_action_penalty=-4,
            layout_name="tiny_corridor",
        )
    )
    assert example["layout_name"] == "tiny_corridor"
    assert "Teacher action hint" not in example["messages"][1]["content"]

    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["right", "right"],
            trajectory_dir=tmp_path,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert reward == 118.0
    assert payload["layout_name"] == "tiny_corridor"
    assert payload["won"] is True
    assert payload["trajectory"][-1]["reason"] == "all_pellets"
    assert all("teacher_action" not in step for step in payload["trajectory"])


def test_episode_workflow_runs_full_text_loop_with_env_reward():
    example = next(generate_episode_specs(episodes=1, max_steps=6))
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["right", "right", "stay", "stay", "stay", "stay"],
        )
    )
    assert isinstance(reward, float)
    assert reward > 0.0


def test_episode_workflow_can_write_trajectory(tmp_path):
    example = next(generate_episode_specs(episodes=1, max_steps=3))
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["right", "right", "stay"],
            trajectory_dir=tmp_path,
        )
    )
    files = list(tmp_path.glob("*.json"))
    assert files
    payload = json.loads(files[0].read_text())
    assert payload["total_reward"] == reward
    assert payload["prompt_style"] == "default"
    assert "Game rules:" in payload["system_prompt"]
    assert payload["illegal_action_penalty"] == 0
    assert payload["trajectory"][0]["model_output"] == "right"
    assert payload["trajectory"][0]["action"] == "right"
    assert payload["trajectory"][0]["exact_action"] is True
    assert payload["trajectory"][0]["legal_action"] is True


def test_episode_workflow_image_text_mode_logs_image_metadata(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=3,
            prompt_style="ghost_legal_strict",
            layout_name="medium_default",
        )
    )
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["right", "right", "stay"],
            trajectory_dir=tmp_path,
            observation_mode="image_text",
            vision_tile_size=16,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["total_reward"] == reward
    assert payload["observation_mode"] == "image_text"
    assert payload["vision_tile_size"] == 16
    first_step = payload["trajectory"][0]
    assert first_step["observation_mode"] == "image_text"
    assert first_step["obs_image_sha256"] is not None
    assert len(first_step["obs_image_sha256"]) == 64
    assert "obs_image_data_url" not in first_step
    assert "Grid:" not in first_step["obs_text"]


def test_image_only_vlm_workflow_forces_strict_image_observation(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=2,
            prompt_style="ghost_legal_strict",
            layout_name="medium_default",
        )
    )
    reward = asyncio.run(
        PacmanImageOnlyVLMWorkflow().run(
            example,
            scripted_actions=["right", "right"],
            trajectory_dir=tmp_path,
            observation_mode="image_text",
            vision_tile_size=16,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    first_step = payload["trajectory"][0]
    assert payload["total_reward"] == reward
    assert payload["observation_mode"] == "image_only"
    assert payload["system_prompt"] == IMAGE_ONLY_SYSTEM_PROMPT
    assert first_step["observation_mode"] == "image_only"
    assert first_step["obs_text"] == "Choose exactly one action token: up, down, left, right, or stay."
    assert "Allowed output tokens now" not in first_step["obs_text"]
    assert "Blue tiles" not in first_step["obs_text"]
    assert "Grid:" not in first_step["obs_text"]


def test_multi_maze_trajectory_preserves_split_and_hash_metadata(tmp_path):
    example = next(
        generate_multi_maze_episode_specs(
            split="test",
            episodes_per_layout=1,
            max_steps=2,
            prompt_style="ghost_legal_strict",
        )
    )
    asyncio.run(
        PacmanImageOnlyVLMWorkflow().run(
            example,
            scripted_actions=["stay", "stay"],
            trajectory_dir=tmp_path,
            vision_tile_size=16,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["maze_suite"] == "multi_maze_v1"
    assert payload["maze_split"] == "test"
    assert payload["layout_hash"] == example["layout_hash"]
    assert payload["topology_hash"] == example["topology_hash"]
    assert payload["rollout_index"] == 0


def test_safe_progress_workflow_logs_potential_components(tmp_path):
    example = next(
        generate_multi_maze_episode_specs(
            split="train",
            episodes_per_layout=1,
            max_steps=2,
            prompt_style="ghost_legal_strict",
            reward_mode="safe_progress",
        )
    )
    asyncio.run(
        PacmanImageOnlyVLMWorkflow().run(
            example,
            scripted_actions=["stay", "stay"],
            trajectory_dir=tmp_path,
            safe_progress_alpha=2.0,
        )
    )

    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["reward_mode"] == "safe_progress"
    assert payload["safe_progress_alpha"] == 2.0
    assert payload["trajectory"][0]["safe_distance_before"] is not None
    assert payload["trajectory"][0]["safe_distance_after"] is not None
    assert "safe_progress_reward" in payload["trajectory"][0]


def test_image_only_vlm_call_uses_visual_legend_system_and_user_multimodal_message(monkeypatch):
    captured = {}

    class FakeMessage:
        content = "right"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        id = "fake-completion-id"

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        chat = FakeChat()

    fake_openai = types.SimpleNamespace(AsyncOpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    obs = [
        {"type": "text", "text": "Choose exactly one action token: up, down, left, right, or stay."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    turn = asyncio.run(
        PacmanImageOnlyVLMWorkflow()._call_model(
            obs,
            model="default",
            temperature=0.0,
            max_completion_tokens=8,
        )
    )
    assert turn.content == "right"
    assert captured["messages"][0] == {"role": "system", "content": IMAGE_ONLY_SYSTEM_PROMPT}
    assert "blue tiles are walls" in captured["messages"][0]["content"]
    assert "red rounded square is the ghost" in captured["messages"][0]["content"]
    assert "Manhattan distance to PacMan is 3 tiles or less" in captured["messages"][0]["content"]
    assert "distance is greater than 3 tiles, it stays still" in captured["messages"][0]["content"]
    assert "moves one legal tile that minimizes its Manhattan distance" in captured["messages"][0]["content"]
    assert "clearing all pellets gives a large win bonus" in captured["messages"][0]["content"]
    assert captured["messages"][1] == {"role": "user", "content": obs}
    assert captured["messages"][1]["content"][0]["text"] == "Choose exactly one action token: up, down, left, right, or stay."
    assert captured["messages"][1]["content"][1]["type"] == "image_url"


def test_shortest_route_vision_baseline_wins_tiny_corridor():
    payload, frames = run_vision_episode(
        agent_name="shortest_route",
        seed=0,
        max_steps=4,
        layout_name="tiny_corridor",
        tile_size=16,
    )
    assert payload["won"] is True
    assert payload["steps"] == 2
    assert len(payload["trajectory"]) == 2
    assert len(frames) == 3
    assert payload["trajectory"][0]["observation_mode"] == "image"
    assert len(payload["trajectory"][0]["obs_image_sha256"]) == 64
    assert payload["trajectory"][0]["action"] == "right"


def test_vision_baseline_frames_can_compose_video_frames():
    payload, frames = run_vision_episode(
        agent_name="greedy",
        seed=0,
        max_steps=4,
        layout_name="tiny_corridor",
        tile_size=16,
    )
    video_frames = compose_vision_video_frames(payload, frames, fps=2)
    assert video_frames
    assert video_frames[0].mode == "RGB"
    assert video_frames[0].height > frames[0].height


def test_teacher_hint_workflow_enforces_teacher_action(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=30,
            prompt_style="teacher_hint",
            illegal_action_penalty=-4,
        )
    )
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["stay"] * 30,
            trajectory_dir=tmp_path,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert reward == 264.0
    assert payload["won"] is True
    assert payload["steps"] == 26
    assert payload["trajectory"][0]["model_output"] == "stay"
    assert payload["trajectory"][0]["teacher_action"] == "down"
    assert payload["trajectory"][0]["action"] == "down"
    assert payload["trajectory"][0]["teacher_corrected"] is True


def test_episode_workflow_records_illegal_action_penalty(tmp_path):
    env = PacmanEnv(max_steps=4, illegal_action_penalty=-4)
    env.reset()
    _, reward, _, info = env.step("up")
    assert info["legal_action"] is False
    assert reward < 0


def test_episode_workflow_parse_failure_is_terminal_negative_reward(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=20,
            prompt_style="ghost_legal_json",
            illegal_action_penalty=-4,
            layout_name="medium_default",
        )
    )
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["```"],
            parse_failure_penalty=-50,
            trajectory_dir=tmp_path,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    first = payload["trajectory"][0]
    assert reward == -50.0
    assert payload["won"] is False
    assert payload["steps"] == 1
    assert first["action"] == PARSE_FAILED_ACTION
    assert first["parse_failed"] is True
    assert first["legal_action"] is False
    assert first["reward"] == -50
    assert first["done"] is True
    assert first["reason"] == "parse_failed"


def test_episode_workflow_masks_parsed_action_to_current_legal_actions(tmp_path):
    example = next(
        generate_episode_specs(
            episodes=1,
            max_steps=1,
            prompt_style="ghost_legal_distance_json",
            illegal_action_penalty=-4,
        )
    )
    reward = asyncio.run(
        PacmanEpisodeWorkflow().run(
            example,
            scripted_actions=["Up is blocked, so choose right."],
            legal_action_mask=True,
            trajectory_dir=tmp_path,
        )
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    first = payload["trajectory"][0]
    assert first["raw_action"] == "up"
    assert first["action"] == "right"
    assert first["action_masked"] is True
    assert first["legal_action"] is True
    assert isinstance(reward, float)


def test_trajectory_summary_counts_actions_and_terminal_reasons():
    summary = summarize(
        [
            {
                "total_reward": 2.0,
                "steps": 2,
                "won": False,
                "trajectory": [
                    {"action": "right", "reason": "running"},
                    {"action": "down", "reason": "max_steps"},
                ],
            },
            {
                "total_reward": 12.0,
                "steps": 2,
                "won": True,
                "trajectory": [
                    {"action": "right", "reason": "running"},
                    {"action": "right", "reason": "all_pellets"},
                ],
            },
        ]
    )
    assert summary["episodes"] == 2
    assert summary["wins"] == 1
    assert summary["avg_reward"] == 7.0
    assert summary["illegal_actions"] == 0
    assert summary["actions"] == {"down": 1, "right": 3}
    assert summary["terminal_reasons"] == {"all_pellets": 1, "max_steps": 1}


def test_step_legal_action_infers_from_observation_when_missing():
    assert (
        step_legal_action(
            {
                "obs": "Step 3/30\nLegal actions: up, down, stay\nGrid:\n#P#",
                "action": "right",
            }
        )
        is False
    )


def test_format_trajectory_is_readable():
    text = format_trajectory(
        {
            "_file": "episode.json",
            "id": "pacman-episode-0",
            "seed": 0,
            "total_reward": -1.0,
            "final_score": -1,
            "steps": 1,
            "won": False,
            "trajectory": [
                {
                    "step": 1,
                    "obs": "Step 0/3\nGrid:\n#P#\nChoose exactly one action from: up, down, left, right, stay.",
                    "model_output": "right",
                    "action": "right",
                    "exact_action": True,
                    "legal_action": True,
                    "reward": -1,
                    "score": -1,
                    "done": False,
                    "reason": "running",
                }
            ],
        },
        max_steps=1,
    )
    assert "file: episode.json" in text
    assert "system_prompt: not recorded in this trajectory file" in text
    assert "model_output: 'right'" in text
    assert "parsed_action: right" in text
    assert "legal_action: True" in text
    assert "Choose exactly one action" in text
    assert "#P#" in text


def test_grid_from_obs_extracts_grid_block():
    obs = "\n".join(
        [
            "Step 0/20",
            "Grid:",
            "#####",
            "#P.G#",
            "#####",
            "Allowed output tokens now: right.",
        ]
    )
    assert grid_from_obs(obs) == ["#####", "#P.G#", "#####"]


def test_trajectory_renderer_composes_side_by_side_frames():
    episode = {
        "total_reward": 9,
        "steps": 1,
        "won": True,
        "trajectory": [
            {
                "step": 0,
                "obs": "\n".join(
                    [
                        "Step 0/20",
                        "Score: 0",
                        "Grid:",
                        "#####",
                        "#P.G#",
                        "#####",
                    ]
                ),
                "model_output": "right",
                "action": "right",
                "reward": 9,
                "score": 9,
                "reason": "all_pellets",
            }
        ],
    }
    frames = compose_frames(episode, episode, "Failure", "Success", fps=2)
    assert frames
    assert frames[0].size[0] > frames[0].size[1]


def test_trajectory_renderer_composes_single_frames():
    episode = {
        "total_reward": -1,
        "steps": 1,
        "won": False,
        "trajectory": [
            {
                "step": 0,
                "obs": "\n".join(
                    [
                        "Step 0/20",
                        "Score: 0",
                        "Grid:",
                        "#####",
                        "#P.G#",
                        "#####",
                    ]
                ),
                "model_output": "right",
                "action": "right",
                "reward": -1,
                "score": 0,
                "reason": "max_steps",
            }
        ],
    }
    frames = compose_single_frames(episode, "Failure", fps=2, title="Test", subtitle="Single")
    assert frames
    assert frames[0].size[0] < 700
    assert frames[0].size[1] < 700
