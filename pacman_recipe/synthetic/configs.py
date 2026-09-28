from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig
from pacman_recipe.level1.recipe import (
    EnvironmentConfig,
    DatasetGenerationConfig,
    PlannerAuditConfig,
)


@dataclass
class PacmanAgentConfig(PPOConfig):
    backplay_experiment: bool = False
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dataset_generation: DatasetGenerationConfig = field(default_factory=DatasetGenerationConfig)
    planner_audit: PlannerAuditConfig = field(default_factory=PlannerAuditConfig)
    recipe_version: str = field(
        default="research-scaffold",
        metadata={"help": "Recipe contract identifier stored with run artifacts."},
    )
    artifact_root: str = field(
        default="run_artifacts",
        metadata={"help": "Configurable root for datasets, trajectories, and checkpoints."},
    )
    trajectory_dir: str | None = field(
        default=None,
        metadata={"help": "Write one verbatim model-response trajectory JSON per rollout episode."},
    )
    step_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty charged for every executed environment step."},
    )
    step_penalty_cleared_ratio_scale: float = field(
        default=0.0,
        metadata={
            "help": (
                "Additional per-step penalty multiplied by the pre-action "
                "normal-pellet cleared ratio."
            )
        },
    )
    wall_penalty: float = field(
        default=1.0,
        metadata={"help": "Non-negative shaped penalty for a move into a wall."},
    )
    use_base_reward: bool = field(
        default=True,
        metadata={"help": "Include the original Pacman score delta in reward."},
    )
    normal_pellet_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating one normal pellet."},
    )
    power_pellet_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating one power pellet."},
    )
    ghost_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating one vulnerable ghost."},
    )
    fruit_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating a fruit event."},
    )
    reward_recipe_version: str = field(
        default="maapacman-level1-event-reward-v3",
        metadata={
            "help": "Versioned event-to-reward contract recorded in trajectories."
        },
    )
    death_penalty: float = field(
        default=0.0,
        metadata={"help": "Terminal penalty for first lethal ghost contact."},
    )
    completion_reward: float = field(
        default=0.0,
        metadata={"help": "Terminal reward for clearing all normal pellets."},
    )
    safety_refusal_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Workflow terminal penalty when Edward cannot advertise a "
                "provably safe next option."
            )
        },
    )
    nearest_pellet_alpha: float = field(
        default=0.0,
        metadata={
            "help": (
                "Alpha for level-1 nearest-normal-pellet distance progress; "
                "zero preserves the sparse reward contract."
            )
        },
    )
    nearest_pellet_remaining_ratio_threshold: float = field(
        default=1.0,
        metadata={
            "help": (
                "Enable nearest-pellet shaping at or below this fraction "
                "of remaining normal pellets."
            )
        },
    )
    nearest_pellet_scale_by_cleared_ratio: bool = field(
        default=False,
        metadata={
            "help": (
                "Scale nearest-pellet distance progress by the cleared "
                "normal-pellet ratio (1 - remaining ratio)."
            )
        },
    )
    nearest_pellet_skip_on_eat: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable nearest-pellet distance shaping on steps that eat "
                "a normal pellet."
            )
        },
    )
    allow_unoffloaded_actor_colocated_ref_for_smoke: bool = field(
        default=False,
        metadata={
            "help": (
                "Smoke-only escape hatch for small models that fit an "
                "actor-colocated reference without native FSDP parameter "
                "offload. Production 9B runs must leave this false."
            )
        },
    )
    workflow: str = field(
        default="pacman_recipe.areal_workflow.PacmanWorkflow",
        metadata={"help": "Workflow class for PacMan training."},
    )
    eval_workflow: str = field(
        default="pacman_recipe.areal_workflow.PacmanWorkflow",
        metadata={"help": "Workflow class for PacMan evaluation."},
    )
    validation_contract: str = field(
        default="greedy1",
        metadata={
            "help": (
                "Level-1 validation/reporting contract. Production follow-up "
                "runs use sampled12_uniform when train, validation, and test "
                "all use the same sampled decoding distribution."
            )
        },
    )
    reward_objective_contract: str = field(
        default="option_return_raw_v1",
        metadata={
            "help": (
                "Training reward/advantage contract. The default "
                "option_return_raw_v1 uses each option's unnormalized "
                "return-to-go. Set episode_return_group_v1 to compare exactly "
                "twelve complete episodes from the same initial maze state "
                "with group-normalized episode returns. Both contracts reduce "
                "policy loss with equal weight per complete episode. Legacy "
                "contracts remain readable for archived recipes."
            )
        },
    )
    enable_thinking: bool | None = field(
        default=None,
        metadata={"help": "Optional chat-template thinking toggle for Qwen-style models."},
    )
    image_prompt_style: str = field(
        default="minimal_v1",
        metadata={
            "help": (
                "Level-1 prompt contract: minimal_v1, live_static_v2, or "
                "live_state_v3 (screenshot plus authoritative engine state "
                "and bounded navigation history)."
            )
        },
    )
    prompt_version: str = field(
        default="legacy",
        metadata={
            "help": (
                "Versioned prompt protocol recorded in manifests and "
                "trajectory audit payloads."
            )
        },
    )
    action_protocol: str = field(
        default="legacy",
        metadata={
            "help": (
                "Model-to-harness action contract: direct open movement "
                "token or Edward option code."
            )
        },
    )
    legal_action_mask: bool = field(
        default=False,
        metadata={"help": "Post-parse PacMan actions to the current legal action set during rollout."},
    )
    open_action_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Constrain each level-1 model generation to the movement actions "
                "that are open in the current live environment state."
            )
        },
    )
    edward_options: bool = field(
        default=False,
        metadata={
            "help": (
                "Ask the model to choose a planner-approved C*/A*/E* "
                "objective, then execute its bounded multi-step option."
            )
        },
    )
    edward_fallback_mode: str = field(
        default="refuse",
        metadata={"help": "refuse preserves legacy emergency handling; risk_ranked advertises all open one-step moves when normal options are empty."},
    )
    objective_encoding: str = field(
        default="legacy",
        metadata={
            "help": (
                "Edward objective output contract. Production v3 uses "
                "edward-option-code-v1 with one tokenizer token per decision."
            )
        },
    )
    guided_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice to constrain rollout completions."},
    )
    legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over the current legal PacMan action tokens."},
    )
    non_stay_legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over current legal non-stay actions when any movement is legal."},
    )
    non_backtracking_legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over legal non-stay actions, excluding immediate reverse when another choice exists."},
    )
    action_token_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over all PacMan action tokens only, without legal-action masking."},
    )
    completion_api: str = field(
        default="chat",
        metadata={"help": "OpenAI API style for rollout calls: 'chat', 'chat_user', or 'completion'."},
    )
    parse_failure_penalty: int = field(
        default=-50,
        metadata={"help": "Deprecated compatibility setting for legacy synthetic workflows."},
    )
    contract_violation_return: float = field(
        default=-1.0,
        metadata={
            "help": "Fail-closed whole-episode return used by the Level-1 "
            "workflow for parse/canonical contract violations; must be -1.0."
        },
    )
    reward_mode: str = field(
        default="sparse",
        metadata={"help": "Environment reward mode, including sparse, route-prefix variants, and safe_progress."},
    )
    route_shaping_scale: float = field(
        default=1.0,
        metadata={"help": "Scale hidden route/progress shaping terms without changing sparse PacMan rewards."},
    )
    safe_progress_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha for safe_progress: original reward + alpha * (safe_distance_before - safe_distance_after)."},
    )
    observation_mode: str = field(
        default="text",
        metadata={"help": "Episode observation mode: text, image, image_text, or image_only."},
    )
    vision_tile_size: int = field(
        default=32,
        metadata={"help": "Tile size in pixels for image/image_text PacMan observations."},
    )
    store_observation_images: bool = field(
        default=False,
        metadata={"help": "Store base64 image data URLs in trajectory JSON for debugging."},
    )
