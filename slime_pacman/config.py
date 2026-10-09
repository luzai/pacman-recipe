"""Explicit CPU-readable training contract."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class PacmanConfig:
    group_size: int = 12
    groups_per_update: int = 4
    max_steps: int = 512
    temperature: float = 0.7
    clip: float = 0.2
    max_input_tokens: int = 2048
    train_seed_start: int = 28
    train_seed_count: int = 40
    validation_seed_start: int = 68
    validation_seed_count: int = 4
    updates: int = 50
    learning_rate: float = 5e-7
    success_speed_bonus: float = 0.0
    edward_fallback_mode: str = "refuse"
    observation_mode: str = "image"
    ascii_map_format: str = "packed"
    vision_image_contract: str = "qwen-min-pixels-537600"

    def __post_init__(self):
        if self.observation_mode not in {'image', 'ascii'}:
            raise ValueError('Unsupported observation mode')
        if self.ascii_map_format not in ("packed", "spaced") or (
                self.ascii_map_format == "spaced" and self.observation_mode != "ascii"):
            raise ValueError("ascii_map_format must be packed, or spaced with observation_mode=ascii")
        from pacman_recipe.level1.vision_prompt import VISION_CELL_CONTRACT, VISION_IMAGE_CONTRACT
        if self.vision_image_contract not in (VISION_IMAGE_CONTRACT, VISION_CELL_CONTRACT) or (
                self.vision_image_contract == VISION_CELL_CONTRACT and self.observation_mode != "image"):
            raise ValueError(f"vision_image_contract must be {VISION_IMAGE_CONTRACT}, "
                             f"or {VISION_CELL_CONTRACT} with observation_mode=image")
        if self.edward_fallback_mode not in {"refuse", "risk_ranked"}:
            raise ValueError("unsupported Edward fallback mode")
        if not math.isfinite(self.success_speed_bonus) or not 0 <= self.success_speed_bonus <= 0.1:
            raise ValueError("success_speed_bonus must be finite and between 0 and 0.1")
        expected_groups, expected_clip = (16, 0.05) if self.observation_mode == 'ascii' else (4, 0.2)
        if self.group_size != 12 or self.groups_per_update != expected_groups:
            raise ValueError("Declared observation mode requires its fixed group count")
        if self.temperature != 0.7 or self.clip != expected_clip:
            raise ValueError("Declared observation mode requires temperature0.7 and its fixed clip")
        for name in (
            "max_steps",
            "max_input_tokens",
            "train_seed_count",
            "validation_seed_count",
            "updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_steps > 512 or self.max_input_tokens > 2048:
            raise ValueError("C2 supports at most 512 steps and 2048 input tokens")
        if set(self.train_seeds) & set(self.validation_seeds):
            raise ValueError("train and validation seeds overlap")
        if self.learning_rate != 5e-7:
            raise ValueError("initial C2 contract fixes learning_rate=5e-7")

    @property
    def train_seeds(self):
        return range(
            self.train_seed_start, self.train_seed_start + self.train_seed_count
        )

    @property
    def validation_seeds(self):
        return range(
            self.validation_seed_start,
            self.validation_seed_start + self.validation_seed_count,
        )

    def as_dict(self):
        return asdict(self)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.pop("schema", None) != "pacman-slime-config-v1":
        raise ValueError("unsupported slime Pacman config schema")
    return PacmanConfig(**raw)


def runner_options(config: PacmanConfig):
    from pacman_recipe.level1.prompts import prompt_contract_metadata
    style = ("live_state_v3" if config.observation_mode != "ascii" else
             "ascii_edward_spaced_v1" if config.ascii_map_format == "spaced" else "ascii_edward_v1")
    prompt = prompt_contract_metadata(style, edward_options=True,
                                     fallback_mode=config.edward_fallback_mode)
    return dict(
        edward_options=True,
        edward_fallback_mode=config.edward_fallback_mode,
        enable_thinking=False,
        image_prompt_style=style,
        prompt_version=prompt['prompt_version'],
        episode_life_mode="single_death",
        ghost_mode="normal",
        environment_max_steps=config.max_steps,
        temperature=config.temperature,
        top_p=1.0,
        max_completion_tokens=1,
        use_base_reward=False,
        normal_pellet_reward=0.0,
        power_pellet_reward=0.0,
        ghost_reward=0.0,
        fruit_reward=0.0,
        death_penalty=0.0,
        completion_reward=1.0,
        safety_refusal_penalty=0.0,
        step_penalty=0.0,
        wall_penalty=0.0,
        nearest_pellet_alpha=0.0,
    )
