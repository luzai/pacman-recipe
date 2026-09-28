from __future__ import annotations

import importlib
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("AREAL_ALLOW_DEFAULT_ADMIN_KEY", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _load_workflow(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^#\r\n]+)", text)
    if not match:
        raise ValueError(f"missing required config field: {key}")
    return match.group(1).strip().strip('"\'')


def _config_override(args: list[str], key: str) -> str | None:
    """Return the final exact-key CLI override, matching AReaL precedence."""

    prefix = f"{key}="
    matches = [argument[len(prefix) :] for argument in args if argument.startswith(prefix)]
    return matches[-1] if matches else None


def _apply_smoke_updates(args: list[str]) -> tuple[list[str], int | None]:
    """Translate the release smoke flag into AReaL's exact step limit."""

    cleaned: list[str] = []
    raw_updates: str | None = None
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--smoke-updates":
            if raw_updates is not None:
                raise ValueError("--smoke-updates may be specified only once")
            if index + 1 >= len(args):
                raise ValueError("--smoke-updates requires a positive integer")
            raw_updates = args[index + 1]
            index += 2
            continue
        if argument.startswith("--smoke-updates="):
            if raw_updates is not None:
                raise ValueError("--smoke-updates may be specified only once")
            raw_updates = argument.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    if raw_updates is None:
        return cleaned, None
    if re.fullmatch(r"[1-9][0-9]*", raw_updates) is None:
        raise ValueError("--smoke-updates requires a positive integer")
    if _config_override(cleaned, "total_train_steps") is not None:
        raise ValueError(
            "--smoke-updates cannot be combined with total_train_steps"
        )
    smoke_updates = int(raw_updates)
    cleaned.append(f"total_train_steps={smoke_updates}")
    return cleaned, smoke_updates


def _parse_exploratory_budget(args: list[str]) -> tuple[list[str], bool]:
    """Consume explicit intent to run an open-ended exploratory budget.

    The release guard pins the training budget: 40 train rows, 5 epochs, a
    50-update total, one checkpoint per update and per-epoch validation. Those
    numbers reproduce the published recipe and must keep failing closed by
    default. An exploratory run that is meant to be stopped by hand needs
    different ones, so it says so on the command line instead of quietly
    editing the constants, which would remove the guard for every formal run
    too. Everything outside the budget -- protocol, reward, learning rate, KL,
    group size, GPU split -- stays pinned.
    """
    cleaned: list[str] = []
    exploratory = False
    for argument in args:
        if argument == "--exploratory-budget":
            if exploratory:
                raise ValueError("--exploratory-budget may be specified only once")
            exploratory = True
        else:
            cleaned.append(argument)
    return cleaned, exploratory


REWARD_ABLATIONS = ("fixed-distance", "binary-outcome")


def _parse_reward_ablation(args: list[str]) -> tuple[list[str], str | None]:
    """Consume explicit experiment intent without adding framework config fields."""
    cleaned: list[str] = []
    ablation = None
    index = 0
    choices = " or ".join(REWARD_ABLATIONS)
    while index < len(args):
        argument = args[index]
        if argument == "--reward-ablation" or argument.startswith("--reward-ablation="):
            if ablation is not None:
                raise ValueError("--reward-ablation may be specified only once")
            if argument == "--reward-ablation":
                index += 1
                if index >= len(args):
                    raise ValueError(f"--reward-ablation requires {choices}")
                ablation = args[index]
            else:
                ablation = argument.split("=", 1)[1]
            if ablation not in REWARD_ABLATIONS:
                raise ValueError(f"--reward-ablation requires {choices}")
        else:
            cleaned.append(argument)
        index += 1
    return cleaned, ablation


def _backend_degree(backend: str) -> int:
    match = re.search(r":d(\d+)p\d+t\d+$", backend)
    if match is None:
        raise ValueError(f"unsupported production backend allocation: {backend}")
    return int(match.group(1))


def _validate_actor_colocated_reference_offload(config) -> None:
    """Require a supported memory policy without conflating TMS and FSDP."""

    ref = config.ref
    if (
        ref is None
        or ref.scheduling_strategy.type != "colocation"
        or ref.scheduling_strategy.target != "actor"
    ):
        return
    if ref.fsdp.offload_params:
        return
    if getattr(config, "allow_unoffloaded_actor_colocated_ref_for_smoke", False):
        return
    # The pinned AReaL trainer offloads both engines initially, onloads/refills
    # the reference only for ref_logp, and offloads it before onloading actor.
    # FSDPEngine's TMS pause/resume is independent of CPUOffloadPolicy. Require
    # both phase transitions and matching FSDP allocations for this extra path.
    phase_offload = (
        config.enable_offload is True
        and config.actor.offload is True
        and ref.offload is True
        and config.actor.backend.startswith("fsdp:")
        and ref.backend == config.actor.backend
    )
    if not phase_offload:
        raise ValueError(
            "an actor-colocated reference requires native FSDP parameter "
            "offload, or enable_offload=true with actor.offload=true and "
            "ref.offload=true on matching FSDP backends, unless the explicit "
            "small-model smoke override is enabled"
        )


def _validate_reward_objective_contract(config) -> None:
    contract = config.reward_objective_contract
    if contract == "legacy":
        return
    if contract not in {
        "step_local_raw_v1",
        "option_return_raw_v1",
        "episode_return_group_v1",
    }:
        raise ValueError(f"unsupported reward_objective_contract: {contract}")
    if (
        contract in {"option_return_raw_v1", "episode_return_group_v1"}
        and not getattr(config, "edward_options", False)
        and not (
            contract == "episode_return_group_v1"
            and getattr(config, "backplay_experiment", False)
        )
    ):
        raise ValueError(f"{contract} requires edward_options=true")
    if not config.workflow.endswith(".PacmanNativeVisionWorkflow"):
        raise ValueError(
            f"{contract} requires PacmanNativeVisionWorkflow"
        )
    if contract in {"option_return_raw_v1", "episode_return_group_v1"}:
        if getattr(config.actor, "use_sapo_loss", False) or getattr(
            config.actor, "use_cispo_loss", False
        ):
            raise ValueError(
                f"{contract} requires the PPO/GRPO surrogate; SAPO and CISPO "
                "do not support equal-episode reduction"
            )
        if int(getattr(config.actor, "ppo_n_minibatches", 1)) != 1:
            raise ValueError(
                f"{contract} requires actor.ppo_n_minibatches=1 so one "
                "optimizer update averages all complete episodes together"
            )
    if contract == "episode_return_group_v1":
        reward_norm = config.actor.reward_norm
        if reward_norm is None:
            raise ValueError(
                "episode_return_group_v1 requires actor.reward_norm"
            )
        n_samples = int(config.gconfig.n_samples)
        if n_samples != 12:
            raise ValueError(
                "episode_return_group_v1 requires exactly 12 complete episodes "
                "per initial maze state"
            )
        if (
            reward_norm.mean_level != "group"
            or reward_norm.std_level != "group"
            or int(reward_norm.group_size) != n_samples
            or bool(reward_norm.mean_leave1out)
            or not bool(getattr(reward_norm, "std_unbiased", False))
            or not math.isclose(
                float(getattr(reward_norm, "eps", 0.0)),
                1.0e-5,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "episode_return_group_v1 requires group mean/std normalization, "
                "group_size=gconfig.n_samples, mean_leave1out=false, "
                "std_unbiased=true, and eps=1e-5"
            )
        if getattr(config.actor, "overlong_reward_penalty", False):
            raise ValueError(
                "episode_return_group_v1 does not support per-completion "
                "overlong reward penalties"
            )
    if contract in {"step_local_raw_v1", "episode_return_group_v1"} and (
        getattr(config, "critic", None) is not None
        or getattr(config, "teacher", None) is not None
    ):
        raise ValueError(f"{contract} requires critic=null and teacher=null")
    if contract != "episode_return_group_v1" and config.actor.reward_norm is not None:
        raise ValueError(
            f"{contract} requires actor.reward_norm=null; normalization across "
            "unrelated maze states changes the sign of state-local returns"
        )
    if config.actor.adv_norm is not None:
        raise ValueError(
            f"{contract} requires actor.adv_norm=null; batch centering creates "
            "a global action-token baseline across unrelated maze states"
        )


def _validate_backplay_contract(config) -> None:
    """Separate experimental primitive GRPO from the published C1/C2 recipes."""
    required = {
        "recipe_version": "maapacman-adaptive-backplay-v1",
        "action_protocol": "direct-open-action-token-v1",
        "prompt_version": "live-state-direct-action-v3",
        "reward_objective_contract": "episode_return_group_v1",
        "edward_options": False,
        "open_action_mask": True,
        "action_token_choice": True,
        "enable_thinking": False,
        "objective_encoding": "direct-action-token-v1",
    }
    for name, expected in required.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f"Backplay requires {name}={expected!r}")
    if config.environment.ghost_mode != "normal":
        raise ValueError("Backplay requires the fixed normal-ghost task")
    if config.environment.max_steps != 512:
        raise ValueError("Backplay preserves the original 512-step horizon")
    if config.environment.episode_life_mode != "original_three_lives":
        raise ValueError("Backplay requires original_three_lives")
    for generation in (config.gconfig, config.eval_gconfig):
        if generation.n_samples != 12 or generation.greedy or generation.temperature <= 0:
            raise ValueError("Backplay requires sampled groups of 12 episodes")
        if generation.top_p != 1.0:
            raise ValueError("Backplay constrained decoding requires top_p=1")
    if config.rollout.max_head_offpolicyness != 0:
        raise ValueError("Backplay frontier probes require the current policy")
    if not math.isfinite(float(config.actor.reward_clip)) or config.actor.reward_clip <= 0:
        raise ValueError("Backplay requires finite positive normalized reward_clip")
    if not (config.gconfig.max_tokens == config.eval_gconfig.max_tokens == config.vllm.max_model_len):
        raise ValueError("Backplay requires matching train/eval/serving token budgets")
    for generation in (config.gconfig, config.eval_gconfig):
        if generation.min_new_tokens != 1 or generation.max_new_tokens != 1:
            raise ValueError("Backplay requires exactly one primitive action token")
    if config.tokenizer_path != config.actor.path or config.rollout.tokenizer_path != config.actor.path:
        raise ValueError("Backplay tokenizer must match actor initialization")
    if config.ref is None or config.ref.path != config.actor.path:
        raise ValueError("Backplay requires the matching reference initialization")
    if config.vllm.logprobs_mode != "processed_logprobs":
        raise ValueError("Backplay requires constrained processed log probabilities")


def _validate_release_stage_contract(
    config,
    *,
    smoke_updates: int | None = None,
    reward_ablation: str | None = None,
    exploratory_budget: bool = False,
) -> None:
    """Fail closed on drift in either public two-stage training recipe."""

    from areal_pacman.level1.recipe import (
        DIRECT_ACTION_PROTOCOL,
        DIRECT_PROMPT_VERSION,
        EDWARD_OPTION_PROTOCOL,
        EDWARD_PROMPT_VERSION,
    )

    protocol = str(getattr(config, "action_protocol", "legacy"))
    if getattr(config, "backplay_experiment", False):
        _validate_backplay_contract(config)
        if reward_ablation is not None or exploratory_budget:
            raise ValueError("Backplay cannot combine unrelated recipe overrides")
        return
    if reward_ablation is not None:
        if reward_ablation == "binary-outcome":
            # The terminal win signal only exists on the Edward option protocol,
            # and it needs a full-length run to show whether the win rate moves.
            if protocol != EDWARD_OPTION_PROTOCOL:
                raise ValueError(
                    "binary-outcome reward ablation requires the C2 Edward option protocol"
                )
        elif reward_ablation != "fixed-distance" or smoke_updates != 4:
            raise ValueError("fixed-distance reward ablation requires --smoke-updates 4")
        elif protocol != DIRECT_ACTION_PROTOCOL:
            raise ValueError("fixed-distance reward ablation requires the C1 direct action protocol")
    if protocol == "legacy":
        return
    if protocol not in {DIRECT_ACTION_PROTOCOL, EDWARD_OPTION_PROTOCOL}:
        raise ValueError(f"unsupported action_protocol: {protocol}")

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"release two-stage recipe requires {message}")

    # binary-outcome deliberately replaces the shaped objective with the terminal
    # win signal, so every shaping coefficient must be free to move. It is checked
    # against its own pinned values further below, not against the release ones.
    binary_outcome = reward_ablation == "binary-outcome"

    require(int(config.environment.max_steps) == 512, "environment.max_steps=512")
    is_c2_overfit = (
        protocol == EDWARD_OPTION_PROTOCOL
        and int(config.dataset_generation.train_episodes) == 4
    )
    expected_train_rows = 80 if protocol == DIRECT_ACTION_PROTOCOL else (4 if is_c2_overfit else 40)
    expected_validation_rows = 8 if is_c2_overfit else 4
    if exploratory_budget:
        # The split may grow and may start above the published seed range so a
        # holdout stays unseen, but it must not shrink below the release size.
        # evaluate_level1.py still refuses a heldout seed that overlaps either
        # split, so the holdout cannot be silently trained on.
        require(
            int(config.dataset_generation.train_episodes) >= expected_train_rows,
            f"at least {expected_train_rows} train rows",
        )
        require(
            int(config.dataset_generation.validation_episodes)
            >= expected_validation_rows,
            f"at least {expected_validation_rows} validation rows",
        )
        require(int(config.dataset_generation.seed) >= 0, "a non-negative dataset seed")
    else:
        require(
            int(config.dataset_generation.train_episodes) == expected_train_rows,
            f"{expected_train_rows} train rows",
        )
        require(
            int(config.dataset_generation.validation_episodes)
            == expected_validation_rows,
            f"{expected_validation_rows} validation rows",
        )
        require(int(config.dataset_generation.seed) == 28, "dataset seed 28")
    require(int(config.train_dataset.batch_size) == 4, "train batch_size=4")
    require(int(config.valid_dataset.batch_size) == 4, "valid batch_size=4")
    require(int(config.gconfig.n_samples) == 12, "gconfig.n_samples=12")
    require(int(config.eval_gconfig.n_samples) == 12, "eval_gconfig.n_samples=12")
    require(int(config.seed) == 1, "training RNG seed 1")
    require(int(config.cluster.n_gpus_per_node) == 8, "8 GPUs")
    require(config.rollout.backend == "vllm:d4p1t1", "4 rollout GPUs")
    require(config.actor.backend == "fsdp:d4p1t1", "4 actor GPUs")
    expected_epochs = 20 if is_c2_overfit else 5
    expected_updates = 100 if protocol == DIRECT_ACTION_PROTOCOL else (20 if is_c2_overfit else 50)
    if exploratory_budget:
        # An open-ended run is stopped by hand, so the epoch count is only a
        # cap. lr_scheduler_type is constant with no warmup, so no schedule
        # depends on the total.
        require(
            int(config.total_train_epochs) >= 1,
            "a positive training epoch cap",
        )
    else:
        require(
            int(config.total_train_epochs) == expected_epochs,
            f"{expected_epochs} formal training epochs",
        )
    require(
        config.total_train_steps is None if smoke_updates is None
        else 1 <= smoke_updates <= expected_updates
        and config.total_train_steps == smoke_updates,
        "total_train_steps=null unless explicitly set by --smoke-updates",
    )
    if not exploratory_budget:
        require(
            expected_train_rows
            // int(config.train_dataset.batch_size)
            * int(config.total_train_epochs)
            == expected_updates,
            f"a {expected_updates}-update full budget",
        )
    require(
        math.isclose(float(config.actor.optimizer.lr), 5.0e-7),
        "actor learning rate 5e-7",
    )
    if not binary_outcome:
        # binary-outcome pins nearest_pellet_alpha to 0.0 in expected_reward above;
        # the release recipe keeps the BFS guidance term at 0.1.
        require(
            math.isclose(float(config.nearest_pellet_alpha), 0.1),
            "nearest_pellet_alpha=0.1",
        )
    require(config.validation_contract == "sampled12_uniform_shaped", "matched sampled validation")
    require(config.image_prompt_style == "live_state_v3", "image_prompt_style=live_state_v3")
    require(config.enable_thinking is False, "enable_thinking=false")
    require(config.actor.init_from_scratch is False, "pretrained actor initialization")
    require(config.tokenizer_path == config.actor.path, "tokenizer_path to follow actor.path")
    require(config.rollout.tokenizer_path == config.actor.path, "rollout tokenizer to follow actor.path")
    require(config.gconfig.greedy is False, "sampled training, not greedy decoding")
    require(config.eval_gconfig.greedy is False, "sampled validation, not greedy decoding")
    require(
        config.gconfig.max_tokens == config.eval_gconfig.max_tokens == config.vllm.max_model_len,
        "matching train/eval/serving token budgets",
    )
    require(int(config.actor.ppo_n_minibatches) == 1, "ppo_n_minibatches=1")
    require(math.isclose(float(config.actor.kl_ctl), 0.01), "KL coefficient 0.01")
    require(config.ref is not None, "a reference model")
    require(config.ref.path == config.actor.path, "ref.path to follow actor.path")
    require(getattr(config, "critic", None) is None, "critic=null")
    require(getattr(config, "teacher", None) is None, "teacher=null")
    if exploratory_budget:
        # A hand-stopped run needs the checkpoint it was stopped for to still
        # exist, which is a retention question rather than a cadence one.
        require(
            config.saver.freq_steps is not None and int(config.saver.freq_steps) >= 1,
            "a positive saver.freq_steps",
        )
        require(
            int(config.saver.keep_last) > 2,
            "saver.keep_last above the release value, so a hand-picked "
            "checkpoint is not pruned before it can be evaluated",
        )
    else:
        require(config.saver.freq_steps == 1, "saver.freq_steps=1")
    if protocol == DIRECT_ACTION_PROTOCOL:
        require(
            all(
                getattr(config.evaluator, field) is None
                for field in ("freq_steps", "freq_epochs", "freq_secs")
            )
            and config.evaluator.eval_before_train is False,
            "C1 validation disabled",
        )
    else:
        if is_c2_overfit:
            require(
                config.dataset_generation.validation_seed_start == 28,
                "C2 overfit validation_seed_start=28",
            )
        if exploratory_budget:
            # One epoch can now be many updates, so per-epoch validation would
            # leave the run unobservable for hours at a time and there would be
            # nothing to stop it on. Require a step interval instead.
            require(
                config.evaluator.freq_epochs is None
                and config.evaluator.freq_steps is not None
                and int(config.evaluator.freq_steps) >= 1
                and config.evaluator.freq_secs is None
                and config.evaluator.eval_before_train is True,
                "C2 exploratory validation before training and at a step interval",
            )
        else:
            require(
                config.evaluator.freq_epochs == (5 if is_c2_overfit else 1)
                and config.evaluator.freq_steps is None
                and config.evaluator.freq_secs is None
                and config.evaluator.eval_before_train is True,
                "C2 validation before training and at the configured epoch interval",
            )
    require(str(config.recover.mode) == "disabled", "recover.mode=disabled")
    require(config.gconfig.min_new_tokens == 1, "one-token train decoding")
    require(config.gconfig.max_new_tokens == 1, "one-token train decoding")
    require(config.eval_gconfig.min_new_tokens == 1, "one-token evaluation decoding")
    require(config.eval_gconfig.max_new_tokens == 1, "one-token evaluation decoding")
    require(math.isclose(float(config.gconfig.temperature), 0.7), "temperature=0.7")
    require(math.isclose(float(config.eval_gconfig.temperature), 0.7), "evaluation temperature=0.7")
    require(math.isclose(float(config.gconfig.top_p), 1.0), "top_p=1.0")
    require(math.isclose(float(config.eval_gconfig.top_p), 1.0), "evaluation top_p=1.0")
    # Pin the shaping coefficients to zero rather than leaving them unchecked:
    # the opt-in names one specific experiment, it is not a licence to set
    # arbitrary rewards.
    if binary_outcome:
        expected_reward = {
            "normal_pellet_reward": 0.0,
            "power_pellet_reward": 0.0,
            "ghost_reward": 0.0,
            "fruit_reward": 0.0,
            "step_penalty": 0.0,
            "step_penalty_cleared_ratio_scale": 0.0,
            "wall_penalty": 0.0,
            "death_penalty": 0.0,
            "safety_refusal_penalty": 0.0,
            "nearest_pellet_alpha": 0.0,
            "nearest_pellet_remaining_ratio_threshold": 1.0,
        }
        require(
            float(config.completion_reward) > 0.0,
            "binary-outcome ablation completion_reward>0 as the only reward",
        )
    else:
        require(math.isclose(float(config.death_penalty), 100.0), "death_penalty=100")
        require(
            math.isclose(float(config.safety_refusal_penalty), 100.0),
            "safety_refusal_penalty=100",
        )
        expected_reward = {
            "normal_pellet_reward": 1.0,
            "power_pellet_reward": 1.0,
            "ghost_reward": 5.0,
            "fruit_reward": 0.0,
            "completion_reward": 50.0,
            "step_penalty": 0.05,
            "step_penalty_cleared_ratio_scale": 0.0,
            "wall_penalty": 0.5,
            "nearest_pellet_remaining_ratio_threshold": 1.0,
        }
    require(config.use_base_reward is False, "use_base_reward=false")
    require(
        config.nearest_pellet_scale_by_cleared_ratio is (reward_ablation is None),
        "scaled nearest-pellet shaping" if reward_ablation is None
        else f"{reward_ablation} ablation nearest_pellet_scale_by_cleared_ratio=false",
    )
    require(config.nearest_pellet_skip_on_eat is True, "nearest-pellet skip-on-eat")
    for field, expected in expected_reward.items():
        require(
            math.isclose(float(getattr(config, field)), expected),
            f"{field}={expected}",
        )

    reward_clip = float(config.actor.reward_clip)
    require(not math.isnan(reward_clip) and reward_clip > 0, "a positive non-NaN reward_clip")
    require(config.actor.adv_norm is None, "actor.adv_norm=null")
    if protocol == DIRECT_ACTION_PROTOCOL:
        require(config.environment.ghost_mode == "disabled", "C1 ghost_mode=disabled")
        require(
            config.environment.episode_life_mode == "single_death",
            "C1 single-death episode mode",
        )
        # A local, validated copy is required on offline training nodes. Model
        # identity/revision is bound by the launcher checkpoint/run manifest.
        require(bool(str(config.actor.path)), "a non-empty C1 initialization model")
        require(config.prompt_version == DIRECT_PROMPT_VERSION, "the C1 prompt version")
        require(config.reward_objective_contract == "step_local_raw_v1", "C1 step-local raw rewards")
        require(config.edward_options is False, "C1 edward_options=false")
        require(config.action_token_choice is True, "C1 action_token_choice=true")
        require(config.open_action_mask is True, "C1 open_action_mask=true")
        require(config.objective_encoding == "direct-action-token-v1", "the direct action encoding")
        require(config.actor.reward_norm is None, "C1 actor.reward_norm=null")
        require(math.isinf(reward_clip) and reward_clip > 0, "C1 actor.reward_clip=.inf")
    else:
        require(config.environment.ghost_mode == "normal", "C2 ghost_mode=normal")
        require(
            config.environment.episode_life_mode
            in {"single_death", "original_three_lives", "three_lives"},
            "C2 episode_life_mode must be single_death, original_three_lives, or three_lives",
        )
        require(config.prompt_version == EDWARD_PROMPT_VERSION, "the C2 prompt version")
        require(config.reward_objective_contract == "episode_return_group_v1", "C2 episode-return group rewards")
        require(config.edward_options is True, "C2 edward_options=true")
        require(config.action_token_choice is False, "C2 action_token_choice=false")
        require(config.open_action_mask is False, "C2 open_action_mask=false")
        require(config.objective_encoding == EDWARD_OPTION_PROTOCOL, "the Edward option-code encoding")
        require(math.isclose(reward_clip, 20.0), "C2 actor.reward_clip=20")


def _build_workflow_kwargs(
    config, generation_config, *, training: bool = True
) -> dict[str, object]:
    from dataclasses import is_dataclass
    from types import SimpleNamespace
    from collections.abc import Mapping
    from omegaconf import OmegaConf
    from areal_pacman.level1.recipe import recipe_contract_metadata

    def plain(value):
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if is_dataclass(value):
            return OmegaConf.to_container(OmegaConf.structured(value), resolve=True)
        if isinstance(value, SimpleNamespace):
            return {key: plain(item) for key, item in vars(value).items()}
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        return value

    raw_config = plain(config)
    kwargs = dict(
        action_protocol=getattr(config, "action_protocol", "legacy"),
        prompt_version=getattr(config, "prompt_version", "legacy"),
        recipe_contract=recipe_contract_metadata(raw_config),
        ghost_mode=config.environment.ghost_mode,
        episode_life_mode=getattr(
            config.environment, "episode_life_mode", "single_death"
        ),
        environment_max_steps=config.environment.max_steps,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=generation_config.max_tokens,
        max_completion_tokens=generation_config.max_new_tokens,
        tokenizer_path=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        image_prompt_style=config.image_prompt_style,
        legal_action_mask=config.legal_action_mask,
        open_action_mask=config.open_action_mask,
        edward_options=getattr(config, "edward_options", False),
        edward_fallback_mode=getattr(config, "edward_fallback_mode", "refuse"),
        objective_encoding=getattr(config, "objective_encoding", "legacy"),
        reward_objective_contract=(
            getattr(config, "reward_objective_contract", "legacy")
            if training
            else "evaluation_only_v1"
        ),
        guided_action_choice=config.guided_action_choice,
        legal_action_choice=config.legal_action_choice,
        non_stay_legal_action_choice=config.non_stay_legal_action_choice,
        non_backtracking_legal_action_choice=config.non_backtracking_legal_action_choice,
        action_token_choice=config.action_token_choice,
        completion_api=config.completion_api,
        parse_failure_penalty=getattr(config, "parse_failure_penalty", -50),
        contract_violation_return=getattr(
            config, "contract_violation_return", -1.0
        ),
        reward_mode=config.reward_mode,
        route_shaping_scale=config.route_shaping_scale,
        safe_progress_alpha=config.safe_progress_alpha,
        step_penalty=config.step_penalty,
        step_penalty_cleared_ratio_scale=getattr(
            config,
            "step_penalty_cleared_ratio_scale",
            0.0,
        ),
        wall_penalty=config.wall_penalty,
        use_base_reward=getattr(config, "use_base_reward", True),
        normal_pellet_reward=getattr(
            config,
            "normal_pellet_reward",
            0.0,
        ),
        power_pellet_reward=getattr(
            config,
            "power_pellet_reward",
            0.0,
        ),
        ghost_reward=getattr(config, "ghost_reward", 0.0),
        fruit_reward=getattr(config, "fruit_reward", 0.0),
        reward_recipe_version=getattr(
            config,
            "reward_recipe_version",
            "maapacman-level1-event-reward-v3",
        ),
        death_penalty=getattr(config, "death_penalty", 0.0),
        completion_reward=getattr(config, "completion_reward", 0.0),
        safety_refusal_penalty=getattr(
            config, "safety_refusal_penalty", 0.0
        ),
        nearest_pellet_alpha=config.nearest_pellet_alpha,
        nearest_pellet_remaining_ratio_threshold=(
            getattr(
                config,
                "nearest_pellet_remaining_ratio_threshold",
                1.0,
            )
        ),
        nearest_pellet_scale_by_cleared_ratio=getattr(
            config,
            "nearest_pellet_scale_by_cleared_ratio",
            False,
        ),
        nearest_pellet_skip_on_eat=getattr(
            config,
            "nearest_pellet_skip_on_eat",
            False,
        ),
        observation_mode=config.observation_mode,
        vision_tile_size=config.vision_tile_size,
        store_observation_images=config.store_observation_images,
        trajectory_dir=(
            os.getenv("PACMAN_TRAJECTORY_DIR") or config.trajectory_dir
        ),
    )
    if getattr(config, "workflow", "").endswith(
        ".PacmanNativeVisionWorkflow"
    ):
        kwargs.update(
            gconfig=generation_config,
            tokenizer=config.tokenizer_path,
            processor=config.tokenizer_path,
        )
    return kwargs


def _validate_release_dataset_inputs(config_path: Path, train_path: str, valid_path: str):
    """Bind formal training inputs to the immutable bundle, not just its rows."""
    import hashlib
    from areal_pacman.level1.recipe import load_recipe_document, recipe_contract_metadata
    from areal_pacman.level1.level1_dataset import environment_metadata, repository_revisions
    from scripts.level1.dataset.prepare_level1_dataset import validate_prepared_dataset_manifest

    raw = load_recipe_document(config_path)
    if raw.get("action_protocol", "legacy") == "legacy":
        return
    root = Path(__file__).resolve().parent
    paths = [Path(value).expanduser() for value in (train_path, valid_path)]
    paths = [(root / value).resolve() if not value.is_absolute() else value.resolve() for value in paths]
    if paths[0].parent != paths[1].parent:
        raise ValueError("release train/validation inputs must belong to the same immutable bundle")
    if paths[0].name != "train_hf" or paths[1].name != "validation_hf":
        raise ValueError("release training requires canonical train_hf/validation_hf bundle paths")
    manifest = validate_prepared_dataset_manifest(
        paths[0].parent / "manifest.json",
        expected_environment=environment_metadata(raw["environment"]["ghost_mode"]),
        expected_source_revisions=repository_revisions(),
        expected_training_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        expected_recipe_contract=recipe_contract_metadata(raw),
    )
    for split in ("train", "validation"):
        if manifest["splits"][split].get("hf") is None:
            raise ValueError(f"release bundle is missing {split} HF data")


def _production_dry_run(
    config_path: Path,
    *,
    validate_areal: bool = False,
    config_args: list[str] | None = None,
    smoke_updates: int | None = None,
    reward_ablation: str | None = None,
    exploratory_budget: bool = False,
) -> bool:
    text = config_path.read_text(encoding="utf-8")
    recipe_version = _yaml_scalar(text, "recipe_version")
    if recipe_version not in {
        "maapacman-level1-v1",
        "maapacman-level1-ghost-v2",
        "maapacman-level1-ghostdoor-v3",
    }:
        return False
    from areal_pacman.level1.level1_dataset import validate_episode_row
    from maapacman.env import PygamePacmanEnv

    epochs = int(_yaml_scalar(text, "total_train_epochs"))
    workflow_path = _yaml_scalar(text, "workflow")
    if epochs < 1:
        raise ValueError("production level-1 training must run at least one epoch")
    workflow_cls = _load_workflow(workflow_path)
    if workflow_cls.__module__ not in {
        "areal_pacman.workflow",
        "areal_pacman.level1.workflow",
    }:
        raise ValueError("production config must use areal_pacman.workflow")

    effective_args = config_args or ["--config", str(config_path)]
    dataset_overrides = [
        _config_override(effective_args, "train_dataset.path"),
        _config_override(effective_args, "valid_dataset.path"),
    ]
    override_present = [value is not None for value in dataset_overrides]
    if any(override_present) and not all(override_present):
        raise ValueError(
            "train_dataset.path and valid_dataset.path must be overridden together"
        )
    if all(override_present):
        if any(not value for value in dataset_overrides):
            raise ValueError("dataset path overrides must be non-empty")
        dataset_matches = [str(path) for path in dataset_overrides]
    else:
        dataset_matches = re.findall(r"(?m)^\s+path:\s*([^#\r\n]+)", text)
        if len(dataset_matches) < 2:
            raise ValueError("config must declare train and validation dataset paths")
    from areal_pacman.level1.recipe import EnvironmentConfig, load_recipe_settings
    from maapacman.env import PygamePacmanEnvConfig

    environment, _ = load_recipe_settings(config_path)
    environment = EnvironmentConfig(
        ghost_mode=(
            _config_override(effective_args, "environment.ghost_mode")
            or environment.ghost_mode
        ),
        max_steps=int(
            _config_override(effective_args, "environment.max_steps")
            or environment.max_steps
        ),
        episode_life_mode=(
            _config_override(effective_args, "environment.episode_life_mode")
            or environment.episode_life_mode
        ),
    )
    _validate_release_dataset_inputs(config_path, *dataset_matches[-2:])
    rows = 0
    for raw_path in dataset_matches[-2:]:
        dataset_path = Path(raw_path.strip().strip('"\''))
        if not dataset_path.is_absolute():
            dataset_path = (Path(__file__).resolve().parent / dataset_path).resolve()
        if dataset_path.suffix == ".jsonl":
            jsonl = dataset_path
        elif dataset_path.name.endswith("_hf"):
            jsonl = dataset_path.with_name(dataset_path.name.removesuffix("_hf") + ".jsonl")
        else:
            jsonl = dataset_path.with_suffix(".jsonl")
        if not jsonl.is_file():
            raise FileNotFoundError(jsonl)
        import json

        for line in jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            validate_episode_row(row)
            if row["env"]["ghost_mode"] != environment.ghost_mode:
                raise ValueError("dataset ghost_mode does not match training config")
            if row["env"]["max_steps"] != environment.max_steps:
                raise ValueError("dataset max_steps does not match training config")
            rows += 1
    env = PygamePacmanEnv(
        PygamePacmanEnvConfig(
            ghost_mode=environment.ghost_mode,
            episode_life_mode=environment.episode_life_mode,
        )
    )
    try:
        spec = env.spec
    finally:
        env.close()
    print("dry_run=ok")
    print(f"workflow={workflow_cls.__module__}.{workflow_cls.__name__}")
    print(f"config={config_path.resolve()}")
    print(f"dataset_rows={rows}")
    print(f"ghost_mode={environment.ghost_mode}")
    print(f"max_steps={environment.max_steps}")
    print(f"total_train_epochs={epochs}")
    print(f"env_api_version={spec.api_version}")
    print(f"level_revision={spec.level_revision}")
    print(f"action_tokens={','.join(spec.action_tokens)}")
    if validate_areal or reward_ablation is not None or exploratory_budget:
        from areal.api.cli_args import load_expr_config
        from areal_pacman.synthetic.configs import PacmanAgentConfig

        config, _ = load_expr_config(effective_args, PacmanAgentConfig)
        _validate_reward_objective_contract(config)
        _validate_release_stage_contract(
            config,
            smoke_updates=smoke_updates,
            reward_ablation=reward_ablation,
            exploratory_budget=exploratory_budget,
        )
        gpu_count = config.cluster.n_gpus_per_node
        if gpu_count not in (4, 6, 8):
            raise ValueError(
                "production config must request an accepted 4-, 6-, or "
                "8-GPU topology"
            )
        rollout_degree = _backend_degree(config.rollout.backend)
        actor_degree = _backend_degree(config.actor.backend)
        if not config.rollout.backend.startswith("vllm:"):
            raise ValueError("production rollout backend must use vLLM")
        if not config.actor.backend.startswith("fsdp:"):
            raise ValueError("production actor backend must use FSDP")
        allocated_gpus = rollout_degree + actor_degree
        if config.gconfig.n_samples < rollout_degree:
            raise ValueError("GRPO group size must be at least the rollout degree")
        if config.gconfig.n_samples % rollout_degree:
            raise ValueError("GRPO group size must divide evenly across rollout workers")
        if config.train_dataset.batch_size % actor_degree:
            raise ValueError("train batch size must divide evenly across actor workers")
        if config.enable_thinking is not False:
            raise ValueError("production image-only config must disable thinking")
        if recipe_version == "maapacman-level1-ghostdoor-v3" and config.edward_options:
            if not config.edward_options:
                raise ValueError("Edward v3 recipe must enable edward_options")
            if config.open_action_mask or config.action_token_choice:
                raise ValueError(
                    "Edward v3 replaces atomic action-token constraints"
                )
            if config.objective_encoding != "edward-option-code-v1":
                raise ValueError(
                    "Edward v3 requires objective_encoding=edward-option-code-v1"
                )
            if (
                config.gconfig.min_new_tokens != 1
                or config.gconfig.max_new_tokens != 1
                or config.eval_gconfig.min_new_tokens != 1
                or config.eval_gconfig.max_new_tokens != 1
            ):
                raise ValueError(
                    "Edward option-code train and evaluation generation must "
                    "use exactly one new token"
                )
            if config.gconfig.top_p != 1.0:
                raise ValueError("Edward objective training requires top_p=1.0")
            if config.actor.temperature != config.gconfig.temperature:
                raise ValueError(
                    "Edward rollout and actor temperatures must match"
                )
        if config.validation_contract == "greedy1":
            if not config.eval_gconfig.greedy:
                raise ValueError("greedy1 validation must use greedy decoding")
            if config.eval_gconfig.temperature != 0.0:
                raise ValueError("greedy1 validation temperature must be zero")
            if config.eval_gconfig.top_p != 1.0:
                raise ValueError("greedy1 validation top_p must be one")
            if config.eval_gconfig.n_samples != 1:
                raise ValueError(
                    "greedy1 validation must use exactly one sample"
                )
        elif config.validation_contract in (
            "sampled12_and_greedy1",
            "sampled12_uniform",
            "sampled12_uniform_shaped",
        ):
            if config.eval_gconfig.greedy:
                raise ValueError(
                    "sampled validation must not use greedy decoding"
                )
            if config.eval_gconfig.n_samples != 12:
                raise ValueError(
                    "sampled12 validation must use exactly 12 samples"
                )
            if config.eval_gconfig.n_samples != config.gconfig.n_samples:
                raise ValueError(
                    "sampled validation count must match the GRPO group size"
                )
            if (
                config.eval_gconfig.temperature
                != config.gconfig.temperature
                or config.eval_gconfig.top_p != config.gconfig.top_p
            ):
                raise ValueError(
                    "sampled validation temperature/top_p must match training"
                )
            if config.validation_contract in (
                "sampled12_uniform",
                "sampled12_uniform_shaped",
            ):
                if config.image_prompt_style != "live_state_v3":
                    raise ValueError(
                        "uniform sampled validation requires live_state_v3"
                    )
            if config.validation_contract == "sampled12_uniform":
                if config.nearest_pellet_alpha != 0.0:
                    raise ValueError(
                        "sampled12_uniform disables nearest-pellet shaping"
                    )
        else:
            raise ValueError(
                "unsupported production validation_contract: "
                f"{config.validation_contract}"
            )
        if config.actor.kl_ctl > 0:
            if config.ref is None:
                raise ValueError("positive KL requires a reference engine")
            if config.actor.optimizer_dtype not in ("float32", "bfloat16"):
                raise ValueError(
                    "the trainable actor optimizer dtype must be float32 or bfloat16"
                )
            if config.ref.optimizer is not None:
                raise ValueError(
                    "the KL reference engine must remain frozen"
                )
            if config.ref.optimizer_dtype != "bfloat16":
                raise ValueError(
                    "positive-KL level-1 training requires bfloat16 reference "
                    "parameter storage for the accepted split 4+4/3+3 topology"
                )
            if config.ref.temperature != config.actor.temperature:
                raise ValueError(
                    "reference and actor temperatures must match"
                )
            if config.enable_offload != config.ref.offload:
                raise ValueError(
                    "positive-KL reference offload requires enable_offload and "
                    "ref.offload to be enabled or disabled together"
                )
            ref_degree = _backend_degree(config.ref.backend)
            if config.ref.scheduling_strategy.type == "colocation":
                if config.ref.scheduling_strategy.target not in ("actor", "rollout"):
                    raise ValueError(
                        "a colocated reference engine must target actor or rollout"
                    )
                _validate_actor_colocated_reference_offload(config)
            else:
                allocated_gpus += ref_degree
        if allocated_gpus != gpu_count:
            raise ValueError(
                "production backend allocation must exactly consume the "
                f"{gpu_count} requested GPUs, got {allocated_gpus}"
            )
        print("areal_config=ok")
        print(f"cluster_gpus={config.cluster.n_gpus_per_node}")
        print(f"rollout_backend={config.rollout.backend}")
        print(f"actor_backend={config.actor.backend}")
        print(f"allocated_gpus={allocated_gpus}")
        print(f"actor_optimizer_dtype={config.actor.optimizer_dtype}")
        print(f"reference_optimizer_dtype={config.ref.optimizer_dtype if config.ref else None}")
        print(f"reference_offload={config.ref.offload if config.ref else False}")
        print(f"validation_contract={config.validation_contract}")
        print(
            "validation_decoding="
            f"n{config.eval_gconfig.n_samples},"
            f"temperature{config.eval_gconfig.temperature},"
            f"top_p{config.eval_gconfig.top_p},"
            f"thinking{config.enable_thinking}"
        )
    return True


# AReaL ships this to the rollout workers over JSON RPC, so it must be an
# import path, never a function object. See areal_pacman/level1/dynamic_filter.py.
_GROUP_REWARD_DEGENERACY_FILTER = (
    "areal_pacman.level1.dynamic_filter.accept_non_degenerate_reward_group"
)


def _build_group_reward_degeneracy_filter(config) -> str | None:
    """Return the import path of the degenerate-group filter, or None.

    Only whole-episode group rewards make a constant-reward group meaningless
    (``reward_norm`` zeroes its advantage), so the filter is enabled for that
    contract alone. The caller passes the result straight to
    ``PPOTrainer.train(dynamic_filter_fn=...)``; a string survives the RPC hop
    to the rollout workers, which resolve it with ``import_from_string``.
    """

    if getattr(config, "reward_objective_contract", None) != "episode_return_group_v1":
        return None
    if int(getattr(config.gconfig, "n_samples", 1)) < 2:
        return None
    return _GROUP_REWARD_DEGENERACY_FILTER


def main(args: list[str]) -> None:
    args, smoke_updates = _apply_smoke_updates(args)
    args, reward_ablation = _parse_reward_ablation(args)
    args, exploratory_budget = _parse_exploratory_budget(args)
    # fixed-distance is a 4-update smoke probe. binary-outcome replaces the whole
    # shaped objective with the terminal win signal, so it needs a full-length run
    # to show whether the win rate moves at all.
    if reward_ablation == "fixed-distance" and smoke_updates != 4:
        raise ValueError("fixed-distance reward ablation requires --smoke-updates 4")
    dry_run = "--dry-run" in args
    args = [arg for arg in args if arg != "--dry-run"]
    validate_areal = "--validate-areal" in args
    args = [arg for arg in args if arg != "--validate-areal"]

    config_path = None
    if "--config" in args:
        index = args.index("--config")
        if index + 1 >= len(args):
            raise ValueError("--config requires a path")
        config_path = Path(args[index + 1])
    if dry_run and config_path is not None and _production_dry_run(
        config_path,
        validate_areal=validate_areal,
        config_args=args,
        smoke_updates=smoke_updates,
        reward_ablation=reward_ablation,
        exploratory_budget=exploratory_budget,
    ):
        if smoke_updates is not None:
            print(f"smoke_updates={smoke_updates}")
        if reward_ablation is not None:
            print(f"reward_ablation={reward_ablation}")
        if exploratory_budget:
            print("exploratory_budget=true")
        return

    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer
    from areal_pacman.synthetic.configs import PacmanAgentConfig
    from datasets import load_from_disk

    config, _ = load_expr_config(args, PacmanAgentConfig)
    _validate_reward_objective_contract(config)
    _validate_release_stage_contract(
        config,
        smoke_updates=smoke_updates,
        reward_ablation=reward_ablation,
        exploratory_budget=exploratory_budget,
    )
    if config_path is not None:
        _validate_release_dataset_inputs(config_path, config.train_dataset.path, config.valid_dataset.path)
    if smoke_updates is not None:
        print(f"smoke_updates={smoke_updates}")
    if reward_ablation is not None:
        print(f"reward_ablation={reward_ablation}")

    if dry_run:
        dataset_path = Path(config.train_dataset.path)
        dataset = load_from_disk(str(dataset_path))
        workflow_cls = _load_workflow(config.workflow)
        print("dry_run=ok")
        print(f"workflow={workflow_cls.__module__}.{workflow_cls.__name__}")
        print(f"train_dataset={dataset_path}")
        print(f"train_rows={len(dataset)}")
        print(f"actor_path={config.actor.path}")
        print(f"scheduler={config.scheduler.type}")
        print(f"reward_mode={config.reward_mode}")
        print(f"safe_progress_alpha={config.safe_progress_alpha}")
        print(f"nearest_pellet_alpha={config.nearest_pellet_alpha}")
        return

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = _build_workflow_kwargs(
        config, config.gconfig, training=True
    )
    eval_workflow_kwargs = _build_workflow_kwargs(
        config, config.eval_gconfig, training=False
    )

    dynamic_filter_fn = _build_group_reward_degeneracy_filter(config)

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=dynamic_filter_fn,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
