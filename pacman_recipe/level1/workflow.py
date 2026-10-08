"""Production image-only AReaL workflow backed by Pacman's public API."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import time
import uuid
from contextvars import ContextVar
from copy import deepcopy
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.utils import stats_tracker
from pacman_env.env import (
    Action,
    PacmanEnvSpec,
    Position,
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
    load_bundled_level,
    nearest_reachable_distance,
)
from pacman_env.planner import (
    EdwardPlanner,
    EdwardSafetyRefusal,
    PlannerCandidate,
    validate_fallback_mode,
)

from ..actions import ActionParseError, parse_action
from .level1_dataset import SUPPORTED_MAX_STEPS, validate_episode_row
from .prompts import (
    build_image_messages,
    edward_system_prompt,
    render_edward_decision_prompt,
    encode_png,
    image_count,
    png_sha256,
    prompt_text,
    prompt_contract_metadata,
    prompt_user_template,
    sent_prompt_sha256,
    text_sha256,
)
from .rewards import RewardConfig, shape_reward
from .image_transport import native_image_data
from .trajectories import audit_trajectory, write_trajectory
from .token_constraints import (
    EDWARD_OPTION_CONSTRAINT,
    ObjectiveParseError,
    ObjectiveTokenConstraint,
)


from .episode import (
    PacmanEpisodeRunner, ModelTurn, EXPECTED_ACTIONS, MOVEMENT_ACTIONS,
    ACTION_MASK_BIT, OPPOSITE_ACTION, LOGGER, _MASK_LEAK_RETRY_ATTEMPTS,
    _compact_edward_decision_prompt, _nearest_reachable_distance_with_diagnostics,
    _normal_pellet_event_position, preferred_open_actions, validate_env_spec,
)


class PacmanImageOnlyWorkflow(PacmanEpisodeRunner):
    """AReaL metrics adapter around the shared episode runner."""

    def _record_metrics(self, payload: Mapping[str, Any]) -> None:
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            win_rate=float(payload["won"]),
            shaped_reward_avg=float(payload["total_shaped_reward"]),
        )


class PacmanNativeVisionWorkflow(PacmanImageOnlyWorkflow, RolloutWorkflow):
    """Native AReaL VLM workflow that keeps every training-time image tensor.

    The legacy agent workflow calls the OpenAI-compatible proxy and returns
    completion IDs plus rewards.  That proxy representation cannot preserve
    Qwen-VL processor outputs.  This workflow instead calls
    ``InferenceEngine.agenerate`` directly and returns the official tensor
    trajectory contract, including ``mm_token_type_ids`` and
    ``multi_modal_input``.
    """

    def __init__(
        self,
        *,
        gconfig: Any,
        tokenizer: Any,
        processor: Any,
        env_factory: Callable[[PygamePacmanEnvConfig], PygamePacmanEnv] = (
            PygamePacmanEnv
        ),
        **workflow_kwargs: Any,
    ) -> None:
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(processor, str):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(processor)
        super().__init__(env_factory=env_factory, **workflow_kwargs)
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.objective_tokenizer = tokenizer
        self.processor = processor
        self.edward_options = bool(
            workflow_kwargs.get("edward_options", False)
        )
        self.constrain_action_tokens = bool(
            workflow_kwargs.get("action_token_choice", False)
        )
        action_token_ids = []
        action_token_id_by_action: dict[str, int] = {}
        for token in MOVEMENT_ACTIONS:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"action {token!r} must map to exactly one tokenizer ID"
                )
            token_id = int(token_ids[0])
            action_token_ids.append(token_id)
            action_token_id_by_action[token] = token_id
        if len(set(action_token_ids)) != len(MOVEMENT_ACTIONS):
            raise ValueError("movement actions must have distinct tokenizer IDs")
        self.action_token_ids = action_token_ids
        self.action_token_id_by_action = action_token_id_by_action
        self.open_action_mask = bool(
            workflow_kwargs.get("open_action_mask", False)
        )
        self._native_engine: ContextVar[Any | None] = ContextVar(
            "pacman_native_engine", default=None
        )
        self._native_turns: ContextVar[
            dict[
                str,
                tuple[
                    dict[str, Any],
                    Any,
                    list[str],
                    list[list[int]],
                ],
            ]
            | None
        ] = ContextVar("pacman_native_turns", default=None)

    @staticmethod
    def _pil_and_chat_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        from .image_transport import pil_and_chat_messages

        return pil_and_chat_messages(messages)

    def _process_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[int]]:
        image, chat_messages = self._pil_and_chat_messages(messages)
        text = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        processed = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        mm_ids = processed.get("mm_token_type_ids")
        if mm_ids is None:
            mm_ids = processed.get("token_type_ids")
        if mm_ids is None:
            raise KeyError(
                "processor did not produce mm_token_type_ids or token_type_ids"
            )
        required = {"input_ids", "pixel_values"}
        missing = required - set(processed)
        if missing:
            raise KeyError(f"processor omitted required VLM fields: {missing}")
        return (
            image,
            chat_messages,
            dict(processed),
            processed["input_ids"].tolist()[0],
        )

    @staticmethod
    def _vllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the JSON-safe vLLM chat form used with separate image_data."""
        converted = deepcopy(messages)
        image_parts = 0
        for message in converted:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image_url"
                ):
                    item["image_url"] = {"url": "placeholder"}
                    image_parts += 1
        if image_parts != 1:
            raise ValueError(
                "native Pacman vLLM request requires exactly one image"
            )
        return converted

    async def _call_model(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> ModelTurn:
        engine = self._native_engine.get()
        native_turns = self._native_turns.get()
        if engine is None or native_turns is None:
            raise RuntimeError("native inference engine is not attached")
        from areal.api import ModelRequest

        image, chat_messages, processed, input_ids = self._process_messages(
            messages
        )
        objective_constraint = options.get("objective_constraint")
        if objective_constraint is not None and not isinstance(
            objective_constraint, ObjectiveTokenConstraint
        ):
            raise TypeError("objective_constraint has the wrong type")
        if objective_constraint is not None:
            allowed_token_ids = objective_constraint.allowed_token_ids
            current_open_actions = []
        elif self.open_action_mask:
            current_open_actions = list(
                options.get("current_open_actions") or []
            )
            if not current_open_actions:
                raise RuntimeError(
                    "open action mask requires current open actions"
                )
            allowed_token_ids = [
                self.action_token_id_by_action[action]
                for action in current_open_actions
            ]
        elif self.constrain_action_tokens:
            allowed_token_ids = self.action_token_ids
            current_open_actions = list(MOVEMENT_ACTIONS)
        else:
            allowed_token_ids = []
            current_open_actions = []

        for attempt in range(1, _MASK_LEAK_RETRY_ATTEMPTS + 1):
            request_id = uuid.uuid4().hex
            request = ModelRequest(
                rid=request_id,
                input_ids=input_ids,
                image_data=native_image_data(messages, image),
                vision_msg_vllm=[self._vllm_messages(messages)],
                gconfig=self.gconfig.new(
                    n_samples=1,
                    min_new_tokens=(
                        1
                        if (
                            objective_constraint is not None
                            or options.get("single_step")
                            or self.open_action_mask
                        )
                        else getattr(self.gconfig, "min_new_tokens", 0)
                    ),
                    max_new_tokens=(
                        objective_constraint.max_new_tokens
                        if objective_constraint is not None
                        else (
                            1
                            if options.get("single_step") or self.open_action_mask
                            else getattr(self.gconfig, "max_new_tokens", 3)
                        )
                    ),
                ),
                tokenizer=self.tokenizer,
                processor=self.processor,
                metadata={
                    "chat_template_kwargs": {"enable_thinking": False},
                    **(
                        {"allowed_token_ids": allowed_token_ids}
                        if allowed_token_ids
                        else {}
                    ),
                },
            )
            self._validate_native_token_budget(input_ids, request.gconfig)
            response = await engine.agenerate(request)
            if response.input_tokens != input_ids:
                raise RuntimeError(
                    "rollout input tokens differ from processor input_ids"
                )
            if (
                len(response.output_tokens)
                != len(response.output_logprobs)
                or len(response.output_tokens) != len(response.output_versions)
            ):
                raise RuntimeError("incomplete native rollout token metadata")
            option_token_ledger: list[list[int]] = []
            try:
                if objective_constraint is not None:
                    token_option = objective_constraint.option_for_tokens(
                        response.output_tokens
                    )
                    decoded_option = objective_constraint.option_for_completion(
                        self.tokenizer.decode(
                            response.output_tokens, skip_special_tokens=True
                        )
                    )
                    if token_option != decoded_option:
                        raise RuntimeError(
                            "objective token sequence and decoded option code disagree"
                        )
                    option_token_ledger = objective_constraint.support_ledger(
                        response.output_tokens
                    )
                elif allowed_token_ids and (
                    len(response.output_tokens) != 1
                    or response.output_tokens[0] not in allowed_token_ids
                ):
                    raise RuntimeError(
                        "native action token is absent from the exact rollout support"
                    )
            except (ObjectiveParseError, RuntimeError) as exc:
                if attempt >= _MASK_LEAK_RETRY_ATTEMPTS:
                    raise
                LOGGER.warning(
                    "MASK_LEAK_RETRY attempt=%d/%d request_id=%s "
                    "output_tokens=%s allowed_token_ids=%s: %s "
                    "(suspected vLLM allowed_token_ids mask drop; resampling "
                    "this decision with a fresh request id)",
                    attempt,
                    _MASK_LEAK_RETRY_ATTEMPTS,
                    request_id,
                    response.output_tokens,
                    allowed_token_ids,
                    exc,
                )
                continue
            break
        native_turns[request_id] = (
            processed,
            response,
            current_open_actions,
            option_token_ledger,
        )
        return ModelTurn(
            completion=self.tokenizer.decode(
                response.output_tokens,
                skip_special_tokens=True,
            ),
            completion_id=request_id,
            messages=messages,
            raw_response={
                "input_len": response.input_len,
                "output_len": response.output_len,
                "stop_reason": response.stop_reason,
            },
            request_extra_body={
                "native_areal_inference": True,
                "enable_thinking": False,
                **(
                    {
                        "allowed_token_ids": allowed_token_ids,
                    }
                    if objective_constraint is not None
                    else {}
                ),
            },
        )

    def _validate_native_token_budget(self, input_ids: list[int], gconfig: Any) -> None:
        """Never submit a silently truncated formal image request to inference."""
        total = getattr(gconfig, "max_tokens", None)
        generated = getattr(gconfig, "max_new_tokens", None)
        formal = self.workflow_kwargs.get("action_protocol") in {
            "direct-open-action-token-v1", "edward-option-code-v1"
        }
        if total is None or generated is None:
            if formal:
                raise ValueError("formal native request requires an explicit token budget")
            return
        if total <= 0 or generated <= 0 or len(input_ids) + generated > total:
            raise ValueError(
                "native multimodal request exceeds the token budget: "
                f"{len(input_ids)} input + {generated} output > {total}; no truncation"
            )

    @staticmethod
    def _tensor_sample(
        processed: dict[str, Any],
        response: Any,
        reward: float,
        allowed_actions: list[str],
        option_token_ledger: list[list[int]] | None = None,
        *,
        rollout_episode_id: int | None = None,
        rollout_episode_return: float | None = None,
        rollout_episode_group_size: int | None = None,
    ) -> dict[str, Any] | None:
        import torch

        if not math.isfinite(float(reward)):
            raise ValueError("training reward must be finite")
        input_ids = list(response.input_tokens)
        output_ids = list(response.output_tokens)
        sequence = input_ids + output_ids
        mm_ids = processed.get("mm_token_type_ids")
        if mm_ids is None:
            mm_ids = processed.get("token_type_ids")
        mm_list = mm_ids.tolist()[0] + [0] * len(output_ids)
        if len(mm_list) != len(sequence):
            raise RuntimeError("multimodal token types do not align with sequence")
        action_mask_bits = 0
        for action in allowed_actions:
            try:
                action_mask_bits |= ACTION_MASK_BIT[action]
            except KeyError as exc:
                raise ValueError(
                    f"unknown masked Pacman action: {action!r}"
                ) from exc
        if allowed_actions and not action_mask_bits:
            raise RuntimeError("allowed Pacman actions produced an empty mask")
        option_token_ledger = option_token_ledger or []
        if allowed_actions and option_token_ledger:
            raise ValueError(
                "legacy action mask and objective token ledger are exclusive"
            )
        if option_token_ledger and len(option_token_ledger) != len(output_ids):
            raise RuntimeError(
                "objective support ledger does not align with output tokens"
            )
        multimodal = [{"pixel_values": processed["pixel_values"]}]
        if "image_grid_thw" in processed:
            multimodal[0]["image_grid_thw"] = processed["image_grid_thw"]
        sample = {
            "input_ids": torch.tensor(
                sequence, dtype=torch.long
            ).unsqueeze(0),
            "mm_token_type_ids": torch.tensor(
                mm_list, dtype=torch.long
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                [0] * len(input_ids) + [1] * len(output_ids),
                dtype=torch.int32,
            ).unsqueeze(0),
            "logprobs": torch.tensor(
                [0.0] * len(input_ids) + list(response.output_logprobs),
                dtype=torch.float32,
            ).unsqueeze(0),
            "versions": torch.tensor(
                [-1] * len(input_ids) + list(response.output_versions),
                dtype=torch.int32,
            ).unsqueeze(0),
            "attention_mask": torch.ones(
                len(sequence), dtype=torch.bool
            ).unsqueeze(0),
            "rewards": torch.tensor(
                [float(reward)], dtype=torch.float32
            ),
            "multi_modal_input": multimodal,
        }
        episode_metadata = (
            rollout_episode_return,
            rollout_episode_group_size,
        )
        if rollout_episode_id is not None:
            sample["rollout_episode_ids"] = torch.tensor(
                [int(rollout_episode_id)], dtype=torch.long
            )
        if any(value is not None for value in episode_metadata):
            if rollout_episode_id is None or any(
                value is None for value in episode_metadata
            ):
                raise ValueError(
                    "whole-episode GRPO requires episode ID, return, and group size"
                )
            if int(rollout_episode_group_size) < 2:
                raise ValueError("whole-episode GRPO group size must be at least 2")
            if float(reward) != float(rollout_episode_return):
                raise ValueError(
                    "training reward must equal the complete episode return"
                )
            sample["rollout_episode_returns"] = torch.tensor(
                [float(rollout_episode_return)], dtype=torch.float32
            )
            sample["rollout_episode_group_sizes"] = torch.tensor(
                [int(rollout_episode_group_size)], dtype=torch.int32
            )
        if option_token_ledger:
            width = max(len(row) for row in option_token_ledger)
            if width <= 0:
                raise RuntimeError("objective support ledger contains no IDs")
            encoded_rows = [[0] * width for _ in input_ids]
            for sampled, support in zip(
                output_ids, option_token_ledger, strict=True
            ):
                if len(set(support)) != len(support):
                    raise ValueError(
                        "objective support ledger contains duplicate IDs"
                    )
                if sampled not in support:
                    raise ValueError(
                        "sampled token is absent from objective support ledger"
                    )
                encoded = [int(token_id) + 1 for token_id in support]
                encoded_rows.append(encoded + [0] * (width - len(encoded)))
            sample["pacman_allowed_token_ids"] = torch.tensor(
                encoded_rows,
                dtype=torch.long,
            ).unsqueeze(0)
        else:
            # The bit mask is placed on each generated token. The causal actor
            # rolls it left once, just like input_ids, so the preceding logit
            # is normalized over the exact actions allowed during rollout.
            sample["pacman_action_mask_bits"] = torch.tensor(
                [0] * len(input_ids)
                + [action_mask_bits] * len(output_ids),
                dtype=torch.uint8,
            ).unsqueeze(0)
        return sample

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> dict[str, Any]:
        from areal.utils.data import concat_padded_tensors

        engine_token = self._native_engine.set(engine)
        turns_token = self._native_turns.set({})
        payload_token = self._episode_payload.set(None)
        try:
            rewards = await super().run(data)
        finally:
            native_turns = self._native_turns.get()
            episode_payload = self._episode_payload.get()
            self._native_turns.reset(turns_token)
            self._native_engine.reset(engine_token)
            self._episode_payload.reset(payload_token)
        if rewards is None:
            if native_turns == {} and episode_payload is None:
                return None
            raise RuntimeError(
                "empty native rollout retained partial processor or episode state"
            )
        if not isinstance(rewards, dict) or not rewards:
            raise RuntimeError(
                "native Pacman rollout did not return per-completion rewards"
            )
        if native_turns is None:
            raise RuntimeError("native processor state was not initialized")
        if episode_payload is None:
            raise RuntimeError(
                "native Pacman rollout did not retain its episode payload"
            )
        reward_ids = set(rewards)
        native_turn_ids = set(native_turns)
        if reward_ids != native_turn_ids:
            raise RuntimeError(
                "native processor data and completion rewards disagree: "
                f"missing={sorted(reward_ids - native_turn_ids)} "
                f"extra={sorted(native_turn_ids - reward_ids)}"
            )
        reward_contract = str(
            self.workflow_kwargs.get("reward_objective_contract", "legacy")
        )
        if reward_contract == "step_local_raw_v1" and self.workflow_kwargs.get("edward_options"):
            raise ValueError("step_local_raw_v1 requires direct actions, not Edward options")
        if any(not math.isfinite(float(value)) for value in rewards.values()):
            raise ValueError("native training completion rewards must be finite")
        episode_kwargs: dict[str, Any] = {}
        if reward_contract in {
            "option_return_raw_v1",
            "episode_return_group_v1",
        }:
            sample_id = str(episode_payload["trajectory_sample_id"])
            digest = hashlib.blake2b(
                sample_id.encode("utf-8"), digest_size=8
            ).digest()
            episode_id = int.from_bytes(digest, "big") & ((1 << 63) - 1)
            episode_kwargs = {"rollout_episode_id": episode_id}
        if reward_contract == "episode_return_group_v1":
            episode_return = float(episode_payload["total_shaped_reward"])
            episode_group_size = int(getattr(self.gconfig, "n_samples", 0))
            if episode_group_size != 12:
                raise ValueError(
                    "episode_return_group_v1 requires exactly 12 complete "
                    "episodes per initial maze state"
                )
            episode_kwargs.update(
                {
                    "rollout_episode_return": episode_return,
                    "rollout_episode_group_size": episode_group_size,
                }
            )
        samples = []
        for completion_id, completion_reward in rewards.items():
            training_reward = (
                episode_kwargs["rollout_episode_return"]
                if "rollout_episode_return" in episode_kwargs
                else completion_reward
            )
            samples.append(
                self._tensor_sample(
                    native_turns[completion_id][0],
                    native_turns[completion_id][1],
                    training_reward,
                    native_turns[completion_id][2],
                    native_turns[completion_id][3],
                    **episode_kwargs,
                )
            )
        return concat_padded_tensors(samples)
