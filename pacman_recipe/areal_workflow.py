from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .synthetic.dataset import system_prompt
from .synthetic.env import ACTIONS, PacmanEnv, PacmanState
from .synthetic.vision import env_png_bytes, image_data_url, image_sha256


PARSE_FAILED_ACTION = "__parse_failed__"
OPPOSITE_ACTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _workflow_trace(event: str, **fields) -> None:
    trace_path = os.getenv("PACMAN_WORKFLOW_TRACE")
    if not trace_path:
        return
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _normalize_allowed_actions(allowed_actions: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if allowed_actions is None:
        return set(ACTIONS)
    normalized = {str(action).strip().lower() for action in allowed_actions}
    return {action for action in normalized if action in ACTIONS}


def _legal_fallback(allowed_actions: set[str]) -> str:
    if "stay" in allowed_actions:
        return "stay"
    return next((action for action in ACTIONS if action in allowed_actions), "stay")


def guided_choices_for_prompt(
    prompt_style: str,
    allowed_actions: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_stay_when_possible: bool = False,
    exclude_reverse_action: str | None = None,
) -> list[str]:
    actions = [action for action in ACTIONS if action in _normalize_allowed_actions(allowed_actions)]
    if exclude_stay_when_possible and any(action != "stay" for action in actions):
        actions = [action for action in actions if action != "stay"]
    reverse_action = OPPOSITE_ACTION.get(str(exclude_reverse_action or "").strip().lower())
    if reverse_action and reverse_action in actions and any(action != reverse_action for action in actions):
        actions = [action for action in actions if action != reverse_action]
    if not actions:
        actions = ["stay"]
    if prompt_style in {
        "ghost_legal_json",
        "ghost_legal_json_concise",
        "ghost_legal_json_first",
        "ghost_legal_json_fast_think",
        "ghost_legal_route_json",
        "ghost_legal_distance_json",
    }:
        return [
            json.dumps({"reason": f"choose {action}", "action": action}, separators=(",", ":"))
            for action in actions
        ]
    return actions


def parse_action(
    text: str,
    allow_word_fallback: bool = True,
    allowed_actions: list[str] | tuple[str, ...] | set[str] | None = None,
    fail_on_no_action: bool = False,
) -> str:
    allowed = _normalize_allowed_actions(allowed_actions)
    lowered = text.strip().lower()
    if lowered in allowed:
        return lowered
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            action = str(parsed.get("action", "")).strip().lower()
            if action in allowed:
                return action
    action_line = re.search(r"(?:^|\n)\s*(?:final\s+)?action\s*[:=]\s*(up|down|left|right|stay)\b", lowered)
    if action_line and action_line.group(1) in allowed:
        return action_line.group(1)
    final_answer = re.search(r"(?:^|\n)\s*final\s+answer\s*[:=]\s*(up|down|left|right|stay)\b", lowered)
    if final_answer and final_answer.group(1) in allowed:
        return final_answer.group(1)
    if not allow_word_fallback:
        return PARSE_FAILED_ACTION if fail_on_no_action else _legal_fallback(allowed)
    for match in re.finditer(r"\b(up|down|left|right|stay)\b", lowered):
        if match.group(1) in allowed:
            return match.group(1)
    return PARSE_FAILED_ACTION if fail_on_no_action else _legal_fallback(allowed)


def parse_action_for_prompt(
    text: str,
    prompt_style: str,
    allowed_actions: list[str] | tuple[str, ...] | set[str] | None = None,
    fail_on_no_action: bool = False,
) -> str:
    strict_styles = {
        "ghost_legal_strict",
        "ghost_legal_reason",
        "ghost_legal_json_concise",
        "ghost_legal_json_first",
        "ghost_legal_route_json",
    }
    return parse_action(
        text,
        allow_word_fallback=prompt_style not in strict_styles,
        allowed_actions=allowed_actions,
        fail_on_no_action=fail_on_no_action,
    )


def action_reward(action: str, answer: str) -> float:
    return 1.0 if action == answer else 0.0


@dataclass(frozen=True)
class ModelTurn:
    content: str
    completion_id: str | None = None
    request_messages: list[dict[str, object]] | None = None
    request_prompt: str | None = None
    request_extra_body: dict[str, object] | None = None
    request_params: dict[str, object] | None = None


@dataclass(frozen=True)
class ObservationTurn:
    text: str
    model_observation: str | list[dict[str, object]]
    image_sha256: str | None = None
    image_data_url: str | None = None


IMAGE_ONLY_SYSTEM_PROMPT = (
    "You control PacMan from a rendered image observation.\n"
    "Visual legend: blue tiles are walls, dark tiles are open floor, the yellow circle is PacMan, "
    "the red rounded square is the ghost, and small pale yellow dots are uneaten pellets. "
    "A white tile/object overlap means PacMan and the ghost collided.\n"
    "Game rules: PacMan moves one tile per turn using up, down, left, right, or stay. "
    "A move into a wall leaves PacMan in place and is not useful. The ghost is deterministic: "
    "it has alternating inactive and movement turns. On an inactive turn it stays still. On a movement turn, "
    "after PacMan moves, the ghost activates only when its Manhattan distance to PacMan is 3 tiles or less; "
    "if the distance is greater than 3 tiles, it stays still. When active, it moves one legal tile that minimizes "
    "its Manhattan distance to PacMan. If PacMan and the ghost occupy the same tile, the episode ends as caught.\n"
    "Scoring: each turn has a small step cost, eating a pellet gives positive reward, being caught gives a large penalty, "
    "clearing all pellets gives a large win bonus, and running out of the step limit gives a penalty.\n"
    "Win condition: clear all pellets. Lose/stop conditions: collide with the ghost or hit the maximum step limit.\n"
    "Each response must be exactly one action token: up, down, left, right, or stay. "
    "Do not explain your reasoning."
)


def observation_metadata_text(env: PacmanEnv) -> str:
    legal_actions = ", ".join(env.legal_actions())
    forbidden_actions = ", ".join(action for action in ACTIONS if action not in env.legal_actions()) or "none"
    unsafe_actions = ", ".join(env.unsafe_actions()) or "none"
    return (
        f"Step {env.state.steps}/{env.max_steps}\n"
        f"Score: {env.state.score}\n"
        f"Pellets left: {len(env.state.pellets)}\n"
        f"Allowed output tokens now: {legal_actions}.\n"
        f"Forbidden output tokens now: {forbidden_actions}.\n"
        f"Unsafe immediate ghost actions: {unsafe_actions}.\n"
        "Choose exactly one allowed token: up, down, left, right, or stay."
    )


def build_observation_turn(
    env: PacmanEnv,
    prompt_style: str,
    observation_mode: str = "text",
    vision_tile_size: int = 32,
    include_image_data_url: bool = True,
) -> ObservationTurn:
    mode = observation_mode.strip().lower()
    if mode == "text":
        text = env.observation_text(prompt_style=prompt_style)
        return ObservationTurn(text=text, model_observation=text)
    if mode not in {"image", "image_text", "image_only"}:
        raise ValueError("observation_mode must be one of: text, image, image_text, image_only")

    png_bytes = env_png_bytes(env, tile_size=vision_tile_size)
    data_url = image_data_url(png_bytes) if include_image_data_url else None
    if mode == "image_only":
        text = "Choose exactly one action token: up, down, left, right, or stay."
    else:
        text = (
            "Use the image as the current PacMan maze observation.\n"
            "Blue tiles are walls, yellow circle is PacMan, red block is the ghost, and small yellow dots are pellets.\n"
        )
    if mode == "image_text":
        text += observation_metadata_text(env)
    elif mode == "image":
        text += "Choose exactly one action token: up, down, left, right, or stay."
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    if data_url is not None:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return ObservationTurn(
        text=text,
        model_observation=content,
        image_sha256=image_sha256(png_bytes),
        image_data_url=data_url,
    )


def raw_completion_prompt(prompt_style: str, observation: str) -> str:
    return (
        f"{system_prompt(prompt_style)}\n\n"
        f"{observation}\n"
        "Answer with exactly one allowed token.\n"
        "Action:"
    )


class PacmanWorkflow:
    """AReaL-compatible async workflow shell.

    AReaL can pass an OpenAI-compatible proxy endpoint through base_url/api_key.
    This class also works as a plain evaluator when a caller supplies completions.
    """

    def __init__(self, **workflow_kwargs):
        self.workflow_kwargs = workflow_kwargs

    async def run(self, data, **extra_kwargs):
        extra_kwargs = {**self.workflow_kwargs, **extra_kwargs}
        completion = extra_kwargs.get("completion")
        if completion is None:
            completion = await self._call_model(data, **extra_kwargs)
        action = parse_action(str(completion))
        return action_reward(action, data["answer"])

    async def _call_model(self, data, **extra_kwargs) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai extra or provide completion=... for dry-run evaluation") from exc

        client = AsyncOpenAI(
            base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=extra_kwargs.get("model", "default"),
            messages=data["messages"],
            temperature=0.0,
            max_tokens=8,
        )
        return response.choices[0].message.content or ""


class PacmanEpisodeWorkflow:
    """Pure text episode rollout.

    One dataset row starts one episode. The model repeatedly sees the current
    text grid, chooses one action, and receives reward from PacmanEnv.step().
    """

    def __init__(self, **workflow_kwargs):
        self.workflow_kwargs = workflow_kwargs
        self.last_trajectory: list[dict[str, object]] = []

    def _trajectory_system_prompt(self, prompt_style: str) -> str:
        return system_prompt(prompt_style)

    async def run(self, data, **extra_kwargs):
        extra_kwargs = {**self.workflow_kwargs, **extra_kwargs}
        episode_id = str(data.get("id") or f"pacman-episode-{data.get('seed', 0)}")
        reward_mode = str(extra_kwargs.get("reward_mode", data.get("reward_mode", "sparse")))
        route_shaping_scale = float(extra_kwargs.get("route_shaping_scale", data.get("route_shaping_scale", 1.0)))
        safe_progress_alpha = float(extra_kwargs.get("safe_progress_alpha", data.get("safe_progress_alpha", 1.0)))
        env = PacmanEnv(
            seed=int(data.get("seed", 0)),
            max_steps=int(data.get("max_steps", 80)),
            illegal_action_penalty=int(data.get("illegal_action_penalty", 0)),
            reward_mode=reward_mode,
            route_shaping_scale=route_shaping_scale,
            safe_progress_alpha=safe_progress_alpha,
            layout_name=str(data.get("layout_name", "default")),
        )
        env.reset()
        prompt_style = str(data.get("prompt_style", "default"))
        parse_failure_penalty = int(extra_kwargs.get("parse_failure_penalty", data.get("parse_failure_penalty", -50)))

        scripted_actions = list(extra_kwargs.get("scripted_actions") or [])
        reward_by_completion: dict[str, float] = {}
        total_reward = 0.0
        trajectory: list[dict[str, object]] = []

        observation_mode = str(extra_kwargs.get("observation_mode", data.get("observation_mode", "text")))
        vision_tile_size = int(extra_kwargs.get("vision_tile_size", data.get("vision_tile_size", 32)))
        store_observation_images = bool(extra_kwargs.get("store_observation_images", data.get("store_observation_images", False)))
        _workflow_trace(
            "episode_start",
            episode_id=episode_id,
            observation_mode=observation_mode,
            prompt_style=prompt_style,
            reward_mode=reward_mode,
            scripted=bool(scripted_actions),
            max_steps=env.max_steps,
        )

        while not env.state.done:
            obs_turn = build_observation_turn(
                env,
                prompt_style=prompt_style,
                observation_mode=observation_mode,
                vision_tile_size=vision_tile_size,
                include_image_data_url=not scripted_actions,
            )
            if scripted_actions:
                turn = ModelTurn(scripted_actions.pop(0))
            else:
                _workflow_trace(
                    "model_call_start",
                    episode_id=episode_id,
                    step=env.state.steps,
                    observation_mode=observation_mode,
                    image_sha256=obs_turn.image_sha256,
                    allowed_actions=env.legal_actions(),
                )
                turn = await self._call_model(
                    obs_turn.model_observation,
                    prompt_style=prompt_style,
                    allowed_actions=env.legal_actions(),
                    previous_action=trajectory[-1]["action"] if trajectory else None,
                    **extra_kwargs,
                )
                _workflow_trace(
                    "model_call_done",
                    episode_id=episode_id,
                    step=env.state.steps,
                    completion_id=turn.completion_id,
                    content_preview=turn.content[:120],
                )
            raw_action = parse_action_for_prompt(
                turn.content,
                prompt_style,
                fail_on_no_action=True,
            )
            parse_failed = raw_action == PARSE_FAILED_ACTION
            if extra_kwargs.get("legal_action_mask") and not parse_failed:
                action = parse_action_for_prompt(
                    turn.content,
                    prompt_style,
                    allowed_actions=env.legal_actions(),
                    fail_on_no_action=True,
                )
            else:
                action = raw_action
            teacher_action = None
            teacher_corrected = False
            if prompt_style == "teacher_hint":
                teacher_action = env.teacher_action()
                teacher_corrected = action != teacher_action
                action = teacher_action
                raw_action = teacher_action if parse_failed else raw_action
                parse_failed = False
            if parse_failed:
                state = PacmanState(
                    pacman=env.state.pacman,
                    ghost=env.state.ghost,
                    pellets=env.state.pellets,
                    steps=env.state.steps + 1,
                    score=env.state.score + parse_failure_penalty,
                    done=True,
                    won=False,
                )
                env.state = state
                reward = parse_failure_penalty
                done = True
                info = {"reason": "parse_failed", "legal_action": False}
            else:
                state, reward, done, info = env.step(action)
            total_reward += float(reward)
            if turn.completion_id:
                reward_by_completion[turn.completion_id] = float(reward)
            step_record = {
                "step": state.steps,
                "obs": obs_turn.text,
                "obs_text": obs_turn.text,
                "observation_mode": observation_mode,
                "obs_image_sha256": obs_turn.image_sha256,
                "model_output": turn.content,
                "request_messages": turn.request_messages,
                "request_prompt": turn.request_prompt,
                "request_extra_body": turn.request_extra_body,
                "request_params": turn.request_params,
                "action": action,
                "raw_action": raw_action,
                "action_masked": action != raw_action,
                "parse_failed": parse_failed,
                "parse_failure_penalty": parse_failure_penalty if parse_failed else 0,
                "exact_action": turn.content.strip().lower() == action,
                "reward": reward,
                "legal_action": info["legal_action"],
                "illegal_action": not bool(info["legal_action"]),
                "route_action": bool(info.get("route_action", False)),
                "safe_distance_before": info.get("safe_distance_before"),
                "safe_distance_after": info.get("safe_distance_after"),
                "safe_progress_reward": float(info.get("safe_progress_reward", 0.0)),
                "done": done,
                "reason": info["reason"],
                "score": state.score,
            }
            if store_observation_images and obs_turn.image_data_url is not None:
                step_record["obs_image_data_url"] = obs_turn.image_data_url
            if teacher_action is not None:
                step_record["teacher_action"] = teacher_action
                step_record["teacher_corrected"] = teacher_corrected
            trajectory.append(step_record)
            _workflow_trace(
                "env_step_done",
                episode_id=episode_id,
                step=state.steps,
                action=action,
                reward=float(reward),
                done=done,
                reason=info["reason"],
                legal_action=bool(info["legal_action"]),
                route_action=bool(info.get("route_action", False)),
                safe_distance_before=info.get("safe_distance_before"),
                safe_distance_after=info.get("safe_distance_after"),
                safe_progress_reward=float(info.get("safe_progress_reward", 0.0)),
                score=state.score,
            )

        self.last_trajectory = trajectory
        self._write_trajectory(
            data,
            env,
            trajectory,
            total_reward,
            extra_kwargs,
            reward_mode,
            route_shaping_scale,
            safe_progress_alpha,
        )
        _workflow_trace(
            "episode_done",
            episode_id=episode_id,
            total_reward=total_reward,
            steps=env.state.steps,
            won=env.state.won,
        )
        if reward_by_completion:
            return reward_by_completion
        return total_reward

    def _write_trajectory(
        self,
        data,
        env: PacmanEnv,
        trajectory: list[dict[str, object]],
        total_reward: float,
        extra_kwargs,
        reward_mode: str,
        route_shaping_scale: float,
        safe_progress_alpha: float,
    ) -> None:
        trajectory_dir = extra_kwargs.get("trajectory_dir") or os.getenv("PACMAN_TRAJECTORY_DIR")
        if not trajectory_dir:
            return

        path = Path(trajectory_dir)
        path.mkdir(parents=True, exist_ok=True)
        episode_id = str(data.get("id") or f"pacman-episode-{data.get('seed', 0)}")
        out = path / f"{episode_id}-{os.getpid()}-{uuid4().hex[:8]}.json"
        final_reason = str(trajectory[-1].get("reason", "unknown")) if trajectory else "unknown"
        payload = {
            "id": episode_id,
            "seed": int(data.get("seed", 0)),
            "max_steps": int(data.get("max_steps", 80)),
            "prompt_style": str(data.get("prompt_style", "default")),
            "layout_name": str(data.get("layout_name", "default")),
            "layout_hash": data.get("layout_hash"),
            "topology_hash": data.get("topology_hash"),
            "maze_suite": data.get("maze_suite"),
            "maze_split": data.get("maze_split"),
            "rollout_index": data.get("rollout_index"),
            "system_prompt": self._trajectory_system_prompt(str(data.get("prompt_style", "default"))),
            "illegal_action_penalty": int(data.get("illegal_action_penalty", 0)),
            "reward_mode": reward_mode,
            "route_shaping_scale": route_shaping_scale,
            "safe_progress_alpha": safe_progress_alpha,
            "observation_mode": str(extra_kwargs.get("observation_mode", data.get("observation_mode", "text"))),
            "vision_tile_size": int(extra_kwargs.get("vision_tile_size", data.get("vision_tile_size", 32))),
            "total_reward": total_reward,
            "final_score": env.state.score,
            "steps": env.state.steps,
            "won": env.state.won,
            "done_reason": final_reason,
            "final_reason": final_reason,
            "trajectory": trajectory,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _workflow_trace("trajectory_written", episode_id=episode_id, path=str(out), steps=len(trajectory))

    async def _call_model(self, observation: str | list[dict[str, object]], **extra_kwargs) -> ModelTurn:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai extra or provide scripted_actions=... for dry-run evaluation") from exc

        client = AsyncOpenAI(
            base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        extra_body = {}
        if extra_kwargs.get("enable_thinking") is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": bool(extra_kwargs["enable_thinking"])
            }
        if (
            extra_kwargs.get("legal_action_choice")
            or extra_kwargs.get("guided_action_choice")
            or extra_kwargs.get("non_stay_legal_action_choice")
            or extra_kwargs.get("non_backtracking_legal_action_choice")
        ):
            extra_body["structured_outputs"] = {
                "choice": guided_choices_for_prompt(
                    extra_kwargs.get("prompt_style", "default"),
                    extra_kwargs.get("allowed_actions"),
                    exclude_stay_when_possible=bool(
                        extra_kwargs.get("non_stay_legal_action_choice")
                        or extra_kwargs.get("non_backtracking_legal_action_choice")
                    ),
                    exclude_reverse_action=(
                        extra_kwargs.get("previous_action")
                        if extra_kwargs.get("non_backtracking_legal_action_choice")
                        else None
                    ),
                )
            }
        elif extra_kwargs.get("action_token_choice"):
            extra_body["structured_outputs"] = {
                "choice": list(ACTIONS),
            }

        completion_api = str(extra_kwargs.get("completion_api", "chat")).lower()
        if completion_api == "completion":
            if not isinstance(observation, str):
                raise ValueError("completion API does not support image observations")
            prompt = raw_completion_prompt(extra_kwargs.get("prompt_style", "default"), observation)
            request_params = {
                "model": extra_kwargs.get("model", "default"),
                "temperature": extra_kwargs.get("temperature", 0.0),
                "top_p": extra_kwargs.get("top_p", 1.0),
                "max_tokens": extra_kwargs.get("max_completion_tokens", 8),
                "stop": ["\n"],
            }
            response = await client.completions.create(
                prompt=prompt,
                extra_body=extra_body or None,
                **request_params,
            )
            return ModelTurn(
                content=response.choices[0].text or "",
                completion_id=getattr(response, "id", None),
                request_prompt=prompt,
                request_extra_body=extra_body or None,
                request_params=request_params,
            )

        if completion_api == "chat_user":
            if not isinstance(observation, str):
                raise ValueError("chat_user API does not support image observations")
            messages = [
                {
                    "role": "user",
                    "content": raw_completion_prompt(extra_kwargs.get("prompt_style", "default"), str(observation)),
                }
            ]
            request_params = {
                "model": extra_kwargs.get("model", "default"),
                "temperature": extra_kwargs.get("temperature", 0.0),
                "top_p": extra_kwargs.get("top_p", 1.0),
                "max_tokens": extra_kwargs.get("max_completion_tokens", 8),
                "stop": ["\n"],
            }
            response = await client.chat.completions.create(
                messages=messages,
                extra_body=extra_body or None,
                **request_params,
            )
            return ModelTurn(
                content=response.choices[0].message.content or "",
                completion_id=getattr(response, "id", None),
                request_messages=messages,
                request_extra_body=extra_body or None,
                request_params=request_params,
            )

        messages = [
            {"role": "system", "content": system_prompt(extra_kwargs.get("prompt_style", "default"))},
            {"role": "user", "content": observation},
        ]
        request_params = {
            "model": extra_kwargs.get("model", "default"),
            "temperature": extra_kwargs.get("temperature", 0.0),
            "top_p": extra_kwargs.get("top_p", 1.0),
            "max_tokens": extra_kwargs.get("max_completion_tokens", 8),
        }
        response = await client.chat.completions.create(
            messages=messages,
            extra_body=extra_body or None,
            **request_params,
        )
        return ModelTurn(
            content=response.choices[0].message.content or "",
            completion_id=getattr(response, "id", None),
            request_messages=messages,
            request_extra_body=extra_body or None,
            request_params=request_params,
        )


class PacmanImageOnlyVLMWorkflow(PacmanEpisodeWorkflow):
    """Strict image-frame VLM episode rollout.

    This workflow exists to avoid accidentally evaluating a metadata-rich
    `image_text` prompt when the intended test is image frame plus action
    instruction only.
    """

    async def run(self, data, **extra_kwargs):
        extra_kwargs = {**extra_kwargs, "observation_mode": "image_only", "completion_api": "chat"}
        return await super().run(data, **extra_kwargs)

    def _trajectory_system_prompt(self, prompt_style: str) -> str:
        return IMAGE_ONLY_SYSTEM_PROMPT

    async def _call_model(self, observation: str | list[dict[str, object]], **extra_kwargs) -> ModelTurn:
        if isinstance(observation, str):
            raise ValueError("PacmanImageOnlyVLMWorkflow requires multimodal image observations")

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai extra or provide scripted_actions=... for dry-run evaluation") from exc

        client = AsyncOpenAI(
            base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )
        extra_body = {}
        if extra_kwargs.get("enable_thinking") is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": bool(extra_kwargs["enable_thinking"])
            }
        if (
            extra_kwargs.get("legal_action_choice")
            or extra_kwargs.get("guided_action_choice")
            or extra_kwargs.get("non_stay_legal_action_choice")
            or extra_kwargs.get("non_backtracking_legal_action_choice")
        ):
            extra_body["structured_outputs"] = {
                "choice": guided_choices_for_prompt(
                    extra_kwargs.get("prompt_style", "default"),
                    extra_kwargs.get("allowed_actions"),
                    exclude_stay_when_possible=bool(
                        extra_kwargs.get("non_stay_legal_action_choice")
                        or extra_kwargs.get("non_backtracking_legal_action_choice")
                    ),
                    exclude_reverse_action=(
                        extra_kwargs.get("previous_action")
                        if extra_kwargs.get("non_backtracking_legal_action_choice")
                        else None
                    ),
                )
            }
        elif extra_kwargs.get("action_token_choice"):
            extra_body["structured_outputs"] = {"choice": list(ACTIONS)}

        messages = [
            {"role": "system", "content": IMAGE_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ]
        request_params = {
            "model": extra_kwargs.get("model", "default"),
            "temperature": extra_kwargs.get("temperature", 0.0),
            "top_p": extra_kwargs.get("top_p", 1.0),
            "max_tokens": extra_kwargs.get("max_completion_tokens", 8),
        }
        response = await client.chat.completions.create(
            messages=messages,
            extra_body=extra_body or None,
            **request_params,
        )
        return ModelTurn(
            content=response.choices[0].message.content or "",
            completion_id=getattr(response, "id", None),
            request_messages=messages,
            request_extra_body=extra_body or None,
            request_params=request_params,
        )


def dry_run_reward() -> float:
    env = PacmanEnv()
    data = {
        "messages": [{"role": "user", "content": env.observation_text()}],
        "answer": "right",
    }
    return action_reward(parse_action("right"), data["answer"])
