from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .areal_workflow import build_observation_turn, parse_action_for_prompt
from .synthetic.dataset import system_prompt
from .synthetic.env import LAYOUTS, PacmanEnv
from .synthetic.vision import env_png_bytes


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
ACTION_TOKENS = {"up", "down", "left", "right", "stay"}
OPPOSITE_ACTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass(frozen=True)
class SftRow:
    row_id: str
    answer: str
    image_path: Path
    prompt_messages: list[dict[str, Any]]
    full_messages: list[dict[str, Any]]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _user_text_and_image_path(row: dict[str, Any]) -> tuple[str, Path]:
    messages = row["messages"]
    user = next(message for message in messages if message["role"] == "user")
    text_parts: list[str] = []
    image_path = row.get("image_path")
    for item in user["content"]:
        if item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        elif item.get("type") == "image_path":
            image_path = item.get("image_path")
    if not image_path:
        raise ValueError(f"row {row.get('id')} has no image_path")
    return "\n".join(part for part in text_parts if part), Path(str(image_path))


def normalize_row(row: dict[str, Any], dataset_root: Path) -> SftRow:
    answer = str(row.get("answer", "")).strip().lower()
    if answer not in ACTION_TOKENS:
        raise ValueError(f"row {row.get('id')} has invalid answer {answer!r}")
    system = next(message for message in row["messages"] if message["role"] == "system")
    user_text, image_path = _user_text_and_image_path(row)
    if not image_path.is_absolute():
        candidates = [(dataset_root / image_path).resolve(), image_path.resolve()]
        image_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    prompt_messages = [
        {"role": "system", "content": str(system["content"])},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
    return SftRow(
        row_id=str(row.get("id", "")),
        answer=answer,
        image_path=image_path,
        prompt_messages=prompt_messages,
        full_messages=full_messages,
    )


def prepare_rows(jsonl: Path, limit: int | None = None) -> list[SftRow]:
    raw_rows = load_jsonl(jsonl, limit=limit)
    dataset_root = jsonl.parent
    return [normalize_row(row, dataset_root) for row in raw_rows]


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _load_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def build_batch(processor: Any, rows: list[SftRow], device: str | None = None) -> dict[str, Any]:
    full_texts = [_apply_chat_template(processor, row.full_messages, add_generation_prompt=False) for row in rows]
    prompt_texts = [_apply_chat_template(processor, row.prompt_messages, add_generation_prompt=True) for row in rows]
    images = [_load_image(row.image_path) for row in rows]
    batch = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    tokenizer = getattr(processor, "tokenizer", processor)
    for index, prompt_text in enumerate(prompt_texts):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        labels[index, : min(len(prompt_ids), labels.shape[1])] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    if device:
        batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    return batch


def iter_batches(rows: list[SftRow], batch_size: int, shuffle: bool, seed: int) -> list[list[SftRow]]:
    ordered = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]


def load_model(model_name: str, dtype: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        AutoModelForMultimodalLM = None

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    model_cls = AutoModelForMultimodalLM or AutoModelForCausalLM
    return model_cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )


def maybe_apply_lora(model: Any, use_lora: bool, rank: int, alpha: int, adapter_dir: Path | None = None) -> Any:
    if adapter_dir is not None:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, adapter_dir, is_trainable=use_lora)
    if not use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)


def _first_device(model: Any) -> str:
    return str(next(model.parameters()).device)


def _decode_action(processor: Any, generated_ids: Any) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def _messages_from_env(env: PacmanEnv, prompt_style: str, vision_tile_size: int) -> tuple[list[dict[str, Any]], bytes]:
    obs = build_observation_turn(
        env,
        prompt_style=prompt_style,
        observation_mode="image_text",
        vision_tile_size=vision_tile_size,
        include_image_data_url=False,
    )
    png_bytes = env_png_bytes(env, tile_size=vision_tile_size)
    text = obs.text
    return (
        [
            {"role": "system", "content": system_prompt(prompt_style)},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "__runtime_env_image__"},
                    {"type": "text", "text": text},
                ],
            },
        ],
        png_bytes,
    )


def _image_from_png_bytes(png_bytes: bytes) -> Any:
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(png_bytes)).convert("RGB")


def generate_action(
    processor: Any,
    model: Any,
    env: PacmanEnv,
    prompt_style: str,
    vision_tile_size: int,
    max_new_tokens: int,
) -> tuple[str, str]:
    import torch

    messages, png_bytes = _messages_from_env(env, prompt_style=prompt_style, vision_tile_size=vision_tile_size)
    prompt = _apply_chat_template(processor, messages, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[_image_from_png_bytes(png_bytes)], return_tensors="pt")
    device = _first_device(model)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None),
        )
    prompt_len = batch["input_ids"].shape[1]
    output_text = _decode_action(processor, generated[0][prompt_len:])
    action = parse_action_for_prompt(
        output_text,
        prompt_style=prompt_style,
        allowed_actions=env.legal_actions(),
        fail_on_no_action=True,
    )
    return action, output_text


def score_action_choice(
    processor: Any,
    model: Any,
    env: PacmanEnv,
    action: str,
    prompt_style: str,
    vision_tile_size: int,
) -> float:
    import torch

    prompt_messages, png_bytes = _messages_from_env(env, prompt_style=prompt_style, vision_tile_size=vision_tile_size)
    full_messages = prompt_messages + [{"role": "assistant", "content": action}]
    full_text = _apply_chat_template(processor, full_messages, add_generation_prompt=False)
    prompt_text = _apply_chat_template(processor, prompt_messages, add_generation_prompt=True)
    image = _image_from_png_bytes(png_bytes)
    batch = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    labels[0, : min(len(prompt_ids), labels.shape[1])] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    device = _first_device(model)
    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    with torch.inference_mode():
        output = model(**batch)
    return float(output.loss.detach().cpu())


def choose_action_by_loss(
    processor: Any,
    model: Any,
    env: PacmanEnv,
    prompt_style: str,
    vision_tile_size: int,
    allowed_actions: list[str] | None = None,
) -> tuple[str, str]:
    legal_actions = allowed_actions or env.legal_actions()
    scores = {
        action: score_action_choice(
            processor=processor,
            model=model,
            env=env,
            action=action,
            prompt_style=prompt_style,
            vision_tile_size=vision_tile_size,
        )
        for action in legal_actions
    }
    action = min(scores, key=scores.get)
    detail = json.dumps(scores, sort_keys=True)
    return action, detail


def evaluate_adapter_policy(processor: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    wins = 0
    total_rewards: list[float] = []
    model.eval()
    for seed in range(args.eval_episodes):
        env = PacmanEnv(
            seed=seed,
            max_steps=args.eval_max_steps,
            layout_name=args.eval_layout_name,
            reward_mode=args.eval_reward_mode,
        )
        total_reward = 0.0
        trajectory = []
        previous_action = None
        while not env.state.done:
            step = env.state.steps
            allowed_actions = env.legal_actions()
            if args.eval_exclude_stay_when_possible and any(action != "stay" for action in allowed_actions):
                allowed_actions = [action for action in allowed_actions if action != "stay"]
            reverse_action = OPPOSITE_ACTION.get(previous_action or "")
            if (
                args.eval_non_backtracking
                and reverse_action in allowed_actions
                and any(action != reverse_action for action in allowed_actions)
            ):
                allowed_actions = [action for action in allowed_actions if action != reverse_action]
            if args.eval_decision_mode == "choice_loss":
                action, model_output = choose_action_by_loss(
                    processor=processor,
                    model=model,
                    env=env,
                    prompt_style=args.eval_prompt_style,
                    vision_tile_size=args.vision_tile_size,
                    allowed_actions=allowed_actions,
                )
            else:
                action, model_output = generate_action(
                    processor=processor,
                    model=model,
                    env=env,
                    prompt_style=args.eval_prompt_style,
                    vision_tile_size=args.vision_tile_size,
                    max_new_tokens=args.eval_max_new_tokens,
                )
                if action == "__parse_failed__":
                    action = "stay"
                if action not in allowed_actions:
                    action = allowed_actions[0]
            state, reward, done, info = env.step(action)
            previous_action = action
            total_reward += float(reward)
            trajectory.append(
                {
                    "step": step,
                    "model_output": model_output,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "reason": info["reason"],
                    "legal_action": info["legal_action"],
                    "route_action": info.get("route_action", False),
                    "score": state.score,
                }
            )
        wins += int(env.state.won)
        total_rewards.append(total_reward)
        episodes.append(
            {
                "seed": seed,
                "won": env.state.won,
                "steps": env.state.steps,
                "total_reward": total_reward,
                "final_score": env.state.score,
                "final_reason": trajectory[-1]["reason"] if trajectory else "empty",
                "trajectory": trajectory,
            }
        )
    return {
        "eval_episodes": args.eval_episodes,
        "eval_wins": wins,
        "eval_reward_min": min(total_rewards) if total_rewards else None,
        "eval_reward_max": max(total_rewards) if total_rewards else None,
        "eval_reward_mean": sum(total_rewards) / len(total_rewards) if total_rewards else None,
        "eval_episodes_detail": episodes,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    rows = prepare_rows(args.jsonl, limit=args.limit)
    summary: dict[str, Any] = {
        "jsonl": str(args.jsonl),
        "rows": len(rows),
        "answers": {answer: sum(row.answer == answer for row in rows) for answer in sorted(ACTION_TOKENS)},
        "first_id": rows[0].row_id if rows else None,
        "first_answer": rows[0].answer if rows else None,
        "first_image_path": str(rows[0].image_path) if rows else None,
    }
    if args.prepare_only:
        return summary

    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.dtype)
    model = maybe_apply_lora(model, args.lora, args.lora_rank, args.lora_alpha, args.adapter_dir)
    if hasattr(model, "gradient_checkpointing_enable") and not args.forward_only:
        model.gradient_checkpointing_enable()
    model.train(not args.forward_only)

    if args.eval_only:
        eval_summary = evaluate_adapter_policy(processor, model, args)
        summary.update(eval_summary)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    device = _first_device(model)
    batch = build_batch(processor, rows[: args.batch_size], device=device)
    with torch.set_grad_enabled(not args.forward_only):
        output = model(**batch)
        loss = output.loss
    summary.update(
        {
            "loss": float(loss.detach().cpu()),
            "input_shape": list(batch["input_ids"].shape),
            "label_tokens": int((batch["labels"] != -100).sum().detach().cpu()),
        }
    )
    if args.forward_only:
        return summary

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr)
    losses: list[float] = []
    total_steps = 0
    for epoch in range(args.epochs):
        for batch_rows in iter_batches(rows, args.batch_size, args.shuffle, seed=args.seed + epoch):
            batch = build_batch(processor, batch_rows, device=device)
            output = model(**batch)
            step_loss = output.loss
            step_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(step_loss.detach().cpu()))
            total_steps += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(args.output_dir / "adapter")
    summary.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_steps": total_steps,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
        }
    )
    if args.eval_after_train:
        eval_summary = evaluate_adapter_policy(processor, model, args)
        summary.update(eval_summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["output_dir"] = str(args.output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step Qwen vision SFT smoke for PacMan image_text rows.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("run_artifacts/vision_sft_smoke"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-after-train", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=30)
    parser.add_argument("--eval-layout-name", choices=tuple(sorted(LAYOUTS)), default="medium_default")
    parser.add_argument("--eval-prompt-style", default="ghost_legal_strict")
    parser.add_argument(
        "--eval-reward-mode",
        choices=("sparse", "route_prefix", "route_prefix_stay_penalty", "route_prefix_progress_penalty"),
        default="sparse",
    )
    parser.add_argument("--eval-max-new-tokens", type=int, default=8)
    parser.add_argument("--eval-decision-mode", choices=("generate", "choice_loss"), default="generate")
    parser.add_argument("--eval-exclude-stay-when-possible", action="store_true")
    parser.add_argument("--eval-non-backtracking", action="store_true")
    parser.add_argument("--vision-tile-size", type=int, default=32)
    args = parser.parse_args()
    summary = run_smoke(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
