from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .areal_workflow import parse_action_for_prompt
from .synthetic.dataset import system_prompt
from .synthetic.env import LAYOUTS, PacmanEnv


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def format_messages(tokenizer, messages: list[dict[str, str]], enable_thinking: bool | None = None) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages) + "\n\nASSISTANT:\n"


def generate_turn(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
) -> str:
    import torch

    prompt = format_messages(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_episode(
    tokenizer,
    model,
    model_name: str,
    seed: int,
    max_steps: int,
    prompt_style: str,
    layout_name: str,
    illegal_action_penalty: int,
    reward_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool | None,
    legal_action_mask: bool,
) -> dict[str, object]:
    env = PacmanEnv(
        seed=seed,
        max_steps=max_steps,
        illegal_action_penalty=illegal_action_penalty,
        reward_mode=reward_mode,
        layout_name=layout_name,
    )
    env.reset()
    total_reward = 0.0
    trajectory: list[dict[str, object]] = []
    sys_prompt = system_prompt(prompt_style)

    while not env.state.done:
        obs = env.observation_text(prompt_style=prompt_style)
        output = generate_turn(
            tokenizer=tokenizer,
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": obs},
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        raw_action = parse_action_for_prompt(output, prompt_style)
        if legal_action_mask:
            action = parse_action_for_prompt(output, prompt_style, allowed_actions=env.legal_actions())
        else:
            action = raw_action
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        trajectory.append(
            {
                "step": state.steps,
                "obs": obs,
                "model_output": output,
                "action": action,
                "raw_action": raw_action,
                "action_masked": action != raw_action,
                "exact_action": output.strip().lower() == action,
                "reward": reward,
                "legal_action": info["legal_action"],
                "done": done,
                "reason": info["reason"],
                "score": state.score,
            }
        )

    return {
        "id": f"pacman-hf-rollout-{seed}",
        "model": model_name,
        "seed": seed,
        "max_steps": max_steps,
        "prompt_style": prompt_style,
        "layout_name": layout_name,
        "system_prompt": sys_prompt,
        "illegal_action_penalty": illegal_action_penalty,
        "reward_mode": reward_mode,
        "enable_thinking": enable_thinking,
        "legal_action_mask": legal_action_mask,
        "total_reward": total_reward,
        "final_score": env.state.score,
        "steps": env.state.steps,
        "won": env.state.won,
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-training Hugging Face PacMan text rollouts.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--prompt-style", default="ghost_legal_reason")
    parser.add_argument("--layout-name", choices=tuple(sorted(LAYOUTS)), default="medium_default")
    parser.add_argument("--illegal-action-penalty", type=int, default=-4)
    parser.add_argument("--reward-mode", choices=("sparse", "route_prefix"), default="sparse")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--legal-action-mask", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model)
    for seed in range(args.episodes):
        payload = run_episode(
            tokenizer=tokenizer,
            model=model,
            model_name=args.model,
            seed=seed,
            max_steps=args.max_steps,
            prompt_style=args.prompt_style,
            layout_name=args.layout_name,
            illegal_action_penalty=args.illegal_action_penalty,
            reward_mode=args.reward_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
            legal_action_mask=args.legal_action_mask,
        )
        model_label = Path(args.model).name or args.model
        safe_model = model_label.replace("/", "__").replace(".", "p")
        model_hash = hashlib.sha1(args.model.encode("utf-8")).hexdigest()[:8]
        safe_model = f"{safe_model}-{model_hash}"[:96]
        path = args.output_dir / f"{safe_model}-seed{seed}-{uuid4().hex[:8]}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote={path}")
        print(f"won={payload['won']} steps={payload['steps']} total_reward={payload['total_reward']}")


if __name__ == "__main__":
    main()
