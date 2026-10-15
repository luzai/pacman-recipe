"""Decisions transport adapter for the unmodified native Pacman episode runner."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pacman_env.planner import resolve_fallback_mode
from pacman_recipe.level1.episode import ModelTurn, PacmanEpisodeRunner
from pacman_recipe.level1.level1_dataset import SUPPORTED_MAX_STEPS, make_episode_row, validate_episode_row
from pacman_recipe.level1.prompts import prompt_contract_metadata
from pacman_recipe.level1.trajectories import audit_trajectory

from .client import MODEL, DecisionError, load_key, send_body


class SymbolCodec:
    """Runner-only reversible symbol IDs, never sent as OpenAI token IDs.

    The Decisions endpoint selects strings rather than Qwen token IDs. Reuse
    ObjectiveTokenConstraint's existing code mapping without loading a model.
    This codec does not claim model tokenization or log-probability parity.
    """
    all_special_ids = ()

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def native_payload(messages, choices):
    choices = tuple(choices)
    if not choices or len(set(choices)) != len(choices):
        raise ValueError("Expected nonempty unique native choices")
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts = [{"type": "input_text", "text": content}]
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                elif part["type"] == "image_url":
                    image = part["image_url"]
                    parts.append({"type": "input_image", "image_url": image["url"]})
                    if "detail" in image:
                        parts[-1]["detail"] = image["detail"]
                else:
                    raise ValueError("Unsupported native content part")
        converted.append({"role": message["role"], "content": parts})
    return {"model": MODEL, "input": converted, "questions": [{
        "type": "choice", "name": "action",
        "instructions": "Follow the supplied game instructions. Select exactly one advertised output code.",
        "choices": [{"value": code, "description": code} for code in choices]}]}


def runner_options(mode, fallback=None):
    fallback = resolve_fallback_mode(fallback, edward_options=mode == "edward")
    if mode not in ("primitive", "edward"):
        raise ValueError("Unknown mode")
    return dict(edward_options=mode == "edward", open_action_mask=mode == "primitive",
        image_prompt_style="live_state_v3", edward_fallback_mode=fallback,
        episode_life_mode="single_death", enable_thinking=False,
        model=MODEL, temperature=0.0, max_completion_tokens=1,
        use_base_reward=False, normal_pellet_reward=0.0, power_pellet_reward=0.0,
        ghost_reward=0.0, fruit_reward=0.0, death_penalty=0.0,
        completion_reward=1.0, safety_refusal_penalty=0.0, step_penalty=0.0,
        wall_penalty=0.0, nearest_pellet_alpha=0.0)


def validate_eval_row(row):
    """Validate the full production row plus an evaluation-only one-step cap.

    Production one-step *training* rows require a teacher-prefix audit. A fresh
    initial-state smoke test has no teacher prefix and must not fabricate one.
    Only this adapter's explicit smoke marker permits the runner's existing
    single-step branch; full comparisons use validate_episode_row unchanged.
    """
    if row.get("baseline_smoke_only") is True:
        if row.get("decision_steps") != 1 or row.get("state_prefix_actions") or row.get("restart_state_path"):
            raise ValueError("Smoke row must start at reset and cap one game step")
        production = dict(row)
        production.pop("baseline_smoke_only")
        production.pop("decision_steps")
        validate_episode_row(production)
    else:
        validate_episode_row(row)


class NativeDecisionRunner(PacmanEpisodeRunner):
    def __init__(self, *, mode, key=None, live=False, output=None, sender=send_body,
                 fallback=None, **kwargs):
        super().__init__(tokenizer=SymbolCodec(), **runner_options(mode, fallback), **kwargs)
        self.key = key
        self.live = live
        self.output = Path(output) if output is not None else None
        self.sender = sender
        self.decisions = []

    async def _call_model(self, messages, **options):
        constraint = options.get("objective_constraint")
        choices = tuple(constraint.rendered_choices if constraint is not None
                        else options["current_open_actions"])
        body = native_payload(messages, choices)
        index = len(self.decisions)
        stem = f"decision{index:04d}"
        if self.output:
            # Prompts and images are evaluator artifacts; no credential is present.
            (self.output / f"{stem}_native_messages.json").write_text(
                json.dumps(messages, indent=2), encoding="utf-8")
            (self.output / f"{stem}_request.json").write_text(
                json.dumps(body, indent=2), encoding="utf-8")
            for msg in body["input"]:
                for part in msg["content"]:
                    if part["type"] == "input_image":
                        (self.output / f"{stem}.png").write_bytes(
                            base64.b64decode(part["image_url"].split(",", 1)[1]))
        record = {"index": index, "choices": choices,
                  "request_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest()}
        if not self.live:
            result = {"action": choices[0], "source": "offline_contract_fixture",
                      "request_seconds": None}
        elif len(choices) == 1:
            # API choice requires 2+ values. Do not add an unadvertised action.
            result = {"action": choices[0], "source": "forced_singleton",
                      "request_seconds": None}
        else:
            record["status"] = "request_started"
            self.decisions.append(record)
            self._write_decisions()
            try:
                result = self.sender(body, self.key, timeout=options.get("decision_timeout", 30))
            except Exception:
                record["status"] = "request_failed"
                self._write_decisions()
                raise
            result["source"] = "live_api"
        if result["action"] not in choices:
            raise DecisionError("Transport returned an unadvertised code")
        record.update(result, status="completed")
        if not self.decisions or self.decisions[-1] is not record:
            self.decisions.append(record)
        self._write_decisions()
        return ModelTurn(result["action"], f"decision-{index}", messages,
                         raw_response={"decisions_adapter": result})

    def _write_decisions(self):
        if self.output:
            (self.output / "decisions.json").write_text(
                json.dumps(self.decisions, indent=2), encoding="utf-8")


def metrics(runner, mode, seed, *, single_step=False):
    episode = runner.last_episode
    if episode is None:
        return {"mode": mode, "seed": seed, "status": "safety_refusal_before_decision",
                "game_steps": 0, "api_calls": 0, "score": None}
    steps = episode["trajectory"]
    live = [d for d in runner.decisions if d.get("source") == "live_api"]
    latency = [d["request_seconds"] for d in live]
    return {"mode": mode, "seed": seed,
        "status": "live" if runner.live else "offline_fixture_not_performance",
        "single_step": single_step, "game_steps": len(steps),
        "decisions": len(runner.decisions), "api_calls": len(live),
        "forced_choices": sum(d.get("source") == "forced_singleton" for d in runner.decisions),
        "score": steps[-1]["score"], "wall_collisions": episode["wall_collisions"],
        "deaths": episode["death_count"],
        "won": episode["won"], "terminal_reason": episode["terminal_reason"],
        "normal_pellets_eaten": episode["normal_pellets_eaten"],
        "pellet_clear_rate": episode["pellet_clear_rate"],
        "normal_pellet_clear_rate": episode["normal_pellet_clear_rate"],
        "request_p50_seconds": float(np.percentile(latency, 50)) if latency else None,
        "request_p95_seconds": float(np.percentile(latency, 95)) if latency else None,
        "input_tokens": sum(d.get("usage", {}).get("input_tokens", 0) for d in live)}


async def run_one(mode, seed, max_steps, single_step, live, key, output, fallback=None):
    fallback = resolve_fallback_mode(fallback, edward_options=mode == "edward")
    output.mkdir(parents=True, exist_ok=False)
    row = make_episode_row(1, split="test", seed=seed, max_steps=max_steps)
    if single_step:
        row["decision_steps"] = 1
        row["baseline_smoke_only"] = True
    runner = NativeDecisionRunner(mode=mode, key=key, live=live, output=output, fallback=fallback,
                                 row_validator=validate_eval_row)
    manifest = {"mode": mode, "row": row, "live": live,
        "prompt": prompt_contract_metadata("live_state_v3", edward_options=mode == "edward", fallback_mode=fallback),
        "runner": "pacman_recipe.level1.episode.PacmanEpisodeRunner (unmodified)",
        "options": runner.workflow_kwargs, "max_api_calls": 1 if single_step else max_steps,
        "row_validation": "full native production validation; smoke-only cap separately validated without fictitious teacher-prefix evidence",
        "differences": ["Decisions choice over native output codes instead of token generation",
            "SymbolCodec is a local code mapper, not the OpenAI/Qwen tokenizer",
            "One supported code is forced locally, with zero API calls",
            "OpenAI controls image encoding and sampling; Qwen processor/temperature parity is not claimed",
            "API renders system messages as developer messages; original role/text is retained in request",
            "Offline choices are fixtures, never performance results"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        await runner.run(row)
        if runner.last_episode is not None:
            audit_trajectory(runner.last_episode)
            (output / "episode.json").write_text(json.dumps(runner.last_episode, indent=2), encoding="utf-8")
        result = metrics(runner, mode, seed, single_step=single_step)
    except Exception as exc:
        # Avoid leaking arbitrary server content or credential-bearing exceptions.
        result = {"mode": mode, "seed": seed, "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc) if isinstance(exc, DecisionError) else "Local harness failure; inspect offline reproduction",
            "api_attempts": sum(d.get("status") in ("request_failed", "request_started") or d.get("source") == "live_api" for d in runner.decisions)}
    finally:
        runner.key = None
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_comparison(output, results):
    headers = ["mode", "seed", "status", "score", "game_steps", "api_calls",
               "wall_collisions", "deaths", "request_p50_seconds", "terminal_reason"]
    lines = ["# Native Decisions baseline comparison", "",
        "Same initial seeds, normal ghosts, single death, identical game-step budgets.",
        "Native primitive and Edward expose different state/action contracts. This is a harness comparison, not a controlled model-only comparison.",
        "Offline fixtures are not performance. A budget cutoff is not a win. Latency is measured over API calls only; game time pauses during requests.", "",
        "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(r.get(h, "—")) for h in headers) + " |" for r in results)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primitive", "edward", "both"), default="both")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--single-step", action="store_true")
    parser.add_argument("--max-steps", type=int, choices=sorted(SUPPORTED_MAX_STEPS), default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fallback", choices=("refuse", "risk_ranked"), default=None, help="Edward fallback (default: risk_ranked); primitive stays refuse")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    output = args.output or Path("artifacts/decisions_vision/native") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    key = load_key() if args.live else None
    modes = ("primitive", "edward") if args.mode == "both" else (args.mode,)
    results = []
    async def run():
        for offset in range(args.episodes):
            for mode in modes:
                result = await run_one(mode, args.seed + offset, args.max_steps,
                    args.single_step, args.live, key,
                    output / f"{mode}_seed{args.seed + offset}", args.fallback if mode == "edward" else "refuse")
                results.append(result)
                write_comparison(output, results)
                print(json.dumps(result), flush=True)
                if result["status"] == "failed":
                    return
    asyncio.run(run())
    print(str((output / "comparison.md").resolve()))
    return int(any(r["status"] == "failed" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
