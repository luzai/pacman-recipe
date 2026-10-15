"""Image smoke test or bounded native pygame rollout; offline by default."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

from .client import DecisionError, PROMPT, decide, load_key, payload


def png_bytes(frame):
    from PIL import Image
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG")
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize bounded API calls")
    parser.add_argument("--image", type=Path, help="PNG-only static decision; no game step")
    parser.add_argument("--steps", type=int, default=1, help="Maximum calls per episode")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1 or args.timeout <= 0:
        parser.error("steps, episodes and timeout must be positive")
    if args.image and (args.steps != 1 or args.episodes != 1):
        parser.error("image mode supports exactly one request")
    output = args.output or Path("artifacts/decisions_vision") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"live": args.live, "seed": args.seed, "episodes": args.episodes,
        "max_calls": args.episodes * args.steps, "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "observation": "current RGB screenshot only; no state, legal mask or history",
        "execution": "chosen primitive action unchanged; native step; single death",
        "game_time": "paused between environment steps; not real-time latency acceptance"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = None
    records = []
    key = None
    try:
        key = load_key() if args.live else None
        if not args.image:
            from pacman_env.env.pygame_environment import PygamePacmanEnv, PygamePacmanEnvConfig
            env = PygamePacmanEnv(PygamePacmanEnvConfig())
        for episode in range(args.episodes):
            if env:
                frame, info = env.reset(seed=args.seed + episode)
                (output / f"initial_info_{episode}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            for turn in range(args.steps):
                start = time.perf_counter()
                png = png_bytes(frame) if env else args.image.read_bytes()
                name = f"episode{episode}_turn{turn}"
                (output / f"{name}.png").write_bytes(png)
                body = payload(png)
                record = {"episode": episode, "turn": turn,
                    "image_sha256": hashlib.sha256(png).hexdigest(),
                    "request_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest()}
                if not args.live:
                    record["status"] = "offline_payload_validated"
                    records.append(record)
                    break
                record.update(decide(png, key, timeout=args.timeout))
                if env:
                    action = record["action"]
                    record["chosen_action_legal"] = action in info["legal_actions"]
                    record["position_before"] = info["pacman_position"]
                    frame, reward, terminated, truncated, info = env.step(action)
                    record.update(executed_action=action, reward=reward,
                        position_after=info["pacman_position"], score=info["score"],
                        wall_collision=info["wall_collision"], death=info["death"],
                        level_cleared=info["level_cleared"], terminated=terminated,
                        truncated=truncated, pellet_clear_rate=info["pellet_clear_rate"])
                    (output / f"{name}_after.png").write_bytes(png_bytes(frame))
                record["turn_seconds"] = time.perf_counter() - start
                records.append(record)
                with (output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                if env and (terminated or truncated):
                    break
        status = "completed" if args.live else "offline_only"
    except DecisionError as exc:
        status = "failed"
        (output / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
    finally:
        key = None
        if env:
            env.close()
    summary = {"status": status, "decisions": len(records), "output": str(output.resolve()),
        "records": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
