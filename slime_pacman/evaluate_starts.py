"""Independent start-state evaluation (dynamic-bank plan): true starts and holdout seeds.

True starts: seeds 0/1/14/15/16 x 48 episodes; holdout: seeds 72-111 x 4 episodes; plain environment
reset (no restart file), same episode code and harness as training, episodes spread round-robin over
the given SGLang endpoints in a process pool. Writes one JSON line per episode; evaluation only, never
training data or bank candidates. Statistics: slime_pacman.eval_stats.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from pacman_recipe.level1.contracts import make_episode_record

from .config import load_config
from .rollout import _init_episode_worker, collect_episode_in_worker, current_sources


def seed_list(text):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--config", default="/workspace/pacman-recipe/configs/slime/c2.yaml")
    parser.add_argument("--true-seeds", default="0,1,14,15,16")
    parser.add_argument("--true-episodes", type=int, default=48)
    parser.add_argument("--holdout-seeds", default="72-111")
    parser.add_argument("--holdout-episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    roots = dict(recipe_root=Path("/workspace/pacman-recipe"), game_root=Path("/workspace/pacman-python"),
                 backend_root=Path("/workspace/slime"))
    jobs = [("true_start", seed, i) for seed in seed_list(args.true_seeds) for i in range(args.true_episodes)]
    jobs += [("holdout", seed, i) for seed in seed_list(args.holdout_seeds) for i in range(args.holdout_episodes)]
    sources = current_sources()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_episode_worker) as pool, args.output.open("x") as out:
        futures = {}
        for n, (group, seed, index) in enumerate(jobs):
            record = make_episode_record(seed, split="test", max_steps=config.max_steps, **roots)
            futures[pool.submit(collect_episode_in_worker, record, args.endpoints[n % len(args.endpoints)],
                                args.model, args.config, sources, "", None, True)] = (group, seed, index)
        for future in as_completed(futures):
            group, seed, index = futures[future]
            episode = future.result()
            out.write(json.dumps(dict(label=args.label, group=group, seed=seed, index=index, reward=episode.reward,
                                      terminal_reason=episode.terminal_reason,
                                      weight_version=episode.weight_version)) + "\n")
            out.flush()
    print(json.dumps(dict(label=args.label, episodes=len(jobs), seconds=time.time() - start)))


if __name__ == "__main__":
    main()
