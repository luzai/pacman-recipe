"""Finalist screening of teacher-bank restart states with the process-parallel client.

Same evidence contract as the shard tools (probe_backplay + merge_finalist_shards): a frozen policy
plays N independent episodes from every candidate state, nothing is trained on, and the merged
summary (purpose "finalist", optimizer_updates 0, probe provenance) is what
backplay_dataset.validate_selection checks. Episodes run in spawn worker processes spread
round-robin over several SGLang endpoints; every decision must carry the expected weight version.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from pacman_recipe.level1.backplay import _wilson, load_restart_bank
from pacman_recipe.level1.contracts import make_episode_record, write_json_new

from .backplay import bind_restart_record
from .backplay_dataset import _sha, probe_provenance
from .config import load_config
from .rollout import _init_episode_worker, collect_episode_in_worker, current_sources


def summarize(state_ids, rows, episodes):
    summaries = []
    for state_id in state_ids:
        state_rows = [row for row in rows if row["state_id"] == state_id]
        if sorted(row["sample_index"] for row in state_rows) != list(range(episodes)):
            raise ValueError(f"incomplete screening evidence for {state_id}")
        successes = sum(int(row["reward"]) for row in state_rows)
        summaries.append(dict(state_id=state_id, samples=episodes, successes=successes,
                              success_rate=successes / episodes, wilson95=_wilson(successes, episodes),
                              terminals=dict(Counter(row["terminal_reason"] for row in state_rows))))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--bank", type=Path, default=Path("/bank"))
    parser.add_argument("--candidate-grid", type=Path, help="candidate-grid.json of the backplay smoke")
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--purpose", choices=("finalist", "evaluation"), default="finalist",
                        help="evaluation: independent re-measurement of fixed states (never selection evidence)")
    parser.add_argument("--weight-version", required=True)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--model", default="/model")
    parser.add_argument("--config", default="/workspace/pacman-recipe/configs/slime/c2.yaml")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_ids = list(args.candidate_id)
    if args.candidate_grid:
        # The smoke candidate grid: {"candidates": [{"state_id": ...}, ...]}.
        state_ids += [c["state_id"] for c in json.loads(args.candidate_grid.read_text())["candidates"]]
    if not state_ids or len(set(state_ids)) != len(state_ids):
        raise ValueError("need distinct candidate state ids")
    config = load_config(args.config)
    template = make_episode_record(0, split="test", recipe_root=Path("/workspace/pacman-recipe"),
                                   game_root=Path("/workspace/pacman-python"),
                                   backend_root=Path("/workspace/slime"), max_steps=config.max_steps)
    bank = load_restart_bank(args.bank)
    provenance = probe_provenance(template, config, bank["bank_id"], _sha(args.server_manifest))
    records = {state_id: bind_restart_record(template, args.bank, state_id) for state_id in state_ids}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_new(args.output / "inputs.json", dict(
        state_ids=state_ids, episodes_per_state=args.episodes, endpoints=args.endpoints, model=args.model,
        expected_weight_version=args.weight_version, purpose=args.purpose,
        training_data=False, provenance=provenance,
        **(dict(optimizer_updates=0) if args.purpose == "finalist" else {})))
    sources = current_sources()
    jobs = [(state_id, index) for index in range(args.episodes) for state_id in state_ids]
    rows, start = [], time.time()
    episodes_path = args.output / "episodes.jsonl"
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_episode_worker) as pool, episodes_path.open("x") as out:
        futures = {pool.submit(collect_episode_in_worker, records[state_id], args.endpoints[n % len(args.endpoints)],
                               args.model, args.config, sources, args.weight_version, None, True): (state_id, index)
                   for n, (state_id, index) in enumerate(jobs)}
        for future in as_completed(futures):
            state_id, index = futures[future]
            episode = future.result()
            if episode.weight_version != args.weight_version:
                raise ValueError("screening server weight version changed or differs")
            row = dict(state_id=state_id, sample_index=index, reward=float(episode.reward),
                       terminal_reason=episode.terminal_reason, weight_version=episode.weight_version)
            if row["reward"] not in (0.0, 1.0):
                raise ValueError("screening reward must be binary")
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            rows.append(row)
    summaries = summarize(state_ids, rows, args.episodes)
    extra = dict(optimizer_updates=0) if args.purpose == "finalist" else {}
    write_json_new(args.output / "summary.json", dict(**extra,
        purpose=args.purpose, training_data=False, provenance=provenance,
        weight_version=args.weight_version, summaries=summaries,
        episodes_file=dict(file=episodes_path.name, sha256=hashlib.sha256(episodes_path.read_bytes()).hexdigest()),
        independent_episode_identity="(state_id, sample_index)", seconds=time.time() - start))
    for summary in summaries:
        print(f'{summary["state_id"][:24]} {summary["successes"]}/{summary["samples"]}', flush=True)


if __name__ == "__main__":
    main()
