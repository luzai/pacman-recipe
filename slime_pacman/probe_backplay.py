"""Frozen-policy options probes; results are never submitted for training."""

import argparse
import asyncio
from collections import Counter
from pathlib import Path

from pacman_recipe.level1.backplay import _wilson, load_restart_bank
from pacman_recipe.level1.contracts import make_episode_record, write_json_new
from .backplay import bind_restart_record
from .backplay_dataset import probe_provenance, _sha
from .config import load_config
from .generation import SGLangGenerator
from .rollout import EpisodeRunner


async def run(args):
    import httpx
    from transformers import AutoProcessor

    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    template = make_episode_record(
        0, split="test", recipe_root=args.recipe, game_root=args.game,
        backend_root=args.slime, max_steps=config.max_steps,
    )
    records = [bind_restart_record(template, args.bank, state_id) for state_id in args.candidate_id]
    provenance = probe_provenance(template, config, load_restart_bank(args.bank)["bank_id"],
                                  _sha(args.server_manifest))
    write_json_new(args.output / "inputs.json", dict(records=records, model=args.model,
                   endpoint=args.endpoint, expected_weight_version=args.weight_version,
                   episodes_per_state=args.episodes, purpose=args.purpose,
                   optimizer_updates=0, training_data=False, provenance=provenance))
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_keepalive_connections=0)) as client:
        generator = SGLangGenerator(processor=processor, endpoint=args.endpoint,
                                    client=client, max_input_tokens=config.max_input_tokens)

        async def generate(messages, constraint):
            decision = await generator(messages, constraint)
            if decision.weight_version != args.weight_version:
                raise ValueError("probe server weight version changed or differs")
            # This is evaluation only; do not retain large training image tensors.
            decision.multimodal_train_inputs = {}
            return decision

        async def episode(record, index):
            async with semaphore:
                runner = EpisodeRunner(record, tokenizer=processor.tokenizer, generate=generate,
                                       config=config, pacman_python_root=args.game)
                result = await runner.collect(empty_weight_version=args.weight_version)
                item = dict(state_id=record["id"], sample_index=index, reward=result.reward,
                            weight_version=result.weight_version, terminal_reason=result.terminal_reason,
                            decision_count=len(result.decisions), trajectory=result.trajectory)
                write_json_new(args.output / f'{record["id"]}-{index:03d}.json', item)
                results.append(item)
                print(f'{record["id"]} sample={index} success={result.reward}', flush=True)

        await asyncio.gather(*(episode(record, i) for record in records for i in range(args.episodes)))
    summaries = []
    for record in records:
        rows = [row for row in results if row["state_id"] == record["id"]]
        successes = sum(int(row["reward"]) for row in rows)
        rate = successes / len(rows)
        summaries.append(dict(state_id=record["id"], samples=len(rows), successes=successes,
                              success_rate=rate, wilson95=_wilson(successes, len(rows)),
                              eligible=args.episodes >= 24 and .3 <= rate <= .7,
                              terminals=dict(Counter(row["terminal_reason"] for row in rows))))
    write_json_new(args.output / "summary.json", dict(summaries=summaries, optimizer_updates=0,
                   purpose=args.purpose, training_data=False, weight_version=args.weight_version,
                   provenance=provenance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "output", "recipe", "game", "slime", "config", "server-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--weight-version", required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--purpose", choices=("coarse", "finalist", "post-update"), required=True)
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.concurrency <= 8:
        parser.error("positive episode count and concurrency in [1,8] required")
    if args.purpose == "finalist" and args.episodes < 24:
        parser.error("finalist evaluation requires at least24 independent episodes")
    if len(args.candidate_id) != len(set(args.candidate_id)):
        parser.error("candidate IDs must be unique")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
