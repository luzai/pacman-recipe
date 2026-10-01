"""Dynamic-bank curriculum rollout (HANDOFF dynamic-bank plan + pre-declared addenda).

Per training rollout `r` (slime rollout id, 0-based; weights are the ones synced after update r):
  * r == 0: initialization -- the current policy plays every training seed's true start
    (PACMAN_INIT_EPISODES_PER_SEED episodes each, not training data); death episodes yield
    pre-death candidates; up to INIT_TRAJECTORIES trajectories are probed (budget INIT_BUDGET).
  * r in {2, 4, 6, 8}: refresh from the death trajectories of training rollouts r-2 and r-1
    (REFRESH_TRAJECTORIES trajectories, budget REFRESH_BUDGET).
  * every r: select PACMAN_TRUE_STARTS rotating true starts + PACMAN_BANK_STARTS bank states,
    freeze bank manifest and selection to dynamic-bank/state/rollout-r.json (resume point), then
    generate one informative group per start with zero-variance resampling (sampling.rollout_slots).
Probe episodes use the frozen current weights and never become training data. Evaluation rollouts
use slime's default path.

Enable with --rollout-function-path slime_pacman.curriculum.generate_rollout.
"""

import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random

from pacman_recipe.level1.backplay import load_restart_bank
from pacman_recipe.level1.contracts import validate_episode_record

from . import dynamic_bank as db
from .backplay import bind_restart_record
from .sampling import check_supported, rollout_slots

REFRESH_ROLLOUTS = (2, 4, 6, 8)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def settings():
    return dict(
        train_seeds=[int(s) for s in os.environ.get("PACMAN_TRAIN_SEEDS", "0,1,14,16").split(",")],
        teacher_bank=Path(os.environ.get("PACMAN_TEACHER_BANK", "/bank")),
        n_true=_env_int("PACMAN_TRUE_STARTS", 1),
        n_bank=_env_int("PACMAN_BANK_STARTS", 3),
        init_episodes_per_seed=_env_int("PACMAN_INIT_EPISODES_PER_SEED", 12),
    )


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.replace(temporary, path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Curriculum:
    def __init__(self, args, data_source):
        self.args = args
        self.cfg = settings()
        if self.cfg["n_true"] + self.cfg["n_bank"] != args.rollout_batch_size:
            raise ValueError("true + bank starts must equal --rollout-batch-size")
        self.root = Path(os.environ["PACMAN_RUN_DIR"]) / "dynamic-bank"
        self.data_source = data_source
        self._template = None

    # ----- records -------------------------------------------------------------------------
    def template(self):
        if self._template is None:
            group = self.data_source.get_samples(1)[0]
            first = group[0][0] if isinstance(group[0], list) else group[0]
            self._template = deepcopy(first.metadata["episode_record"])
        return self._template

    def true_start_record(self, seed):
        bank = load_restart_bank(self.cfg["teacher_bank"])
        ids = [e["restart_state_id"] for e in bank["restart_states"]
               if e["seed"] == seed and e["is_true_initial_state"]]
        if len(ids) != 1:
            raise ValueError(f"teacher bank needs exactly one true start for seed {seed}")
        return bind_restart_record(self.template(), self.cfg["teacher_bank"], ids[0])

    def bank_record(self, entry):
        path = (self.root / "states" / entry["file"]).resolve()
        if _sha(path) != entry["file_sha256"]:
            raise ValueError("bank state file checksum changed")
        record = deepcopy(self.template())
        record["id"] = entry["state_id"]
        record["environment"]["seed"] = entry["seed"]
        record["restart"] = dict(path=str(path), sha256=entry["file_sha256"], id=entry["state_id"])
        validate_episode_record(record)
        return record

    def start_record(self, manifest, start):
        if start["kind"] == "true_start":
            return self.true_start_record(start["seed"])
        return self.bank_record(manifest["states"][start["state_id"]])

    # ----- episodes outside training (initialization and probes) ---------------------------------
    async def run_episodes(self, records, capture_dir=None):
        from slime.rollout.sglang_rollout import GenerateState, get_model_url

        from .config import load_config
        from .rollout import (_collect_episode, _episode_pool, collect_episode_in_worker, current_sources,
                              episode_workers)

        endpoint, sources = get_model_url(self.args, "policy"), current_sources()
        config_path = os.environ["PACMAN_SLIME_CONFIG"]

        def capture_for(index, record):
            if capture_dir is None:
                return None
            name = f"init-{record['id'][:24]}-{index:03d}"
            return dict(path=str(Path(capture_dir) / f"{name}.json"),
                        source=dict(artifact_id=name, start_id=record["id"], rollout_id="init"))

        workers = episode_workers()
        if workers:
            loop = asyncio.get_running_loop()
            futures = [loop.run_in_executor(_episode_pool(workers), collect_episode_in_worker, record, endpoint,
                                            self.args.hf_checkpoint, config_path, sources, "",
                                            capture_for(i, record), True)
                       for i, record in enumerate(records)]
        else:
            state = GenerateState(self.args)
            futures = [_collect_episode(record, endpoint=endpoint, tokenizer=state.tokenizer,
                                        processor=state.processor, config=load_config(config_path),
                                        expected_sources=sources, version="", capture=capture_for(i, record))
                       for i, record in enumerate(records)]
        return await asyncio.gather(*futures, return_exceptions=True)

    async def probe(self, record, count):
        results = await self.run_episodes([record] * count)
        errors = [repr(r) for r in results if isinstance(r, BaseException)]
        if errors:
            return None, errors
        return sum(int(r.reward) for r in results), [r.weight_version for r in results[:1]]

    # ----- bank refresh --------------------------------------------------------------------------
    def trajectories(self, directories):
        found = []
        for directory in directories:
            for path in sorted(Path(directory).glob("*.json")):
                payload = json.loads(path.read_text())
                if payload.get("schema") != db.CANDIDATE_SCHEMA:
                    continue
                found.append(dict(episode_id=payload["source"]["artifact_id"], seed=payload["seed"],
                                  death_position=payload["last_choice_position"], path=str(path)))
        return found

    async def refresh(self, manifest, update, directories, count, budget, label):
        rng = random.Random(f"pacman-bank-{self.args.seed}-{label}")
        pool = [t for t in self.trajectories(directories) if t["seed"] in self.cfg["train_seeds"]]
        chosen = db.choose_trajectories(pool, manifest["seed_processed_counts"], count, rng)
        log = dict(label=label, update=update, available_trajectories=len(pool), budget=budget,
                   chosen=[t["episode_id"] for t in chosen], probes=[])
        for trajectory in chosen:
            key = str(trajectory["seed"])
            manifest["seed_processed_counts"][key] = manifest["seed_processed_counts"].get(key, 0) + 1
            payload = json.loads(Path(trajectory["path"]).read_text())
            for candidate in sorted(payload["candidates"], key=lambda c: c["distance"]):
                entry_log = dict(episode_id=trajectory["episode_id"], distance=candidate["distance"])
                log["probes"].append(entry_log)
                if budget < db.FIRST_PROBE:
                    entry_log["result"] = "budget_exhausted"
                    break
                boundary = candidate["boundary"]
                state_id = db.state_identity(boundary)
                state_path = self.root / "states" / f"{state_id}.json"
                if not state_path.exists():
                    _write_json(state_path, db.state_file_payload(boundary))
                entry = dict(state_id=state_id, file=state_path.name, file_sha256=_sha(state_path),
                             seed=payload["seed"], bucket=db.bucket_key(boundary["features"]),
                             features=boundary["features"], distance=candidate["distance"],
                             sources=[dict(payload["source"], distance=candidate["distance"])])
                record = self.bank_record(entry)
                first, info = await self.probe(record, db.FIRST_PROBE)
                budget -= db.FIRST_PROBE
                if first is None:
                    entry_log.update(result="probe_error", errors=info[:3])
                    continue
                entry_log["first_wins"] = first
                decision = db.probe_decision(first)
                if decision == "extend":
                    if budget < db.SECOND_PROBE:
                        entry_log["result"] = "incomplete_budget"
                        break
                    second, info = await self.probe(record, db.SECOND_PROBE)
                    budget -= db.SECOND_PROBE
                    if second is None:
                        entry_log.update(result="probe_error", errors=info[:3])
                        continue
                    entry_log["second_wins"] = second
                    decision = db.probe_decision(first, second)
                entry_log["result"] = decision
                if decision == "accept":
                    entry["probe"] = dict(wins=first + entry_log["second_wins"], episodes=24, update=update,
                                          weight_version=info[0] if info else None)
                    entry_log["events"] = db.add_state(manifest, entry, update)
                    break
            if budget < db.FIRST_PROBE:
                break
        log["budget_left"] = budget
        log["bank_size"] = len(manifest["states"])
        _write_json(self.root / "refresh" / f"{label}.json", log)
        return log

    async def initialize(self, manifest):
        capture_dir = self.root / "candidates" / "init"
        records = [self.true_start_record(seed) for seed in self.cfg["train_seeds"]
                   for _ in range(self.cfg["init_episodes_per_seed"])]
        results = await self.run_episodes(records, capture_dir=capture_dir)
        errors = [repr(r) for r in results if isinstance(r, BaseException)]
        if errors:
            raise RuntimeError(f"initialization episodes failed: {errors[:3]}")
        outcome = dict(episodes=len(results), wins=sum(int(r.reward) for r in results),
                       terminals={reason: sum(r.terminal_reason == reason for r in results)
                                  for reason in sorted({r.terminal_reason for r in results})})
        log = await self.refresh(manifest, 0, [capture_dir], db.INIT_TRAJECTORIES, db.INIT_BUDGET, "init")
        log["initial_rollout"] = outcome
        _write_json(self.root / "refresh" / "init.json", log)

    # ----- per-rollout entry ----------------------------------------------------------------------
    def state_path(self, rollout_id):
        return self.root / "state" / f"rollout-{rollout_id:04d}.json"

    def latest_manifest(self, rollout_id):
        for previous in range(rollout_id - 1, -1, -1):
            if self.state_path(previous).exists():
                return json.loads(self.state_path(previous).read_text())["manifest"]
        return None

    async def plan(self, rollout_id):
        if self.state_path(rollout_id).exists():  # resumed inside this rollout: reuse its frozen plan
            return json.loads(self.state_path(rollout_id).read_text())
        manifest = self.latest_manifest(rollout_id)
        if manifest is None:
            if rollout_id != 0:
                raise RuntimeError("dynamic bank state missing for a resumed run")
            manifest = db.new_manifest()
            await self.initialize(manifest)
        elif rollout_id in REFRESH_ROLLOUTS:
            directories = [self.root / "candidates" / f"rollout-{r:04d}" for r in (rollout_id - 2, rollout_id - 1)]
            await self.refresh(manifest, rollout_id, directories, db.REFRESH_TRAJECTORIES, db.REFRESH_BUDGET,
                               f"rollout-{rollout_id:04d}")
        starts = db.select_starts(manifest, rollout_id, self.cfg["train_seeds"], self.cfg["n_true"],
                                  self.cfg["n_bank"])
        db.record_selection(manifest, starts)
        plan = dict(schema=db.STATE_SCHEMA, rollout_id=rollout_id, manifest=manifest, starts=starts)
        _write_json(self.state_path(rollout_id), plan)
        return plan

    async def rollout(self, rollout_id):
        plan = await self.plan(rollout_id)
        records = [self.start_record(plan["manifest"], start) for start in plan["starts"]]
        self.args._pacman_capture_dir = str(self.root / "candidates" / f"rollout-{rollout_id:04d}")
        self.args._pacman_rollout_id = rollout_id
        try:
            return await rollout_slots(self.args, rollout_id, self.data_source.get_samples, records,
                                       self.root / "groups" / f"rollout-{rollout_id:04d}.json")
        finally:
            self.args._pacman_capture_dir = None


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout import sglang_rollout
    from slime.utils.async_utils import run

    if evaluation:
        return sglang_rollout.generate_rollout(args, rollout_id, data_source, evaluation=True)
    check_supported(args)
    return run(Curriculum(args, data_source).rollout(rollout_id))
