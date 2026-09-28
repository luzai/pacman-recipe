"""Bounded, synchronous Backplay experiments around the existing PPOTrainer.

No framework training loop is copied: the adapter replaces its batch producer
and evaluation hook. Each batch is fully consumed before changing restart state.
GPU imports are deferred so the experiment rules can be tested on a CPU host.
"""
from __future__ import annotations

import copy
import hashlib
import getpass
import json
import math
from pathlib import Path
from typing import Any

from .backplay import (
    NoLearnableRestartState, aggregate_probe_results, load_restart_bank, load_restart_state,
    select_restart_state,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def redact_config(value: Any) -> Any:
    """Keep experiment settings while excluding resolved authentication values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            sensitive = (any(word in name for word in ("api_key", "password", "secret", "private_key"))
                         or name in {"token", "access_token", "auth_token", "admin_token", "api_token", "authorization", "credentials"})
            result[key] = "[REDACTED]" if sensitive else redact_config(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value


def bind_restart_row(template: dict, entry: dict, bank_dir: Path, group_id: str) -> dict:
    """Materialize one immutable group input, never change an in-flight row."""
    saved = load_restart_state(bank_dir, entry)
    row = copy.deepcopy(template)
    if row.get("decision_steps") == 1 or row.get("state_prefix_actions"):
        raise ValueError("Backplay requires complete primitive episodes without prefixes")
    row["id"] = group_id
    row["env"]["seed"] = int(entry["seed"])
    if int(row["env"]["max_steps"]) != int(saved["payload"]["identity"]["max_steps"]):
        raise ValueError("restart and learner must preserve the same original horizon")
    row.update(restart_state_path=str((bank_dir / entry["state_path"]).resolve()),
               restart_state_sha256=entry["state_file_sha256"],
               restart_state_id=entry["restart_state_id"])
    return row


def pilot_gate(before: dict, after: dict, actual_advantages: list[dict]) -> dict:
    """A conservative predeclared gate; nonzero variance alone is not learning."""
    enough = before["samples"] >= 24 and after["samples"] >= 24
    success_improved = after["success_rate_wilson95"][0] > before["success_rate_wilson95"][1]
    progress_improved = (after["mean_reward"] > before["mean_reward"] or
                         after["mean_pellets_eaten"] > before["mean_pellets_eaten"])
    actual_signal = any(item["count"] > 1 and item["variance"] > 1e-12
                        for item in actual_advantages)
    dispersion = before["group_reward_variance"] is not None and before["group_reward_variance"] > 0
    return {"passed": bool(enough and success_improved and progress_improved and actual_signal and dispersion),
            "adequate_samples": enough, "nonoverlapping_success_wilson95": success_improved,
            "reward_or_progress_improved": progress_improved,
            "actual_training_advantage_variance_nonzero": actual_signal,
            "baseline_reward_dispersion_nonzero": dispersion,
            "interpretation": "A pass supports a shorter-horizon learning benefit; it does not prove the primary cause of cold start."}


def tensor_advantage_moments(batch: list[dict]) -> dict:
    """Measure the actual actor advantage tensor only on trainable tokens."""
    from areal.infra.rpc.rtensor import RTensor, clear_fetch_buffer, flatten_shard_ids
    count, total, squares = 0, 0.0, 0.0
    minimum, maximum = math.inf, -math.inf
    for shard in batch:
        # localize mutates RTensor.data. Keep PPO's original wrappers as meta
        # references so subsequent RPC dispatch does not inline fetched tensors.
        fields = {key: copy.copy(shard[key]) if isinstance(shard[key], RTensor) else shard[key]
                  for key in ("advantages", "loss_mask")}
        try:
            local = RTensor.localize(fields)
            values = local["advantages"][local["loss_mask"].bool()].detach().double().cpu()
            if not values.isfinite().all().item():
                raise ValueError("non-finite actual training advantages")
            n = values.numel()
            if n:
                count += n
                total += values.sum().item()
                squares += values.square().sum().item()
                minimum = min(minimum, values.min().item())
                maximum = max(maximum, values.max().item())
        finally:
            # Do not delete remote shards: PPO still needs them.
            clear_fetch_buffer(flatten_shard_ids(fields))
    if not count:
        raise ValueError("no trainable tokens in actor advantage batch")
    return {"source": "actor.compute_advantages/loss_mask", "count": count,
            "mean": total / count, "variance": max(0.0, squares / count - (total / count) ** 2),
            "minimum": minimum, "maximum": maximum, "weighting": "trainable_token_population"}


def continuation_lineage(parent_path: Path, *, source_policy_version: int,
                         allow_optimizer_reset: bool) -> dict:
    """Validate a completed staged parent; an HF load does not recover optimizer state."""
    if not allow_optimizer_reset:
        raise ValueError("continuation requires explicit --allow-optimizer-reset")
    raw = parent_path.read_bytes()
    parent = json.loads(raw)
    if parent.get("phase") != "staged" or parent.get("status") != "no_current_learnable_frontier":
        raise ValueError("continuation requires a staged parent stopped with no_current_learnable_frontier")
    if parent.get("pilot_gate", {}).get("passed") is not True:
        raise ValueError("parent pilot gate did not pass")
    pilot = int(parent["pilot_updates"])
    budget = int(parent["adaptive_updates"])
    batches = [int(e["policy_version"]) for e in parent.get("events", []) if e.get("event") == "train_batch"]
    if sorted(batches) != list(range(source_policy_version)):
        raise ValueError("parent training batches do not prove the requested source policy lineage")
    finals = [e for e in parent.get("events", []) if e.get("event") == "probe"
              and e.get("label") == "true-start-no-frontier"
              and str(e.get("policy_version")) == str(source_policy_version)]
    if not finals or not parent.get("true_start_final"):
        raise ValueError("parent final current-policy evaluation is missing")
    completed = source_policy_version - pilot
    if completed < 0 or completed >= budget:
        raise ValueError("parent has no remaining adaptive update budget")
    return {"parent_report_path": str(parent_path.resolve()),
            "parent_report_sha256": hashlib.sha256(raw).hexdigest(),
            "source_policy_version": source_policy_version,
            "original_pilot_updates": pilot, "original_adaptive_update_budget": budget,
            "parent_completed_adaptive_updates": completed,
            "remaining_adaptive_updates": budget - completed,
            "optimizer_reset": True, "exact_resume": False,
            "parent_pilot_gate": copy.deepcopy(parent["pilot_gate"]),
            "parent_report": parent}


def require_fresh_continuation_storage(config, parent_config: dict) -> None:
    """Reject prior recovery material before PPOTrainer can automatically load it."""
    if config.trial_name == parent_config.get("trial_name"):
        raise ValueError("continuation requires an independent trial_name")
    for section in (config.recover, config.saver):
        root = (Path(section.fileroot) / "checkpoints" / getpass.getuser()
                / section.experiment_name / section.trial_name)
        if root.exists():
            raise FileExistsError(f"continuation requires a fresh checkpoint/recovery root: {root}")
    if config.recover.mode != "auto" or config.recover.no_save_optim or config.recover.freq_steps != 1:
        raise ValueError("continuation must save model and optimizer recovery material every update")


def validate_continuation_checkpoint(checkpoint: Path, parent_config: dict, source_version: int) -> dict:
    """Bind HF weights to the parent's exact saved update and hash every artifact."""
    saver = parent_config["saver"]
    root = (Path(saver["fileroot"]) / "checkpoints" / getpass.getuser()
            / saver["experiment_name"] / saver["trial_name"] / "default")
    matches = [p.resolve() for p in root.glob(f"epoch*epochstep*globalstep{source_version - 1}") if p.is_dir()]
    checkpoint = checkpoint.resolve()
    if len(matches) != 1 or checkpoint != matches[0]:
        raise ValueError("checkpoint is not the unique parent checkpoint for the declared source policy")
    if not (checkpoint / "config.json").is_file():
        raise ValueError("checkpoint config.json missing")
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        if not mapping:
            raise ValueError("empty checkpoint weight map")
        shards = set(mapping.values())
        for shard in shards:
            path = checkpoint / shard
            if path.resolve().parent != checkpoint or not path.is_file() or path.stat().st_size == 0:
                raise ValueError("checkpoint weight shard missing or outside checkpoint")
    elif not (checkpoint / "model.safetensors").is_file():
        raise ValueError("complete single-file model.safetensors or indexed shards required")
    else:
        shards = {"model.safetensors"}
    for shard in shards:
        path = checkpoint / shard
        with path.open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            if not 0 < header_length <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError("truncated safetensors header")
            header = json.loads(stream.read(header_length))
        offsets = sorted(item["data_offsets"] for name, item in header.items() if name != "__metadata__")
        end = 0
        for start, stop in offsets:
            if start != end or stop < start:
                raise ValueError("invalid safetensors data offsets")
            end = stop
        if not offsets or end != path.stat().st_size - 8 - header_length:
            raise ValueError("truncated safetensors data")
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files[str(path.relative_to(checkpoint))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return {"path": str(checkpoint), "source_policy_version": source_version, "files": files}


def validate_continuation_config(parent: dict, current: dict) -> None:
    """Reject every config difference except declared segment storage and recovery changes."""
    allowed = {"actor.path", "tokenizer_path", "rollout.tokenizer_path", "vllm.model",
        "artifact_root", "trajectory_dir", "trial_name", "cluster.fileroot",
        "actor.trial_name", "ref.trial_name", "rollout.trial_name", "rollout.fileroot",
        "cluster.name_resolve.nfs_record_root", "stats_logger.swanlab.name",
        "total_train_steps", "total_train_epochs", "recover.mode", "recover.no_save_optim",
        "recover.freq_steps"}
    for section in ("saver", "recover", "evaluator", "stats_logger"):
        allowed.update({section + ".trial_name", section + ".fileroot"})
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
            return result
        return {prefix: value}
    before, after = flatten(parent), flatten(current)
    changed = [key for key in before.keys() | after.keys()
               if key not in allowed and (key not in before or key not in after or before[key] != after[key])]
    if changed:
        raise ValueError("undeclared continuation config changes: " + ", ".join(sorted(changed)))


class BackplayStopped(RuntimeError):
    """Intentional bounded experiment stop, with its outcome already persisted."""


class BackplayExperiment:
    def __init__(self, *, trainer, template: dict, bank_dir: Path, output_dir: Path,
                 workflow: str, probe_kwargs: dict, pilot_updates: int = 10,
                 adaptive_updates: int = 40, probe_samples: int = 24,
                 probe_every: int = 5, phase: str = "staged", candidate_ids: list[str] | None = None,
                 probe_state_batch_size: int = 1):
        if phase not in {"pilot", "staged"}:
            raise ValueError("phase must be pilot or staged")
        if min(pilot_updates, adaptive_updates, probe_every) < 1 or probe_samples < 24 or probe_samples % 12:
            raise ValueError("positive budgets and probe_samples >=24 divisible by12 required")
        if probe_state_batch_size not in (1, 2):
            raise ValueError("probe_state_batch_size must be 1 or 2")
        self.probe_state_batch_size = probe_state_batch_size
        self.trainer, self.template = trainer, template
        self.bank_dir, self.output_dir = bank_dir.resolve(), output_dir.resolve()
        self.workflow, self.probe_kwargs = workflow, copy.deepcopy(probe_kwargs)
        self.pilot_updates, self.adaptive_updates = pilot_updates, adaptive_updates
        self.probe_samples, self.probe_every, self.phase = probe_samples, probe_every, phase
        manifest = load_restart_bank(self.bank_dir)
        all_entries = manifest["restart_states"]
        if len({entry["restart_state_id"] for entry in all_entries}) != len(all_entries):
            raise ValueError("duplicate restart IDs")
        selected_ids = set(candidate_ids or [entry["restart_state_id"] for entry in all_entries])
        if selected_ids - {entry["restart_state_id"] for entry in all_entries}:
            raise ValueError("unknown candidate ID")
        self.candidates = [entry for entry in all_entries if entry["restart_state_id"] in selected_ids]
        self.true_starts = [entry for entry in all_entries if entry["env_step"] == 0]
        self.candidates += [entry for entry in self.true_starts if entry["restart_state_id"] not in selected_ids]
        if not self.true_starts or not self.candidates:
            raise ValueError("bank requires candidates and a true initial state")
        for entry in self.candidates + self.true_starts:
            bind_restart_row(template, entry, self.bank_dir, "preflight")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.version, self.current, self.pilot_state = 0, None, None
        self.before, self.gate = None, None
        self.advantage_history: list[dict] = []
        self.probe_counter = 0
        self.report = {"phase": phase, "status": "initialized", "bank_id": manifest["bank_id"],
                       "actor_path": str(trainer.config.actor.path), "pilot_updates": pilot_updates,
                       "adaptive_updates": adaptive_updates, "probe_samples": probe_samples,
                       "probe_state_batch_size": probe_state_batch_size,
                       "selection_interval": [0.3, 0.7], "events": [],
                       "candidate_ids": [entry["restart_state_id"] for entry in self.candidates],
                       "scope": "fixed maze; changed restart distribution; primitive actions; original remaining horizon"}
        self.persist()

    def persist(self):
        write_json(self.output_dir / "experiment.json", self.report)

    def event(self, kind: str, **fields):
        self.report["events"].append({"event": kind, "policy_version": str(self.version), **fields})
        self.persist()

    def stop(self, status: str):
        self.report["status"] = status
        self.persist()
        raise BackplayStopped(status)

    def probe(self, entries: list[dict], label: str) -> list[dict]:
        """Submit complete groups on the synchronized current inference weights."""
        from areal.infra.utils.concurrent import run_async_task
        from areal.trainer.rl_trainer import _clear_eval_result
        policy_version = self.version
        if self.trainer.eval_rollout.get_version() != policy_version:
            raise RuntimeError("probe engine is not on the current experiment policy version")
        if len({entry['restart_state_id'] for entry in entries}) != len(entries):
            raise ValueError("duplicate probe candidate ID")
        self.probe_counter += 1
        directory = self.output_dir / "probes" / f"v{self.version}-{self.probe_counter}-{label}"
        records = []
        # Each group gets its own path; group attribution never depends on arrival order.
        for offset in range(0, len(entries), self.probe_state_batch_size):
            groups = []
            for entry in entries[offset:offset + self.probe_state_batch_size]:
                for index in range(self.probe_samples // 12):
                    group = f"probe-v{policy_version}-{self.probe_counter}-{entry['restart_state_id']}-{index}"
                    groups.append((entry, group, directory / group))
            # With two states, keep at most four groups outstanding, even when
            # a caller requests more than the normal 24 samples per state.
            limit = 4 if self.probe_state_batch_size == 2 else len(groups)
            for start in range(0, len(groups), limit):
                if self.version != policy_version or self.trainer.eval_rollout.get_version() != policy_version:
                    raise RuntimeError("probe policy version changed during scan")
                batch = groups[start:start + limit]
                for entry, group, group_dir in batch:
                    kwargs = copy.deepcopy(self.probe_kwargs)
                    kwargs["trajectory_dir"] = str(group_dir)
                    row = bind_restart_row(self.template, entry, self.bank_dir, group)
                    self.trainer.eval_rollout.submit(row, self.workflow, kwargs, group_size=12, is_eval=True)
                for _ in batch:
                    result = self.trainer.eval_rollout.wait(1, timeout=None)
                    run_async_task(_clear_eval_result, result)
                if self.version != policy_version or self.trainer.eval_rollout.get_version() != policy_version:
                    raise RuntimeError("probe policy version changed during scan")
            for entry, group, group_dir in groups:
                payloads = [json.loads(path.read_text(encoding="utf-8")) for path in group_dir.glob("*.json")]
                if len(payloads) != 12:
                    raise RuntimeError(f"probe requires12 complete episodes; {group} has{len(payloads)}")
                for payload in payloads:
                    if payload.get("restart_state", {}).get("id") != entry["restart_state_id"]:
                        raise RuntimeError("probe trajectory came from a different restart state")
                    records.append({"restart_state_id": entry["restart_state_id"],
                                    "policy_version": str(self.version), "group_id": group,
                                    "sample_id": str(payload["trajectory_sample_id"]),
                                    "success": bool(payload["won"]),
                                    "reward": float(payload["total_shaped_reward"]),
                                    "pellets_eaten": int(payload["suffix_normal_pellets_eaten"]),
                                    "steps": int(payload["steps"])})
        summary = aggregate_probe_results(records, policy_version=str(self.version), minimum_samples=self.probe_samples)
        write_json(directory / "records.json", records)
        write_json(directory / "summary.json", summary)
        self.event("probe", label=label, summary_path=str(directory / "summary.json"), summaries=summary)
        return summary

    def initial_probe(self):
        self.report["true_start_before"] = self.probe(self.true_starts, "true-start-baseline")
        if any(item["success_rate"] >= 0.3 for item in self.report["true_start_before"]):
            self.stop("initial_cold_start_assumption_not_supported")
        pilot_candidates = [entry for entry in self.candidates if entry["env_step"] > 0]
        summaries = self.probe(pilot_candidates, "frontier-selection") if pilot_candidates else []
        try:
            self.current = select_restart_state(pilot_candidates, summaries, policy_version="0", minimum_samples=self.probe_samples)
        except NoLearnableRestartState:
            self.stop("no_initial_learnable_restart_state")
        self.pilot_state = self.current
        # Independent pre-training sample avoids using a lucky selection sample as baseline.
        self.before = self.probe([self.current], "pilot-independent-baseline")[0]
        self.report["pilot_before"] = self.before
        self.event("pilot_selected", restart_state_id=self.current["restart_state_id"])

    def record_true_start(self, key: str, label: str):
        measured = self.probe(self.true_starts, label)
        self.report[key] = measured
        baseline = {item["restart_state_id"]: item for item in self.report.get("true_start_before", [])}
        comparisons = []
        for item in measured:
            before = baseline.get(item.get("restart_state_id"))
            if before is not None:
                comparisons.append({"restart_state_id": item["restart_state_id"],
                    "success_rate_before": before["success_rate"], "success_rate_after": item["success_rate"],
                    "success_rate_change": item["success_rate"] - before["success_rate"],
                    "success_wilson95_nonoverlap_improvement": item["success_rate_wilson95"][0] > before["success_rate_wilson95"][1],
                    "complete_task_successes": item["successes"], "samples": item["samples"]})
        self.report[key + "_comparison"] = comparisons
        self.persist()
        return measured

    def prepare_batch(self, dataloader, workflow, workflow_kwargs=None,
                      should_accept_fn=None, group_size=1, dynamic_bs=False):
        if group_size != 12 or dynamic_bs or should_accept_fn is not None:
            raise ValueError("Backplay uses fixed complete groups, no dynamic filtering or batching")
        if self.current is None:
            self.initial_probe()
        if self.trainer.rollout.get_version() != self.version:
            raise RuntimeError("training rollout engine has stale policy weights")
        rows = [bind_restart_row(self.template, self.current, self.bank_dir,
                                 f"train-v{self.version}-group{i}")
                for i in range(int(dataloader.batch_size))]
        kwargs = copy.deepcopy(workflow_kwargs or {})
        kwargs["trajectory_dir"] = str(self.output_dir / "train" / f"v{self.version}")
        result = self.trainer.rollout.rollout_batch(rows, workflow, kwargs, group_size=12)
        if len(result) != len(rows):
            raise RuntimeError("an incomplete/filtered Backplay group cannot become an optimizer batch")
        self.event("train_batch", restart_state_id=self.current["restart_state_id"], groups=len(rows))
        return result

    def after_update(self, *, global_step: int, **unused):
        self.version = global_step + 1
        if self.version <= self.pilot_updates:
            if self.version % self.probe_every == 0 or self.version == self.pilot_updates:
                after = self.probe([self.pilot_state], "pilot-learning-curve")[0]
            if self.version != self.pilot_updates:
                return
            self.gate = pilot_gate(self.before, after, self.advantage_history)
            self.report.update(pilot_gate=self.gate, pilot_after=after)
            self.record_true_start("true_start_after_pilot", "true-start-after-pilot")
            self.persist()
            if not self.gate["passed"]:
                self.stop("pilot_inconclusive_debug_before_curriculum")
            if self.phase == "pilot":
                self.report["status"] = "pilot_passed"
                self.persist()
                return
        if self.phase == "staged":
            total = self.pilot_updates + self.adaptive_updates
            if self.version >= total:
                self.record_true_start("true_start_final", "true-start-final")
                self.report["status"] = "bounded_adaptive_run_completed"
                self.persist()
                return
            if (self.version - self.pilot_updates) % self.probe_every == 0:
                summaries = self.probe(self.candidates, "adaptive-frontier")
                try:
                    chosen = select_restart_state(self.candidates, summaries,
                        policy_version=str(self.version), minimum_samples=self.probe_samples)
                except NoLearnableRestartState:
                    final = self.record_true_start("true_start_final", "true-start-no-frontier")
                    if final and all(item["success_rate"] > 0.7 for item in final):
                        self.stop("true_start_above_frontier")
                    self.stop("no_current_learnable_frontier")
                self.current = chosen
                self.event("adaptive_selected", restart_state_id=chosen["restart_state_id"], env_step=chosen["env_step"])

    def install(self):
        from areal.utils.environ import is_single_controller
        if not is_single_controller():
            raise ValueError("Backplay runner requires single-controller local scheduler")
        if self.trainer.eval_rollout is None or self.trainer.recover_info is not None:
            raise ValueError("Backplay requires an evaluation rollout and a fresh experiment")
        if hasattr(self.trainer.rollout, "data_generator"):
            raise ValueError("refuse to attach Backplay to an already-prefetching rollout")
        original_advantages = self.trainer.actor.compute_advantages

        def compute_advantages(*args, **kwargs):
            batch = original_advantages(*args, **kwargs)
            moments = tensor_advantage_moments(batch)
            moments["policy_version"] = str(self.version)
            self.advantage_history.append(moments)
            self.event("actual_training_advantages", **moments)
            return batch

        self.trainer.actor.prepare_batch = self.prepare_batch
        self.trainer.actor.compute_advantages = compute_advantages
        self.trainer._evaluate = self.after_update


class AdaptiveContinuationExperiment(BackplayExperiment):
    """A new optimizer segment with inherited evidence and a bounded remaining budget."""

    def __init__(self, *, lineage: dict, **kwargs):
        kwargs["probe_state_batch_size"] = 2
        super().__init__(**kwargs)
        self.lineage = copy.deepcopy(lineage)
        self.source_version = int(lineage["source_policy_version"])
        self.remaining_updates = int(lineage["remaining_adaptive_updates"])
        self.gate = copy.deepcopy(lineage["parent_pilot_gate"])
        parent = lineage["parent_report"]
        self.report.update(phase="adaptive-continuation", optimizer_reset=True, exact_resume=False,
            lineage={k: v for k, v in lineage.items() if k != "parent_report"},
            source_policy_version=str(self.source_version), engine_policy_version="0",
            cumulative_policy_version=str(self.source_version), segment_completed_updates=0,
            pilot_updates=0, adaptive_updates=self.remaining_updates,
            pilot_gate=self.gate, true_start_before=copy.deepcopy(parent.get("true_start_before", [])),
            parent_true_start_final=copy.deepcopy(parent["true_start_final"]))
        self.report["recovery_material"] = {"configured_format": "dcp", "save_model": True, "save_optimizer": True,
            "frequency_updates": 1, "scheduler_rng_recovery_verified": False, "bitwise_resume_verified": False}
        write_json(self.output_dir / "parent-experiment.json", parent)
        self.persist()

    def event(self, kind: str, **fields):
        fields.update(engine_policy_version=str(self.version),
                      cumulative_policy_version=str(self.source_version + self.version))
        super().event(kind, **fields)

    def choose_current_frontier(self):
        summaries = self.probe(self.candidates, "continuation-frontier")
        try:
            self.current = select_restart_state(self.candidates, summaries,
                policy_version=str(self.version), minimum_samples=self.probe_samples)
        except NoLearnableRestartState:
            self.record_true_start("true_start_final", "true-start-no-frontier")
            self.stop("no_current_learnable_frontier")
        self.event("adaptive_selected", restart_state_id=self.current["restart_state_id"],
                   env_step=self.current["env_step"])

    def initial_probe(self):
        self.choose_current_frontier()
        # Selection and baseline are separate complete samples, as in the pilot.
        measured = self.probe([self.current], "continuation-independent-baseline")
        self.report["continuation_before"] = measured[0]
        self.persist()

    def after_update(self, *, global_step: int, **unused):
        self.version = global_step + 1
        self.report.update(engine_policy_version=str(self.version),
            cumulative_policy_version=str(self.source_version + self.version),
            segment_completed_updates=self.version)
        self.event("continuation_update_completed", segment_update=self.version)
        if self.version >= self.remaining_updates:
            self.record_true_start("true_start_final", "true-start-final")
            self.report["status"] = "bounded_adaptive_continuation_completed"
            self.persist()
        elif self.version % self.probe_every == 0:
            self.choose_current_frontier()


def main(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "staged"], default="staged")
    parser.add_argument("--pilot-updates", type=int, default=10)
    parser.add_argument("--adaptive-updates", type=int, default=40)
    parser.add_argument("--probe-samples", type=int, default=24)
    parser.add_argument("--probe-every", type=int, default=5)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--continue-from-report", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--source-policy-version", type=int)
    parser.add_argument("--allow-optimizer-reset", action="store_true")
    args, config_args = parser.parse_known_args(argv)
    lineage = None
    if args.continue_from_report is not None:
        if args.actor_checkpoint is None or args.source_policy_version is None:
            parser.error("continuation requires --actor-checkpoint and --source-policy-version")
        if args.phase != "staged":
            parser.error("continuation cannot run a pilot")
        lineage = continuation_lineage(args.continue_from_report,
            source_policy_version=args.source_policy_version, allow_optimizer_reset=args.allow_optimizer_reset)
        checkpoint = args.actor_checkpoint.resolve()
        # Retain the original KL reference rather than rebasing it on the new actor.
        parent_config = json.loads((args.continue_from_report.parent / "resolved-config.json").read_text())
        lineage["checkpoint_manifest"] = validate_continuation_checkpoint(checkpoint, parent_config,
                                                                         args.source_policy_version)
        reference = parent_config.get("ref")
        config_args = [*config_args, f"actor.path={checkpoint}", f"tokenizer_path={checkpoint}",
                       f"vllm.model={checkpoint}", "recover.mode=auto", "++recover.no_save_optim=false",
                       "recover.freq_steps=1"]
        if reference is not None:
            config_args.append(f"ref.path={reference['path']}")
        lineage["source_actor_checkpoint"] = str(checkpoint)
        lineage["parent_config_sha256"] = hashlib.sha256(
            (args.continue_from_report.parent / "resolved-config.json").read_bytes()).hexdigest()
        lineage["reference_path"] = reference["path"] if reference else None
    elif args.actor_checkpoint is not None or args.source_policy_version is not None or args.allow_optimizer_reset:
        parser.error("continuation flags require --continue-from-report")
    # Preserve the normal entrypoint's runtime environment defaults before
    # importing the framework or Transformers.
    from train_areal import (
        _build_workflow_kwargs, _validate_reward_objective_contract,
        _validate_backplay_contract, _validate_actor_colocated_reference_offload,
    )
    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from datasets import Dataset
    from pacman_recipe.level1.level1_dataset import make_episode_row
    from pacman_recipe.synthetic.configs import PacmanAgentConfig
    config, _ = load_expr_config(config_args, PacmanAgentConfig)
    _validate_reward_objective_contract(config)
    _validate_backplay_contract(config)
    _validate_actor_colocated_reference_offload(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    if lineage:
        require_fresh_continuation_storage(config, parent_config)
        if config.actor.use_lora:
            raise ValueError("continuation requires full/merged HF weights")
    bank = load_restart_bank(args.bank)
    if config.evaluator.eval_before_train:
        raise ValueError("runner owns baseline evaluation; set evaluator.eval_before_train=false")
    if config.dynamic_bs:
        raise ValueError("Backplay requires dynamic_bs=false")
    total = (lineage["remaining_adaptive_updates"] if lineage else
             args.pilot_updates + (args.adaptive_updates if args.phase == "staged" else 0))
    config.total_train_steps = total
    seed = int(bank["restart_states"][0]["seed"])
    template = make_episode_row(1, split="train", seed=seed,
        max_steps=config.environment.max_steps, ghost_mode=config.environment.ghost_mode,
        action_protocol=config.action_protocol)
    rows = [dict(copy.deepcopy(template), id=f"backplay-placeholder-{index}")
            for index in range(config.train_dataset.batch_size)]
    train, valid = Dataset.from_list(rows), Dataset.from_list(rows)
    config.total_train_epochs = max(config.total_train_epochs, total)
    if lineage:
        from omegaconf import OmegaConf
        validate_continuation_config(parent_config, redact_config(
            OmegaConf.to_container(OmegaConf.structured(config), resolve=True)))
    train_kwargs = _build_workflow_kwargs(config, config.gconfig, training=True)
    probe_kwargs = _build_workflow_kwargs(config, config.eval_gconfig, training=False)
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        experiment_cls = AdaptiveContinuationExperiment if lineage else BackplayExperiment
        experiment = experiment_cls(**({"lineage": lineage} if lineage else {}),
            trainer=trainer, template=dict(train[0]),
            bank_dir=args.bank, output_dir=args.output, workflow=config.workflow,
            probe_kwargs=probe_kwargs, pilot_updates=args.pilot_updates,
            adaptive_updates=args.adaptive_updates, probe_samples=args.probe_samples,
            probe_every=args.probe_every, phase=args.phase, candidate_ids=args.candidate_id)
        experiment.install()
        write_json(experiment.output_dir / "input-template.json", template)
        from omegaconf import OmegaConf
        write_json(experiment.output_dir / "resolved-config.json", redact_config(
            OmegaConf.to_container(OmegaConf.structured(config), resolve=True)))
        if lineage:
            write_json(experiment.output_dir / "parent-resolved-config.json", redact_config(parent_config))
        try:
            trainer.train(workflow=config.workflow, workflow_kwargs=train_kwargs,
                          eval_workflow=config.workflow, eval_workflow_kwargs=probe_kwargs,
                          dynamic_filter_fn=None)
        except BackplayStopped as exc:
            print(f"backplay_stop={exc}")
        except Exception:
            experiment.report["status"] = "failed"
            experiment.persist()
            raise
    print(f"backplay_report={args.output.resolve() / 'experiment.json'}")
