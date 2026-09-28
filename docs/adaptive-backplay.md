# Primitive-action Adaptive Backplay experiment

This is a bounded experiment, not evidence that cold start is solved. The maze
stays fixed; the distribution of restart states changes. Failure-only rollouts
can still have shaped-reward dispersion, so both the current true-start policy
and restart candidates are measured before training.

## Contract

- The Edward advertised-options teacher supplies successful trajectories only.
- The learner selects primitive `U/D/L/R` actions and uses complete episode
  return groups of 12, with 4 groups per optimizer update.
- Restart files retain original simulator time, RNG, lives and remaining horizon.
  The workflow starts fresh policy context and reward tracking at that boundary;
  rewards and progress credited to the learner cover its suffix only.
- Every probe uses the current synchronized policy: at least 24 sampled
  episodes per state (two groups of 12). Wilson95 intervals accompany estimates.
- Selection uses only success in `[0.3,0.7]`, then earliest teacher timestep.
  No reward-weighted sampler, PLR, extra baseline training or options for learner.
- Probe normalized-return variance is a labeled proxy. Training logs separately
  record the actual actor advantage tensor's variance over trainable tokens.

## Execution

Build a bank with `scripts/level1/evaluate/build_restart_bank.py --help`. Then,
from the recipe root on a prepared AReaL GPU runtime:

```bash
python scripts/level1/train/train_backplay.py \
  --bank /absolute/restart-bank --output /absolute/new-experiment \
  --config configs/level1/experiments/backplay.yaml \
  actor.path=/absolute/Qwen3.5-9B \
  --pilot-updates 10 --adaptive-updates 40 --probe-samples 24 --probe-every 5
```

Use a new output directory. This runner requires single-controller scheduling,
`evaluator.eval_before_train=false`, a fresh experiment (`recover.mode=disabled`),
and fixed batch sizes. It creates its own row templates from source provenance;
the YAML's dataset paths are placeholders. Existing shared-node safety and
vLLM runtime validation still apply before launching.

`--candidate-id ID` can be repeated to declare a smaller candidate subset;
true-start entries are always included. The exact subset is recorded. Selection
is earliest **among this subset**, not proof of a globally earliest frontier.
Default probing covers every candidate, which can be expensive for dense banks.

`--phase pilot` stops after the fixed-state experiment. Default `--phase staged`
runs the two experiments consecutively in one PPOTrainer lifecycle, preserving
weights and optimizer state. This is not an automatic resume interface.

## Pilot gate and stopping

The selected restart state remains fixed for the pilot. Its baseline is a fresh
sample independent of the candidate-selection sample. After the bounded pilot,
adaptive training proceeds only when:

1. Final success Wilson95 lower bound exceeds the baseline upper bound.
2. Mean suffix reward or pellets eaten increased.
3. Actual training advantage variance was nonzero, and baseline reward groups
   showed dispersion.

This conservative gate can be inconclusive with 24 samples; an inconclusive
pilot stops the curriculum for diagnosis. Nonzero variance alone never opens
the gate. A positive result supports a shorter-horizon learning benefit but does
not establish cold start as the sole or primary bottleneck.

The pilot never selects a true initial state. If its initial probe already shows
at least 30% true-start success, the runner records
`initial_cold_start_assumption_not_supported` and stops before any training.

During adaptive training, fresh probes choose the frontier every configured
number of updates. The cadence triggers measurement, not mechanical progression.
If no measured state is in range, the runner evaluates true start and stops with
`no_current_learnable_frontier`; it does not silently select an impossible state.
If all measured true-start states instead exceed 70%, it records
`true_start_above_frontier`, with the baseline/final comparison. This label alone
does not establish statistically supported improvement.

The report is `experiment.json`, with per-probe complete trajectories, records,
summaries, actual advantage diagnostics, and training trajectories. True-start
results are measured before training, after pilot, and at adaptive completion.
`bounded_adaptive_run_completed` means the budget finished; it is **not** a
claim of improved true-start performance or task completion. Assess those final
measurements separately.

## Why a synchronous batch adapter

AReaL's normal producer prefetches future batches and retains an iterator.
Changing a dataset alone could train on an old frontier after a switch. This
recipe instead submits exactly one complete optimizer batch through
`rollout_batch`, waits for every group, and changes the state only after the
existing trainer synchronizes updated weights. It leaves PPO, optimizer,
checkpointing and weight updates in the upstream trainer.

## Frozen checkpoint diagnostics

To test additional bank states after a run ends, use
`scripts/level1/evaluate/probe_backplay_checkpoint.py --config configs/level1/experiments/backplay.yaml`
with `--bank BANK --output NEW_OUTPUT --actor-checkpoint HF_CHECKPOINT`
and `--source-policy-version 15 --candidate-id RESTART_ID` (repeat candidate IDs).
The default is 24 episodes per requested state, in two groups of 12, using the
same evaluation workflow, action masks, rewards and remaining horizon as training.
Only explicitly requested states are probed. This is a diagnostic evaluation,
not an additional training experiment or an automatic change to the curriculum.

The entrypoint uses PPOTrainer initialization and therefore needs the configured
actor/reference/rollout allocation even though it never calls training or updates
the optimizer. Use a complete full/merged HF checkpoint, a fresh output directory,
and unique trial/artifact directories. Checkpoint, tokenizer and vLLM model paths
are bound together; recovery is disabled. The fresh engine uses version 0, while
`source_policy_version` records which completed training policy supplied its weights.
The saved `experiment.json` records `optimizer_updates: 0` and a terminal evaluation
status. A completed report establishes evaluation completion, not model improvement.

## Explicit adaptive continuation from HF weights

`train_backplay.py` can prepare a new adaptive segment using
`--continue-from-report PARENT/experiment.json --actor-checkpoint HF_CHECKPOINT`
`--source-policy-version 15 --allow-optimizer-reset` and a fresh `--output`,
independent `trial_name` and artifact directory. Omitting the explicit optimizer
reset flag rejects continuation. This loads model weights with a new optimizer;
it is **not exact resume** and never reruns the pilot.

The parent must be a staged experiment stopped at `no_current_learnable_frontier`,
with a passed pilot gate, contiguous training-batch lineage and a final evaluation
at the declared source version. The original adaptive budget is retained: a parent
with 10 pilot updates, 40 allowed adaptive updates and final policy 15 leaves
exactly 35 updates. The new report stores the parent path and SHA256, a parent report
copy, inherited gate, source checkpoint, and both engine-local and cumulative
policy versions. Engine version 0 initially represents source policy 15.

`--candidate-id` may restrict the new bank scan. Before the first update the segment
probes candidates with its loaded policy, selects using the existing 30–70% rule,
and performs a separate independent baseline on the selected state. No eligible
state means a final true-start evaluation followed by a stop without training.
Subsequent scans and final true-start evaluation follow the original interval and
remaining budget. The KL reference path comes from the parent's resolved config.

The segment enables `recover.mode=auto`, `recover.no_save_optim=false` and
`recover.freq_steps=1` to save DCP model/optimizer material after every update.
Existing checkpoint/recovery roots are rejected before trainer initialization;
any actual auto-recovery is also rejected by the normal adapter guard. The pinned
trainer saves recovery material before the evaluation hook can stop the segment.
Scheduler/RNG recovery and bitwise resume remain unverified. This entrypoint does
not implement resumption of a previously started continuation segment.

Preflight binds the checkpoint to the unique parent saver-directory checkpoint
for `source_policy_version - 1`, checks every indexed shard and safetensors byte
extent, and records SHA256 hashes for all checkpoint files. A foreign trial,
wrong update, missing shard or truncated weight file is rejected before GPU
initialization. It compares the complete resolved config with the parent using
an explicit allowlist for checkpoint/tokenizer paths, independent output/trial
paths, recovery settings and the remaining step budget. Any other difference,
including optimizer, KL, sampling, environment, reward or workflow settings,
is rejected rather than silently creating a different experiment.

Continuation probes schedule at most two states together, with at most four
12-episode groups outstanding. They submit the entire bounded batch before waiting,
clear each returned result once, and attribute episodes by immutable group paths.
The original Backplay experiment retains one-state scheduling. This can fill four
rollout workers for the usual 24-sample probes; measured speedup is not established.
