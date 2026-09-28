# Scripts

## Production Level-1

- `level1/train/`: the shared two-stage launcher and the historical overfit
  pipeline.
- `level1/evaluate/`: checkpoint evaluation, prompt A/B, and comparison tools.
- `level1/dataset/`: deterministic dataset and manifest generation.
- `level1/report/`: trajectory auditing, checkpoint assembly, demo export, and
  report helpers.

Simulator save-state sanity check (CPU, no model service):

```bash
python scripts/level1/evaluate/check_simulator_saved_state.py \
  --output-dir /absolute/new/output-directory
```

Writes `state.json` and `summary.json`; closes the original worker, restores
from the JSON file, and compares RGB, full simulator/RNG state, rewards and
transition events after each recorded action. See
[save-state semantics](../docs/simulator-save-state.md).

`level1/report/export_level1_rollout_video.py` labels videos with the recorded
terminal reason, final score, and remaining pellets. Replay accepts a final
planner `safety_refusal` truncation only while the game environment remains
live; inconsistent refusal flags and other terminal-state mismatches fail
before video encoding.

Primary training entry point:

```bash
export OWNER_ROOT=/path/to/writable/owner-root
export ENV_ROOT="${CONDA_PREFIX:?activate maapacman-rl first}"
export PYTHON="${ENV_ROOT}/bin/python"
export AREAL_ROOT=/path/to/AReaL
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
export MODEL_PATH=/path/to/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/curriculum1.yaml"
unset TRAIN_EPISODES VALIDATION_EPISODES DATASET_MAX_STEPS
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID="curriculum1-${RUN_TS}"
export DATASET_OUTPUT_ROOT="$PWD/artifacts/datasets/${RUN_ID}"
export ARTIFACT_ROOT="$PWD/run_artifacts/${RUN_ID}"

bash scripts/level1/train/run_level1_training.sh
```

Run a two-update smoke test without a separate config (use fresh output paths):

```bash
unset RUN_ID ARTIFACT_ROOT DATASET_OUTPUT_ROOT
bash scripts/level1/train/run_level1_training.sh --smoke-updates 2
```

vLLM uses Unix-domain sockets whose full path is limited to 107 bytes. Its
directory plus `/` and a 36-character UUID must fit. A long `TMPDIR` under a run
artifact directory can exceed this limit even when the directory is writable.
Before launching, explicitly create a short directory within your own task area,
then set `VLLM_RPC_BASE_PATH` (supported by the pinned vLLM 0.22.1), for example:

```bash
mkdir -p /path/to/own/short-ipc
chmod 700 /path/to/own/short-ipc
export VLLM_RPC_BASE_PATH=/path/to/own/short-ipc
```

Replace that example with your own task-owned path; do not use another person's
directory. The launcher checks the byte limit before any model/GPU checks and
briefly binds a test socket only in an explicitly selected directory. It never
creates or rewrites a temporary-directory setting. A valid short override lets
you retain a long `TMPDIR` and long artifact paths. Without an explicit temporary
path, the existing library default is preserved and only its length is checked.

For Curriculum 2, select `configs/level1/train/curriculum2.yaml` and point
`MODEL_PATH` at the same complete local Qwen3.5-9B base checkpoint used for the
baseline. Curriculum 2 does not read a Curriculum 1 checkpoint. The launcher
loads config, tokenizer, and processor offline and verifies every indexed
weight shard before creating run output.

The launcher reads seed counts and horizon from YAML. Both stages use 512
underlying environment steps. C1 uses 80/4 train/validation rows (seeds 28–107/108–111),
5 epochs and 100 updates; C2 uses 40/4 rows (seeds 28–67/68–71), 5 epochs and 50 updates. C1 has no ghosts or Edward: one legal U/D/L/R per
model decision, raw step-local reward and no reward normalization/clipping of
finite rewards (`reward_clip: .inf`). C2 uses normal ghosts, one advertised
Edward option code, full-episode group-12 normalization then clipping to ±20.
Both use batch 4, 12 samples, learning rate 5e-7, shaping alpha 0.1 and KL 0.01
with a reference model. C2 validates once before training and after every epoch;
its real GPU outputs remain to be checked.

C2 smoke reuses its YAML with `--smoke-updates 2` and a real complete C1 smoke
checkpoint. The structural checkpoint validator is not a full model load or
image-inference test. Do not label base placeholders or smoke weights as the
final trained agent. `recover.mode=disabled` also disables recovery-state saving
in the pinned AReaL; ordinary model weights do not restore optimizer state.

### Temporary C2 test from legacy Iter25 gs24

This explicit compatibility-test entry point does not change the formal C2
default or replace the new C1 -> C2 acceptance gate. Confirm that the supplied
complete checkpoint is actually Iter25 **globalstep24**; a directory name alone
does not prove its identity. No server-specific checkpoint path is assumed.
After the usual environment setup and live GPU/ownership checks, run:

```bash
bash scripts/level1/train/run_c2_legacy_iter25_gs24_smoke.sh \
  /absolute/path/to/verified-iter25-gs24-complete-checkpoint \
  /absolute/path/to/new-legacy-smoke-output
```

The wrapper fixes C2 YAML, two updates, a `legacy-iter25-gs24-smoke-*` run name,
and a fresh output/dataset directory. It clears inherited dataset overrides,
uses a fresh optimizer, and retains the shared launcher's model validation and
GPU safeguards. Existing output paths are rejected. This runs a training smoke,
not an MP4-only rollout; legacy video evaluation is a separate task. This
wrapper does not itself certify checkpoint provenance or successful inference.

Data preparation creates a new immutable bundle containing `train.jsonl`,
`validation.jsonl`, `train_hf/`, `validation_hf/`, `manifest.json` and
`manifest.sha256`. Keep the bundle together and retain these canonical names.
V4 manifests now require recipe/protocol hashes and current three-repository
provenance; stale bundles must be regenerated. Seed/count/horizon CLI overrides
must match YAML. Validation compares sidecar, source hashes, split seeds,
canonical paths and HF rows against audited JSONL.

`prepare_level1_dataset.py` records one real initial-state anchor per row:
C1 executes the first legal U/D/L/R without constructing Edward; C2 records the
advertised/selected Edward candidate. These anchors are neither model rollout
nor training samples. `prepare_level1_v3_audits.py` is an Edward-only baseline
diagnostic and explicitly rejects C1 configurations.

### Actor/reference offload validation

For an actor-colocated reference, native FSDP parameter offload and phase-level
TMS offload are separate policies. The pinned trainer also supports disabling
`fsdp.offload_params` when `enable_offload`, `actor.offload`, and `ref.offload`
are all true and both engines use the same FSDP backend allocation. The launcher
rejects other phase-only combinations; do not use the legacy small-model smoke
override for the 9B recipe. Passing this check is not a GPU memory or training
completion guarantee: validate the real smoke before a full run.

### Recipe-driven evaluation

`evaluate_level1.py --config` reads the actual stage ghost/harness/reward/prompt
and one-token decoding settings. Conflicting legacy horizon, ghost or prompt
flags are rejected. Temperature/top-p overrides are explicit, separately
recorded protocols. Real inference-server verification is still pending.

The following is an evaluator invocation, **not a service startup command**.
It requires an already running compatible image-capable service that loads the
same complete checkpoint and advertises the supplied exact model ID at
`/v1/models`. Replace placeholders, freeze the test plan and use a new output:

```bash
export C2_CHECKPOINT=/path/to/selected-complete-c2-checkpoint
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export SERVED_MODEL_ID=exact-model-id-from-service
export EVAL_OUTPUT=/path/to/new-evaluation/c2-heldout.json

"$PYTHON" scripts/level1/evaluate/evaluate_level1.py \
  --config configs/level1/train/curriculum2.yaml \
  --model "$SERVED_MODEL_ID" --checkpoint-path "$C2_CHECKPOINT" \
  --tokenizer-path "$C2_CHECKPOINT" --base-url "$OPENAI_BASE_URL" \
  --purpose heldout --seed 112 --episodes 20 --samples-per-seed 3 \
  --generation-seed-base 0 --output "$EVAL_OUTPUT"
```

This proposed 20×3 held-out plan has not been executed or proven reproducible
by a real backend. `--purpose validation` restricts environment seeds to
108–111; only validation reports may select release checkpoints. Preserve
held-out seeds for final reporting rather than tuning.

`evaluate_level1_run.sh` defaults to C2 `CONFIG`, `EVAL_SEED=108`,
`EVAL_EPISODES=4` and `EVAL_SAMPLES_PER_SEED=12`. It accepts any nonzero number
of checkpoint directories; optional `CHECKPOINT_LIST` is a text file with one
checkpoint path per line to select a subset. The sampled runner delegates to
the shared runner. Full backend/resource requirements remain subject to the
real-service gate, not merely shell syntax or mocked tests.

The current saver retains `keep_last=2` plus checkpoints selected by training
reward and runs before evaluation. `CHECKPOINT_LIST`/checkpoint discovery can
only evaluate candidates still present; record their update IDs and select
the best non-base trained candidate using validation reports. An overall
comparison may favor base, but base is not a trained release checkpoint.
Do not claim a validation-best model over all 100 updates unless all relevant
candidates were actually protected and evaluated.

The evaluator creates a manifest before running and appends every attempt to
`.attempts.jsonl`; it defaults to two retries per infrastructure-failed episode,
without erasing earlier errors. Reports distinguish `evaluation_completed`,
`full_completions`, `attempts`, `error_attempts`, `win_rate` (wins/all attempts)
and `planned_trial_win_rate` (wins/planned trials). Compare Base/C1/C2 only
under one fixed ghost/harness/seed/512-step/decoding protocol. A completed
evaluation is not a successful game; report the measured rate without a
minimum threshold and keep failure videos if no completion is observed.

`checkpoint.weight_identity_verified=false` is intentional: matching a local
manifest and an API model alias is not proof of the server's loaded weights.
Keep server launch/load logs and verify the real model loading gate separately.

For download-and-run delivery, export the selected complete C2 weights with
tokenizer/processor, matching config/harness, file checksums and C1 parent
lineage. The exporter currently rejects all missing tensors, including vision:
the pinned training implementation has no verified frozen-vision evidence.
Never silently replace trained parameters with base weights. The checkpoint
validator checks required architecture keys/shapes using a meta model, without
allocating full weights; this is still not an actual model-loading gate.
Verify a real image rollout and full games, then download the published
artifact into a new directory and repeat loading/rollouts. Weight hosting and
inference hardware are not yet specified; current delivery is source/recipe-only.

The 2026-09-04 targeted CPU run passed 53 tests across
`test_dataset_stage_contract.py`, `test_level1_v3_audits.py` and
`test_curriculum_ghost_modes.py`; two parse-failure workflow tests were excluded
because Windows lacks `uvloop`. This is not Linux full-suite, GPU smoke,
complete-checkpoint inference or game-completion evidence.

Run this from the repository root after activating the `maapacman-rl` Conda
environment. Replace owner, AReaL, pacman-python and model placeholders with
your own paths and the fixed revisions described in the root README.

## Historical synthetic experiments

- `synthetic/`: multi-maze generation, rollout summaries, reports, and legacy
  remote supervisors.

The synthetic scripts are retained for reproducibility and are not the
production bundled-environment Level-1 launcher.
