# AReaL Pacman RL Recipe Design

Status: two-stage source/recipe integration; final C2 inference and game-clear
acceptance remain unverified. Dated evidence below is historical, not new gates.
Environment provider: bundled `pacman_env.PygamePacmanEnv` API `3.0`
Environment ID: `pacman-python-level1-ghostdoor-v3`
Dataset contract: `maapacman-level1-dataset-v4`
Production C1/C2 dataset default: `512` underlying `env.step` calls
Supported episode caps: `32`, `256`, `512`, and `2000` actions
Reusable `PygamePacmanEnvConfig` default: `512` actions; immutable rows still
determine the actual horizon
Remote Linux backend: SDL dummy
Contract synchronized with code: `2026-09-04`
Historical remote evidence last verified: `2026-07-23`

The headless environment implementation is bundled in this repository under
`pacman_env/`; it uses the sibling `pacman-python` game checkout.

Example checkout layout:

```text
${WORKSPACE_ROOT}/AReaL
${WORKSPACE_ROOT}/pacman-recipe
${WORKSPACE_ROOT}/pacman-python
```

## 0. Current two-stage release contract

| Setting | C1 | C2 |
| --- | --- | --- |
| Initialization | Qwen3.5-9B | Complete C1 weights, fresh optimizer/scheduler |
| Ghosts / harness | Disabled; no Edward in data, training or native evaluation | Normal; Edward options |
| Output protocol | One legal U/D/L/R, `direct-open-action-token-v1` | One advertised code, `edward-option-code-v1`, mapped to C*/A*/E* |
| Prompt version | `live-state-direct-action-v3` | `edward-option-code-v1` |
| Objective / normalization / clip | `step_local_raw_v1` / none / `.inf` | `episode_return_group_v1` / group-12 mean/sample std / 20 |

Both stages use 512 underlying steps, batch 4, and 12 samples. C1 uses 80/4
train/validation rows, seeds 28–107/108–111, and 5 epochs = 100 updates; C2 uses
40/4 rows, seeds 28–67/68–71, and 5 epochs = 50 updates. Training RNG seed 1 is
separate. Learning rate is 5e-7 and shaping alpha is 0.1. Both retain
KL 0.01 with a reference model, `ppo_n_minibatches=1`, no critic/teacher, and no
advantage normalization. Seeds are not new maps or a guarantee of task diversity.

C1 assigns only each direction's own shaped reward, without episode broadcast,
group normalization or cross-game-step GAE, preserving its original loss
reduction. C2 normalizes whole-episode returns over 12 same-state episodes and
then clips, distributing that signal to model decisions with equal total loss
weight per episode; option returns are audit records, not additive rewards.
The sample standard deviation uses `std_unbiased=true`, `mean_leave1out=false`
and `eps=1e-5`. `.inf` is only C1's clipping bound, serialized as `"inf"` in
JSON; task rewards still must be finite and PPO/gradient clipping stays active.

Current delivery is source/recipe-only. A download-and-run agent requires the
selected complete C2 checkpoint, tokenizer/processor, exact Edward harness,
config and tested inference backend. Record file hashes, C1 parent lineage,
base revision, run/update, three source revisions and export settings; no
external symlinks to training-machine files. C1 weights are recommended for
reproducing C2; optimizer recovery state is a separate artifact.

Select release weights using a declared validation-only rule among actual
retained trained candidates, record their update IDs, retain last, then freeze
held-out evaluation. Current `keep_last=2` and training-reward `keep_best_metric`
run before evaluation; saving each update does not retain all 100 or protect
the global validation best. A validation-best protection mechanism or all-model
retention needs separate implementation/storage planning. The proposed test is seeds 112–131 × three
fixed generation RNG seeds (60 games), not a completed result. Measure normal
ghosts + Edward + 512-step games, report all attempts/errors and actual win
rate **without a minimum threshold**. A zero rate must be reported honestly;
provide a win video only when one is observed. Loading success is not a win.
Complete-model loading, real-image end-to-end gameplay, full-game reporting,
and re-download verification are distinct release gates; all remain required.

The 2026-09-04 targeted CPU check passed 53 tests across
`test_dataset_stage_contract.py`, `test_level1_v3_audits.py` and
`test_curriculum_ghost_modes.py`; two workflow parse-failure tests were excluded
because Windows lacks `uvloop`. This does not establish Linux full-suite,
distributed GPU, final checkpoint inference or game-completion results.

## 1. Ownership

```text
AReaL fork
  owns distributed training, rollout workers and checkpoint orchestration

pacman-recipe repository
  maapacman package owns the external process wrapper, frame-boundary action
  protocol, RGB Surface extraction and stable environment API
  pacman_recipe package owns episode datasets, prompts, model calls, parsing,
  reward shaping, trajectory records, AReaL configuration and evaluation

pacman-python
  owns the original game rules, resources and pygame renderer
```

`pacman-python` is a sibling dependency and must remain source-clean. It must
not import `pacman_env` or AReaL.

### 1.1 Production environment API

Production level-1 files import only `PygamePacmanEnv`:

- `pacman_recipe/level1/workflow.py`
- `pacman_recipe/level1/level1_dataset.py`
- `train_areal.py` production dry-run path
- `scripts/level1/dataset/write_level1_manifest.py`
- `tests/test_level1_recipe.py`

The accepted environment import is
`from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig`. The recipe
does not define or expose an alternative Pacman environment contract.

## 2. Episode architecture

```text
episode row
  -> PacmanNativeVisionWorkflow (training) / PacmanImageOnlyWorkflow (evaluation)
  -> PygamePacmanEnv.reset()
  -> original pacman-python process and pygame Surface
  -> RGB observation
  -> AReaL multimodal rollout endpoint
  -> VLM completion
  -> C1: constrained U/D/L/R -> one PygamePacmanEnv.step(action)
     C2: constrained option code -> Edward option -> one or more env.step calls
  -> original score delta and state metrics
  -> recipe reward adapter
  -> C1 step-local task reward / C2 whole-episode task objective
  -> repeat until terminated or truncated
```

The environment is a local Python object. The OpenAI-compatible HTTP API is
only between AReaL and the model server; it is not the game protocol.

After reset and after every completed action transaction, the pygame worker is
paused at the returned `display.flip()` boundary while AReaL waits for the
model response. Pacman, ghosts, timers, and animation do not advance during
model inference. Model latency therefore changes rollout wall time but does not
create hidden game frames.

## 3. Required installation and mirror

The production training node needs three fixed source layers. The
`pacman-recipe` checkout contains both Python packages, so a standalone
Pacman repository or installation is not required:

```text
${AREAL_ROOT}
${PACMAN_RECIPE_ROOT}
${PACMAN_PYTHON_ROOT}
```

Choose an owner-controlled persistent `OWNER_ROOT` on each node. It may be a
symlink to a data volume, but validate its ownership and target locally instead
of copying a machine-specific path.

The existing AReaL fork checkout is intentionally outside the deployable
recipe mirror and is separated from robotics development:

```text
${AREAL_ROOT}
  release branch: release/pacman-v0.1.0
  pinned revision: ee872bae29152f4b553349385aced59abd1651ba
  origin: https://github.com/luzai/AReaL.git

${UNRELATED_AREAL_ROOT}
  local branch: <unrelated-development-branch>
  purpose: preserve unrelated development and its existing dirty state
```

The physical AReaL fork path is node-local `${AREAL_ROOT}`. The
`maapacman-rl` editable binding and production launcher must resolve `areal`
from this worktree. The launcher prepends `AREAL_ROOT` to `PYTHONPATH` and
rejects an import resolved outside it. An unrelated AReaL development worktree
is not a Pacman training dependency.

Use a project-specific Conda prefix rather than system Python or an unrelated
existing environment:

```bash
OWNER_ROOT="${OWNER_ROOT:?set an owner-controlled project root}"
CODE_ROOT="${CODE_ROOT:-$OWNER_ROOT/maapacman-stack}"
AREAL_ROOT="${AREAL_ROOT:-$OWNER_ROOT/AReaL}"
PACMAN_RECIPE_ROOT="${PACMAN_RECIPE_ROOT:-$CODE_ROOT/pacman-recipe}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-$CODE_ROOT/pacman-python}"
ENV_ROOT="${ENV_ROOT:-$OWNER_ROOT/.conda/envs/maapacman-rl}"
PYTHON="${PYTHON:-$ENV_ROOT/bin/python}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="$PACMAN_PYTHON_ROOT"
: "${BASE_ENV:?set BASE_ENV to a compatible source Conda prefix}"

conda create -y -p "$ENV_ROOT" --clone "$BASE_ENV"

"$PYTHON" -m pip install "pygame==2.6.1"
"$PYTHON" -m pip install \
  -e "$AREAL_ROOT" \
  -e "$PACMAN_RECIPE_ROOT"
```

Pin and record the AReaL, pacman-recipe, and pacman-python Git revisions before
training. The bundled `pacman_env` package shares the pacman-recipe revision.
Treat the `pacman-python` mirror as read-only during rollout. Per-worker copies,
`agent_state.json`, pygame processes, and IPC remain disposable under `/tmp`.

The historical H100 validation used `/tmp + pip --target` so it could be
removed without touching persistent environments. That method proves runtime
compatibility but is not the production installation recipe.

An unrelated legacy `pacman_gym` Conda environment was audited but not
modified. It used Python `3.11.15` and lacked pygame and the bundled
`pacman_env` module, so it was not the accepted recipe environment.

The launcher sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.
Xvfb is not part of the recipe runtime or deployment dependencies. Dated
API-v1 evidence recorded original-pygame worker and oracle gates on node1 and
node5 with SDL dummy; those results establish the display choice but are not a
current API-v3 acceptance result.

## 4. Environment construction

```python
import os

from pacman_env.env import PygamePacmanEnv, PygamePacmanEnvConfig

env = PygamePacmanEnv(
    PygamePacmanEnvConfig(
        pacman_python_root=os.environ["MAAPACMAN_PACMAN_PYTHON_ROOT"],
        level=1,
        max_steps=512,
        ghost_mode="normal",  # C2; C1 explicitly selects "disabled".
        video_driver="dummy",
        audio_driver="dummy",
    )
)
```

Both formal recipes and row generation default to `512` underlying steps.
Generic/historical API-v3 rows still support `32`, `256`, `512`, or `2000`;
the workflow uses the immutable row value and formal recipe checks reject a
different horizon. This count is not the number of C2 option selections.

The workflow validates before rollout:

```python
if env.spec.api_version != "3.0":
    raise RuntimeError("unsupported original-pygame environment API")
if env.spec.env_id != "pacman-python-level1-ghostdoor-v3":
    raise RuntimeError("wrong Pacman environment")
if env.spec.action_tokens != ("U", "D", "L", "R", "S"):
    raise RuntimeError("incompatible action contract")
if env.config.max_steps not in {32, 256, 512, 2000}:
    raise RuntimeError("unsupported level-1 episode cap")
```

It must import only the public `pacman_env.env` API, not the worker module.

## 5. Dataset contract

One abbreviated row constructs one complete original-game episode:

```json
{
  "id": "level1-normal-seed28-train-0001",
  "split": "train",
  "dataset_contract_version": "maapacman-level1-dataset-v4",
  "action_protocol": "edward-option-code-v1",
  "prompt_version": "edward-option-code-v1",
  "env": {
    "ghost_mode": "normal",
    "name": "pacman-python-level1-ghostdoor-v3",
    "api_version": "3.0",
    "backend": "original-pygame",
    "pacman_python_revision": "cbb97115e407abc86a44adc82a1b8f360b3e8da0",
    "level_revision": "36116c17c6c0805fdb1a07216357ac64c88d2c3108a0e37dce2a01b4ea2a8b97",
    "level": 1,
    "seed": 28,
    "max_steps": 512,
    "observation_mode": "rgb"
  }
}
```

Do not reuse historical rows labeled `maapacman-level1-v1` or API `1.0`
without checking their `env_id`: those rows describe the older private AReaL
environment, not the current API-v3 original-pygame wrapper. Current rows must
use dataset contract `maapacman-level1-dataset-v4`, environment API `3.0`, and
environment ID `pacman-python-level1-ghostdoor-v3`.

Dataset v4 requires `env.ghost_mode`. `disabled` records expose no ghosts;
`normal` records expose the four live ghosts. The mode is part of the ruleset
revision and must match the selected training recipe, runtime state, audit
anchor, and trajectory evidence.

This abbreviated example is not a complete accepted row. Formal bundles also
bind three-repository provenance, recipe/prompt/reward metadata and hashes,
audited real one-step anchors, JSONL/HF content and a manifest checksum sidecar.
C1 anchors execute the first legal U/D/L/R with no Edward construction or
planner provenance; C2 anchors retain candidates and the selected option.
Neither anchor is a model rollout or training sample. Existing v4 data without
the new metadata is rejected; regenerate into a new directory, never patch it.
Keep canonical bundle paths and pass CLI counts/seeds/horizon matching YAML.

Production level-1 dataset validation accepts only `env.max_steps` values
`32`, `256`, `512`, and `2000`; row generation defaults to `512`. The selected
cap is part of the immutable row contract, so incompatible horizons must not be
mixed into one training/evaluation split. If a terminal transition lands
exactly on the cap, `terminated=True` takes precedence and that successful
action must not also be reported as `truncated=True`.

## 6. Observation and action protocol

Each model turn contains:

1. One fixed system prompt.
2. Exactly one PNG encoded from the current `(400,336,3)` RGB Surface.
3. Stage-specific live-state context and a single-token instruction; C2 also
   advertises the current Edward candidates and their code mapping.

C1 allows only currently open `U/D/L/R`, not `S` or JSON. C2 permits only an
advertised option code, not a primitive direction or JSON; the harness executes
the mapped option. Both masks are dynamic. Preserve fail-closed parsing and
the documented −1 episode target; never hide model failure by substituting an
automatic planner's choice. The environment's generic `S` operation below is
not an allowed formal C1 model response.

One environment step is one action transaction ending at an original pygame
frame boundary. Directional input moves one cell in the committed level-1
game; the transaction may run more than one internal frame until that grid
transition completes. Completion is detected from game state and the
corresponding `display.flip()`, never from a fixed sleep. `S` sends no
directional event and advances exactly one original frame.

The worker blocks inside the wrapped `display.flip()` after reset and after a
completed transaction. It resumes only when AReaL sends the next action. This
pause is part of the environment contract: an action selected after a `50 ms`
or a `5 s` model call is applied to the same returned state.

Every rollout worker owns a unique temporary runtime directory, original script
copy, resource link/private copy, pygame process, IPC channel, state file, and
Surface. Workers do not own an X server or `DISPLAY`; pygame renders through
SDL dummy and Pacman reads the completed pygame Surface directly.

### Current IPC contract and historical timing evidence

The current implementation uses newline-delimited JSON over subprocess pipes.
Actions and state use small messages. Each RGB frame is copied from the pygame
Surface, compressed with zlib level 1, base64 encoded into JSON, and decoded by
the parent. Cross-process shared memory is not currently used.

The unchanged original loop still executes `clock.tick(60)` after the paused
`flip()` resumes. `tick(60)` measures elapsed wall time since its previous
call, including time spent paused inside `flip()` waiting for the model. It
waits only when that total elapsed time is less than approximately `16.67 ms`;
it does not add another `16.67 ms` after a slower model call.

The following dated API-v1, 287-action local profiling used real `L/R` movement
and the then-committed renderer:

```text
1 worker env.step:                p50 16.81 ms, p95 18.58 ms
16 worker env.step:               p50 40.52 ms, p95 61.80 ms
zlib+base64+JSON frame roundtrip: p50  1.38 ms, p95  2.07 ms
shared-memory two-copy estimate:  p50  0.024 ms, p95 0.037 ms
small pipe notification:          p50  0.053 ms, p95 0.091 ms
1 worker, simulated 50 ms model wait:
  action to observation:          p50 11.80 ms, p95 14.98 ms
  complete model+environment turn:p50 62.13 ms, p95 65.18 ms
16 workers, simulated 50 ms model wait:
  action to observation:          p50 14.52 ms, p95 21.09 ms
  complete model+environment turn:p50 64.88 ms, p95 71.49 ms
```

The immediate-action `16.81 ms` result was the rate-limited scripted-agent
case and could not be added directly to a `50 ms` model call. With the simulated
`50 ms` wait, the historical single-worker p50 was `62.13 ms` per complete
turn, or approximately `17.83 s` for that run's 287 actions. The corresponding
16-worker local p50 was `64.88 ms`, or approximately `18.62 s` per worker
episode. These are historical local measurements, not current H100 guarantees.

In that profiling, the RGB codec represented about `0.40 s` of serial CPU work
across 287 frames, and shared-memory payloads were estimated to save about
`0.38 s` per worker episode. The general conclusion remains that
`clock.tick(60)` has little or no sleep to remove when model latency already
exceeds the `16.67 ms` frame budget; it is a bottleneck only for faster scripted
or low-latency policies.

The recipe keeps both pipe IPC and the original `clock.tick(60)` behavior.
H100 profiling with the real model and selected rollout concurrency decides
only whether the shared-memory IPC optimization is justified. Changing or
bypassing the original pygame clock is outside the recipe design.

A later hybrid protocol may keep pipe messages for actions, state, request IDs,
and frame sequence while placing each worker's RGB buffer in shared memory.
This is required only if H100 profiling at the selected rollout concurrency
shows frame serialization causing CPU saturation; it does not block the first
training gate.

## 7. Reward ownership

Pacman returns the original game score delta:

| Original event | Base reward |
|---|---:|
| Empty move, wall or `S` | `0` |
| Normal pellet | `10` |
| Power pellet | `100` |

The shared event-reward-v3 recipe sets `use_base_reward=false`: normal/power
pellets +1 each, ghosts +5, fruit 0, completion +50, death −100, actual step
−0.05, wall collision −0.5, plus nearest-pellet shaping with alpha 0.1,
threshold 1, cleared-ratio scaling and skip-on-eat. The original score remains
auditable but is not added again. Edward safety refusal subtracts 100 exactly
once from the episode/last decision; initial refusal without a model decision
rejects the sample. AVOID itself has no penalty. C1 has neither ghosts nor
Edward, so those corresponding events do not arise.

Every step record stores both `base_reward` and `shaped_reward`.

The committed original game enters mode `6` when normal pellets reach zero;
power pellets are not part of that internal clear counter. For this revision,
the terminal reason is therefore `all_normal_pellets`. The recipe must not
reinterpret it as `all_pellets`.

## 8. Trajectory provenance

Each trajectory records at least:

```text
env_api_version
env_id
backend
pacman_python_revision
level_revision
renderer_revision
seed and max_steps
ghost_mode and three-repository provenance
action_protocol, prompt_version, template and actual prompt hashes
reward objective, reward clipping/normalization contract
RGB frame hashes where requested
action and parse status
base_reward and shaped_reward
score and collectible counts
pygame_mode
terminated, truncated and terminal_reason
```

Training outputs from different `pacman_python_revision` values must not be
merged as if they used the same environment.

## 9. Current executable and evaluation contract

The production workflow constructs `PygamePacmanEnv` directly and validates
API `3.0`, environment ID `pacman-python-level1-ghostdoor-v3`, and dataset
contract `maapacman-level1-dataset-v4`. It takes the episode cap from the
validated row; formal C1/C2 and row generation default to `512`, while generic
rows support `32`, `256`, `512`, and `2000`. No environment default overrides
the immutable row or selected recipe contract.

The former `107`-test/`13`-subtest claim and the 287-action oracle were part of
the dated API-v1 evidence. They are not a current API-v3 acceptance result;
Sections 10-12 retain those experiments only as historical evidence.

The H100 recipe uses eight GPUs:

```text
actor:   fsdp:d4p1t1  -> four actor workers on GPUs 0-3
rollout: vllm:d4p1t1  -> four rollout workers on GPUs 4-7
```

The launcher explicitly exports the pinned
`MAAPACMAN_PACMAN_PYTHON_ROOT`, places `maapacman-rl/bin` first on `PATH`,
refuses busy GPUs, and never preempts unrelated work.

Every rollout has a random `trajectory_sample_id` and is persisted as:

```text
<dataset-row-id>--sample-<trajectory-sample-id>.json
```

Exclusive file creation prevents repeated rollout samples of one dataset row from
replacing each other. The earlier group-12 run wrote only the last sample for
each row; its two validation files are not a 24-sample mean.

Training and evaluation decoding are independent:

```text
training: sampled12, temperature 0.7, top_p 1.0, exactly one token
per-update validation: freq_steps=1, sampled12, temperature 0.7, top_p 1.0
saving: freq_steps=1 independently of validation
post-run: explicit recipe/config; sampled and greedy results remain separate
```

`PacmanImageOnlyWorkflow` defaults to `enable_thinking=false`, rejects
`enable_thinking=true`, and explicitly sends
`chat_template_kwargs.enable_thinking=false`. Trajectories record the prompt
style, decoding contract, request body, verbatim response, and reasoning
content. Both sampled and greedy validation require thinking to be disabled and
the report rejects any observed reasoning content.

Use the selected recipe for ghost/harness/reward/prompt settings. Fair
Base/C1/C2 comparison fixes one common evaluation protocol; unlike native-stage
returns are not directly comparable. Full-game success requires
`terminal_reason=all_normal_pellets`, zero normal pellets and the matching
clear event, not merely high reward. Evaluation completion and game completion
are separate fields. No minimum win rate is imposed.

The same YAML supports `--smoke-updates 2`; C2 smoke must use a real complete
C1 smoke checkpoint. Full training remains 100 updates per stage. Initial
`recover.mode=disabled` also disables recovery-state saving in this AReaL;
switching to auto after interruption cannot recover unsaved optimizer state.
Enable recovery from the first run if complete recovery is required. Final
model loading, GPU evaluation artifacts and downloadable weights remain
unverified; source delivery alone is not a directly runnable trained agent.

## 10. Historical corrected group-12 evidence

Historical run:

```text
${ARTIFACT_ROOT}/level1-group12-20260723b
```

This API-v1, 287-action run predated the current API-v3 contract. Its original
post-training test omitted `enable_thinking=false`; in-training
validation also sampled at temperature `1.0`. The corrected evaluator rebuilt
complete VLM checkpoints for base/epoch0/epoch1/epoch2/epoch3/final, ran one
true greedy episode on each, then ran 24 matched sampled episodes on final.
All used seed `0`, 287 steps, original pygame RGB, and thinking disabled.

Evidence:

```text
.../corrected_eval_20260723/comparison.json
```

| Policy | Decode | Score | Pellet clear | Walls | Actions |
|---|---|---:|---:|---:|---|
| base | greedy | 0 | 0.00% | 287 | `U x287` |
| epoch0 | greedy | 20 | 1.02% | 283 | `L x287` |
| epoch1-3 | greedy | 20 | 1.02% | 283 | `L x287` |
| final | greedy | 20 | 1.02% | 283 | `L x287` |
| final | sampled, 24 episodes | 20 avg | 1.02% avg | 283 avg | `L x6888` |

Every corrected episode has zero reasoning turns. The first update caused
single-action collapse and later updates did not recover.

## 11. Historical static image-only prompt A/B

The A/B compared:

- `minimal_v1`, the original short image-only prompt;
- `live_static_v2`, which borrows only static visual landmarks, absolute screen
  directions, the nearby-pellet objective, and the blue-wall rule.

`live_static_v2` has no coordinates, pellet counts, legal action sets,
OPEN/BLOCKED directions, route hints, cell history, or live-controller action
veto.

Base and collapsed-final policies each ran one greedy plus 12 sampled episodes
for both prompts. Evidence:

```text
.../prompt_ab_20260723b/comparison.json
```

The live-inspired prompt changed base greedy from score `0` to `20`, but the
training-relevant sampled distribution was worse:

| Base sampled policy | Avg score | Pellet clear | Avg walls |
|---|---:|---:|---:|
| `minimal_v1` | 389.17 | 15.65% | 130.50 |
| `live_static_v2` | 357.50 | 14.80% | 151.75 |

Both prompts left the collapsed final checkpoint at score `20` with `L` on
every turn. The next training gate therefore freezes `minimal_v1`; the
live-inspired prompt remains a rejected ablation.

## 12. Historical anti-collapse gate and conditional next gates

Historical API-v1, 287-action config:

```text
configs/level1/archive/level1_image_anticollapse_4update_group12_8gpu.yaml
```

It restarts from base Qwen3.5-9B with:

```text
max_steps = 287
group size = 12
rollout + actor workers = 4 + 4
reference = BF16, actor-colocated with native FSDP CPU parameter offload
optimizer updates = exactly 4
rollout temperature = 0.7
learning rate = 1.5e-6
KL coefficient = 0.01
actor storage = FP32 master weights
frozen reference storage = BF16, resident
prompt = minimal_v1
validation = greedy temperature 0 after every update
reward = unchanged score_delta - step_penalty - wall_penalty
```

No progress shaping or Oracle action is active in this gate. Acceptance
requires retaining sampled exploration and improving the best corrected greedy
checkpoint without parse failures or reasoning content.

The first `20260723a` launch completed rollout and PPO compute but OOMed during
the first xccl weight sync: the newly enabled reference engine and actor left
only `1.63 GiB`, while the FSDP full-tensor gather required `3.79 GiB`.
`20260723b` tested reference offload, but AReaL's nested `stdbuf` wrappers made
the first TMS preload string invalid. A recipe-owned `sitecustomize.py` shim
fixed that parsing issue. `20260723c` then completed all `48` first-batch
rollouts and the initial reference offload, but TMS failed to restore the
large colocated FSDP reference with `CUDA error: invalid argument`.

TMS is therefore rejected for this production gate. `20260723d` kept the
colocated reference resident in BF16. It completed all `48` rollouts, ref-logp,
PPO, xccl weight synchronization, and wrote a complete `17.9 GB` checkpoint.
The actor plus reference nevertheless peaked at about `79.75 / 81.56 GiB`;
an asynchronous allocation failure surfaced in the checkpoint
`torch.cuda.synchronize()` before validation.

An isolated `20260723e` attempt used `3 rollout + 4 actor + 1 independent
reference`. It proved the physical separation, but this AReaL controller
produced only three of the four dataset groups and waited indefinitely at
`36/48` trajectories: one rollout worker is required per synchronized consumer
item in this configuration.

An attempted `20260723f` topology kept `4 + 4` and targeted the reference at
rollout, but AReaL initializes reference before rollout and rejected the job
with `WorkerNotFoundError`; changing upstream initialization order is out of
scope.

The accepted eight-GPU topology therefore remains `4 rollout + 4 actor`, with
the BF16 reference actor-colocated but configured with native FSDP2
`fsdp.offload_params: true`. It keeps frozen parameter shards on CPU outside
reference forward passes, then streams them to GPU as FSDP layers execute.
This is distinct from AReaL's TMS engine `offload`: global `enable_offload` and
`ref.offload` remain false. The actor retains `optimizer_dtype: float32`; the
reference uses `optimizer_dtype: bfloat16`. The design preserves the required
four synchronized rollout groups and four-way actor sharding while freeing
reference storage during PPO, xccl, and checkpoint synchronization. The TMS
smoke script and shim remain diagnostic evidence only and are not activated by
the formal config.

The accepted topology completed as run:

```text
${ARTIFACT_ROOT}/level1-anticollapse-4update-20260723h
```

It finished all four optimizer updates, saved four complete `17.9 GB`
checkpoints, and persisted 200 unique trajectories: 192 sampled training
episodes plus eight historical greedy validation episodes. Native FSDP
reference offload reduced static reference GPU use from about `19.9 GiB` to
`2.3 GiB`; PPO later plateaued around `72.22 / 79.19 GiB` without unbounded
growth or OOM.

The corrected matched evaluation nevertheless failed the quality gate:

| Policy | Decode | Avg score | Pellet clear | Avg walls |
|---|---|---:|---:|---:|
| base | greedy1 | 0 | 0.00% | 287.00 |
| update02 | greedy1 | 20 | 1.02% | 283.00 |
| base | sampled12, `0.7/0.95` | 248.33 | 9.99% | 165.42 |
| update02 | sampled12, `0.7/0.95` | 213.33 | 8.97% | 180.42 |

All corrected episodes had thinking disabled, zero reasoning turns, and zero
parse failures. Greedy selected update02 only because it repeated `R` for all
287 actions; under the training-matched sampled distribution, update02 was
worse than base. Sparse training must therefore not be scaled.

The next isolated change is
`alpha * (nearest_normal_pellet_distance_before -
nearest_normal_pellet_distance_after)` with `alpha=1`, using BFS over the level
representation owned by the bundled `pacman_env` package. Distance and reward
terms must be logged and independently audited; `pacman-python` remains
unchanged.

That follow-up is implemented by
`configs/level1/archive/level1_image_progress_4update_group12_8gpu.yaml`. It changes only the
auditable alpha-1 reward term and the validation contract:

```text
in-training sampled validation:
  n_samples = 12
  temperature = 0.7
  top_p = 0.95
  enable_thinking = false

post-run checkpoint report:
  sampled12 = the same 0.7 / 0.95 contract
  greedy1 = temperature 0 / top_p 1
  enable_thinking = false for both
```

Sampled validation is the primary checkpoint-selection distribution because it
matches training. Greedy remains a separate collapse/determinism diagnostic;
the report exposes both `best_sampled_label` and `best_greedy_label` rather
than merging the two results.

The recorded fallback plan was to use Oracle SFT curricula at horizons `64`,
`128`, and the then-full `287`, followed by RL. Those values belong to this
historical run plan and are not the current API-v3 supported-cap contract.

## 13. Historical official-main migration gate

On 2026-07-23 both remote nodes were moved to clean official AReaL worktrees at
commit `4d7ee11479d61ebe6c6f020e2bdcda5d76c6a76b`; the former checkouts and all
Morgan/robotics changes remain untouched on `robotics/morgan-vla`.

The node5 eight-GPU diagnostic used
`configs/level1/archive/level1_official_areal_smoke_3b_4gpu.yaml` with Qwen2.5-VL-3B. It
proved that the official scheduler, four vLLM workers, four actor workers, and
the real SDL-dummy `PygamePacmanEnv` screenshot path initialize and generate
valid game trajectories. It then failed before the first optimizer update:

```text
ref.compute_logp
  -> FSDPEngine._prepare_mb_list
  -> KeyError: 'mm_token_type_ids'
```

This is an interface mismatch, not an OOM and not a Pygame failure. The
official OpenAI proxy caches token IDs, log-probs, versions, masks, and rewards,
but its `InteractionWithTokenLogpReward.to_tensor_dict()` does not preserve
the image tensors or Qwen-VL `mm_token_type_ids`. Official Qwen-VL FSDP
correctly requires both. The vLLM `awex_adapter` warning about a missing
Megatron package is optional-plugin noise; all four inference servers became
ready and served the RGB requests.

Do not patch the official AReaL worktree or silently train these samples as
text-only. The production migration must replace the OpenAI-proxy export with
a recipe-owned native multimodal `RolloutWorkflow` that returns the official
tensor contract:

```text
input_ids
attention_mask
loss_mask
logprobs
versions
rewards
mm_token_type_ids
multi_modal_input[pixel_values, image_grid_thw]
```

By the end of this recorded migration, the recipe provided
`pacman_recipe.level1.workflow.PacmanNativeVisionWorkflow`, and the official 3B
smoke config selected it. The workflow called `InferenceEngine.agenerate()`
directly, required exact equality between processor `input_ids` and rollout
response `input_tokens`, and returned one complete multimodal training row per
Pacman action. It did not modify the official AReaL worktree.

The dated isolated node5 verification passed the complete recipe suite
(`138 passed, 13 subtests`) and the real cached
`Qwen2.5-VL-3B-Instruct` processor produced:

```text
input_ids          (1, 410)
mm_token_type_ids  (1, 410)
pixel_values       (1344, 1176)
image_grid_thw     (1, 3) = [[1, 32, 42]]
```

At the end of this 2026-07-23 record, the revised smoke had not yet passed a
real reference-logp, actor update, checkpoint save/reload, and fixed evaluation.
Sections 10-12 therefore remain historical results from the AReaL-VLA-based
stack rather than proof of current API-v3 training compatibility.

## Appendix A. Superseded 2026-07-22 snapshot

The remainder is retained only as historical evidence. Its test counts,
six-GPU topology, filename-overwrite behavior, and early-run conclusions are
superseded by the current contract in Sections 4, 5, and 9. Sections 10-13 also
retain later dated evidence rather than current API-v3 acceptance results.

At that superseded snapshot, the executable level-1 recipe constructed
`PygamePacmanEnv` directly, validated API `1.0` and
`pacman-python-level1-pygame-v1`, recorded environment provenance and
collectible state, and used the original game's `all_normal_pellets` terminal
reason. Local, node1, and node5 recipe suites had passed `99` tests plus `13`
subtests, including the historical 287-step oracle and cancellation while
awaiting a model response.

The launcher in that snapshot used a six-GPU `d3` actor plus `d3` rollout
topology. It placed the selected Conda environment first on `PATH` because
AReaL launched nested workers with `python3`; it refused busy selected GPUs and
did not preempt unrelated processes.

Trajectory JSON filenames in that snapshot used the dataset episode ID, so
repeated samples of the same ID replaced the earlier file. The directory was
therefore an auditable final sample set, but its file count was not the total
rollout count. Epoch and global-step counts had to be read from AReaL metrics
and checkpoint names.

### Historical acceptance gates

### Local gate — completed

- `pacman-python` source checkout was clean at commit `d258122e...`.
- Pacman wrapper used the original pygame Surface.
- Windows native and SDL dummy produced identical reset and `L,L,L,S` hashes.
- `PygamePacmanEnv` tests passed `6/6`, including four concurrent workers.
- The complete Pacman unittest suite passed `23/23`.
- The complete pacman-recipe pytest suite passed `98` tests and `11` subtests.

### Linux display gate — completed on an 8×H100 test node

- Actual pygame driver: SDL dummy, pygame `2.6.1`, SDL `2.28.4`.
- Three repeated `L,L,L,S` runs matched Windows raw RGB hashes exactly.
- Four- and sixteen-worker concurrent tests passed with unique runtime IDs.
- Read-only source-resource concurrency passed.
- The 287-step original-game oracle matched Windows state and RGB hash.
- No worker process or temporary worker directory remained.

Xvfb was not installed or used and was not part of that historical recipe.

### Recipe CPU gate — completed locally and on both remote nodes

- The node5 mirror and dedicated Conda prefix were present.
- Direct PygamePacmanEnv tests, RGB parity, 4/16-worker isolation, and the
  complete oracle had passed in that persistent environment.
- This historical Node1 gate used the earlier three-editable-package layout.
  The current release folds `pacman_env` into the `pacman-recipe` checkout; the
  recorded direct pygame gates and deployed recipe suite result (`98` tests and
  `11` subtests) are retained here as run history.
- Node1 used a node-local AReaL development checkout at base `da645a37...`
  with 26 dirty entries; node5 used its own physical checkout. The paths were
  intentionally node-local rather than shared.
- Both nodes regenerated identical 8-row train and 2-row validation datasets
  with `max_steps=287` and canonical level revision `36116c17...`.
- Both nodes passed AReaL config loading for six GPUs, `vllm:d3p1t1`, and
  `fsdp:d3p1t1`.
- The real model request, canonical parser, reward mapping, trajectory audit,
  and process cleanup were exercised on node5.

### GPU gate — training passed; greedy improvement failed

Node5 run `impl-20260722b` used GPUs 1–6 and completed in 856.94 seconds.
GPU0 and GPU7 workloads were left untouched. Evidence:

- three vLLM rollout servers loaded Qwen3.5-9B and served real RGB requests;
- each saved trajectory contained 287 original-pygame environment steps;
- three PPO optimizer updates completed, ending at `v_theta=3`;
- checkpoints existed for epoch 0/global step 1 and epoch 1/global step 3;
- all nine final trajectory files passed independent reward recomputation;
- the deployable final checkpoint contained 427 trained keys and 333 restored
  frozen visual keys.

The frozen greedy seed-0 evaluation did **not** improve after this short run:

```text
                 baseline   after 2 epochs
base reward          0.0          0.0
pellet clear rate    0.0          0.0
shaped reward     -287.0       -287.0
parse failures       0            0
```

At that point, the recipe was accepted as an executable RL training path, but
the two-epoch overfit-quality criterion had failed. The proposed next experiment
was to reduce the 287-turn credit horizon or use demonstration-biased initial
actions, then compare several fixed seeds and sampled as well as greedy policy
metrics. That historical run did not demonstrate policy improvement.
