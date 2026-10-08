# Production Level-1 package

This package owns the Pacman Level-1 reinforcement-learning recipe:

- `workflow.py`: AReaL rollout workflows, no-thinking decoding and stage-specific
  dynamic direction/option masking.
- `level1_dataset.py`: deterministic rows backed by
  `pacman_env.env.PygamePacmanEnv`.
- `prompts.py`: production image/live-state prompt contracts.
- `rewards.py`: Level-1 reward composition and audit.
- `trajectories.py`: Level-1 trajectory validation and summaries.

The headless environment is the bundled sibling package `pacman_env` in the
same repository. A separate Pacman checkout is not required.

The formal stages both use a 512 `env.step` horizon, 80/4 train/validation rows
(seeds 28–107/108–111), batch 4, 12 samples and 5 epochs = 100 updates. Training
RNG seed 1 is separate. Both save and request validation every update.
The saver keeps the latest two plus training-reward-selected checkpoints,
before validation runs. Select release weights only among actual retained
validation-tested candidates and report their step IDs; saving every update
does not establish global validation-best retention.

- C1: ghost-disabled, no Edward in data/training/native evaluation; one legal
  U/D/L/R token under `direct-open-action-token-v1`, with
  `live-state-direct-action-v3+fixed-image-dynamic-v2` prompts. `step_local_raw_v1` uses each action's
  own shaped reward, `reward_norm/adv_norm=null`, and `reward_clip=.inf`.
- C2: normal ghosts, advertised Edward option-code token under
  `edward-option-code-v1`; complete C1 initialization and fresh optimizer/scheduler.
  `episode_return_group_v1` normalizes full returns across 12 same-state episodes,
  then clips to ±20, with equal loss weight per episode. Option returns are
  audit records, not extra reward terms.

Both retain reference KL=0.01 and no critic. Dataset/run/trajectory metadata
bind actual protocols, prompts, reward settings, seeds and source identities.
V4 bundles need the new explicit recipe metadata; regenerate rather than patch
old data. C1 finite rewards above 20 remain intact and JSON stores the clip
bound as `"inf"`, while non-finite rewards remain invalid.

Current delivery is source/recipe-only. The final downloadable agent requires
complete C2 inference weights, processor/tokenizer and the matching normal-ghost
Edward harness/runtime. Full-game evaluation reports observed success rate
without a minimum threshold; loading or CPU tests alone do not prove a win.
See the root README for GPU, complete-model and download-verification gates.

New code should import from `pacman_recipe.level1`. Root modules with the same
names are compatibility shims for existing Python callers and YAML workflow
paths.


Next-run layout correction (2026-10-07, working tree): ASCII now uses
`edward-ascii-option-code-v2+fixed-map-dynamic-v1`. Its fixed user explanations
are derived from the VLM fixed prefix, replacing only screenshot/image terms
with map terms. State fields, candidate fields, navigation and metrics are all
explained before the map; dynamic state/candidate rows follow it. No cache padding
is added. SFT and RL should use this same formatter; existing data and running
snapshots retain their own identities. Re-tokenize and freeze the actual prefix
length for the next run; the prior 384-token candidate is superseded.

Previous prompt layout update (2026-10-07, source commit `45a89dd`): image Edward uses
`edward-option-code-v2+fixed-image-dynamic-v2`; ASCII retains its separate
`edward-ascii-option-code-v1` version with a new template fingerprint. Renderers,
dataset versions, YAML, trajectory audits, and Slime runner declarations agree.
`vision_prompt.py` shares fixed-image-dynamic layout with grounding clients;
ASCII renders map, state, candidate rows, and output as separate sections.
Regenerate prepared data and runtime identities after this change; never patch
old hashes. The 18 focused CPU checks passed, including a real game episode.
The paused 200-update job has not been redeployed with these prompts.
