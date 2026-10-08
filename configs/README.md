# Configurations

## slime C2

`slime/c2.yaml` fixes the new binary-reward, single-death contract with 40 train
seeds and 4 validation seeds. This differs from the historical AReaL recipes
below. See [slime acceptance](../docs/slime-migration.md); GPU validation is pending.

## Production Level-1

The two formal recipes share the original Level-1 map and a **512 underlying
environment-step** horizon, not 512 model calls or option selections.

| Setting | `curriculum1.yaml` | `curriculum2.yaml` |
| --- | --- | --- |
| Initialization | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-9B`; fresh optimizer/scheduler |
| Ghost mode | `disabled` | `normal` |
| Episode lives | First death ends episode | Three reserve lives; fourth death is game over |
| Action protocol | `direct-open-action-token-v1` | `edward-option-code-v1` |
| Prompt version | `live-state-direct-action-v4` | `edward-option-code-v2` |
| Output | One legal `U/D/L/R`; no `S`, JSON, or Edward | One advertised option code mapped to `C*/A*/E*`; no direction or JSON |
| `edward_options` | `false` | `true` |
| `action_token_choice` / `open_action_mask` | `true` / `true` | `false` / `false`; dynamic option-candidate mask instead |
| Reward objective | `step_local_raw_v1` | `episode_return_group_v1` |
| `actor.reward_norm` / `actor.reward_clip` | `null` / `.inf` | Group mean/sample std over 12 episodes / `20.0` |

Both use 80 training rows (seeds 28–107), 4 validation rows (108–111), batch
size 4, 12 episodes per state, 5 epochs and 100 optimizer updates: 20 updates
per epoch and 48 episodes per update. Training RNG seed 1 differs from dataset
seed 28. Seeds are not different maps; ghost-disabled initial states may repeat.
This training budget is not proof that either stage learns to clear the game.

Shared settings: learning rate `5e-7`, shaping alpha `0.1`, 8 GPUs (4 rollout +
4 actor), `ppo_n_minibatches=1`, KL `0.01`, reference initialized with actor,
and `critic/teacher/adv_norm=null`. Train/validation decoding uses temperature
`0.7`, top-p `1.0`, one token and thinking disabled. `saver.freq_steps=1` and
`evaluator.freq_steps=1` separately request saving and validation every update;
actual GPU artifacts remain to be verified.

Saving every update does not retain every checkpoint. The current
`keep_last=2` plus training-reward `keep_best_metric` policy runs before
evaluation; validation-based selection is therefore restricted to candidates
that actually remain on disk. Record their update list and selection rule,
retain last, and do not claim a best-over-all-100 validation checkpoint.
Protecting a global validation best or retaining all weights remains separate
work with an explicit storage budget.

C1 uses each action's raw shaped step reward, without group normalization,
episode-return broadcast, cross-game-step GAE or C2's episode-equal loss
reduction. It is critic-free, step-local PPO-style training, not standard
group-relative GRPO. `.inf` preserves finite rewards above 20; NaN/Inf rewards
remain invalid, PPO/gradient clipping remains active, and JSON records `"inf"`.

C2 normalizes full episode return within each group of 12 using
`mean_leave1out=false`, `std_unbiased=true`, `eps=1e-5`, then clips to ±20.
The episode signal reaches its model decisions with equal total loss weight
per episode. `option_return` is retained for audit, not added to the return again.

Reward v3 uses `use_base_reward=false`: normal/power pellets +1 each, ghosts +5,
fruit 0, completion +50, death −100, executed step −0.05, wall collision −0.5,
plus nearest-pellet shaping (alpha 0.1, threshold 1, scaled by cleared ratio,
skipped on pellet eating). Edward safety refusal subtracts 100 once from the
episode/last decision; refusal before any model decision rejects the sample.
Selecting AVOID itself has no penalty. Existing parse-contract failure keeps
the fail-closed episode target of −1.

Smoke reuses either selected YAML with `--smoke-updates 2`; there is no third
formal configuration. C2 smoke needs a complete checkpoint from real C1 smoke,
not a base placeholder. Initial runs retain `recover.mode=disabled`, which also
disables recovery-state saving in the pinned AReaL. Ordinary model checkpoints
do not include complete optimizer/dataloader recovery state.

The v4 bundle labels now require explicit recipe/protocol metadata and hashes.
Regenerate data in new directories; never patch old manifests. Dataset CLI
seed/count/horizon must agree with YAML. C1 data, training and native evaluation
must not invoke Edward. `planner_audit.max_steps=512` is a separate Edward
baseline diagnostic horizon, not permission to run that baseline for C1.

- `level1/train/`: the two active training configurations.
- `level1/eval/`: evaluation-only configurations.
- `level1/archive/`: earlier Level-1 gates, the former standalone smoke gate,
  and the log-prob probe retained for reproducibility.

Current delivery is source/recipe-only. A directly runnable final agent also
needs complete C2 inference weights and the exact harness/runtime. Report actual
full-game success rate without a minimum threshold; do not claim solved gameplay
before observing it. See the root README for weight/download-verification gates.

## Historical experiments

Machine-specific synthetic/text and early vision experiment configurations are
not shipped on this release branch. They are not part of the Level-1
reproduction path.

Production Level-1 uses the bundled `pacman_env.PygamePacmanEnv`; it does not
require a separate Pacman checkout. Configurations under `level1/archive/`
are not the default production path.
