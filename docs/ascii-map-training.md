# ASCII map training integration

Based on pacman-recipe a34e54a; retains the visual-grounding SFT recipe.

## Planner terminal collection

The last normal pellet ends the episode. COLLECT still requires safe arrival,
but does not require an escape or return after that terminal move. Power pellets
and nonterminal targets retain the existing checks. Option continuation uses
the same candidate generator.

## CPU curriculum contracts

`slime_pacman.map_training` provides TrainingPolicy / TrainingWinController,
FormalSampler, and durable checkpoint publish / restore functions.

The intended policy is threshold 0.8, window_batches 3, min_games 384,
consecutive_checks 2, min_maps 8. Each batch has 16 groups of 12 replicas;
tier 0 uses 16 current groups, subsequent tiers use 13 current and 3 earlier
ones. Only completed training batches drive promotion. Every start is initial;
there is no automatic demotion, failure bank, or backplay. Use at least 16
distinct layouts per tier to avoid wrapping within a batch. Validation and test
layouts cannot share a layout SHA with training.

Checkpoint receipts bind controller state to payload hashes and the completed
iteration marker. Call publish only after a synchronous model/Adam/RNG save;
restore verifies bytes and iteration identity. It does not prove actual GPU
optimizer or RNG restoration.

## Integration boundary

These reusable contracts do not install a rollout entry point or launch a run.
Remaining integration includes custom map game/planner/reward binding, catalog
generation, worker dispatch, lifecycle hook wrappers, scheduler/budget recovery,
reference KL, fixed validation panels, and container preflight. Historical run
scripts and their absolute server paths are deliberately not launch defaults.

Related dependency branches already published:
- luzai/pacman-python: codex/map-curriculum-game-fixes, 8b2bd556bd6cd707a55b3746f21b6c7bfdf30c88
- luzai/slime: codex/map-curriculum-runtime-hooks, 40e4c44af03445bd18948e7d378f20242dfa7b6e

CPU tests cover the contracts; new GPU training/recovery acceptance is pending.
