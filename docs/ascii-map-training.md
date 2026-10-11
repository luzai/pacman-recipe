# ASCII map training integration

Based on pacman-recipe a34e54a; retains the visual-grounding SFT recipe.

## Primitive spaced ASCII support

`action_mode: primitive` with `observation_mode: ascii` and
`ascii_map_format: spaced` selects indexed ASCII maps and direct legal U/D/L/R
tokens. Edward option prompts and fallback remain separate and unchanged.
Episode records bind the action protocol, ghost mode and life mode; workers
preserve those settings. Runtime and dataset identities must be regenerated
when changing prompt contracts.

Small-map generation can optionally add pellets to existing empty floor using
`small_floor_pellets=True`; this is restricted to maps below 11 rows and retains
protected spawn, portal, door and power-pellet tiles. A 7x7 template is included
in `configs/slime/maps/template-7.txt`.

The training adapter supports the pinned `slime.data.batch_builder` module and
falls back to the older rollout module only when the new module is absent.
These changes do not enable reference KL, replay, backplay or reward shaping.
SGLang overlap tensor/list fixes are independent runtime patches and are not
included in this recipe commit.

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

## Portable entry points

Run from the recipe root in the locked Linux training container.

- `python -m slime_pacman.map_training.catalog --spec maps.json --output /private/catalog --game-root /pinned/game --slime-root /pinned/slime`: generate private game roots and initial-stage records. The JSON spec contains tiers with template paths, ghosts, seed, game_seeds, and train/validation/test counts. Templates for 11, 13, 15 are in configs/slime/maps; supply the original game's 21x25 level for the final tier. Freeze catalogs before model evaluation. Reusing existing held-out maps requires importing their unchanged rows rather than regenerating replacements.
- Put catalog rows into PACMAN_FORMAL_CONFIG, together with policy, max_tier, seed and optimizer_budget. Set PACMAN_SLIME_CONFIG to the existing ASCII/spaced recipe config, and PACMAN_RUN_DIR to a private run directory.
- `python -m slime_pacman.map_training.runtime_check --container NAME --lock runtime-lock.json --output receipts --slime-root /pinned/slime`: verify the exact image ID, CPU dependency versions and installed sampler/overlap file hashes. Use --install-patches only before starting inference; it invokes the guarded installers in slime. The runtime lock requires image_id, versions and runtime_files mappings. This generates IMAGE_CONTENT_VERIFIED.json and RUNTIME_CPU_VERIFIED.json; it does not start GPU inference.
- `python -m slime_pacman.map_training.launch --spec launch.json`: launch an explicitly configured slime argv. The spec includes argv, environment, source_root, source_manifest (relative file -> SHA), runtime_receipts, image_id, wall_seconds, opening_only=true, dynamic_bank=false, failure_bank=false, backplay=false. The argv supplies --num-rollout, --load, --save and the unchanged model/batch/optimizer settings. Cold start uses the merged SFT; checkpoint recovery adds --restore and retains the exact original spec and deadline. Native checkpoint scheduler, Adam and RNG loading are retained; no experiment-specific LR rewrite is installed.

The entry point sets the three lifecycle hooks, synchronous saves and resume validation. Overlap remains enabled. Reference KL is intentionally excluded and rejected by this launch path. Existing generic recipe capabilities remain available to other experiments.

Validation rows may label panel=original8 or panel=new16; both reward and win rate are recorded separately, with common frozen per-decision seeds. Test rows do not drive promotion or checkpoint selection. Map generation deduplicates rotations/reflections and rejects disconnected pellet layouts; this is a CPU gate, not a claim about task difficulty or policy wins.

## Dependency pins and acceptance

- luzai/pacman-python: codex/map-curriculum-game-fixes, 8b2bd556bd6cd707a55b3746f21b6c7bfdf30c88
- luzai/slime: main, 33758cf5b98f5a55e100eb3c8f7aa0c39b32e87e

The slime merge retains main's straw restore handoff and paused checkpoint admission. Use exact SHAs and regenerate records/manifests after source changes. GPU training, real tokenizer/SGLang serving and full Adam/RNG recovery must be accepted with these newly integrated sources before a new run; no new training was started during integration.
