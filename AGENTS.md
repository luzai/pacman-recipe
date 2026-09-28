# AGENTS.md - pacman-recipe

Pacman Level 1 reinforcement-learning recipe with AReaL and a slime adapter.
The slime adapter has CPU acceptance coverage; GPU acceptance is pending.
Source/recipe-only delivery; `release/maapacman-v0.1.0` is the daily
development and delivery branch. `backup/2026-09-04/*` is kept for history
only.

## Layout

- `pacman_recipe/` - datasets, native multi-modal workflow, Reward v3,
  trajectory accounting, and evaluation tools
- `pacman_env/` - bundled headless pygame Level 1 environment; uses the
  sibling `pacman-python` game checkout
- `configs/level1/`, `scripts/level1/` - training configs and run entry points
- `slime_pacman/`, `configs/slime/`, `patches/` - pinned slime adapter and acceptance configuration
- `tests/` - pytest suite
- `docs/` - recipe and handoff documentation; the largest handoff records live
  at the workspace root one level up

## Rules

- Run `python -m pytest tests/ -x` for code changes; add tests for new
  functionality (pytest config: `testpaths = ["tests"]`,
  `pythonpath = ["."]`).
- Reproducibility pins source revisions by SHA (see README "源码边界");
  do not rely on moving branch names. After updating a dependency revision,
  sync the manifest and re-verify.
- `pacman_recipe.synthetic.*` is reserved for historical synthetic-maze
  experiments.
- Formal reports go to the workspace-root `reports\` directory (archive:
  `reports-archive\`); do not create a new `reports/` inside this repo.
- GPU work on the shared H100 nodes follows the workspace-root safety rules in
  `..\AGENTS.md` ("Remote GPU And H100 Rules"); treat those rules as loaded
  even when a harness only discovers this file.
- The `AReaL\` sibling is an upstream framework clone with its own AGENTS.md;
  do not refactor it casually from this project.
