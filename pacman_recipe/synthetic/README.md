# Historical synthetic package

This package contains the deterministic text/maze Pacman environment and its
historical research utilities. It is separate from production Pacman
Level-1:

- `env.py`: synthetic `PacmanEnv`.
- `dataset.py`: text and multi-maze prompt/spec generation.
- `maze_suite.py`: deterministic maze registry and split metadata.
- `baselines.py`, `evaluate.py`, and `vision.py`: synthetic baselines,
  evaluation, and rendering.
- `configs.py`: historical synthetic AReaL configuration adapter.

New synthetic code should import from `pacman_recipe.synthetic`. Root modules
remain compatibility shims.
