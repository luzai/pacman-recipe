# New slime jobs: defaults and local-first SwanLab

New `PacmanConfig` and `configs/slime/c2.yaml` default to:

- ClipCOV enabled, ratio `0.002`, covariance interval `(1, 5)`.
- Four total lives (`original_three_lives`: the engine name means three spare lives).
- Successful reward `1 + 0.1*(1-deaths/3) + 0.05*(1-steps/H)`; failures zero.
- SwanLab `auto` mode, project `pacman-slime`.

These defaults apply to the `python -m slime_pacman.launch` entry point. The
launcher materializes the ClipCOV declaration into the environment inherited by
Ray workers. Explicit `PACMAN_EXPERIMENTAL_CLIP_COV` overrides remain supported;
tracking records both the requested config and the effective declaration.
`clip_cov: false` or the explicit environment value `off` disables it. Unset
environment variables in direct low-level loss tests still mean disabled.
Invalid declarations fail in the existing ClipCOV validation; no silent fallback.

Dataset generation now forwards life/action/observation contracts from the same
config. An old single-death dataset must be regenerated or paired with an
explicit `episode_life_mode: single_death`. Historical frozen source/config
snapshots and checkpoint identities must not be rewritten.

## Tracking and outages

Install the optional SDK with `pip install -e '.[tracking]'`. Set credentials
through `SWANLAB_API_KEY`, never in the config/argv or tracked files. Configure
`HTTPS_PROXY`/`ALL_PROXY` in the launcher environment when required; install SOCKS
transport support for a SOCKS proxy. Credentials are not forwarded to Ray.

The launcher saves `tracking/<session>/config.json`, `metrics.jsonl`, `status.json`
and `DONE.json`. This includes effective ClipCOV, source hashes, model/load argv,
reward/lives, curriculum identity and panels. Numeric train/rollout/eval/performance
metrics and ClipCOV selection counts are collected from the existing log stream.
Raw trajectories and large checkpoints remain in their existing experiment
directories; they are not automatically uploaded.

A separate uploader process uses SwanLab cloud mode when an environment API key
is available, otherwise offline mode. SDK initialization failure attempts offline
mode. Missing SDK or logging failure leaves the replayable JSONL journal intact.
Network hangs do not block training; shutdown waits at most seven seconds before
terminating only the uploader. Training's actual exit code is preserved.
`status.json` cloud means the SDK initialized in cloud mode, not proof that every
event reached the server. The journal is the recovery source of truth, including
when the SDK internally buffers a network failure without raising an exception.

SwanLab's step axis is a monotonic logging event index. `source/rollout_index`
preserves the upstream index; it must not be interpreted as an audited completed
optimizer-update count. Initial validation and training events stay distinct.

Replay after an outage (creates a new SwanLab experiment; do not replay twice
and treat the resulting runs as independent experiments):

```bash
python -m slime_pacman.tracking --directory /absolute/run/tracking/SESSION --mode auto
```

For SDK offline files, `swanlab sync` is also available. Explicit
`swanlab_mode: offline` avoids cloud initialization; `disabled` bypasses this
tracking wrapper entirely. No mode changes training hyperparameters.

API reference: https://docs.swanlab.cn/en/api/py-init.html
