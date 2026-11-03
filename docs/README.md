# Documentation

- [Checkpoint 审计工具](checkpoint-audit-helpers.md): exact loaded-state comparisons and pure latest/best/pending selection.
- [Action prefix isolation](action-prefix-isolation.md): opt-in per-request namespaces with strict zero-cache-hit checks.
- [Engine-assisted experiment components](experimental-engine-assisted.md): explicit training/curriculum/evaluation contracts and optional PyTorch LoRA/loss helpers; no deployment entry point.

- [名称迁移与 prompt 审计](preparation-migration.md): current package names, compatibility and prompt v2.
- [slime 迁移与验收](slime-migration.md): independent vision requests, episode weighting, source pins, and remaining GPU gates.
- `architecture/`: AReaL recipe architecture in English and Chinese.
- [`../RUN_ARTIFACTS.md`](../RUN_ARTIFACTS.md): artifact retention policy and
  the latest completed production-run summary.

Generated reports, rendered slides, videos, and raw trajectories stay outside
Git under the ignored `artifacts/` directory. See `../RUN_ARTIFACTS.md` for the
repository policy and reproducibility guidance.
