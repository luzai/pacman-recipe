# Formal jobs: exact final-reward group filtering

Preparation only; no training launch or live source replacement. Tier start is
pending the initial D5+LoRA A 64-episode-per-tier validation.

New formal jobs (Primitive and Vision) default to `exact_final_reward` with
`max_rounds: 16` and `retry_task: same_tier_new_task` when `reward_group_filter` is absent.
Explicit historical policies without `retry_task` retain fixed-task behavior;
set the new field explicitly when migrating. The explicit example is
`configs/slime/maps/next-vision-reward-filter.json`. Set `reward_group_filter: null`
in JSON to disable it. Runtime and restore launcher resolve the same default.
The resolved policy is included in checkpoint/controller identity; an old unfiltered
checkpoint cannot silently resume with filtering enabled. Use its original frozen
code/config, or explicitly disable the policy in a separately prepared compatible
configuration. Do not rewrite old manifests or hot-patch running jobs.

The formal rollout path rejects a complete 12-episode group iff its **final shaped
rewards are exactly equal**, including all-zero and constant nonbinary rewards.
All-win groups with differing speed/life bonuses are retained. Rewards, advantage
calculation, Clip-Cov and loss normalization are unchanged. No tolerance/rounding
or binary-win conversion is used. Nonfinite rewards and mixed model versions fail.

Keep 16 fixed slots and their tier/replay quotas. Replace rejected slots with
same-tier training tasks, preferring layouts not yet sampled in this batch. Once
the layout pool is exhausted, cycle deterministically while avoiding the slot's
previous layout. In group_randomized mode each replacement gets a fresh environment
seed (also bound into episode_record), shared by its twelve replicas. Fixed_catalog
mode preserves the replacement catalog task's environment seed. Policy sampling
seeds and sample IDs are fresh, under the same weights. No task is deleted or
blacklisted: normal future first-pass sampling can revisit hard tasks (with fresh
environment seeds in group_randomized mode). This is not a guaranteed scheduled
revisit of the exact failed environment. Wait for every group in each round;
never prefer whichever finishes first. Return exactly 16 effective groups (192
episodes) to training. Further decision/token counts remain variable as before.

Safety limit: 16 total rounds (initial plus 15 retries), at most 3072 sampled
episodes. Exhaustion raises an explicit shortage with audit evidence; no partial
batch, constant-group fallback, optimizer update or curriculum advancement occurs.
No downgrade to easier tiers or partial-batch fallback is permitted. Replacement
is deterministic and recorded, and does not mutate the controller or catalog.

`resampling/rollout-*-round-*.json` retains all sampled episode references, seeds,
final rewards and rejection decisions. Full episodes retain their existing files.
Rejected in-memory sample payloads are released between rounds; finalization only
receives retained groups. `effective-batches/rollout-*.json` names retained episodes.

`map-batches` and `curriculum/episode_win_rate` use the unfiltered **first** 192
episodes, and curriculum commits only after successful optimization. This avoids
using rejection-conditioned win rates for promotion. `rollout/effective_episode_win_rate`
reports the retained batch separately; reward-filter metrics report generated,
rejected/effective groups and rounds. Independent validation is never filtered.

CPU tests cover strict selection, exact shaped equality, complete batches,
deterministic seeds, binding/version/finite guards, exhaustion, first-pass versus
effective metrics and checkpoint identity. Real GPU acceptance remains required
before claiming end-to-end deployment readiness; no GPU training is started here.
