# Edward default: continuous routes, public ghost motion, risk-ranked fallback

Selecting Edward in `PacmanEpisodeRunner` now selects `LatestEdwardPlanner`.
No extra experimental flag is required. `PacmanConfig` also defaults to
`edward_harness="latest"`. Default fallback remains `risk_ranked`; LURE is off
and is not offered by the default planner. No training is started by this change.

## Behavior

- Bind each option to a full route ID, action sequence, target and portal transitions.
- Show distinct junction/portal/approach summaries to the model.
- Continue a still-valid route through junctions, checking short collision risk and
  nonfinal-pellet escape on each step. Return control on completion, risk, invalid
  geometry, changed life/target or an unusable next action.
- Whitelist current ghost coordinates, direction, speed, state and edible time.
  Never read internal ghost routes or RNG. This is environment-assisted visual
  control, not screenshot-only perception.
- Respect observed direction on the first frame; enumerate legal directions at
  subsequent tile centers, including corridor reversal. Bounded prediction and
  escape checks are not a full-route safety guarantee. Fallback can still be risky.
- Safety dispatch is instance-local: production does not patch module globals or
  import the experiment package. Portal transitions remain enabled; LURE is absent.
- Persist route registry, execution/interruption events and harness selection for
  audit. Prompt rendering and trajectory reconstruction share route summaries.

## Reproduction and deployment

Set runner option `edward_harness="legacy"` (or the same field in `PacmanConfig`)
for the old native Edward implementation. Historical replay adapters select this
explicitly before substituting their frozen planner class. Existing source-pinned
remote evaluations are untouched; these source changes apply only when deploying
this revision. Old recipe contracts without a harness selector are interpreted as
legacy and reject an accidental latest run. New recipe metadata records selection.
Plain latest saved-state restarts reconstruct remaining pellets from the live
snapshot; recorded latest boundaries must agree with it. Missing or invalid pellet
coordinates fail rather than silently recreating already-eaten targets.

The isolated optional `LURE_PORTAL` experiment remains available under
`experiments/edward_route_options`; its context patches require a dedicated process.
It is not imported or activated by the default path.

## Evidence boundary

CPU tests cover route continuation/interruption, legacy compatibility, independent
instances, hidden-field exclusion, portal retention, prompt/runner integration and
parity with the frozen V3 candidate. These tests do not establish model performance.
The earlier 11 correlated suffix states all passed for the latest frozen candidate,
but LURE was never offered there. Earlier no-path V2 had 44 deaths versus the
escape-step baseline's 26 in 64 games despite equal 60/64 wins. Full-game V3
comparison remains separately evaluated; do not call safety parity established.
See [candidate evidence](../experiments/edward_route_options/CANDIDATE.md).

## Validation commands

```text
python -m pytest tests/env/ tests/test_latest_edward_episode.py tests/test_latest_edward_contract.py tests/test_slime_fallback_config.py tests/test_vision_prompt_layout.py -q
python -m pytest tests/ -x -q
```

The full repository command is blocked in this local environment at collection by
missing `areal`; some other suites also require `torch`. Focused CPU/native tests
run without these dependencies. The repository's existing `test_level1_recipe.py`
also contains malformed repeated `config = (` syntax outside this change.
No full-suite pass is claimed. Checkpoint/GPU inference was not rerun for packaging;
frozen remote evaluation directories and running containers were not modified.

Local results: 162 tests and 3 subtests passed in the combined environment run;
its one fruit-appearance test initially failed because the external game defaults
to fruit disabled. Rerunning that test in a child process with
`PACMAN_DISABLE_FRUIT=0` passed (1 test), without changing training/evaluation
settings. The complete latest native integration file passed all 5 tests, including
exact middle-boundary restore and cross-harness rejection. An earlier fresh-worker
identity mismatch during a concurrent local Git commit disappeared after rerunning
with a stable revision; source-pin checks were retained.

## Shuffled single-letter presentation (2026-10-22)

New native `latest` and legacy runs default to `edward_shuffle_options=False`.
Set it explicitly to `True` only for an opt-in presentation experiment.
When enabled, at each Agent boundary independently permute candidate display order and the
single-token output-code mapping. Models see one letter per candidate, without
internal A0/C0 IDs. Route summaries use the same letters; original option/route
IDs stay in logs and execution. Explicit risk ranks remain unchanged, described
as fallible estimates rather than instructions. Candidate contents, safety checks,
portal behavior and route continuation are unchanged; LURE remains off.

The versioned permutation derives from episode seed and absolute environment
step with SHA-256 domains; it consumes no game/model/global RNG. Restoring a
boundary reproduces its mapping. Each executed option logs its shuffle key and
code map; trajectory audit recomputes the mapping and validates its boundary key.
Only reshuffle when requesting a new choice, not during route continuation.
Risk-fallback audit validates unique contiguous ranks against internal candidate
IDs, independently of display order. Rank tampering, omitted legal directions,
and executed action/target mismatches remain rejected.

These local source changes do not hot-update frozen evaluations. Deploy with new
source/prompt identities; old fixed-code rollouts are not new shuffled training
data. Existing checkpoints may react differently to new code assignments: CPU
correctness tests are not evidence of preserved win rate. Real-model paired
replay found no added deaths but introduced one timeout in each of two
shuffled arms (11 correlated suffixes per model/arm). It does not establish a
performance benefit or full-game non-regression, so shuffle stays off by default.
