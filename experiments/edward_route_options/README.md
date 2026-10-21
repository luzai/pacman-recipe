# Edward route-option experiment

Opt-in, inference-only planner prototypes. Production Edward defaults are unchanged.
No LURE_PORTAL implementation is included, and this module does not start training.

## Variants

| Arm | Implementation | Changed factor |
| --- | --- | --- |
| A | `pacman_env.planner.EdwardPlanner` | Native control |
| B | `route_bound.NativeStep1` | Native candidates, commit length one |
| C | `repair_v1.RepairedPlanner` | Earlier multi-route + short-risk single-step prototype |
| D | `repair_adapter.RepairedPlanner` | C plus bounded nonfinal-pellet escape check |
| E | `route_bound.BoundPlanner` | Full route binding, geometry summaries, short segments |

A/B isolates commitment; C/D isolates the escape check. D/E changes route
retention, candidate diversity, summaries, auxiliary routes and execution together.
Do not attribute all differences to multi-route generation alone.

## Integration contract

This is not a drop-in training configuration. The replay adapter explicitly adds
these hooks to the native runner, preserving its image rendering, constrained
option-token decoding, game state, RNG, budgets and trajectory audit:

1. Generate advertised candidates, copy `planner.route_menu` into the decision
   context before rendering the prompt. Use `install_prompt` only in an isolated
   replay process; it wraps the native renderer and audit renderer together.
2. After a model chooses an option, call `planner.bind(option, live_snapshot)`.
   The option ID resolves to a content-hashed route ID and full route registry.
3. Before executing each action, call `planner.before_action(action)`; then the
   native runner executes and calls `record_action`. Do not advance twice.
4. Call `continue_option` on the resulting observation. It follows the saved
   action sequence while valid, without regenerating the menu. On completion,
   invalidation or a boundary, clear the active option and ask the model again.
5. Run the native trajectory audit and `audit_bound`; persist the route registry
   and events alongside episodes so advertised and executed routes can be audited.

A segment ends after at most four moves, or at a junction, portal exit or route
end. Each continuation checks position, legal action, short-term collision risk
and (near a COLLECT target) escape viability. This is **not full-route safety**.
The initial action is checked during menu construction; the simulator does not
advance between selection and execution. The native risk fallback is explicitly
risky and stays one step. Moving edible-ghost targets have not been qualified by
this late-pellet experiment; do not generalize this acceptance to hunting skills.

Route summaries contain junction choices, turns, portal entry/exit, approach side
and cell, next segment boundary and route ID. Summary text need not reconstruct
the full route: the registry is authoritative. Pacman routes always retain portal
transitions. Ghost motion projection follows the native no-portal ghost topology.
The projection uses exported ghost pixel position, speed, direction and remaining
path; this is the existing harness-information protocol, not image-only perception.
It never reads future simulator RNG.

The escape guard looks ahead when a target is within two moves and seeks an exit
to a junction within four more moves. The last normal pellet is exempt because it
wins immediately. It is conservative, bounded and may reject viable alternatives.

## Tests and evidence

From the repository root:

```text
python -m pytest tests/env/test_route_options_experiment.py -q
python -m pytest tests/env/ -x -q
python -m pytest tests/ -x -q
```

The portable tests cover identity, summary distinction, continuation, interruption,
portals, the nonfinal escape guard and native commit-only control. Paired GPU
results use frozen checkpoint160 on 11 saved states from six failed source
episodes (updates160/163), original RNG and remaining 512-step budgets. CPU
selectors and real Vision model calls must be reported separately. This correlated
failure-state sample is diagnostic, not a generalization or training acceptance.
See [paired results and rejected-route evidence](RESULTS.md) and `DESIGN.txt`. Historical evidence remains
in the workspace reports; no source trajectories, image payloads or model weights
are committed.

## Frozen replay entry point

`python -m experiments.edward_route_options.replay --cpu` runs the first-boundary
CPU gate. With `CPU_ROLLOUT=1` it runs full CPU selector suffixes. Without `--cpu`
it requires a passing CPU receipt and uses the real frozen Vision model. Run in
an isolated process and verified idle GPU environment; it never launches services.
Set `EDWARD_MODEL_ROOT` (contains model160), `EDWARD_BANK_ROOT` (cases.json and
saved states), `EDWARD_OUTPUT_ROOT` (existing dedicated writable directory), and
`REPLAY_PHASE` (distinct output subdirectory per attempt). Bank paths and their
saved-state source identities must remain valid. Preserve old attempt directories.
The bank and weights are external artifacts, not bundled demo fixtures.
The guarded in-memory native runner adapter fails on incompatible source anchors;
it does not rewrite installed source. Relative imports and configurable roots are
the only replay packaging changes from the executed frozen script.

## Future LURE_PORTAL (disabled)

Finish the base experiment first, then add a separate optional arm. Require
observed approaching ghosts and an exit that increases separation or opens a
pellet route. Retain explicit lure/cross/return phases across safe steps; interrupt
when danger increases or the route is invalid. End on success or budget and use
cooldown to avoid portal loops. Check risk at every step, including the portal
exit. The model can decline the option.

Use paired states/RNG/weights/budgets and report wins, deaths, remaining pellets,
steps, model calls and latency. Rough parity may justify keeping an experimental
choice but does not establish benefit. Check candidate crowding and model
selection distraction; predefine acceptable noninferiority margins before a
larger test. Do not infer no harm from this small sample alone.
