# Paired route-option result (2026-10-21)

**Historical A¨CE result; later candidates are documented in [CANDIDATE.md](CANDIDATE.md).**

**Historical decision for the original short-segment version: keep escape-only single-step as the stronger diagnostic baseline.
Do not promote the route-bound version to the default or start training.**
The route-bound implementation passes binding/execution checks but fails the
requested no-win-regression criterion on real Vision replay.

All totals below cover the same 11 saved states, paired original RNG,
checkpoint160 and remaining original512-step budgets. These states come from six
source episodes, not 11 independent episodes or a generalization sample.
Deaths are newly incurred during suffix replay. CPU choices are deterministic
selector decisions; only the second table counts actual Vision calls.

## CPU selector replay

| Arm | Wins | Deaths | Pellets left | Decisions | Steps | Objective switches | Reverse actions | ABA returns | Route switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 8/11 | 7 | 6 | 212 | 336 | 189 | 78 | 71 | N/A |
| Native commit1 | 8/11 | 7 | 6 | 338 | 338 | 205 | 79 | 72 | N/A |
| Multi-route single-step | 11/11 | 3 | 0 | 198 | 198 | 27 | 11 | 8 | N/A |
| Escape single-step | 11/11 | 0 | 0 | 232 | 232 | 27 | 7 | 7 | N/A |
| Bound segments | 11/11 | 0 | 0 | 164 | 232 | 27 | 7 | 7 | 21 |

Escape and bound CPU action sequences match in all11 cases; the latter avoids68
selector calls. This does not establish real model success.

## Real Vision replay

| Arm | Wins | Deaths | Pellets left | Model calls | Steps | Objective switches | Reverse actions | ABA returns | Route switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native | 6/11 | 7 | 9 | 475 | 548 | 460 | 73 | 66 | N/A |
| Native commit1 | 6/11 | 7 | 9 | 552 | 552 | 488 | 75 | 68 | N/A |
| Multi-route single-step | 11/11 | 2 | 0 | 337 | 337 | 181 | 53 | 51 | N/A |
| Escape single-step | 11/11 | 0 | 0 | 343 | 343 | 146 | 44 | 44 | N/A |
| Bound segments | 9/11 | 0 | 2 | 268 | 373 | 143 | 44 | 44 | 109 |

Bound segments reduce model calls by75 (21.9%) versus escape single-step but
increase executed steps by30 (8.7%) and lose two wins. Fewer calls are not evidence
of a better policy. Reverse/ABA counts include potentially useful escape moves.
Legacy arms do not bind full routes, so exact route-switch counts are unavailable;
objective changes are a separate proxy, not interchangeable with route switches.

## Retained failure evidence

- `u163-g1-r1-s445`: escape single-step clears in67 steps; bound segments exhaust
  the same68-step allowance with one pellet left, zero deaths. Decisions include
  target changes, reversals and repeated portal detours.
- `u163-g1-r6-s481`: escape single-step clears in31 steps; bound segments exhaust
  the same32-step allowance with one pellet left, zero deaths;17 route switches.
- `u163-g1-r1-s373`: both clear without deaths, but escape uses27 steps/27 calls
  versus bound68 steps/49 calls. CPU success did not predict this model inefficiency.

These observations do not uniquely identify the cause: route diversity, richer
summaries, auxiliary-option execution and segment boundaries changed together.
Next isolate these factors against escape single-step before adding LURE. Do not
relax collision checks simply to improve this selected small sample.

## Validation and limitations

- All55 CPU suffixes and55 real-model suffixes passed native trajectory audits.
- 22 bound episodes passed independent summary-to-route-to-executed-action
  readback. Ten frozen-image route tests and55 first-boundary gates passed.
- Native and earlier multi-route CPU/GPU results exactly reproduce prior actions,
  wins, deaths and lengths; escape CPU also exactly reproduces its earlier run.
- Portable planner tests:30 passed, including10 new contract checks.
- Broader environment suite:90 passed before an existing active-fruit fixture
  failed to activate fruit in128 steps. Full suite collection is blocked by missing
  local `areal`; neither suite is claimed fully passing.
- GPU and final readback containers exited0, OOMKilled=false; no GPU compute
  processes remained on the checked host at final verification.
- Published planner helpers are AST-equivalent to executed code after relative
  imports/formatting and removal of an unused historical selector containing LURE.
  Replay packaging additionally makes data/output roots configurable. The live
  experiment files were never overwritten.

No training, optimizer update, production-default change or LURE trial was made.
The source manifest records the frozen executed scripts. Full evidence remains in
workspace `reports/vision-edward-routebound-20261021`; remote full trajectories are
retained in the matching experiment root. Model weights/images are not published.
