# Edward candidate evidence and frozen replay package

The native Edward runner now defaults to the latest continuous-route,
no-internal-ghost-path V3 implementation, with `risk_ranked` fallback and LURE off,
as requested by the user. See [default integration](../../docs/edward-default.md).
This directory retains isolated replay variants and their evidence. Default
selection is not a claim of broad safety acceptance. Old A-E failures remain intact.

## Components and boundaries

- `continuity.ContinuousPlanner`: keep a valid bound route instead of stopping at
  every junction. Continue only after geometry, life-state, next-action and short
  collision/escape checks. Route completion, invalidation and actual risk return
  control to the model. The registry retains full actions and portal transitions.
- `motion_guard.NoPathPlanner` inside `guard_mode('final_base')`: whitelist current
  ghost id/state/position/pixel_position/direction/speed/edible_ticks. Internal
  path_remaining, path_target, currentPath and RNG do not enter the planner.
  Coordinates and motion still come from the environment: **not image-only**.
- The first simulated frame respects observed direction. Future tile centers
  enumerate every legal direction, including corridor reversal. V2 assumed ongoing
  corridor direction; the native game can expire/replan its path there. The new
  envelope fixes that motion-model omission, not a guarantee of overall safety.
- The collision horizon and escape search remain bounded. Unknown/overlarge
  branches fail closed; risk-ranked fallback remains potentially dangerous.
- `portal_skill.LurePortalPlanner`: explicit opt-in only. The menu describes the
  approach, portal crossing and target; the registry binds the full route. It
  uses a spare AVOID code, never replaces an ordinary candidate, and is withheld
  when the native menu is a risk fallback. The model may decline it.
- LURE requires an approaching ghost and predicted separation gain, with route
  budget/cooldown/use limits. Every step can interrupt for risk, invalidation,
  lost approach, completed target or budget. There is no waiting command and no
  promise the ghost will actually chase. Spare-slot scarcity can suppress it.

Guard contexts patch shared planner helpers. Use a dedicated **single-threaded
replay process**, not concurrent arms/production workers in one interpreter.
Restoration after the context exits is tested. Integrate the same bind,
before_action, record_action, continue_option and route-menu hooks described in
README. Using the class without its guard context does not establish isolation.

## Executable replay

From the repository root, install the usual evaluation dependencies including
PyTorch/Transformers. External frozen checkpoint and saved-state bank are required;
no weights, credentials or raw image/trajectory payloads are shipped here.

Set `EDWARD_MODEL_ROOT` (contains `model160` and prior `episodes/`),
`EDWARD_BANK_ROOT` (cases.json and cases/<id>/state.json), and
`EDWARD_OUTPUT_ROOT` to an existing dedicated writable directory. For a paired
LURE comparison set `EDWARD_ENABLE_LURE=1` **before both CPU and GPU gates**;
omit it for the base-only default. Use the same selected cases on both gates.

```text
python -m experiments.edward_route_options.candidate_replay --cpu
# Only in a verified idle GPU environment, with a unique REPLAY_PHASE:
python -m experiments.edward_route_options.candidate_replay
```

The GPU entry requires a matching CPU receipt (native adapter, helper hashes,
arms, cases, runner source and bank manifest). Keep the frozen model payload read-only and verify
its full SHA receipt separately; this CPU receipt is not a model-weight digest.
The native in-memory adapter fails on incompatible anchors. It does not edit
installed source, launch a service, or construct an optimizer. Packaging changed
imports/configurable roots, made LURE opt-in, strengthened the CPU receipt and
fixed telemetry; reported GPU results below used the frozen report scripts, not
an independently qualified GPU run of this packaged entry point.

Telemetry counts LURE offers from model-called trajectory observation_context,
not the boundary context (which lacks route_menu). Skill safety interruptions are
included with ordinary short/escape risk interruptions. Preserve raw events.

## Evidence snapshot

All five checkpoint policies passed the same 11 saved suffix states with zero
new deaths. This correlated regression panel cannot establish full-game success.
HF bf16 SDPA greedy legal-option argmax; probability reporting temperature0.7.

| Policy | Wins | Steps | Calls | Reversals |
| --- | ---: | ---: | ---: | ---: |
| Raw Qwen3.5-9B | 11/11 | 296 | 68 | 18 |
| D5 + LoRA A initial | 11/11 | 328 | 76 | 11 |
| Primitive Adam160 | 11/11 | 302 | 72 | 12 |
| Primitive Adam170 | 11/11 | 251 | 57 | 10 |
| Edward Adam1 | 11/11 | 292 | 62 | 9 |

LURE-enabled checkpoint arms had zero offered LURE candidates; identical outcomes
are **not evidence of skill benefit**. Synthetic CPU/native-processor trigger tests
only establish reachability and contract behavior. They are not model wins.
Codex CLI gpt-6.1-sol low effort was user-stopped after9 completed pairs, all wins,
zero deaths (base188steps/37calls/5reversals). Compare only those same9 cases;
identical across-arm requests reused cached choices, not independent samples.
Raw Qwen is not being expanded to64 and Codex must not be resumed automatically.
Per-case results and matched summaries are adjacent CSV/JSON files.

The earlier V2 full-game checkpoint160 comparison remains an unresolved warning:

| Harness | Wins/64 | Deaths | Episodes with death | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| escape_step1 (internal ghost path) | 60 | 26 | 21 | 4 |
| original continuous (internal ghost path) | 58 | 32 | 25 | 5 |
| no-internal-path V2 | 60 | 44 | 32 | 4 |

39 of V2's44 deaths occurred in games that eventually won. Thus equal win rate
cannot establish safety parity. The direct V2→latest motion-guard64 comparison
and D5A/Adam170/Edward1 full-game64 assessments are running separately, LURE off.
Each is8 T5 maps×8 seeds, not64 maps, with512 steps/four lives. This reused diagnostic
set is not fresh independent generalization evidence. The user requested the latest implementation as the default, but this does not
resolve the safety comparison or show that RL is the only remaining problem.
No new training has been launched.
