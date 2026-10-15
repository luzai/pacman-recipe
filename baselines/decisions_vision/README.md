# Decisions vision baseline

Native primitive and Edward baselines, plus a legacy image-only smoke test, using OpenAI
`POST https://api.openai.com/v1/decisions`, model `gpt-6-luna`.
Official schema: https://developers.openai.com/api/docs/guides/decisions

Run from the `pacman-recipe` root with Python containing numpy, Pillow and
pygame 2.6.1. The native sibling `pacman-python` checkout must be available.

```powershell
# Pin the intended no-fruit evaluation condition, even if the shell overrides it.
$env:PACMAN_DISABLE_FRUIT = "1"
# Native offline contract rollout: fixture choices, NOT performance scores.
python -m baselines.decisions_vision.native --mode both --max-steps 32
# Native live decision + one game step per mode (explicit smoke-only cap).
python -m baselines.decisions_vision.native --live --mode both --single-step
# Native score comparison: same seed and at most 32 game steps per mode.
python -m baselines.decisions_vision.native --live --mode both --max-steps 32 --seed 0
# More seeds, only when explicitly desired (up to 192 calls).
python -m baselines.decisions_vision.native --live --mode both --max-steps 32 --episodes 3
```

## Native modes (main comparison)

`native.py` subclasses the existing `PacmanEpisodeRunner` and replaces only
`_call_model`. It does not copy or change the episode loop. Both modes use
the existing `live_state_v3` visual prompts, `vision_model_image`, fixed/image/
dynamic message layout, structured state, history and native trajectory audit.
Primitive keeps the existing open-action mask U/D/L/R (no S). Edward keeps
`EdwardPlanner`, authoritative candidate order and option-code map, commit
length, safety checks after each step, continuation and replanning. Fallback
defaults to `risk_ranked` for Edward; `--fallback refuse` reproduces legacy handling.
Primitive always uses `refuse`; the CLI fallback flag applies only to Edward.
The archived 50-pair comparison below used explicit `refuse` and has not been rerun
with the new default. Both modes retain their existing prompt renderers.
The two modes deliberately retain different observation/action contracts.
This is not a comparison isolating the effect of a prompt or model alone.

Normal ghosts and single-death episodes are shared; the comparison budget is
an immutable native environment horizon (32/256/512/2000), not a decision cap.
Rewards use binary level completion, with other shaping disabled. Reported
score is the native game score, separate from training reward. The one-step
smoke mode uses the runner's existing `single_step_complete` branch; an
evaluation-only marker permits this cap after validating the original full
production row. It does not fabricate a teacher-prefix training audit.

API differences are explicit: original message text, roles, ordering and image
bytes are preserved, but content block names become `input_text`/`input_image`.
The native output support becomes a Decisions `choice` question. OpenAI renders
system messages as developer messages. Its image encoding, sampler, tokenizer
and probabilities are not the Qwen backend's. A reversible local `SymbolCodec`
only lets the runner reuse native objective code mapping; its IDs are never
sent as model token IDs. When only one code is supported, it is executed
locally and counted as `forced_singleton`, with no fabricated extra choice
and no API call. No confidence threshold, new tactical filter or retry is added.

Outputs under ignored `artifacts/decisions_vision/native/<UTC timestamp>/`
include `comparison.md` and JSON, plus per-mode manifests with source/prompt
fingerprints and row identity, exact native messages, converted API bodies
(without authorization), images at each decision, API latency/usage records,
and the complete audited `episode.json`. API failures preserve the pending
request and safe error summary. Offline choices always select the first
advertised code and are explicitly marked as fixtures, never live performance.

```powershell
python -m pytest tests/test_decisions_native.py tests/test_decisions_vision.py -q
python -m pytest tests/env/test_edward_planner.py tests/env/test_edward_risk_fallback.py -q
```

Tests compare native and adapter messages byte-for-byte for both one-step and
multi-step real-game runs, plus state/history, candidate/code maps, executed
actions, option status and scores. Additional tests cover continuation,
replanning, singleton support, malformed results, refusals, HTTP failure,
no retries, redirects and credential input. Full-suite collection currently
requires the unavailable local `areal` dependency.

## No-fruit 50-pair evaluation (2026-10-15)

Fixed seeds 0–49, both modes, at most 512 game steps, normal ghosts, single death,
and `PACMAN_DISABLE_FRUIT=1`; eight seeds ran concurrently. Prompts, supports,
navigation and episode runner were unchanged. The game checkout now disables
fruit by default; an explicit environment value `0` still enables it.

| Mode | Clears / planned attempts | Mean score (complete episodes) | API latency mean / p50 / p95 / p99 |
| --- | --- | --- | --- |
| Primitive | 0/50 | 490.2 (n=50) | 0.308 / 0.262 / 0.639 / 1.050 s |
| Edward | 10/50 | 5104.5 (n=47) | 0.306 / 0.266 / 0.563 / 1.059 s |

Edward seeds 1, 12 and 35 stopped on API `refusal`, without a retry or substitute
action. These three attempts count as non-clears for the operational primary
endpoint; unavailable complete-episode scores are not imputed. The refusal
convention was specified after the first refusal, so complete-case sensitivity
is reported too. Two-sided exact McNemar p=0.001953125, unchanged when the three
failed pairs are excluded. Mean paired score difference over 47 complete pairs
is 4611.3 (paired bootstrap 95% CI 3621.5–5606.8; 20,000 resamples).
This supports a difference for this fixed layout and seed distribution, not
unseen-map generalization or trained-checkpoint performance.

Latency pools successful API decisions, including those before an episode later
refused. Forced singleton choices and refused requests are excluded (refusal
elapsed time was not retained). It includes connection/network/response work,
but not screenshots, game steps or video encoding. Concurrent evaluation and
background video export shared the host. Local evidence and the detailed table:
`artifacts/decisions_vision/native512_no_fruit50_20261015_v1/comparison.md`.
Earlier fruit-enabled experiments below are archived separately and not pooled.

## Historical fruit-enabled live results (2026-10-15)

Both modes passed one live decision/one executed game-step smoke test at seed 0,
with no wall collision. Separate short comparison: same seed 0, normal ghosts,
single death, at most 32 game steps, `gpt-6-luna`:

| Mode | Game score | Game steps | API calls | Forced choices | Wall collisions | Deaths | API p50 / p95 | End |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Native primitive | 110 | 13 | 13 | 0 | 0 | 1 | 0.189 / 0.239 s | Death |
| Native Edward | 260 | 32 | 24 | 3 | 0 | 0 | 0.173 / 0.209 s | Step budget |

Primitive ate 11 normal pellets; Edward ate 26. Edward made 27 policy-boundary
decisions (24 API + 3 forced choices); five other game steps continued an option
without asking the model again. Neither run cleared the level. This is one seed,
not evidence of stable superiority or generalization. Existing trained agent
scores were not rerun, so this table compares the API model under two native
harnesses, not API versus trained-checkpoint performance.

Local evidence: `artifacts/decisions_vision/native/20261015T185017264764Z/`
(single-step smoke) and `artifacts/decisions_vision/native/20261015T185028925852Z/`
(comparison, exact prompts, images and audited trajectories).

Validation: 64 tests passed across baseline and Edward planner suites; 19 passed,
1 skipped and 7 subtests passed across image transport, layout and token-constraint
suites. The skip requires a real local model processor. The full suite is blocked
at collection by `ModuleNotFoundError: No module named 'areal'`.

## Legacy image-only smoke (not the main baseline)

```powershell
# Offline: render a real screenshot and validate the image request; no API call.
python -m baselines.decisions_vision.evaluate
# One live image decision and one unchanged native environment step.
python -m baselines.decisions_vision.evaluate --live --steps 1 --seed 0
# Bounded rollout: at most 96 requests; explicitly opt in when needed.
python -m baselines.decisions_vision.evaluate --live --steps 32 --episodes 3 --seed 0
# One supplied PNG, decision only.
python -m baselines.decisions_vision.evaluate --live --image C:\path\frame.png
```

Credentials: `OPENAI_API_KEY` takes precedence over the local file
`baselines/decisions_vision/api_key.local.txt`. This file is ignored by Git;
never stage it with force. Logs do not contain keys, authorization headers,
server error bodies. Native request bodies contain game inputs only and are saved
without authorization; the key is not included. One attempt per decision, no retry
and no redirect. Refusal, timeout or invalid response stops the rollout;
there is no substitute action.

Inputs contain the current screenshot and fixed English rules only. All five
actions U/D/L/R/S are offered; no legal-action filtering, hidden game state,
planner, history or tactical hints are supplied. Ground-truth legal actions
are used only after inference to label the chosen action. This is an image-only
baseline, not the structured-state Edward/options protocol. Environment
defaults remain normal ghosts, single death, and a 512-step episode horizon.
`--steps` caps API decisions separately; hitting it is a budget cutoff, not a win.
Game time pauses while awaiting API responses, so wall-clock latency is not
equivalent to live Windows game control. No global keyboard input is used.

Each unique ignored `artifacts/decisions_vision/<UTC timestamp>/` directory
contains prompt/hash/budget manifest, initial provenance/state, before/after
PNGs, decision records (action, confidence, probabilities, token usage,
request latency, turn latency, legal-action label, score, wall collision,
death and pellet clear rate), and a summary. Failed API calls preserve the
input screenshot and a credential-safe error. Initial state is an evaluator
artifact and is never sent to the model. One successful turn proves transport
and execution only, not visual accuracy or game performance.

Validation: `python -m pytest tests/test_decisions_vision.py -q`.
For performance studies use fixed seeds, the same observation/action contract
and budgets across agents; report decision and game-step counts separately.
Static labeled-dataset accuracy and aggregate p50/p95 reports are future work.

## First live check (2026-10-15)

Seed 0, one screenshot and one native step: API returned D (down), confidence
0.21, input tokens 346, request time 0.412 s, turn time 0.715 s. Down was
illegal at [16,10]; the unchanged action hit a wall, position remained [16,10],
score delta 0, no death. This verifies the image transport/action execution
chain, with an incorrect first decision. Evidence is in the local ignored
`artifacts/decisions_vision/20261015T183125178207Z/` directory.

Seven focused tests passed. The required full `tests/ -x` run stopped during
collection at `tests/test_backplay_workflow.py`: `ModuleNotFoundError: No module
named 'areal'`. Full-suite acceptance remains unverified in this local Python.
