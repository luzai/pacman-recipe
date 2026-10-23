# Compact Vision Edward cache/TIS runtime

This is the frozen experiment runtime used with recipe commit `365106d`, pinned
Slime `89bfada990a00663846e0ac804de1454685ceed3` and image digest
`sha256:6a8d21365e2d5f44f067666f5985a30c450b246ef29b5911a73d40d238709aa1`.
It is an experiment overlay, not a general deployment launcher or a change to
recipe defaults. `runtime-sha256.json` records the exact active runtime bytes.

## Declared run

Initial D5+LoRA A, fresh Adam/scheduler/RNG, T5 through T7, 200 optimizer updates.
16 effective nonconstant-final-reward groups ×12 episodes, resampling same-tier
new tasks for up to16 rounds. MB1/full-block32, CPU optimizer offload,8 GPUs.
Frozen vision/merger, full language, no KL; ClipCov .002,1,5; native TIS upper2.
Latest Edward, portals retained, risk_ranked, shuffle/LURE disabled, four lives,
no fruit. `compact_v1`, input4096/context8192, fixed text prefix512, KV65536,
max-running-requests8, FP32 Mamba state, no CUDA graph.

Formal curriculum JSON sets `evaluation_replicas: 1`,
`final_evaluation_replicas: 12`, `vqa_probe: true`:24 held-out maps,24 regular
validation episodes and288 final episodes. Fixed VQA is48 wall/pellet-cell
questions and is a narrow diagnostic. Its generator explicitly opts out of the
policy cache contract and requires observed integer `cached_tokens=0`; policy
requests retain full fixed-prefix checks.

## Mount and runtime contract

Mount the frozen recipe, game, pinned Slime, initial model and runtime read-only.
Mount `formal_train.py` over `/workspace/slime/train.py`; launch
`/runtime/train_with_hooks.py`. `/acceptance` owns configuration/proofs and
`/run` owns this run's artifacts. All writable payloads belong to its Docker
root. The launcher must independently verify live GPU ownership, initial model
SHA, exact mounted-file hashes, no existing run/checkpoint, and adequate space.
This directory deliberately contains no automatic launcher or model weights.

`PACMAN_CACHE_CONTRACT=/cache/contract.json` and
`PACMAN_FIXED_PREFIX_FILE=/cache/prefix.json` must refer to matching freshly
processor-derived declarations. Install the repository's
`patches/sglang/pacman_fixed_prefix_cache.py` once into the new container's
MambaRadixCache; never patch a shared or running engine. Only the exact text
warmup may insert a node. Image/dynamic prefixes must never enter this cache.
`hooked_manager` and `cache_boundary` warm every engine after KV onload and at
initial/eval/generate/weight-update boundaries, including same-version onload.

## Acceptance and limitations

`first-update-pre-sync-gate.json` is not a final pass. The runtime marker also
requires Adam1 readback, scheduler/RNG/moments, effective192 single-version
batch, visual-exact/language-changed export, actual TIS/ClipCov evidence and
a durable joint archive. The historical .05 logprob comparison is recorded;
it is not enforced under the separately declared TIS/cache-aware policy.
No exact backend parity claim is made.

**A runtime marker alone is not complete cache-invalidation acceptance.**
An exact512-token warm request can only match511 tokens in the pinned scheduler;
therefore its `prewarm_cached_tokens=0` cannot independently prove a flush.
A513-token canary returning512 proves rewarm, not invalidation. Independent
same-process tensor and disk tests must first show full real-image requests
longer than512 return cold0 after changed weights, then hot512 after warming.
For the actual Adam1 boundary, `audit_live_flush.sh` additionally reads all8
engine-specific raw Ray logs, verifies flush HTTP200 precedes tensor-update
HTTP200 in the version1→2 window, and matches all8 endpoints/new versions/hot512.
It currently targets the named experiment root; inspect and change that explicit
root before using it for another run. `PENDING` or `AUDIT_REQUIRED` is not a pass.
The pinned successful flush implementation resets the radix tree and clears
both pools. This supplemental audit must pass before claiming first-gate success.

A fixed-prefix guard changes numerical distributions in this hybrid model.
TIS does not prove cache correctness. The unit tests use mocks and do not replace
real GPU weight updates, optimizer acceptance, or long-run quality evaluation.

## Verification

`python experiments/vision_edward_compact_cache/test_runtime.py` runs the ten
focused mock runtime checks. Real native acceptance separately regenerated384
records and exercised8 tiers with the actual image processor. Only complete
collections should appear in plots; keep filtered training results and
independent validation separate and compare the same tier. Small baseline
samples are not broad generalization evidence.

Archive reservation and cleanup preserve the newest complete durable joint
checkpoint while the next archive is in flight. Only task-owned old payloads
may be removed after verification; initial weights and unrelated jobs stay
protected. The current runtime is included verbatim, including its narrow
model/layout assumptions, for reproducibility and audit.