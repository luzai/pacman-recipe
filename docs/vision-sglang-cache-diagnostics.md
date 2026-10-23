# SGLang rollout cache optimization — completed independent diagnostics

Production training was not stopped, restarted, or changed. Tests used H100_2_3 GPU1 only; unrelated GPU0 and parallel evaluation GPU2/3 were preserved. All independent inference containers completed and GPU1 was verified free (0 MiB).

## Configurations

A: production-equivalent SGLang context default262144, automatic KV pool, static fraction0.5, radixoff, concurrency8.
B: same weights/backend/prompts/constraints and all other flags, explicit context8192 and max_total_tokens65536, radixoff.
C: B plus exact fixed-text-prefix Mamba cache, fresh Vision declaration576tokens before vision_start; no image/state tokens cached.

Model initial D5+LoRAA SHA3ff9e8b4e6baab851fc8ac98612f706707641ab5e959217925f911f880e75a2e; source52d536a; image6a8d21365e2d5f44f067666f5985a30c450b246ef29b5911a73d40d238709aa1. Real PNG requests captured from11 saved-state first boundaries. These are diagnostics, not11 new games and not independent generalization evidence.

## Warmed throughput diagnostic

Same GPU, one engine at a time, sequenceABBAABBAAB. Each engine first ran32 warmup requests, then32 measured requests at concurrency8;32requests repeat the same11images in fixed order. Five measured rounds perarm,160requests perarm. No cache reuse in eitherarm.

| Metric | A baseline | B bounded | B change |
|---|---:|---:|---:|
| Median round request/s |14.646|14.323|-2.2%|
| Median request latency s |0.559|0.544|-2.7%|
| P95 request latency s |0.733|0.800|+9.1%|
| Median GPU memory MiB |44366|24662|-19704MiB (-44.4%)|
| Round request/s range |12.387–15.105|11.060–15.974|overlap|

The earlier single11-request run showed7.02 versus5.27request/s. Its25%drop did not persist at the warmed median, but tail latency remains worse. This is a memory-saving tradeoff, not a demonstrated throughput improvement or proof of no regression. Five rounds on repeated11images do not establish long-run p95 or workload-wide performance.

## Probability verification

Initial A/B11requests: every legal-support logprob exactly equal; greedy11/11; token counts exact, no truncation; normalized legal support verified.
Repeated diagnostics: B all5rounds/160requests exactly matched firstA legal-support distributions. One baselineA round(index7) had maxdifference0.116860 and mean0.0019565; allotherA rounds exactly matched. Across320requests greedy320/320, mean candidate difference0.00019565. This baseline-only variation must not be attributed to the bounded configuration; its cause was not isolated.

## Prefix reuse: FAILED

C all11requests hit exactly576 cached tokens. Flush produced0hit; rewarm restored576. However candidate-logprob maxdifference0.356922 and mean0.086355 versusB; greedy11/11does not qualify the distribution. Do not enable prefix in production. No tolerance was relaxed.
C allocated212 Mamba states (conv0.23GB+SSM9.98GB) in addition to2GB KV; measured GPU32536MiB. Small-run2.894request/s is not a stable throughput estimate. Actual weight-update invalidation was not tested, only flush invalidation. A smaller Mamba pool would not address the unqualified numerical parity and was not mixed into this experiment.

## Recommendation and boundaries

Keep production prefix/radixoff. B is a viable independently tested low-memory rollout configuration with modest median throughput cost and a measuredp95cost; do not label it a speed optimization. Production integration requires a deliberate safe checkpoint/engine-rebuild boundary and a fresh formal train-vs-rollout gate, preserving Adam/scheduler/RNG/controller and optimizerbudget. The engine API does not provide proven hot-resizing of its allocation; release/resume is notresize.

Artifacts: local repeated/COMPLETE.json, repeated/PARITY.json and10perroundJSON; AB_SUMMARY.json; fixed.json,prefix.json,PREFIX_FAILED.md. Remote /docker/pacman-vision-edward-sglang-cache-20261022-v1 retains all scripts,requests,logs andcontainer IDs. No training launched by these scripts.

## Legal-action importance correction

Pacman custom loss enables TIS by default, without an extra CLI flag. The upper bound is `DEFAULT_TIS_UPPER_BOUND = 2.0`: `min(2, exp(old_actor_logprob - behavior_logprob))`, detached per decision. Both probabilities use the same legal candidate support and temperature. Existing behavior is unchanged; this is not a newly enabled correction and is not MIS.

Do not add upstream `--use-tis`: the custom loss rejects it to prevent double correction. Exactly one optimizer update per rollout is required. `train/tis_weight_mean` and `train/tis_truncate_fraction` report actual execution. TIS corrects distribution mismatch under its assumptions; it does not validate stale cache state, image identity, candidate support, or cache invalidation after weight updates.
