# Pacman fixed-prefix cache

This opt-in patch targets the frozen SGLang image/cache implementation recorded
in the experiment runtime spec. It is for the ASCII Edward path.

1. Render real requests with the production formatter and chat template.
   `declare_prefix` caps the common token prefix by the fixed instructions before
   the MAP and rounds down to a multiple of 64. Never pad or rewrite the prompt.
2. Freeze the token IDs, their SHA, source/model/dataset identities, cache mode,
   patch SHA, startup arguments, and worker/concurrency counts in the runtime spec.
3. Mount `fixed_prefix_cache.py` as
   `sglang/srt/mem_cache/pacman_fixed_prefix_cache.py`, and append to the pinned
   `mamba_radix_cache.py`:

   ```python
   from .pacman_fixed_prefix_cache import install
   install(MambaRadixCache)
   ```

4. For the fixed arm set `PACMAN_FIXED_PREFIX_FILE` to the frozen prefix JSON.
   Enable deterministic inference, use FP32 Mamba SSM, disable chunked prefill,
   and set `--max-running-requests 32` (64 requires its own frozen acceptance).
   Radix remains enabled underneath the guard, which permits exactly one node.
   For the off arm unset this variable and use `--disable-radix-cache`.
5. Set `PACMAN_CACHE_CONTRACT` for the driver/workers. Call
   `prefix_warmup.warm_router(args, weight_version)` at the synchronized preparation
   boundary before dispatching initialization, evaluation, or training games,
   and again after each weight/cache change. It warms every engine and verifies
   the exact hit length and policy version with a real prompt canary.

The guard rejects partial/foreign matches and non-fixed insertions. Only an
exact-length warmup may donate its Mamba state. Ordinary requests free suffix
KV and transient states through the upstream no-insert path. Key slicing
preserves SGLang's native token-array representation.

Acceptance requires the real repeated probability replay and paired real
optimizer updates. CPU tests or a warmed cache alone do not establish adoption.
The bounded experiment harness and receipts live in
`reports/slime-migration-20260928/round2/rollout-throughput-prefix-cache` in the
parent workspace. Existing long-training snapshots retain their original mode.
