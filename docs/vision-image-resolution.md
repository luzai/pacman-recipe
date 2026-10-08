# VLM screenshot resolution contract

`VISION_IMAGE_CONTRACT = qwen-min-pixels-537600` (`pacman_recipe/level1/vision_prompt.py`).

Qwen3.5 merges 2x2 vision patches of 16 px, so one merged token covers 32x32 input
pixels. The 336x400 screenshot has 16 px board cells. With the checkpoint default
(`shortest_edge=65536`) the image stays unscaled: grid `[1,24,20]`, 120 merged tokens,
one token per ~2x2 cells. With `min_pixels=537600` it is resized 2x to 672x800: grid
`[1,50,42]`, 525 merged tokens, exactly one per board cell. Visual-grounding SFT
(`reports/.../visual-grounding-*`) is trained at 537600, so VLM RL must use the same value.

## Where it is applied

- Training inputs: `slime_pacman/rollout.py` wraps slime's `load_processor` with
  `configure_image_processor`, so rollout `input_ids` and `pixel_values` sent to
  Megatron use 537600.
- SGLang inference: `slime_pacman/sglang_processors/qwen35.py` calls the same helper on
  the HF processor before the base class is built. It rejects `min_pixels`, `max_pixels`
  or `size` in `--mm-process-config` image settings: transformers 5.12.1 silently
  ignores those `images_kwargs` (verified: they left the grid at `[1,24,20]`).
- `slime_pacman/preflight.py check_model` asserts a 336x400 image gives `[[1,50,42]]`;
  both prompt budget scripts measure at this resolution.

## Verification (2026-10-07)

- `tests/test_slime_processor.py`, `tests/test_vision_prompt_layout.py`: 16 passed locally.
- Runtime image `29399fd0a593` (transformers 5.12.1, pinned SGLang), CPU only: the real
  `PacmanQwen35Processor.process_mm_data` and the training-side processor both produced
  grid `[[1,50,42]]`, 540 input tokens, identical `pixel_values` and `input_ids`.

## Live server check (2026-10-07)

Real SGLang server (H100_2_4 GPU0, base model, mRoPE and ragged-logprob patches,
`slime_pacman.sglang_processors`) on 16 real screenshots with the latest Edward prompt:
server `prompt_tokens` equalled the training-side input length (1673) for all 16;
argmax option agreed 16/16 with an HF bf16 forward on the training-side inputs; on the
support-normalized temperature-0.7 distribution the max probability difference was
0.033 mean / 0.063 max, KL(HF||SGLang) 0.0055 mean / 0.012 max. Repeating the same
SGLang request (non-deterministic mode) already moved log-probs by up to 0.21. Worst-case
Edward prompt budget at this resolution: 1673 / 2048 (risk fallback 1639). Receipt:
`reports/slime-migration-20260928/round2/vlm-resolution-live-probe/README.md`.

## Not yet verified / comparability

- Megatron vs SGLang at this resolution with the base model through a full slime 1-update
  run. Earlier v11/v12 runs used a merged grounding checkpoint whose own processor config
  already set `shortest_edge=537600`; their mean `masked_logprob_abs_diff` was about
  0.050-0.056, an existing unresolved VLM mismatch.
- VLM runs that loaded the base model directly used the default 1x resolution
  (`[1,24,20]`); their data, prompt lengths and checkpoints are not comparable with runs
  after this change. Runs on the merged grounding checkpoint were already at 2x.
