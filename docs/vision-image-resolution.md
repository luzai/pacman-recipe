# VLM screenshot resolution contract

`VISION_IMAGE_CONTRACT = qwen-min-pixels-537600` (`pacman_recipe/level1/vision_prompt.py`).

Qwen3.5 merges 2x2 vision patches of 16 px, so one merged token covers 32x32 input
pixels. The 336x400 screenshot has 16 px board cells. With the checkpoint default
(`shortest_edge=65536`) the image stays unscaled: grid `[1,24,20]`, 120 merged tokens,
one token per ~2x2 cells. With `min_pixels=537600` it is resized 2x to 672x800: grid
`[1,50,42]`, 525 merged tokens, exactly one per board cell. Visual-grounding SFT
(`reports/.../visual-grounding-*`) is trained at 537600, so VLM RL must use the same value.

## Opt-in cell contract for mazes up to 25x21 (2026-10-09)

`vision_image_contract: cell-2x-bicubic-v1` in the slime config (default stays
`qwen-min-pixels-537600`). The single `min_pixels` above only aligns the 336x400 level-1
screenshot; an 11x11, 13x13 or 15x15 maze frame is upscaled to a `[1,46,46]` grid that does
not line up with its cells. Under the cell contract:

- `vision_model_image` upscales the RGB frame exactly 2x with PIL bicubic (32 px per cell)
  before PNG encoding, so the training processor and SGLang receive the same image.
- `configure_image_processor` sets `min_pixels=4096`, `max_pixels=537600`; a 2x board passes
  through unresized, giving `[1, 2*rows, 2*cols]`. Mazes larger than 25 rows or 21 columns are
  rejected.
- `slime_pacman.launch` exports `PACMAN_VISION_IMAGE_CONTRACT` from the config; rollout
  workers and the SGLang processor read it. The SGLang processor rejects an image that is not
  32 px per cell (a screenshot that skipped the upscale) with HTTP 400.
- Preflight checks 25x21, 11x11 and 15x21 boards under this contract.

Verified in the runtime image with D4b update-3000 (H100_2_1, 2026-10-09): on 16 real
level-1 screenshots the cell contract gives bit-identical `pixel_values` and `input_ids` to
the 537600 contract, so checkpoints trained under 537600 see unchanged inputs; live SGLang
matched training-side prompt tokens 19/19 (including 11/13/15 mazes at 1269/1317/1373
tokens), option argmax 19/19 vs HF bf16, max support-normalized probability difference
0.045, mean KL 0.0030; an unscaled screenshot was rejected. Receipt:
`reports/slime-migration-20260928/round2/vg-cell-contract-probe/README.md`.
An SFT export for this contract must write `shortest_edge=4096`, `longest_edge=537600`;
`scripts/level1/report/check_hf_export.py --min-pixels 4096` checks the former.

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
