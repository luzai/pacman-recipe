# Visual-grounding SFT (D5 + language LoRA A)

Teaches Qwen3.5-9B to read Pacman screenshots of any maze up to 25x21 cells before image RL:
stage 1 trains the vision encoder + merger (language frozen), stage 2 adds a small language LoRA.
The result initializes the image RL policy (`observation_mode: image`,
`vision_image_contract: cell-2x-bicubic-v1`).

Everything is regenerated from the game; no screenshots are stored in git. A rebuilt dataset is
not byte-identical to the original D5 data (its earlier versions were assembled over several
rounds), but it follows the same layouts, labeling rules, question mix and sampling.

## Image contract

Screenshots are upscaled exactly 2x (PIL bicubic, 32 px per 16 px cell) before the processor,
which then passes them through (`min_pixels` 4096, `max_pixels` 537600): one visual token per
board cell for any board up to 25x21. Training, evaluation and the RL rollout/SGLang processor
all call `pacman_recipe.level1.vision_prompt` (see `docs/vision-image-resolution.md`).

## Prerequisites

- The `maapacman-rl` environment (torch, transformers 5.12, peft, pygame) and a checkout of
  `pacman-python` (the game). Run every command from the repository root.
- Base model: a complete local Qwen3.5-9B HF checkpoint.
- GPUs: 4 for stage 1 (FSDP2, about 22 GB each), 1 for stage 2 and for evaluation.

```bash
export GAME=/path/to/pacman-python BASE=/path/to/Qwen3.5-9B WORK=/path/to/new/vg-run
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Steps

| Step | Command | Time |
|---|---|---|
| 1. Data | `python -m vg_sft.make_data --game $GAME --out $WORK/data` | ~10 min, 48 CPU workers |
| 2. Stage-1 plan | `python -m vg_sft.make_plan --dataset $WORK/data --updates 6000 --out $WORK/plan-vision.json` | seconds |
| 3. Stage 1: vision + merger | `torchrun --nproc_per_node 4 -m vg_sft.train_vision --model $BASE --dataset $WORK/data --plan $WORK/plan-vision.json --out $WORK/vision` | ~4.5 h on 4 H100 |
| 4. Stage-2 plan | `python -m vg_sft.make_plan --dataset $WORK/data --updates 1000 --seed 20261010 --out $WORK/plan-lora.json` | seconds |
| 5. Stage 2: language LoRA | `python -m vg_sft.train_lora --model $WORK/vision/update-6000 --dataset $WORK/data --plan $WORK/plan-lora.json --out $WORK/lora` | ~2 h on 1 H100 |
| 6. Merge + export check | `python -m vg_sft.merge --base $WORK/vision/update-6000 --adapter $WORK/lora/update-1000 --out $WORK/merged` | ~5 min, CPU |
| 7. Evaluate | `python -m vg_sft.evaluate --model $WORK/merged --dataset $WORK/data --sets test-family test-size test-stream --raw --out $WORK/eval` then `python -m vg_sft.score --dataset $WORK/data --evals $WORK/eval --sets test-family test-size test-stream` | ~2-3 h on 1 GPU (split sets across GPUs to go faster) |

Step 1 writes `train.jsonl` / `train-frames.jsonl` / `validation.jsonl`, three test files and
`manifest.json` (row and frame counts, sizes, hard-frame categories, file and code hashes).
Use the final checkpoints (`update-6000`, `update-1000`); validation CE does not track
cross-family accuracy, so it is not used to pick checkpoints.

### Test files

- `test-family`: native levels 3 and 11 (never trained) and unseen procedural 25x21 layouts.
- `test-size`: unseen procedural layouts of 18 sizes from 11x11 to 25x21.
- `test-stream`: unfiltered frames of the shipped level 1 (map transcription only), the closest
  match to RL screenshots.

### Optional checks before RL

- `python -m vg_sft.decision_drift collect --game $GAME --slime /path/to/slime --model $BASE --out $WORK/decisions`
  then `drift` for the stage-1 and merged models and `compare`: how far the option-code choice
  moved on real image decision prompts. Reference: merged vs stage 1 top-1 agreement 96%,
  KL 0.011; allowed-code probability 0.991, entropy over the allowed codes 0.92.
- Serve the merged model with the pinned SGLang image and the `slime_pacman.sglang_processors`
  package (`PACMAN_VISION_IMAGE_CONTRACT=cell-2x-bicubic-v1`); prompt tokens must equal the
  training-side processor and option-code argmax must match an HF bf16 forward.

## Reference results (original D5 + LoRA A, 2026-10-11)

Exact full-map transcriptions with the schema grammar:

| Set | D4b (25x21 only) | D5 (stage 1) | D5 + LoRA A |
|---|---:|---:|---:|
| Level-1 screenshot stream | 256/531 | 401/531 | 427/531 |
| Unseen family (native 11 etc.) | 36/64 | 50/64 | 51/64 |
| Unseen sizes, 11x11 to 25x21 | 133/216 | 182/216 | 189/216 |

A rebuilt run is expected to land close to these, not on them exactly.

## Known limits

- Layouts whose top row is a wall touching both image edges above empty edge cells (native
  level-3 family) are misread: an extra column in row 1 and walls in the empty edge cells. Level 1
  and the procedural layouts are not affected. Fix (not applied): train on layouts that break the
  closed-frame pattern.
- Overlapping sprites and cells under the score/lives bar are the hardest frames.

## Design choices that matter

- Language frozen in stage 1: full-model SFT memorizes layouts and fails on unseen families.
- Map answers separate cells with single spaces; otherwise the tokenizer merges runs and the
  model has to count run lengths.
- Labels follow pixels: a cell is `?` only where something is actually drawn over it.
- Stage 2 trains on VQA only. Mixing in decision prompts labeled with the model's own choice
  collapsed the option-code distribution (entropy 0.85 to 0.06).
