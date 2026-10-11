"""Shared model I/O for VG SFT. Images follow the recipe's cell contract (one visual token per
board cell, any board up to 25x21): the same pacman_recipe.level1.vision_prompt helpers that RL
rollouts and the SGLang processor use, so training, evaluation and serving see identical inputs."""
import json
from pathlib import Path

import numpy as np

from pacman_recipe.level1.vision_prompt import (VISION_CELL_CONTRACT, VISION_CELL_MAX_PIXELS, VISION_CELL_MIN_PIXELS,
                                                 configure_image_processor, vision_model_image)
from vg_sft.labels import SYSTEM

CONTRACT = VISION_CELL_CONTRACT
MIN_PIXELS, MAX_PIXELS = VISION_CELL_MIN_PIXELS, VISION_CELL_MAX_PIXELS


def read_rows(path, limit=0):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_processor(model_path):
    from transformers import AutoProcessor
    return configure_image_processor(AutoProcessor.from_pretrained(model_path, local_files_only=True), CONTRACT)


def load_model(path, device='cuda:0'):
    import torch
    from transformers import AutoModelForImageTextToText
    # cuDNN SDPA fails to build a vision attention plan on these GPUs; keep the override everywhere.
    torch.backends.cuda.enable_cudnn_sdp(False)
    return AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                       attn_implementation='sdpa', device_map={'': device})


def prompt_text(processor, question):
    return processor.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM},
         {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def model_image(image):
    """The screenshot as the model sees it: exactly 2x bicubic (32 px per cell)."""
    from PIL import Image
    return Image.fromarray(vision_model_image(np.asarray(image.convert('RGB')), CONTRACT))


def images_for(root, rows):
    from PIL import Image
    return [model_image(Image.open(Path(root) / r['image'])) for r in rows]


def make_train_batch(processor, root, rows):
    """Prompt+answer tokens with the prompt (including the expanded image tokens) masked out."""
    import torch
    processor.tokenizer.padding_side = 'right'
    prompts = [prompt_text(processor, r['question']) for r in rows]
    answers = [r['answer'] for r in rows]
    texts = [prompt + answer + processor.tokenizer.eos_token for prompt, answer in zip(prompts, answers)]
    images = images_for(root, rows)
    batch = processor(text=texts, images=images, return_tensors='pt', padding=True)
    prefixes = processor(text=prompts, images=images, return_tensors='pt', padding=True)
    labels = batch['input_ids'].clone()
    for i, n in enumerate(prefixes['attention_mask'].sum(dim=1).tolist()):
        assert torch.equal(batch['input_ids'][i, :n], prefixes['input_ids'][i, :n]), 'Non-prefix chat template'
        labels[i, :n] = -100
    labels[batch['attention_mask'] == 0] = -100
    assert bool((labels != -100).any(dim=1).all())
    keep = int(batch['input_ids'].shape[1] - min(prefixes['attention_mask'].sum(dim=1).tolist()) + 1)
    return batch, labels, keep, answers


def move_batch(batch, device='cuda:0'):
    return {k: v.to(device) for k, v in batch.items()}


def completion_loss(model, batch, labels, keep):
    import torch.nn.functional as F
    logits = model(**batch, use_cache=False, logits_to_keep=keep).logits[:, :-1, :].float()
    targets = labels[:, -keep+1:]
    assert logits.shape[:2] == targets.shape
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def lr_factor(update, total):
    """10% linear warmup, then cosine to 0.1x."""
    import math
    warmup = max(1, math.ceil(total*0.1))
    return update/warmup if update <= warmup else 0.1+0.9*0.5*(1+math.cos(math.pi*(update-warmup)/max(1, total-warmup)))
