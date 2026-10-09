"""Versioned prefix-friendly layout shared by vision decision and VQA clients."""

import os

VISION_LAYOUT_VERSION = "fixed-image-dynamic-v2"

# One visual token per 16px board cell: Qwen3.5 merges 2x2 patches of 16px, so the
# 336x400 screenshot must be resized 2x (672x800 = 537600 px). The default
# preprocessor minimum (65536) leaves it unscaled, one token per 2x2 cells.
# Training (HF processor) and SGLang inference must both use this value.
VISION_IMAGE_MIN_PIXELS = 537600
VISION_IMAGE_MAX_PIXELS = 16777216
VISION_IMAGE_CONTRACT = f"qwen-min-pixels-{VISION_IMAGE_MIN_PIXELS}"

# Opt-in contract for mazes of any size up to 25x21 cells: the screenshot is upscaled
# exactly 2x (PIL bicubic, 32 px per cell) before it reaches either processor, and the
# processor then passes it through unresized, so the grid is [1, 2*rows, 2*cols]. At
# 25x21 the pixel_values are bit-identical to the min-pixels contract (checked in the
# runtime image), so models trained under it stay valid. The legacy single min_pixels
# cannot do this: an 11x11 maze would be upscaled to a 46x46 grid, not 22x22.
VISION_CELL_CONTRACT = "cell-2x-bicubic-v1"
VISION_CONTRACT_ENV = "PACMAN_VISION_IMAGE_CONTRACT"
VISION_CELL_PIXELS = 16
VISION_CELL_MAX_ROWS, VISION_CELL_MAX_COLS = 25, 21
VISION_CELL_MIN_PIXELS = 4 * 32 * 32
VISION_CELL_MAX_PIXELS = VISION_CELL_MAX_ROWS * 32 * VISION_CELL_MAX_COLS * 32


def vision_image_contract():
    """The active contract; the launcher exports it so SGLang's processor agrees."""
    contract = os.environ.get(VISION_CONTRACT_ENV) or VISION_IMAGE_CONTRACT
    if contract not in (VISION_IMAGE_CONTRACT, VISION_CELL_CONTRACT):
        raise ValueError(f"unknown {VISION_CONTRACT_ENV}={contract!r}")
    return contract


def configure_image_processor(processor, contract=None):
    """Pin the Pacman image resolution on an HF processor; returns the processor."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("processor has no image processor")
    if (contract or vision_image_contract()) == VISION_CELL_CONTRACT:
        low, high = VISION_CELL_MIN_PIXELS, VISION_CELL_MAX_PIXELS
    else:
        low, high = VISION_IMAGE_MIN_PIXELS, max(VISION_IMAGE_MIN_PIXELS, VISION_IMAGE_MAX_PIXELS)
    size = dict(image_processor.size)
    size["shortest_edge"] = low
    size["longest_edge"] = high
    image_processor.size = size
    return processor


def check_cell_image_size(width, height):
    """Under the cell contract every model image is a 2x board: 32 px per cell, <=25x21."""
    if width % 32 or height % 32:
        raise ValueError(f"cell-contract image {width}x{height} is not 32 px per cell; upscale it first")
    if height // 32 > VISION_CELL_MAX_ROWS or width // 32 > VISION_CELL_MAX_COLS:
        raise ValueError(f"maze {height // 32}x{width // 32} exceeds {VISION_CELL_MAX_ROWS}x{VISION_CELL_MAX_COLS} cells")


def vision_model_image(rgb, contract=None):
    """The RGB array the model sees: unchanged, or 2x bicubic under the cell contract."""
    if (contract or vision_image_contract()) != VISION_CELL_CONTRACT:
        return rgb
    import numpy as np
    from PIL import Image

    height, width = rgb.shape[:2]
    if width % VISION_CELL_PIXELS or height % VISION_CELL_PIXELS:
        raise ValueError(f"screenshot {width}x{height} is not a whole number of 16 px cells")
    check_cell_image_size(2 * width, 2 * height)
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize((2 * width, 2 * height), Image.BICUBIC))

GROUNDING_PROMPT_VERSION = "visual-grounding-sections-v1"
GROUNDING_SYSTEM = (
    "You read visible facts from the current Pacman screenshot. "
    "Do not choose tactical actions or infer hidden simulator state."
)
GROUNDING_FIXED_TEXT = (
    "[VISUAL GROUNDING]\n"
    "Use only the current screenshot. Grid coordinates are [row,column], "
    "starting at 0 at the board's top-left cell; rows increase downward and "
    "columns increase rightward. Coordinates refer to board cells, not pixels. "
    "Report only visible facts; do not infer future motion, remaining edible "
    "time, or hidden objects. Follow the question's conventions for absent or "
    "uncertain objects; never invent a position.\n\n[CURRENT IMAGE]\n"
)


def vision_content(image_block, *, fixed_text="", dynamic_text=""):
    """Keep stable instructions before the image and per-request text after it."""
    if image_block.get("type") not in ("image", "image_url"):
        raise ValueError("vision_content requires one image block")
    content = []
    if fixed_text:
        content.append({"type": "text", "text": fixed_text})
    content.append(dict(image_block))
    if dynamic_text:
        content.append({"type": "text", "text": dynamic_text})
    return content


def grounding_messages(system, question, *, fixed_text=None, answer_format=None):
    """Processor-ready VQA messages; callers supply the actual image separately."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty visual grounding question is required")
    output = answer_format or "Return only the JSON requested by the question; no explanation or Markdown."
    return [{"role": "system", "content": system or GROUNDING_SYSTEM},
            {"role": "user", "content": vision_content(
                {"type": "image"},
                fixed_text=GROUNDING_FIXED_TEXT if fixed_text is None else fixed_text,
                dynamic_text=f"[QUESTION]\n{question}\n\n[OUTPUT]\n{output}")}]
