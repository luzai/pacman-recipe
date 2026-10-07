"""Versioned prefix-friendly layout shared by vision decision and VQA clients."""

VISION_LAYOUT_VERSION = "fixed-image-dynamic-v2"

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
