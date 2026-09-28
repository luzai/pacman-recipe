import base64
import os
import statistics
import sys
import time
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from pacman_recipe.level1.image_transport import native_image_data
from pacman_recipe.level1.prompts import build_image_messages, encode_png


def old_encode(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return [base64.b64encode(buffer.getvalue()).decode("ascii")]


def decode(data):
    return Image.open(BytesIO(base64.b64decode(data[0]))).convert("RGB")


@pytest.mark.parametrize("seed", range(5))
def test_rgb_reuse_exact_bytes_and_pixels(seed, monkeypatch):
    pixels = np.random.default_rng(seed).integers(0, 256, (400, 336, 3), dtype=np.uint8)
    messages = build_image_messages(encode_png(pixels))
    image = Image.fromarray(pixels)
    expected = old_encode(image)
    def forbidden(*args, **kwargs):
        raise AssertionError("RGB fast path must not encode PNG again")
    monkeypatch.setattr(Image.Image, "save", forbidden)
    actual = native_image_data(messages, image)
    # PNG compression differs; the reused bytes must match the source,
    # while decoded pixels must match the previous RGB re-encoding.
    assert actual == [messages[1]["content"][1]["image_url"]["url"].split(",", 1)[1]]
    np.testing.assert_array_equal(np.asarray(decode(actual)), np.asarray(decode(expected)))
    np.testing.assert_array_equal(np.asarray(decode(actual)), pixels)


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "metadata"])
def test_noncanonical_keeps_old_normalization(mode, monkeypatch):
    source = Image.new("RGB", (13, 17), (10, 20, 30))
    if mode != "metadata":
        source = source.convert(mode)
    buffer = BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("description", "test")
    source.save(buffer, format="PNG", **({"pnginfo": metadata} if mode == "metadata" else {}))
    messages = build_image_messages(buffer.getvalue())
    rgb = Image.open(BytesIO(buffer.getvalue())).convert("RGB")
    calls = []
    def fallback(image):
        calls.append(image)
        return old_encode(image)
    monkeypatch.setitem(sys.modules, "areal.utils.image", SimpleNamespace(image2base64=fallback))
    assert native_image_data(messages, rgb) == old_encode(rgb)
    assert calls == [rgb]


def test_real_processor_equivalence():
    model = os.environ.get("PACMAN_TEST_PROCESSOR_PATH")
    if not model:
        pytest.skip("requires local production processor, no model weights loaded")
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    pixels = np.random.default_rng(17).integers(0, 256, (400, 336, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    messages = build_image_messages(encode_png(pixels))
    chat = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Choose U D L R"}]}]
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    before = processor(text=[text], images=[decode(old_encode(image))], return_tensors="pt")
    after = processor(text=[text], images=[decode(native_image_data(messages, image))], return_tensors="pt")
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_transport_microbenchmark():
    timings = {}
    for name, pixels in [
        ("flat", np.zeros((400, 336, 3), dtype=np.uint8)),
        ("noise", np.random.default_rng(0).integers(0, 256, (400, 336, 3), dtype=np.uint8)),
    ]:
        image = Image.fromarray(pixels)
        messages = build_image_messages(encode_png(pixels))
        for label, operation in [("old", lambda: old_encode(image)), ("reuse", lambda: native_image_data(messages, image))]:
            samples = []
            for _ in range(30):
                start = time.perf_counter()
                operation()
                samples.append((time.perf_counter() - start) * 1000)
            timings[f"{name}_{label}_median_ms"] = statistics.median(samples)
    print(timings)
