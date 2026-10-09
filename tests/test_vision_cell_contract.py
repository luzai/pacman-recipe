"""Opt-in cell contract: 2x bicubic screenshots, one visual token per maze cell up to 25x21."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from pacman_recipe.level1 import vision_prompt as vp
from tests.test_slime_processor import _hf_processor, processor_class  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_default_contract_is_unchanged(monkeypatch):
    monkeypatch.delenv(vp.VISION_CONTRACT_ENV, raising=False)
    assert vp.vision_image_contract() == vp.VISION_IMAGE_CONTRACT == "qwen-min-pixels-537600"
    rgb = np.zeros((400, 336, 3), np.uint8)
    assert vp.vision_model_image(rgb) is rgb
    hf = _hf_processor()
    vp.configure_image_processor(hf)
    assert hf.image_processor.size == {"shortest_edge": 537600, "longest_edge": 16777216}


def test_env_selects_cell_contract_and_rejects_unknown(monkeypatch):
    monkeypatch.setenv(vp.VISION_CONTRACT_ENV, vp.VISION_CELL_CONTRACT)
    assert vp.vision_image_contract() == vp.VISION_CELL_CONTRACT
    hf = _hf_processor()
    vp.configure_image_processor(hf)
    assert hf.image_processor.size == {"shortest_edge": 4096, "longest_edge": 537600}
    monkeypatch.setenv(vp.VISION_CONTRACT_ENV, "cell-3x")
    with pytest.raises(ValueError, match="unknown"):
        vp.vision_image_contract()


@pytest.mark.parametrize("rows,cols", [(25, 21), (11, 11), (7, 9), (21, 15)])
def test_cell_image_is_exact_2x_bicubic(rows, cols):
    rgb = np.random.default_rng(rows * cols).integers(0, 256, (rows * 16, cols * 16, 3), dtype=np.uint8)
    out = vp.vision_model_image(rgb, vp.VISION_CELL_CONTRACT)
    assert out.shape == (rows * 32, cols * 32, 3) and out.dtype == np.uint8
    expected = np.asarray(Image.fromarray(rgb).resize((cols * 32, rows * 32), Image.BICUBIC))
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("shape,match", [
    ((400, 330, 3), "16 px cells"),
    ((26 * 16, 21 * 16, 3), "exceeds"),
    ((21 * 16, 25 * 16, 3), "exceeds"),
])
def test_cell_image_rejects_partial_cells_and_large_mazes(shape, match):
    with pytest.raises(ValueError, match=match):
        vp.vision_model_image(np.zeros(shape, np.uint8), vp.VISION_CELL_CONTRACT)


def test_cell_size_check_catches_screenshot_that_skipped_upscale():
    vp.check_cell_image_size(672, 800)
    with pytest.raises(ValueError, match="32 px"):
        vp.check_cell_image_size(336, 400)


def test_sglang_processor_enforces_upscaled_images(processor_class, monkeypatch):  # noqa: F811
    monkeypatch.setenv(vp.VISION_CONTRACT_ENV, vp.VISION_CELL_CONTRACT)
    args = SimpleNamespace(disable_fast_image_processor=False, mm_process_config={"image": {}})
    hf = _hf_processor()
    server = processor_class(None, args, hf)
    assert hf.image_processor.size == {"shortest_edge": 4096, "longest_edge": 537600}
    server.process_mm_data("x", images=[Image.new("RGB", (672, 800))])
    with pytest.raises(ValueError, match="32 px"):
        server.process_mm_data("x", images=[Image.new("RGB", (336, 400))])


def _config(tmp_path, **overrides):
    raw = yaml.safe_load((ROOT / "configs/slime/c2.yaml").read_text(encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_config_validates_contract_and_launcher_exports_it(tmp_path):
    from slime_pacman.config import load_config
    from slime_pacman.launch import build_environment

    legacy = _config(tmp_path)
    assert load_config(legacy).vision_image_contract == vp.VISION_IMAGE_CONTRACT
    env = build_environment(slime_root=tmp_path, config=legacy, run_dir=tmp_path)
    assert env[vp.VISION_CONTRACT_ENV] == vp.VISION_IMAGE_CONTRACT
    cell = _config(tmp_path, vision_image_contract=vp.VISION_CELL_CONTRACT)
    env = build_environment(slime_root=tmp_path, config=cell, run_dir=tmp_path)
    assert env[vp.VISION_CONTRACT_ENV] == vp.VISION_CELL_CONTRACT
    with pytest.raises(ValueError, match="vision_image_contract"):
        load_config(_config(tmp_path, vision_image_contract="cell-3x"))
