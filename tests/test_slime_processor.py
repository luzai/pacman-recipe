import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def processor_class(monkeypatch):
    class Base:
        def __init__(self, config, args, processor):
            self.server_args = args
            self.image_config = args.mm_process_config.get("image", {})
            self.processor = processor

        def process_mm_data(self, text, **kwargs):
            if not self.server_args.disable_fast_image_processor:
                kwargs["device"] = "cuda:0"
            return self.processor, kwargs

    model = ModuleType("sglang.srt.models.qwen3_5")
    model.Qwen3_5ForConditionalGeneration = type("Qwen3_5ForConditionalGeneration", (), {})
    base = ModuleType("sglang.srt.multimodal.processors.qwen_vl")
    base.QwenVLImageProcessor = Base
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    name = "slime_pacman.sglang_processors.qwen35"
    sys.modules.pop(name, None)
    cls = importlib.import_module(name).PacmanQwen35Processor
    yield cls
    sys.modules.pop(name, None)


def test_cpu_processor_preserves_implementation_and_global_args(processor_class):
    args = SimpleNamespace(disable_fast_image_processor=False,
                           mm_process_config={"image": {"device": "cpu"}})
    original = _hf_processor()
    processor = processor_class(None, args, original)
    selected, kwargs = processor.process_mm_data("prompt", images=["image"], device="cuda:0")
    assert selected is original and kwargs["device"] == "cpu"
    assert args.disable_fast_image_processor is False
    assert args.mm_process_config["image"]["device"] == "cpu"
    assert processor.image_config == {}
    assert original.image_processor.size["shortest_edge"] == 537600


@pytest.mark.parametrize("slow,device", [(True, "cpu"), (False, "cuda:0")])
def test_cpu_processor_rejects_conflicting_settings(processor_class, slow, device):
    args = SimpleNamespace(disable_fast_image_processor=slow,
                           mm_process_config={"image": {"device": device}})
    with pytest.raises(ValueError):
        processor_class(None, args, _hf_processor())


def test_command_exports_processor_registration(monkeypatch, capsys):
    from slime_pacman.command import main, PROCESSOR_ENV

    monkeypatch.setattr(sys, "argv", ["command", "--model", "/model", "--dataset", "/data",
                                     "--run-dir", "/run", "--updates", "1"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert PROCESSOR_ENV.items() <= result["environment"].items()
    assert "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=slime_pacman.sglang_processors" in result["bash"]


def test_runtime_rejects_missing_processor_registration(monkeypatch):
    from slime_pacman.preflight import check_runtime

    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    with pytest.raises(ValueError, match="SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"):
        check_runtime()


def _hf_processor():
    return SimpleNamespace(image_processor=SimpleNamespace(
        size={"shortest_edge": 65536, "longest_edge": 16777216}))


def test_sglang_processor_pins_training_image_resolution(processor_class):
    args = SimpleNamespace(disable_fast_image_processor=False, mm_process_config={"image": {}})
    hf = _hf_processor()
    processor_class(None, args, hf)
    assert hf.image_processor.size == {"shortest_edge": 537600, "longest_edge": 16777216}
    assert args.mm_process_config["image"] == {}


@pytest.mark.parametrize("key", ["min_pixels", "max_pixels", "size"])
def test_sglang_processor_rejects_ignored_resolution_kwargs(processor_class, key):
    args = SimpleNamespace(disable_fast_image_processor=False,
                           mm_process_config={"image": {key: 537600}})
    with pytest.raises(ValueError, match="resolution"):
        processor_class(None, args, _hf_processor())


def test_configure_image_processor_sets_one_token_per_cell_area():
    from pacman_recipe.level1.vision_prompt import configure_image_processor

    processor = _hf_processor()
    assert configure_image_processor(processor) is processor
    assert processor.image_processor.size == {"shortest_edge": 537600, "longest_edge": 16777216}
