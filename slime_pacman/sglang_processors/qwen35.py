"""Keep Qwen3.5 image processing on the same CPU path as training inputs."""

import copy

from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


class PacmanQwen35Processor(QwenVLImageProcessor):
    models = [Qwen3_5ForConditionalGeneration]

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        if server_args.disable_fast_image_processor:
            raise ValueError("Pacman requires the checkpoint's fast image processor")
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        # The HF processor has already been constructed. This private copy only
        # prevents the base class from injecting device=cuda on each call; it
        # does not select a different processor or mutate global ServerArgs.
        self.server_args = copy.copy(server_args)
        self.server_args.disable_fast_image_processor = True
        self.image_config = dict(self.image_config)
        device = self.image_config.pop("device", "cpu")
        if str(device) != "cpu":
            raise ValueError("Pacman image processing must use CPU")

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        kwargs["device"] = "cpu"
        return super().process_mm_data(
            input_text, images=images, videos=videos, audios=audios, **kwargs
        )
