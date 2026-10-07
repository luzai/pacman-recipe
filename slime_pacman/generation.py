"""Independent single-image requests, using slime's native vision format."""

from dataclasses import dataclass
import math
import uuid

import torch

from pacman_recipe.level1.image_transport import pil_and_chat_messages
from pacman_recipe.level1.prompts import png_sha256
from .probability import PacmanLogitProcessor


@dataclass
class Decision:
    prompt: str
    input_ids: list[int]
    action_id: int
    completion: str
    allowed_token_ids: list[int]
    behavior_log_prob: float
    weight_version: str
    multimodal_train_inputs: dict
    image_sha256: str | None
    candidate_log_probs: list[float]


def process_request(processor, messages, constraint, max_input_tokens):
    image, chat = pil_and_chat_messages(messages)
    if len(chat) != 2 or [m["role"] for m in chat] != ["system", "user"]:
        raise ValueError("Pacman requires a fresh system/user pair per decision")
    prompt = processor.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    processed = processor(
        text=[prompt], images=[image], return_tensors="pt", truncation=False
    )
    ids = processed["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= max_input_tokens:
        raise ValueError(
            "prompt exceeds budget or has invalid input shape; truncation forbidden"
        )
    if "pixel_values" not in processed or "image_grid_thw" not in processed:
        raise ValueError("processor did not produce Qwen vision tensors")
    # mm_token_type_ids is not a Megatron model input: the Qwen3.5-VL plugin rebuilds
    # MRoPE positions from input_ids and image_grid_thw (upstream slime disables it too).
    mm = {
        k: v
        for k, v in processed.items()
        if k not in {"input_ids", "attention_mask", "mm_token_type_ids"}
    }
    for key, value in mm.items():
        if isinstance(value, torch.Tensor):
            mm[key] = value.cpu()
    # The vision tower casts pixel_values to its bf16 dtype before anything else
    # (slime_plugins/models/qwen3_5_vl.py), so casting here hands it the same tensor
    # and halves the 12.9 MB each decision carries to training.
    mm["pixel_values"] = mm["pixel_values"].to(torch.bfloat16)
    return prompt, ids[0].tolist(), mm, image


def _image_url(messages):
    urls = [
        p["image_url"]["url"]
        for m in messages
        if isinstance(m["content"], list)
        for p in m["content"]
        if p["type"] == "image_url"
    ]
    if len(urls) != 1 or not urls[0].startswith("data:image/png;base64,"):
        raise ValueError("Pacman request requires exactly one inline PNG image")
    return urls[0]


def process_text_request(processor, messages, max_input_tokens):
    if len(messages) != 2 or [m['role'] for m in messages] != ['system', 'user']:
        raise ValueError('Text policy requires a fresh system/user pair')
    chat = []
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            if not content or any(p.get('type') != 'text' for p in content):
                raise ValueError('ASCII policy refuses image or nontext content')
            content = '\n'.join(p['text'] for p in content)
        if not isinstance(content, str):
            raise ValueError('ASCII content must be text')
        chat.append(dict(role=message['role'], content=content))
    tokenizer = getattr(processor, 'tokenizer', processor)
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    processed = tokenizer(prompt, return_tensors='pt', truncation=False)
    ids = processed['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= max_input_tokens:
        raise ValueError('ASCII prompt exceeds budget; truncation forbidden')
    return prompt, ids[0].tolist(), {}, None


class SGLangGenerator:
    def __init__(self, *, processor, endpoint, client, max_input_tokens=2048, observation_mode='image'):
        if observation_mode not in ('image', 'ascii'):
            raise ValueError('Unsupported observation mode')
        self.observation_mode = observation_mode
        self.processor, self.endpoint, self.client = processor, endpoint, client
        self.max_input_tokens = max_input_tokens
        self.serialized_processor = PacmanLogitProcessor.to_str()

    async def __call__(self, messages, constraint):
        import base64

        if self.observation_mode == 'ascii':
            prompt, ids, mm, image = process_text_request(self.processor, messages, self.max_input_tokens)
            image_url = None
        else:
            prompt, ids, mm, image = process_request(self.processor, messages, constraint, self.max_input_tokens)
            image_url = _image_url(messages)
        support = constraint.allowed_token_ids
        body = {
            "rid": uuid.uuid4().hex,
            "text": prompt,
            # Send the episode's own lossless RGB PNG (encode_png) as is. slime's
            # encode_image_for_rollout_engine would re-encode the same pixels at
            # level 6 (~3 ms/decision); SGLang decodes either to identical pixels.
            "image_data": [image_url],
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
                "custom_params": {
                    "pacman_allowed_token_ids": support,
                    "pacman_temperature": 0.7,
                },
            },
            "custom_logit_processor": self.serialized_processor,
            "return_logprob": True,
            "token_ids_logprob": support,
        }
        if image_url is None:
            del body['image_data']
        response = await self.client.post(self.endpoint, json=body)
        response.raise_for_status()
        result = response.json()
        meta = result["meta_info"]
        entries = meta.get("output_token_logprobs")
        candidates = meta.get("output_token_ids_logprobs")
        if not entries or len(entries) != 1 or not candidates or len(candidates) != 1:
            raise ValueError(
                "SGLang must return one sampled token and its complete support log-probs"
            )
        logp, action = float(entries[0][0]), int(entries[0][1])
        constraint.option_for_tokens([action])
        candidate_map = {int(row[1]): float(row[0]) for row in candidates[0]}
        if len(candidates[0]) != len(support) or set(candidate_map) != set(support):
            raise ValueError(
                "SGLang candidate probability support differs from request"
            )
        probabilities = [candidate_map[i] for i in support]
        if (
            not math.isfinite(logp)
            or not all(math.isfinite(p) for p in probabilities)
            or abs(sum(math.exp(p) for p in probabilities) - 1) > 2e-4
            or abs(logp - candidate_map[action]) > 2e-4
        ):
            raise ValueError(
                "SGLang log-probs are not normalized on the sampled support"
            )
        version = (
            "" if meta.get("weight_version") is None else str(meta["weight_version"])
        )
        if not version or int(meta.get("prompt_tokens", -1)) != len(ids):
            raise ValueError(
                "missing weight version or HF/SGLang multimodal token-count mismatch"
            )
        completion = constraint.code_for_option(constraint.option_for_tokens([action]))
        if result.get("text") != completion:
            raise ValueError("SGLang decoded text differs from exact selected token")
        raw = base64.b64decode(image_url.split(",", 1)[1]) if image_url else None
        return Decision(
            prompt,
            ids,
            action,
            completion,
            support,
            logp,
            version,
            mm,
            png_sha256(raw) if raw is not None else None,
            probabilities,
        )
