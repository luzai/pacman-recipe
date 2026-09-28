"""Reuse canonical RGB PNG payloads without changing model-visible pixels."""

import base64
from io import BytesIO
from typing import Any

from PIL import Image


def native_image_data(messages, rgb_image):
    """Reuse plain RGB PNGs; retain RGB re-encoding for other image formats.

    Inspect the source header, not the processor output: RGBA, palettes and
    metadata may affect a server's decoding, so those keep the old path.
    """
    urls = [
        item["image_url"]["url"]
        for message in messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item.get("type") == "image_url"
    ]
    if len(urls) != 1:
        raise ValueError("native Pacman request requires exactly one image")
    prefix = "data:image/png;base64,"
    if not urls[0].startswith(prefix):
        raise ValueError("native Pacman vision request requires an inline PNG")
    encoded = urls[0][len(prefix):]
    with Image.open(BytesIO(base64.b64decode(encoded))) as source:
        if source.format == "PNG" and source.mode == "RGB" and not source.info:
            return [encoded]
    # Preserve the previous normalization for non-canonical inputs.
    from areal.utils.image import image2base64

    return image2base64(rgb_image)


def pil_and_chat_messages(
    messages: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    image = None
    chat_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            chat_messages.append(
                {"role": message["role"], "content": content}
            )
            continue
        converted: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                converted.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                if image is not None:
                    raise ValueError("native Pacman request requires exactly one current image")
                url = item["image_url"]["url"]
                prefix = "data:image/png;base64,"
                if not url.startswith(prefix):
                    raise ValueError(
                        "native Pacman vision workflow requires an inline PNG"
                    )
                image = Image.open(
                    BytesIO(base64.b64decode(url[len(prefix) :]))
                ).convert("RGB")
                converted.append({"type": "image", "image": image})
            else:
                raise ValueError(
                    f"unsupported multimodal message item: {item!r}"
                )
        chat_messages.append(
            {"role": message["role"], "content": converted}
        )
    if image is None:
        raise ValueError("native Pacman vision request has no image")
    return image, chat_messages
