"""Fixed official endpoint, one attempt, no redirects; never log credentials."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
KEY_FILE = Path(__file__).with_name("api_key.local.txt")
ACTIONS = {"U": "Move up", "D": "Move down", "L": "Move left",
           "R": "Move right", "S": "Stay still"}
PROMPT = (
    "Play Pacman using only this current screenshot. Pacman is yellow; walls "
    "block movement. Survive first, then collect pellets and power pellets. "
    "Avoid dangerous ghosts; blue vulnerable ghosts can be eaten. "
    "Choose one primitive action, lasting about 16 game logic ticks. "
    "Ghosts also move. Do not move into walls."
)


class DecisionError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DecisionError("Redirect refused")


def load_key(path=KEY_FILE):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            key = Path(path).read_text(encoding="utf-8-sig").strip()
        except OSError:
            raise DecisionError("Missing local API key file or OPENAI_API_KEY") from None
    if not key or any(c.isspace() for c in key):
        raise DecisionError("Invalid credential format")
    return key


def payload(png: bytes):
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Expected PNG image bytes")
    return {"model": MODEL, "input": [{"role": "user", "content": [
        {"type": "input_text", "text": PROMPT},
        {"type": "input_image", "image_url": "data:image/png;base64," +
         base64.b64encode(png).decode("ascii")},
    ]}], "questions": [{"type": "choice", "name": "action",
        "instructions": "Choose the best next primitive action for Pacman.",
        "choices": [{"value": k, "description": v} for k, v in ACTIONS.items()]}]}


def parse_answer(result, choices=None):
    choices = ACTIONS if choices is None else choices
    if not isinstance(result, dict) or result.get("model") != MODEL:
        raise DecisionError("Unexpected response model")
    answers = result.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        raise DecisionError("Expected exactly one answer")
    answer = answers[0]
    if not isinstance(answer, dict) or answer.get("name") != "action":
        raise DecisionError("Unexpected answer name")
    if answer.get("type") == "refusal":
        raise DecisionError("Model refused; no action executed")
    if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str) or answer.get("choice") not in choices:
        raise DecisionError("Invalid action response")
    confidence = answer.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise DecisionError("Invalid confidence")
    # Whitelist fields rather than recording arbitrary server content.
    probabilities = answer.get("probabilities", [])
    if not isinstance(probabilities, list):
        raise DecisionError("Invalid probabilities")
    clean = []
    for item in probabilities:
        if not isinstance(item, dict):
            raise DecisionError("Invalid probability item")
        value, p = item.get("value"), item.get("probability")
        if not isinstance(value, str) or value not in choices or type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise DecisionError("Invalid probability item")
        clean.append({"value": value, "probability": p})
    return {"action": answer["choice"], "confidence": confidence,
            "probabilities": clean}


def decide(png: bytes, key: str, *, timeout=30, opener=None):
    return send_body(payload(png), key, timeout=timeout, opener=opener)


def send_body(body, key, *, timeout=30, opener=None):
    choices = [item["value"] for item in body["questions"][0]["choices"]]
    request = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST")
    open_request = opener or urllib.request.build_opener(NoRedirect()).open
    start = time.perf_counter()
    try:
        with open_request(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise DecisionError(f"HTTP status: {exc.code}; no retry") from None
    except (urllib.error.URLError, OSError, ValueError):
        raise DecisionError("Network or JSON response failure; no retry") from None
    parsed = parse_answer(result, choices)
    usage = result.get("usage", {})
    parsed.update(model=MODEL, request_seconds=time.perf_counter() - start,
                  usage={k: v for k, v in usage.items()
                         if k in ("input_tokens", "output_tokens", "total_tokens")
                         and type(v) is int} if isinstance(usage, dict) else {})
    return parsed
