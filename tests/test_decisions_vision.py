import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from baselines.decisions_vision.client import (
    ACTIONS, MODEL, URL, DecisionError, NoRedirect, decide, load_key, parse_answer,
)


def response(action="L"):
    return {"model": MODEL, "answers": [{"type": "choice", "name": "action",
        "choice": action, "confidence": 0.8, "probabilities": [
            {"value": action, "probability": 1.0}]}], "usage": {"input_tokens": 10}}


def test_image_transport_one_request_no_network():
    calls = []
    png = b"\x89PNG\r\n\x1a\nfixture"
    def fake(req, timeout):
        calls.append(req)
        body = json.loads(req.data)
        assert req.full_url == URL
        assert req.get_method() == "POST"
        assert body["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        assert [x["value"] for x in body["questions"][0]["choices"]] == list(ACTIONS)
        assert timeout == 30
        return io.BytesIO(json.dumps(response()).encode())
    with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
        result = decide(png, "offline-placeholder", opener=fake)
    assert len(calls) == 1
    assert result["action"] == "L"
    assert result["usage"] == {"input_tokens": 10}


@pytest.mark.parametrize("result", [{}, response("X"),
    {"model": MODEL, "answers": [{"type": "refusal", "name": "action"}]}])
def test_invalid_response_never_yields_action(result):
    with pytest.raises(DecisionError):
        parse_answer(result)


def test_http_error_redacted_and_not_retried():
    calls = []
    def failed(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(URL, 401, "sensitive server message", {}, None)
    with pytest.raises(DecisionError, match="HTTP status: 401; no retry"):
        decide(b"\x89PNG\r\n\x1a\nfixture", "secret", opener=failed)
    assert len(calls) == 1


def test_redirect_refused():
    with pytest.raises(DecisionError, match="Redirect refused"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com")


def test_local_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "key.txt"
    path.write_text("offline-placeholder\n")
    assert load_key(path) == "offline-placeholder"
