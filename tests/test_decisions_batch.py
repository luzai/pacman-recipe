import json

import pytest

from baselines.decisions_vision.batch import mcnemar, summarize, validate_artifacts
from baselines.decisions_vision.native import native_payload
import hashlib


def test_exact_paired_test_matches_known_discordance():
    assert mcnemar(10, 0) == pytest.approx(.001953125)
    assert mcnemar(0, 0) == 1
    assert mcnemar(2, 2) == 1


def test_summary_keeps_failed_pairs_out_of_score_and_latency(tmp_path):
    rows = [dict(seed=0, mode="primitive", status="live", won=False, score=10,
                 game_steps=2, normal_pellet_clear_rate=.01, terminal_reason="death"),
            dict(seed=0, mode="edward", status="live", won=True, score=20,
                 game_steps=3, normal_pellet_clear_rate=1., terminal_reason="level_complete"),
            dict(seed=1, mode="primitive", status="failed"),
            dict(seed=1, mode="edward", status="live", won=True, score=100,
                 game_steps=5, normal_pellet_clear_rate=1., terminal_reason="level_complete")]
    folder = tmp_path / "seed1/primitive_seed1"
    folder.mkdir(parents=True)
    (folder / "decisions.json").write_text(json.dumps([
        dict(source="live_api", request_seconds=.3),
        dict(source="forced_singleton", request_seconds=None),
        dict(status="request_failed")]))
    report = summarize(tmp_path,dict(seeds=[0,1],observation="ascii",max_steps=512),rows)
    a = report["aggregates"]["primitive"]
    assert a["mean_score"] == 10 and a["failed"] == 1
    assert a["api_successes"] == a["api_failures"] == a["forced_choices"] == 1
    assert a["latency_mean_seconds"] == .3
    assert report["paired_statistics"]["complete_pairs"] == 1
    assert report["paired_statistics"]["mean_paired_score_difference"] == 10
    assert report["paired_statistics"]["win_rate_difference"] == 1


def test_validation_checks_failed_attempt_and_rejects_modified_body(tmp_path):
    messages = [dict(role="user",content="MAP\n# . _ P G\n")]
    body = native_payload(messages,["L","R"])
    record = dict(index=0,choices=["L","R"],status="request_failed",
                  request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest())
    (tmp_path / "decisions.json").write_text(json.dumps([record]))
    (tmp_path / "decision0000_native_messages.json").write_text(json.dumps(messages))
    request_path = tmp_path / "decision0000_request.json"
    request_path.write_text(json.dumps(body))
    assert validate_artifacts(tmp_path,"ascii")["verified_requests"] == 1
    body["input"][0]["content"][0]["text"] = "changed whitespace"
    request_path.write_text(json.dumps(body))
    with pytest.raises(AssertionError):
        validate_artifacts(tmp_path,"ascii")
