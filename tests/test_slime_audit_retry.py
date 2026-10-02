import asyncio
import json

import pytest

from pacman_recipe.level1.trajectories import TrajectoryAuditError
from slime_pacman import rollout


class FakeRunner:
    calls = 0

    def __init__(self, record, **kwargs):
        self.record = record

    async def collect(self, *, empty_weight_version):
        FakeRunner.calls += 1
        if FakeRunner.calls <= FakeRunner.fail_times:
            raise TrajectoryAuditError("trajectory lives do not reconcile (step 3)", {"trajectory": [{"step": 3}]})
        return rollout.EpisodeResult(1.0, [], "v", None, "all_normal_pellets")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "EpisodeRunner", FakeRunner)
    monkeypatch.setattr(rollout, "SGLangGenerator", lambda **kwargs: None)
    monkeypatch.setattr(rollout, "pacman_python_root", lambda: None)
    monkeypatch.setenv("PACMAN_RUN_DIR", str(tmp_path))
    FakeRunner.calls = 0
    return tmp_path


def collect():
    record = {"id": "restart-x", "environment": {"seed": 7}}
    return asyncio.run(rollout._collect_episode(
        record, endpoint="http://x", tokenizer=None, processor=None,
        config=type("C", (), {"max_input_tokens": 2048})(), expected_sources=None, version="v"))


def test_audit_failure_is_dumped_and_episode_replayed(fake):
    FakeRunner.fail_times = 1
    episode = collect()
    assert episode.reward == 1.0 and FakeRunner.calls == 2
    (dump,) = (fake / "audit-failures").glob("*.json")
    body = json.loads(dump.read_text())
    assert body["record_id"] == "restart-x" and body["seed"] == 7 and "step 3" in body["error"]
    assert body["payload"]["trajectory"] == [{"step": 3}]
    assert rollout.audit_failure_count() == 1


def test_repeated_audit_failures_still_raise(fake):
    FakeRunner.fail_times = rollout.AUDIT_RETRIES + 1
    with pytest.raises(TrajectoryAuditError):
        collect()
    assert rollout.audit_failure_count() == rollout.AUDIT_RETRIES + 1
