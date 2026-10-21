import json
import sys
from types import SimpleNamespace

import pytest

from slime_pacman.config import PacmanConfig
from slime_pacman.launch import build_environment
from slime_pacman.tracking import metric_events, safe_argv, upload, run_logged


def test_defaults_and_explicit_disable_reach_workers(tmp_path, monkeypatch):
    monkeypatch.delenv("PACMAN_EXPERIMENTAL_CLIP_COV", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text("schema: pacman-slime-config-v1\n")
    cfg = PacmanConfig()
    assert cfg.episode_life_mode == "original_three_lives"
    assert (cfg.success_life_bonus, cfg.success_speed_bonus) == (.1, .05)
    assert cfg.swanlab_mode == "auto"
    args = dict(slime_root=tmp_path, config=config, run_dir=tmp_path / "run")
    assert build_environment(**args)["PACMAN_EXPERIMENTAL_CLIP_COV"] == "0.002,1.0,5.0"
    config.write_text("schema: pacman-slime-config-v1\nclip_cov: false\n")
    assert build_environment(**args)["PACMAN_EXPERIMENTAL_CLIP_COV"] == "off"
    monkeypatch.setenv("PACMAN_EXPERIMENTAL_CLIP_COV", "0.01,1,5")
    assert build_environment(**args)["PACMAN_EXPERIMENTAL_CLIP_COV"] == "0.01,1,5"


def test_metrics_and_secret_redaction():
    events = list(metric_events("step 9: {'train/masked_entropy': 0.03, 'train/lr': 5e-7}"))
    assert events[0]["source_step"] == 9
    assert events[0]["metrics"]["train/masked_entropy"] == .03
    assert safe_argv(["--wandb-key", "secret-value", "--api-key=abc", "--lr", "5e-7"]) == [
        "--wandb-key", "[REDACTED]", "--api-key=[REDACTED]", "--lr", "5e-7"]


def prepare(root):
    (root / "config.json").write_text(json.dumps({"config": {"swanlab_project": "test"}}))
    (root / "metrics.jsonl").write_text(json.dumps({"source_step": 3, "metrics": {"train/loss": .2}}) + "\n")
    return root


def test_cloud_init_failure_falls_back_offline(tmp_path, monkeypatch):
    calls = []
    def init(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "cloud":
            raise ConnectionError("secret must not appear in status")
    monkeypatch.setenv("SWANLAB_API_KEY", "secret")
    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=init, log=lambda *a, **k: None, finish=lambda: None))
    upload(prepare(tmp_path), "auto")
    assert calls == ["cloud", "offline"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "offline"
    assert "secret" not in (tmp_path / "status.json").read_text()


def test_log_failure_preserves_replay_journal(tmp_path, monkeypatch):
    def log(*a, **kw):
        raise ConnectionError("offline")
    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=lambda **k: None, log=log, finish=lambda: None))
    prepare(tmp_path)
    before = (tmp_path / "metrics.jsonl").read_bytes()
    upload(tmp_path, "offline")
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "local_only"
    assert (tmp_path / "metrics.jsonl").read_bytes() == before


def test_training_exit_and_metrics_survive_missing_sdk(tmp_path):
    # Real child process, no Ray or GPU, exercises journaling and return-code propagation.
    code = "print(\"step 0: {'train/loss': 0.5}\", flush=True); raise SystemExit(7)"
    result = run_logged([sys.executable, "-c", code], PacmanConfig(swanlab_mode="offline"),
                        {"PACMAN_RUN_DIR": str(tmp_path)}, ["--load", "/model"])
    assert result == 7
    root = next((tmp_path / "tracking").iterdir())
    assert json.loads((root / "metrics.jsonl").read_text())["metrics"]["train/loss"] == .5
    assert json.loads((root / "DONE.json").read_text())["exit_code"] == 7


def test_uploader_start_failure_does_not_prevent_training(tmp_path, monkeypatch):
    import subprocess
    real_popen = subprocess.Popen
    def popen(command, **kwargs):
        if "slime_pacman.tracking" in command:
            raise OSError("cannot start uploader")
        return real_popen(command, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", popen)
    assert run_logged([sys.executable, "-c", "print('training completed')"], PacmanConfig(),
                      {"PACMAN_RUN_DIR": str(tmp_path)}, []) == 0
    root = next((tmp_path / "tracking").iterdir())
    assert json.loads((root / "status.json").read_text())["reason"] == "uploader_start_failed"


@pytest.mark.parametrize("kwargs", [{"clip_cov": "false"}, {"clip_cov_ratio": 0}, {"clip_cov_low": 5}, {"swanlab_mode": "bogus"}])
def test_invalid_tracking_config(kwargs):
    with pytest.raises(ValueError):
        PacmanConfig(**kwargs)
