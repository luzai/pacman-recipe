import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from slime_pacman import launch


def test_environment_uses_cli_paths_and_preserves_dependencies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["megatron", "extra", "megatron"]))
    monkeypatch.setenv("PACMAN_RUN_DIR", "/stale")
    monkeypatch.setenv("PACMAN_SLIME_CONFIG", "/stale.yaml")
    monkeypatch.setenv("PACMAN_PYTHON_ROOT", "game")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    env = launch.build_environment(slime_root="slime", config="c2.yaml", run_dir="run")
    assert env["SLIME_ROOT"] == str(tmp_path / "slime")
    assert env["PACMAN_SLIME_CONFIG"] == str(tmp_path / "c2.yaml")
    assert env["PACMAN_RUN_DIR"] == str(tmp_path / "run")
    assert env["PACMAN_PYTHON_ROOT"] == str(tmp_path / "game")
    assert env["HF_HUB_OFFLINE"] == "1"
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path / "megatron") in paths and len(paths) == len(set(paths))
    assert all(Path(path).is_absolute() for path in paths)


@pytest.mark.parametrize("initialized", [False, True])
def test_launcher_initializes_before_upstream_and_strips_wrapper_args(monkeypatch, tmp_path, initialized):
    (tmp_path / "train.py").write_text("# fake upstream")
    (tmp_path / "c2.yaml").write_text("# fake config")
    events = []
    ray = SimpleNamespace(
        is_initialized=lambda: initialized,
        init=lambda **kw: events.append(("init", kw)),
        shutdown=lambda: events.append(("shutdown",)),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["launch", "--slime-root", str(tmp_path),
        "--config", str(tmp_path / "c2.yaml"), "--run-dir", str(tmp_path / "run"),
        "--", "--spec", "provider", "factory", "--num-rollout", "1"])
    for key in launch.build_environment(slime_root=tmp_path, config=tmp_path / "c2.yaml", run_dir=tmp_path / "run"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    def run(path, run_name):
        assert events[0][0] == "init"
        assert sys.argv == [str(tmp_path / "train.py"), "--spec", "provider", "factory", "--num-rollout", "1"]
        assert run_name == "__main__"
        assert events[0][1]["runtime_env"]["env_vars"]["PACMAN_RUN_DIR"] == str(tmp_path / "run")
        events.append(("run",))
    monkeypatch.setattr(launch.runpy, "run_path", run)
    if initialized:
        with pytest.raises(RuntimeError, match="already initialized"):
            launch.main()
        assert events == []
    else:
        launch.main()
        assert [event[0] for event in events] == ["init", "run", "shutdown"]


def test_command_reports_matching_launcher_environment(monkeypatch, tmp_path, capsys):
    from slime_pacman.command import main
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PACMAN_RUN_DIR", "/stale")
    monkeypatch.setattr(sys, "argv", ["command", "--model", "/model", "--dataset", "/data",
        "--config", str(root / "configs/slime/c2.yaml"), "--run-dir", "new-run"])
    main()
    result = json.loads(capsys.readouterr().out)
    argv, env = result["argv"], result["environment"]
    assert argv[:3] == ["python", "-m", "slime_pacman.launch"]
    assert env["PACMAN_RUN_DIR"] == str(tmp_path / "new-run")
    assert argv[argv.index("--run-dir") + 1] == env["PACMAN_RUN_DIR"]
    assert argv[argv.index("--config") + 1] == env["PACMAN_SLIME_CONFIG"]
    assert argv[argv.index("--slime-root") + 1] == env["SLIME_ROOT"]
    assert argv[argv.index("--save") + 1] == str(tmp_path / "new-run/checkpoints")
