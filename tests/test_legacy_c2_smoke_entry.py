"""Test wrapper routing with a fake launcher; never start training or use GPUs."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAME = "run_c2_legacy_iter25_gs24_smoke.sh"


def test_legacy_entry(tmp_path):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash or not Path(bash).exists():
        pytest.skip("bash unavailable")
    scripts = tmp_path / "scripts/level1/train"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/pacman_paths.sh", tmp_path / "scripts/pacman_paths.sh")
    wrapper = scripts / NAME
    shutil.copyfile(ROOT / "scripts/level1/train" / NAME, wrapper)
    (scripts / "run_level1_training.sh").write_text(
        'set -eu\n'
        '[[ "$*" == "--smoke-updates 2" ]]\n'
        '[[ "$CONFIG" == */configs/level1/train/curriculum2.yaml ]]\n'
        '[[ "$RUN_ID" == legacy-iter25-gs24-smoke-* ]]\n'
        '[[ "$DATASET_OUTPUT_ROOT" == "$ARTIFACT_ROOT/dataset" ]]\n'
        '[[ -d "$CURRICULUM1_CHECKPOINT" ]]\n'
        '[[ ! -v TRAIN_EPISODES && ! -v MODEL_PATH ]]\n'
        'echo MOCK_ROUTING_OK\n', encoding="utf-8", newline="\n",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    # Translate paths in Bash so this also exercises real Git Bash on Windows.
    command = 'p=$(cd "$1" && pwd); bash "$p/scripts/level1/train/' + NAME + '" "$p/checkpoint" "$p/output"'
    env = {**os.environ, "TRAIN_EPISODES": "1", "MODEL_PATH": "wrong"}
    result = subprocess.run([bash, "-c", command, "test", tmp_path.as_posix()],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "MOCK_ROUTING_OK" in result.stdout
    (tmp_path / "output").mkdir()
    result = subprocess.run([bash, "-c", command, "test", tmp_path.as_posix()],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "MOCK_ROUTING_OK" not in result.stdout
    result = subprocess.run([bash, wrapper.as_posix()], capture_output=True, text=True)
    assert result.returncode == 2
