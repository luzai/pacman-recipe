"""Failed restore must preserve an already running episode."""

import copy
import json
import shutil

import pytest

from maapacman.env import InvalidConfigurationError, PygameWorkerError
from maapacman.env._saved_state import checksum
from .test_saved_state import ROOT, make_env, transition_record


@pytest.mark.parametrize("damage", ["missing_mode", "bad_surface", "bad_reference", "runtime"])
def test_worker_rejection_rolls_back_and_continues(damage):
    with make_env() as env:
        env.reset(seed=7)
        env.step("L")
        saved = env.save_state()
        corrupt = copy.deepcopy(saved)
        worker = corrupt["payload"]["worker"]
        graph = worker["graph"]
        if damage == "missing_mode":
            game = next(n for n in graph["nodes"] if n.get("name") == "game")
            attrs = graph["nodes"][game["attrs"]["ref"]]
            attrs["items"] = [[k, v] for k, v in attrs["items"] if k != "mode"]
        elif damage == "bad_surface":
            surface = next(n for n in graph["nodes"] if n["kind"] == "surface")
            surface["pixels"] = "bm90LXpsaWI="
        elif damage == "bad_reference":
            graph["root"] = {"ref": len(graph["nodes"]) + 1}
        else:
            worker["runtime"]["pygame"] = "incompatible"
        # Exercise semantic validation rather than just a checksum rejection.
        corrupt["sha256"] = checksum(corrupt["payload"])
        with pytest.raises(PygameWorkerError, match="invalid saved state"):
            env.restore_state(corrupt)
        assert env.save_state() == saved
        result = transition_record(env.step("S"))
        end = env.save_state()
        env.restore_state(json.loads(json.dumps(saved)))
        assert transition_record(env.step("S")) == result
        assert env.save_state() == end


def test_changed_font_resource_is_rejected(tmp_path):
    # Work on a private copy; never alter the configured simulator resources.
    local_root = tmp_path / "pacman-python"
    shutil.copytree(ROOT / "pacman", local_root / "pacman")
    from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig

    with PygamePacmanEnv(PygamePacmanEnvConfig(pacman_python_root=local_root)) as env:
        env.reset(seed=0)
        saved = env.save_state()
        font = local_root / "pacman" / "res" / "VeraMoBd.ttf"
        with font.open("ab") as stream:
            stream.write(b"changed resource")
        with pytest.raises(InvalidConfigurationError, match="source/config/schema"):
            env.restore_state(saved)
