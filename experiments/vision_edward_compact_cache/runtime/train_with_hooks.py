"""Experiment-local actor class replacement; pinned framework files unchanged."""
import runpy
import sys
import slime.utils.arguments as arguments
_original_parse_args = arguments.parse_args
_HOOKS = {
 "custom_rollout_training_completed_hook_path": "slime_pacman.formal_rollout.optimizer_completed_hook",
 "custom_rollout_checkpoint_saved_hook_path": "curriculum_runtime.checkpoint_saved_hook",
 "custom_rollout_checkpoint_load_hook_path": "slime_pacman.formal_rollout.checkpoint_load_hook",
}
def parse_with_required_hooks(*a, **kw):
    parsed = _original_parse_args(*a, **kw)
    for name, expected in _HOOKS.items():
        flag = "--" + name.replace("_", "-")
        assert sys.argv.count(flag) == 1, flag
        assert sys.argv[sys.argv.index(flag)+1] == expected, flag
        actual = getattr(parsed, name, None)
        assert actual in (None, expected), (name, actual)
        setattr(parsed, name, expected)
    print("REQUIRED_FORMAL_HOOK_ARGUMENTS_BOUND", _HOOKS, flush=True)
    return parsed
arguments.parse_args = parse_with_required_hooks
import slime.ray.rollout as original
from hooked_manager import RolloutManager
original.RolloutManager = RolloutManager
print("EXPERIMENT_LOCAL_ROLLOUT_HOOKS_INSTALLED", flush=True)
from checkpoint_space import install_checkpoint_space_guard
install_checkpoint_space_guard()
runpy.run_path("/workspace/slime/train.py", run_name="__main__")
