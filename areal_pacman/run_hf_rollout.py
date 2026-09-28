"""Compatibility entrypoint for pacman_recipe.run_hf_rollout."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('pacman_recipe.run_hf_rollout', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('pacman_recipe.run_hf_rollout')
