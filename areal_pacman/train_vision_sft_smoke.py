"""Compatibility entrypoint for pacman_recipe.train_vision_sft_smoke."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('pacman_recipe.train_vision_sft_smoke', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('pacman_recipe.train_vision_sft_smoke')
