"""Compatibility entrypoint for pacman_recipe.synthetic.vision."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('pacman_recipe.synthetic.vision', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('pacman_recipe.synthetic.vision')
