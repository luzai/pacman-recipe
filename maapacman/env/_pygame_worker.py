"""Compatibility entrypoint for pacman_env.env._pygame_worker."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module('pacman_env.env._pygame_worker', run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module('pacman_env.env._pygame_worker')
