"""Compatibility imports; use pacman_recipe in new code."""
from importlib import import_module as _import_module

_canonical = _import_module('pacman_recipe')
__all__ = getattr(_canonical, "__all__", [n for n in vars(_canonical) if not n.startswith("_")])
globals().update({n: getattr(_canonical, n) for n in __all__})
