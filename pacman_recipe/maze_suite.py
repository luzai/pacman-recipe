"""Backward-compatible imports for the synthetic maze suite."""

from .synthetic.maze_suite import (
    SPLIT_SIZES,
    SUITE_NAME,
    SUITE_PATH,
    layout_hash,
    load_layout_registry,
    load_suite,
    maze_records,
    split_layout_names,
    suite_summary,
    topology_hash,
)

__all__ = [
    "SPLIT_SIZES",
    "SUITE_NAME",
    "SUITE_PATH",
    "layout_hash",
    "load_layout_registry",
    "load_suite",
    "maze_records",
    "split_layout_names",
    "suite_summary",
    "topology_hash",
]
