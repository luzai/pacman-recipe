"""Backward-compatible imports for synthetic evaluation."""

from .synthetic.evaluate import evaluate, main, run_episode

__all__ = ["evaluate", "main", "run_episode"]


if __name__ == "__main__":
    main()
