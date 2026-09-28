"""Run the bounded primitive-action pilot and gated Adaptive Backplay."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from areal_pacman.level1.backplay_runner import main
    main()
