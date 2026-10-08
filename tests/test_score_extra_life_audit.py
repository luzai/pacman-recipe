import pytest
from pacman_recipe.level1.trajectories import expected_lives_after_step


@pytest.mark.parametrize('before,after,lives,death,expected', [
    (24000, 26500, 3, False, 4),  # Real baseline fruit crossing 25k.
    (24000, 26500, 3, True, 3),   # Award and death in one macro step.
    (25000, 27500, 4, False, 4),  # No duplicate award at exact boundary.
    (24990, 25000, 3, False, 4),
    (49000, 151000, 3, False, 6),
    (150000, 160000, 3, True, 2),
    (0, 24999, 3, False, 3),
])
def test_original_score_awards(before, after, lives, death, expected):
    assert expected_lives_after_step(lives, death, before, after) == expected
