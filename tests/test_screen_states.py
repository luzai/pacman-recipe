import pytest

from slime_pacman.screen_states import summarize


def row(state_id, index, reward, terminal="death"):
    return dict(state_id=state_id, sample_index=index, reward=reward, terminal_reason=terminal)


def test_summarize_counts_each_state_once_in_finalist_format():
    rows = [row("a", i, float(i < 3), "all_normal_pellets" if i < 3 else "death") for i in range(4)]
    rows += [row("b", i, 0.0) for i in range(4)]
    a, b = summarize(["a", "b"], rows, 4)
    assert (a["samples"], a["successes"], a["success_rate"]) == (4, 3, 0.75)
    assert a["terminals"] == {"all_normal_pellets": 3, "death": 1}
    assert b["successes"] == 0 and len(a["wilson95"]) == 2


def test_summarize_rejects_missing_or_duplicate_episodes():
    with pytest.raises(ValueError, match="incomplete"):
        summarize(["a"], [row("a", 0, 1.0), row("a", 0, 1.0)], 2)
