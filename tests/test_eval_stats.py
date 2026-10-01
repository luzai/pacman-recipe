import pytest

from slime_pacman import eval_stats as es
from slime_pacman.evaluate_starts import seed_list


def rows(group, rates, episodes=48):
    out = []
    for seed, rate in rates.items():
        wins = round(rate * episodes)
        for i in range(episodes):
            won = i < wins
            out.append(dict(group=group, seed=seed, index=i, reward=1.0 if won else 0.0,
                            terminal_reason="all_normal_pellets" if won else ("death" if i % 4 else "max_steps")))
    return out


def test_summary_counts_and_bootstrap_interval():
    data = rows("true_start", {0: 0.25, 1: 0.5, 15: 0.75}) + rows("holdout", {72: 0.5}, episodes=4)
    report = es.summarize(data, "true_start")
    assert report["episodes"] == 144 and report["wins"] == 72 and report["pass_rate"] == 0.5
    assert report["seeds"]["1"]["wins"] == 24 and report["seeds"]["15"]["pass_rate"] == 0.75
    low, high = report["bootstrap95"]
    assert low < 0.5 < high and high - low < 0.2
    assert report == es.summarize(data, "true_start")  # fixed bootstrap seed
    assert report["death_rate"] + report["timeout_rate"] + 0.5 == pytest.approx(1.0)
    assert es.summarize(data, "holdout")["episodes"] == 4


def test_clear_improvement_requires_ten_points_and_positive_lower_bound():
    base = rows("true_start", {s: 0.25 for s in (0, 1, 14, 15, 16)})
    better = rows("true_start", {s: 0.45 for s in (0, 1, 14, 15, 16)})
    slight = rows("true_start", {s: 0.29 for s in (0, 1, 14, 15, 16)})
    result = es.compare(base, better, "true_start")
    assert result["difference"] == pytest.approx(0.2, abs=0.01) and result["clear_improvement"]
    assert result["difference_bootstrap95"][0] > 0
    assert not es.compare(base, slight, "true_start")["clear_improvement"]
    with pytest.raises(ValueError, match="different seeds"):
        es.compare(base, rows("true_start", {0: 0.5}), "true_start")


def test_wilson_and_seed_parsing():
    low, high = es.wilson(12, 48)
    assert low < 0.25 < high
    assert es.wilson(0, 0) == [0.0, 0.0]
    assert seed_list("0,1,14,15,16") == [0, 1, 14, 15, 16]
    assert seed_list("72-75,80") == [72, 73, 74, 75, 80]
