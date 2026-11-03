import pytest

from slime_pacman.map_training.checkpoint_retention import select_retained_updates


def test_latest_best_pending_and_exact_fraction_tie():
    # 2/3 and 4/6 tie exactly, independent of count magnitudes or float rounding.
    p = select_retained_updates([10, 20, 30, 40], {10: (2, 3), 20: (4, 6), 30: (1, 2)}, [30])
    assert p.keep_updates == {10, 30, 40}
    assert p.latest_update == 40 and p.best_update == 10


def test_external_best_does_not_claim_missing_local_payload_is_retained():
    p = select_retained_updates([334, 339], {269: (77, 96), 334: (60, 96), 339: (70, 96)})
    assert p.best_update == 269
    assert p.keep_updates == {339}


def test_without_validation_keeps_latest_and_pending():
    assert select_retained_updates([0, 5, 10], {}, [5]).keep_updates == {5, 10}
    p = select_retained_updates([], {})
    assert p.keep_updates == set() and p.best_update is None and p.latest_update is None


def test_missing_pending_fails_closed():
    with pytest.raises(ValueError, match='pending'):
        select_retained_updates([10], {}, [9])


@pytest.mark.parametrize('scores', [{1: (0, 0)}, {1: (2, 1)}, {1: (-1, 2)},
                                    {1: (True, 2)}, {1: (1, 2.0)}, {1: (1, 2, 3)}])
def test_invalid_validation_counts_are_rejected(scores):
    with pytest.raises(ValueError):
        select_retained_updates([1], scores)


@pytest.mark.parametrize('updates', [[-1], [True], [1.0]])
def test_update_labels_require_nonnegative_integers(updates):
    with pytest.raises(ValueError, match='update'):
        select_retained_updates(updates, {})
