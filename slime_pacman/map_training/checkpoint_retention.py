"""Pure latest/best/pending selection; never reads or deletes checkpoint files."""
from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class RetentionSelection:
    keep_updates: frozenset[int]
    latest_update: int | None
    best_update: int | None


def _update(value):
    if type(value) is not int or value < 0:
        raise ValueError('checkpoint update must be a nonnegative integer')
    return value


def select_retained_updates(available_updates: Iterable[int],
                            scores: Mapping[int, tuple[int, int]],
                            pending_updates: Iterable[int] = ()) -> RetentionSelection:
    """Select latest complete, available global best, and pending payloads.

    ``available_updates`` must already be verified complete by the caller. Labels
    are used verbatim; this does not convert zero-based iterations to update counts.
    Scores are (wins, episodes) from one comparable validation protocol. Scores for
    external ancestors are permitted: if an external ancestor is globally best,
    ``best_update`` names it but it is not included in this scope's keep set. The
    caller must protect that ancestor separately. Equal rates choose earlier labels.

    A pending label outside the available set is rejected rather than pretending
    its payload has been protected. This is a plan, not permission to delete the
    complement: active readers, mounts, identity and file checks remain external.
    """
    available = {_update(u) for u in available_updates}
    pending = {_update(u) for u in pending_updates}
    if not pending <= available:
        raise ValueError('pending checkpoint payload is not available')
    rates = {}
    for u, pair in scores.items():
        _update(u)
        try:
            wins, episodes = pair
        except (TypeError, ValueError) as exc:
            raise ValueError('score must be a (wins, episodes) pair') from exc
        if (type(wins) is not int or type(episodes) is not int
                or episodes <= 0 or not 0 <= wins <= episodes):
            raise ValueError('score requires integer 0 <= wins <= positive episodes')
        rates[u] = Fraction(wins, episodes)
    if not available:
        return RetentionSelection(frozenset(), None, None)
    latest = max(available)
    best = max(rates, key=lambda u: (rates[u], -u)) if rates else None
    keep = {latest, *pending}
    if best in available:
        keep.add(best)
    return RetentionSelection(frozenset(keep), latest, best)
