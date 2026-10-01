"""Dynamic bank of pre-death decision boundaries: pure selection and bookkeeping rules.

Implements the curriculum in reports/slime-migration-20260928/HANDOFF.md (dynamic-bank plan and its
two pre-declared addenda). No I/O or model calls here; the rollout orchestration lives in
slime_pacman.curriculum. All tie-breaks are deterministic or use a caller-supplied seeded RNG.
"""

import hashlib
import json

SCHEMA = "pacman-dynamic-bank-v1"
CANDIDATE_SCHEMA = "pacman-bank-candidate-v1"
STATE_SCHEMA = "pacman-dynamic-bank-state-v1"
DISTANCES = (4, 8, 16)  # decision points before the last model choice (= 0) of a death episode
RING_SIZE = max(DISTANCES) + 1
CAPACITY, PER_SEED, PER_BUCKET = 128, 16, 4
FIRST_PROBE, SECOND_PROBE = 8, 16
ACCEPT_WINS = (3, 21)  # inclusive, out of FIRST_PROBE + SECOND_PROBE
REFRESH_TRAJECTORIES, REFRESH_BUDGET = 2, 96
INIT_TRAJECTORIES, INIT_BUDGET = 6, 288  # addendum 3: >=3 states for 1 true + 3 bank starts


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def death_candidates(boundaries, terminal_reason):
    """[(distance, boundary)] for a death episode; distance 0 is the last model choice."""
    if terminal_reason != "death" or not boundaries:
        return []
    boundaries = list(boundaries)
    return [(d, boundaries[-1 - d]) for d in DISTANCES if len(boundaries) > d]


def state_identity(boundary):
    """Joint hash of game state and planner/runner context (identical -> same bank state)."""
    context = boundary["context"]
    material = canonical({"env": boundary["env_state"]["sha256"], "planner": context["planner"],
                          "runner": context["runner"]})
    return "dyn-" + hashlib.sha256(material.encode()).hexdigest()


def state_file_payload(boundary):
    """Restart file: env saved state plus the context the episode runner restores and verifies."""
    return dict(boundary["env_state"], boundary_context=boundary["context"])


def distance_bin(distance):
    if distance is None or distance >= 6:
        return "6+"
    return "0-2" if distance <= 2 else "3-5"


def pellet_bin(pellets):
    return "0-10" if pellets <= 10 else ("11-40" if pellets <= 40 else "41+")


def bucket_key(features):
    row, col = features["pacman_position"]
    return f"{row // 4}|{col // 4}|{distance_bin(features['nearest_lethal_ghost_distance'])}|" \
           f"{pellet_bin(features['normal_pellets_remaining'])}"


def new_manifest():
    return {"schema": SCHEMA, "states": {}, "bucket_selection_counts": {}, "seed_processed_counts": {},
            "events": []}


def _victim(entries):
    # Evict the most used, then the stalest probe, then the smallest state ID.
    return min(entries, key=lambda e: (-e["usage_count"], e["last_probe_update"], e["state_id"]))


def add_state(manifest, entry, update):
    """Insert a probed state, merging duplicates and enforcing bucket/seed/total capacity."""
    states, events = manifest["states"], []
    existing = states.get(entry["state_id"])
    if existing is not None:
        existing["sources"].extend(entry["sources"])
        events.append(dict(update=update, event="merged", state_id=entry["state_id"]))
        manifest["events"].extend(events)
        return events
    entry = dict(entry, usage_count=0, added_update=update, last_probe_update=update)
    states[entry["state_id"]] = entry
    events.append(dict(update=update, event="added", state_id=entry["state_id"], bucket=entry["bucket"],
                       seed=entry["seed"]))

    def evict(group, reason):
        victim = _victim(group)
        del states[victim["state_id"]]
        events.append(dict(update=update, event="evicted", state_id=victim["state_id"], reason=reason))

    while (same := [e for e in states.values() if e["bucket"] == entry["bucket"]]) and len(same) > PER_BUCKET:
        evict(same, "bucket_full")
    while (same := [e for e in states.values() if e["seed"] == entry["seed"]]) and len(same) > PER_SEED:
        evict(same, "seed_full")
    while len(states) > CAPACITY:
        buckets = {}
        for e in states.values():
            buckets.setdefault(e["bucket"], []).append(e)
        largest = min(buckets, key=lambda key: (-len(buckets[key]), key))
        evict(buckets[largest], "capacity_full")
    manifest["events"].extend(events)
    return events


def true_start_seeds(update, train_seeds, count=2):
    """`count` true-start seeds per update, rotating through the training seeds."""
    offset = (count * update) % len(train_seeds)
    return [train_seeds[(offset + i) % len(train_seeds)] for i in range(count)]


def select_starts(manifest, update, train_seeds, n_true=1, n_bank=3):
    """Four distinct starts: n_true rotating true starts plus n_bank bank states.

    Bank slot 1 prefers states from the latest refresh, slot 2 older ones; each falls back to the
    other pool, then to an unselected training-seed true start. Within a pool prefer a new seed and
    a new bucket, then the least selected bucket, the least used state, then key/state ID.
    """
    starts = [dict(kind="true_start", seed=seed) for seed in true_start_seeds(update, train_seeds, n_true)]
    states = list(manifest["states"].values())
    newest = max((e["added_update"] for e in states), default=None)
    pools = [[e for e in states if e["added_update"] == newest], [e for e in states if e["added_update"] != newest]]
    chosen = []
    counts = manifest["bucket_selection_counts"]
    for slot in range(n_bank):
        used_ids = {c["state_id"] for c in chosen}
        used_seeds = {s["seed"] for s in starts} | {c["seed"] for c in chosen}
        used_buckets = {c["bucket"] for c in chosen}
        primary, secondary = pools[slot % 2], pools[(slot + 1) % 2]
        pick = None
        for pool in (primary, secondary):
            available = [e for e in pool if e["state_id"] not in used_ids]
            if available:
                pick = min(available, key=lambda e: (e["seed"] in used_seeds, e["bucket"] in used_buckets,
                                                     counts.get(e["bucket"], 0), e["usage_count"], e["bucket"],
                                                     e["state_id"]))
                break
        if pick is not None:
            chosen.append(pick)
            starts.append(dict(kind="bank", state_id=pick["state_id"], seed=pick["seed"], bucket=pick["bucket"]))
            continue
        used = {s["seed"] for s in starts if s["kind"] == "true_start"}
        spare = [seed for seed in train_seeds if seed not in used]
        if not spare:
            raise ValueError("no distinct start left to fill the batch")
        starts.append(dict(kind="true_start", seed=spare[0], filled_for_bank_slot=slot))
    return starts


def record_selection(manifest, starts):
    for start in starts:
        if start["kind"] == "bank":
            manifest["states"][start["state_id"]]["usage_count"] += 1
            counts = manifest["bucket_selection_counts"]
            counts[start["bucket"]] = counts.get(start["bucket"], 0) + 1


def choose_trajectories(trajectories, seed_processed_counts, count, rng):
    """Pick death trajectories: fewest-processed seed first, then new seeds and new death cells.

    trajectories: dicts with episode_id, seed, death_position. Ties broken by a seeded shuffle.
    """
    pool = list(trajectories)
    rng.shuffle(pool)
    pool.sort(key=lambda t: seed_processed_counts.get(str(t["seed"]), 0))
    picked = []
    while pool and len(picked) < count:
        seeds = {t["seed"] for t in picked}
        cells = {tuple(t["death_position"]) for t in picked}
        best = min(range(len(pool)), key=lambda i: (pool[i]["seed"] in seeds,
                                                    tuple(pool[i]["death_position"]) in cells, i))
        picked.append(pool.pop(best))
    return picked


def probe_decision(first_wins, second_wins=None):
    """'extend' after 8 games with 1-7 wins; 'reject' at 0/8 or 8/8; after 24 'accept' or 'reject'."""
    if second_wins is None:
        if not 0 <= first_wins <= FIRST_PROBE:
            raise ValueError("invalid first-probe wins")
        return "extend" if 0 < first_wins < FIRST_PROBE else "reject"
    total = first_wins + second_wins
    return "accept" if ACCEPT_WINS[0] <= total <= ACCEPT_WINS[1] else "reject"
