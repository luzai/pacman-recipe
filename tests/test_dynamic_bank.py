import random

import pytest

from slime_pacman import dynamic_bank as db


def boundary(i, env_sha=None, remaining=None):
    return {"env_state": {"sha256": env_sha or f"env{i}"},
            "context": {"planner": {"remaining": remaining or [[1, i]], "last_action": "L", "fallback_mode": "risk_ranked"},
                        "runner": {"recent_actions": ["L"] * i}, "verify": {"png_sha256": str(i)}}}


def entry(state_id, seed=0, bucket="0|0|6+|41+", usage=0):
    return dict(state_id=state_id, seed=seed, bucket=bucket, sources=[dict(episode=state_id)],
                usage_count=usage)


def test_death_candidates_use_distances_from_last_choice():
    ring = list(range(17))  # boundary 16 is the last model choice
    assert db.death_candidates(ring, "death") == [(4, 12), (8, 8), (16, 0)]
    assert db.death_candidates(ring[:10], "death") == [(4, 5), (8, 1)]
    assert db.death_candidates(ring, "all_normal_pellets") == []
    assert db.death_candidates([], "death") == []


def test_state_identity_joins_env_and_context_but_not_verification():
    a, b = boundary(1), boundary(1)
    b["context"]["verify"] = {"png_sha256": "other"}
    assert db.state_identity(a) == db.state_identity(b)
    assert db.state_identity(a) != db.state_identity(boundary(1, remaining=[[2, 2]]))
    assert db.state_identity(a) != db.state_identity(boundary(1, env_sha="x"))
    payload = db.state_file_payload(a)
    assert payload["boundary_context"] is a["context"] and payload["sha256"] == "env1"


@pytest.mark.parametrize("distance, expected", [(None, "6+"), (0, "0-2"), (2, "0-2"), (3, "3-5"), (5, "3-5"), (6, "6+")])
def test_distance_bins(distance, expected):
    assert db.distance_bin(distance) == expected


def test_bucket_key():
    features = dict(pacman_position=[9, 14], nearest_lethal_ghost_distance=4, normal_pellets_remaining=40)
    assert db.bucket_key(features) == "2|3|3-5|11-40"
    assert db.pellet_bin(10) == "0-10" and db.pellet_bin(41) == "41+"


def test_add_state_merges_duplicates_and_evicts_full_bucket():
    manifest = db.new_manifest()
    for i in range(4):
        db.add_state(manifest, entry(f"s{i}"), update=i)
    manifest["states"]["s1"]["usage_count"] = 3
    events = db.add_state(manifest, entry("s1"), update=5)
    assert events[0]["event"] == "merged" and len(manifest["states"]["s1"]["sources"]) == 2
    events = db.add_state(manifest, entry("s9"), update=6)
    assert [e["event"] for e in events] == ["added", "evicted"]
    assert events[1]["state_id"] == "s1" and events[1]["reason"] == "bucket_full"  # most used
    assert len(manifest["states"]) == 4


def test_eviction_tie_breaks_by_stale_probe_then_state_id():
    manifest = db.new_manifest()
    db.add_state(manifest, entry("b"), update=1)
    db.add_state(manifest, entry("a"), update=1)
    db.add_state(manifest, entry("c"), update=0)
    db.add_state(manifest, entry("d"), update=2)
    events = db.add_state(manifest, entry("e"), update=3)
    assert events[-1]["state_id"] == "c"  # oldest probe
    events = db.add_state(manifest, entry("f"), update=3)
    assert events[-1]["state_id"] == "a"  # same probe age -> smallest ID


def test_seed_and_total_capacity():
    manifest = db.new_manifest()
    for i in range(db.PER_SEED + 1):
        db.add_state(manifest, entry(f"s{i:02d}", seed=0, bucket=f"b{i:02d}"), update=i)
    assert sum(e["seed"] == 0 for e in manifest["states"].values()) == db.PER_SEED
    manifest = db.new_manifest()
    for i in range(db.CAPACITY + 1):
        db.add_state(manifest, entry(f"t{i:03d}", seed=i, bucket=f"b{i % 64:02d}"), update=i)
    assert len(manifest["states"]) == db.CAPACITY
    assert manifest["events"][-1]["reason"] == "capacity_full"


def test_true_start_rotation():
    seeds = [0, 1, 14, 16]
    assert [db.true_start_seeds(u, seeds) for u in range(3)] == [[0, 1], [14, 16], [0, 1]]
    assert [db.true_start_seeds(u, seeds, 1) for u in range(5)] == [[0], [1], [14], [16], [0]]


def test_select_starts_prefers_new_then_old_states_and_distinct_seeds():
    manifest = db.new_manifest()
    db.add_state(manifest, entry("old-a", seed=14, bucket="x"), update=0)
    db.add_state(manifest, entry("new-a", seed=0, bucket="y"), update=2)
    db.add_state(manifest, entry("new-b", seed=16, bucket="z"), update=2)
    starts = db.select_starts(manifest, update=0, train_seeds=[0, 1, 14, 16], n_true=2, n_bank=2)
    assert [s["kind"] for s in starts] == ["true_start", "true_start", "bank", "bank"]
    assert starts[2]["state_id"] == "new-b"  # newest pool, seed not already used (0 is a true start)
    assert starts[3]["state_id"] == "old-a"
    db.record_selection(manifest, starts)
    assert manifest["states"]["new-b"]["usage_count"] == 1 and manifest["bucket_selection_counts"] == {"z": 1, "x": 1}


def test_select_starts_fills_with_true_starts_when_bank_is_short():
    manifest = db.new_manifest()
    db.add_state(manifest, entry("only", seed=16), update=0)
    starts = db.select_starts(manifest, update=0, train_seeds=[0, 1, 14, 16], n_true=2, n_bank=2)
    assert starts[2]["state_id"] == "only"
    assert starts[3] == dict(kind="true_start", seed=14, filled_for_bank_slot=1)
    empty = db.select_starts(db.new_manifest(), update=1, train_seeds=[0, 1, 14, 16], n_true=2, n_bank=2)
    assert [s["seed"] for s in empty] == [14, 16, 0, 1]


def test_choose_trajectories_prefers_fresh_seeds_and_cells():
    trajectories = [dict(episode_id="a", seed=0, death_position=[1, 1]),
                    dict(episode_id="b", seed=0, death_position=[2, 2]),
                    dict(episode_id="c", seed=1, death_position=[1, 1]),
                    dict(episode_id="d", seed=14, death_position=[3, 3])]
    picked = db.choose_trajectories(trajectories, {"0": 2, "1": 0, "14": 0}, 2, random.Random(0))
    assert {t["seed"] for t in picked} == {1, 14}
    again = db.choose_trajectories(trajectories, {"0": 2, "1": 0, "14": 0}, 2, random.Random(0))
    assert [t["episode_id"] for t in picked] == [t["episode_id"] for t in again]
    assert len(db.choose_trajectories(trajectories[:1], {}, 2, random.Random(0))) == 1


@pytest.mark.parametrize("first, second, expected", [(0, None, "reject"), (8, None, "reject"), (1, None, "extend"),
                                                     (7, None, "extend"), (1, 1, "reject"), (1, 2, "accept"),
                                                     (7, 14, "accept"), (7, 15, "reject")])
def test_probe_decision(first, second, expected):
    assert db.probe_decision(first, second) == expected


def test_default_is_one_true_start_and_three_bank_states():
    manifest = db.new_manifest()
    for i, (seed, bucket, update) in enumerate([(0, "a", 0), (1, "b", 0), (14, "c", 2), (16, "d", 2)]):
        db.add_state(manifest, entry(f"s{i}", seed=seed, bucket=bucket), update=update)
    starts = db.select_starts(manifest, update=1, train_seeds=[0, 1, 14, 16])
    assert [s["kind"] for s in starts] == ["true_start", "bank", "bank", "bank"]
    assert starts[0]["seed"] == 1
    assert len({s.get("state_id", s["seed"]) for s in starts}) == 4
    assert starts[1]["state_id"] in ("s2", "s3") and starts[2]["state_id"] in ("s0", "s1")
