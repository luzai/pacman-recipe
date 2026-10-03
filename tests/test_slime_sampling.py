import asyncio
import random
from types import SimpleNamespace

import pytest

from slime_pacman.sampling import generate_rollout, is_zero_variance, resample_slots, select_start_records

MIXED, WIN, LOSS = [1.0] * 6 + [0.0] * 6, [1.0] * 12, [0.0] * 12


def test_fixed_starts_ignore_resampling_cursor_and_epoch_shuffle():
    class Dataset:
        samples = [SimpleNamespace(metadata={"episode_record": {"id": s}}) for s in "ABCD"]

        def __len__(self):
            return len(self.samples)

    def get_samples(n):
        # A cursor crossing shuffled epochs can return A,A,B,C.
        return [[SimpleNamespace(metadata={"episode_record": {"id": s}})] for s in "AABC"[:n]]

    source = SimpleNamespace(dataset=Dataset(), get_samples=get_samples)
    for _ in range(4):
        source.get_samples(3)  # Simulate template allocation for retry rounds.
        assert {r["id"] for r in select_start_records(source, 4)} == set("ABCD")
        source.dataset.samples.reverse()


def test_fixed_starts_reject_duplicate_dataset_rows():
    class Dataset:
        samples = [SimpleNamespace(metadata={"episode_record": {"id": "A"}})] * 4

        def __len__(self):
            return 4

    with pytest.raises(ValueError, match="duplicate"):
        select_start_records(SimpleNamespace(dataset=Dataset()), 4)


def test_larger_dataset_keeps_source_selection():
    source = SimpleNamespace(dataset=list(range(8)), get_samples=lambda n: [
        [SimpleNamespace(metadata={"episode_record": {"id": s}})] for s in "BCDE"[:n]
    ])
    assert [r["id"] for r in select_start_records(source, 4)] == list("BCDE")


def group(index, rewards, decisions=2):
    return [[SimpleNamespace(reward=r, group_index=index) for _ in range(decisions)] for r in rewards]


def launcher(plan):
    """plan[slot] = list of reward lists returned for that slot on successive launches."""
    calls, counter, cursor = [], iter(range(1000)), {slot: 0 for slot in plan}

    async def launch(slots):
        calls.append(list(slots))
        out = []
        for slot in slots:
            out.append(group(next(counter), plan[slot][cursor[slot]]))
            cursor[slot] += 1
        return out

    return launch, calls


def test_zero_variance_detection():
    assert is_zero_variance(group(0, WIN)) and is_zero_variance(group(0, LOSS))
    assert not is_zero_variance(group(0, MIXED))
    bad = group(0, MIXED)
    bad[0][0].reward, bad[0][1].reward = 1.0, 0.0
    with pytest.raises(ValueError, match="different rewards"):
        is_zero_variance(bad)


def test_all_slots_informative_needs_one_round():
    launch, calls = launcher({s: [MIXED] for s in range(4)})
    kept, dropped, stats, _ = asyncio.run(resample_slots(4, launch, 4, random.Random(0)))
    assert calls == [[0, 1, 2, 3]] and len(kept) == 4 and dropped == []
    assert stats == dict(rounds=1, generated_groups=4, zero_variance_groups=0, filled_with_zero_variance=0)


def test_zero_variance_slots_are_rerun_from_the_same_start():
    launch, calls = launcher({0: [MIXED], 1: [WIN, WIN, MIXED], 2: [MIXED], 3: [LOSS, MIXED]})
    kept, dropped, stats, log = asyncio.run(resample_slots(4, launch, 4, random.Random(0)))
    assert calls == [[0, 1, 2, 3], [1, 3], [1]]
    assert not any(is_zero_variance(g) for g in kept) and len(dropped) == 3
    assert [len(log[s]) for s in range(4)] == [1, 3, 1, 2]
    assert stats["rounds"] == 3 and stats["generated_groups"] == 7 and stats["filled_with_zero_variance"] == 0


def test_round_cap_fills_a_slot_with_its_own_zero_variance_group():
    def run(seed):
        launch, calls = launcher({0: [MIXED], 1: [WIN, LOSS], 2: [MIXED], 3: [MIXED]})
        return asyncio.run(resample_slots(4, launch, 2, random.Random(seed))), calls

    (kept, dropped, stats, log), calls = run(7)
    assert calls == [[0, 1, 2, 3], [1]] and stats["filled_with_zero_variance"] == 1
    assert is_zero_variance(kept[1]) and len(dropped) == 1 and log[1][-1]["filled_with_zero_variance"]
    (again, _, _, _), _ = run(7)
    assert kept[1][0][0].group_index == again[1][0][0].group_index


def test_slot_results_do_not_depend_on_completion_order():
    async def launch(slots):
        await asyncio.sleep(0)
        return [group(10 + s, MIXED) for s in slots]

    kept, _, _, _ = asyncio.run(resample_slots(3, launch, 4, random.Random(0)))
    assert [g[0][0].group_index for g in kept] == [10, 11, 12]


def test_generate_rollout_rejects_conflicting_slime_sampling_options():
    args = SimpleNamespace(dynamic_sampling_filter_path="x.filter", partial_rollout=False,
                           rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
                           rollout_data_transport="object-store")
    pytest.importorskip("slime.rollout.sglang_rollout")
    with pytest.raises(ValueError, match="replaces these rollout options"):
        generate_rollout(args, 0, SimpleNamespace(get_samples=None))


def test_rollout_slots_waits_for_rounds_reruns_slot_and_discards(monkeypatch, tmp_path):
    sglang_rollout = pytest.importorskip("slime.rollout.sglang_rollout")
    base_types = pytest.importorskip("slime.rollout.base_types")
    transport = pytest.importorskip("slime.utils.rollout_transport")
    from slime_pacman import sampling

    outcomes = {"A": [MIXED], "B": [WIN, MIXED], "C": [MIXED], "D": [MIXED]}
    seen = []

    class State:
        def __init__(self, args):
            self.pendings = set()

        def submit_generate_tasks(self, groups):
            async def finish(g, delay):
                await asyncio.sleep(delay)
                start = g[0][0].metadata["episode_record"]["id"]
                seen.append(start)
                rewards = outcomes[start].pop(0)
                return [[SimpleNamespace(reward=r, group_index=g[0][0].group_index)] for r in rewards]

            for i, g in enumerate(groups):
                self.pendings.add(asyncio.ensure_future(finish(g, 0.01 * (len(groups) - i))))

        def reset(self):
            pass

    counter = iter(range(100))

    def get_samples(n):
        out = []
        for _ in range(n):
            gi = next(counter)
            out.append([[SimpleNamespace(group_index=gi, metadata={"episode_record": {"id": "template"}})]])
        return out

    discarded, finalized = [], {}
    monkeypatch.setattr(sglang_rollout, "GenerateState", State)
    monkeypatch.setattr(transport, "discard_rollout_group", lambda g, args, reason: discarded.append(reason))
    monkeypatch.setattr(base_types, "finalize_rollout_groups",
                        lambda args, rid, groups, metrics: finalized.update(groups=groups, metrics=metrics) or "out")
    records = [{"id": s} for s in "ABCD"]
    out = asyncio.run(sampling.rollout_slots(SimpleNamespace(seed=1), 3, get_samples, records,
                                             tmp_path / "log.json"))
    assert out == "out" and discarded == ["zero_variance_group"]
    assert sorted(seen) == ["A", "B", "B", "C", "D"]  # slot B rerun from the same start
    assert len(finalized["groups"]) == 4 and finalized["metrics"]["rollout/resample_rounds"] == 2
    assert (tmp_path / "log.json").exists()
