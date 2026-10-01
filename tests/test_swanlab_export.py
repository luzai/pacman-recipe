import json

from slime_pacman.swanlab_export import parse_log, write_jsonl

LOG = """\
\x1b[36m(MegatronTrainRayActor pid=1)\x1b[0m [2026-10-01 20:00:00] model.py:900 - step 0: {'train/loss': -0.01, 'train/grad_norm': 44.9}
(RolloutManager pid=2) [2026-10-01 20:00:01] rollout.py:430 - Pacman rollout metrics: {'rollout/win_rate': 0.5, 'rollout/episodes': 48, 'rollout/step': 0}
(MegatronTrainRayActor pid=1) perf 0: {'perf/train_time': 400.0, 'perf/note': 'x'}\x1b[32m [repeated 7x across cluster]\x1b[0m
(RolloutManager pid=2) Pacman evaluation metrics: {'schema': 'pacman-eval-metrics-v1', 'rollout_id': 1, 'weight_versions': ['2'], 'metrics': {'eval/pacman/win_rate': 0.25, 'eval/step': 1}}
step 1: {'train/loss': nan, 'train/grad_norm': 12.2, 'train/flag': True}
unrelated line: {'not': 'parsed'
Ports for engine 3: {'host': 'h', 'port': 15006}
Reloading 97 process groups in pid 16849: {'nccl': 94}
"""


def test_parse_log_merges_metrics_per_rollout(tmp_path):
    steps = parse_log(LOG.splitlines())
    assert steps == {
        0: {"train/loss": -0.01, "train/grad_norm": 44.9, "rollout/win_rate": 0.5,
            "rollout/episodes": 48.0, "perf/train_time": 400.0},
        1: {"eval/pacman/win_rate": 0.25, "train/grad_norm": 12.2},
    }
    write_jsonl(steps, tmp_path / "m.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1]
