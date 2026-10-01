"""Export a slime run's metrics from its container log to a SwanLab offline run.

slime only logs to wandb/tensorboard, but every metric it reports is also printed to the log
(``step N: {...}``, ``perf N: {...}``, ``<metric> N: {...}``, ``Pacman rollout metrics: {...}``,
``Pacman evaluation metrics: {...}``). This reads those lines after the fact, so training code and
running jobs are untouched. Metrics are merged per rollout id (one optimizer step per rollout) and
written to ``metrics.jsonl``; with ``--swanlab-dir`` they are also logged to a SwanLab run in
offline mode, which ``swanlab sync <run dir>`` uploads later from any machine with network access.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
INDEXED = re.compile(r"(?:^|[\s\]])([A-Za-z_][\w/.-]*) (\d+): (\{.*\})\s*$")
ROLLOUT = re.compile(r"Pacman rollout metrics: (\{.*\})\s*$")
EVALUATION = re.compile(r"Pacman evaluation metrics: (\{.*\})\s*$")
REPEATED = re.compile(r"\s*\[repeated \d+x across cluster\]\s*$")
NONFINITE = re.compile(r"(?<![\w'\"])-?(?:nan|inf)(?![\w'\"])")


def _numbers(metrics):
    # Metric keys are namespaced (train/, rollout/, perf/, eval/...); this drops look-alike log
    # lines such as "Ports for engine 0: {'port': ...}".
    return {k: float(v) for k, v in metrics.items()
            if isinstance(k, str) and "/" in k
            and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def _literal(text):
    try:
        # Python prints non-finite floats as bare nan/inf, which literal_eval rejects.
        value = ast.literal_eval(NONFINITE.sub("None", text))
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def parse_log(lines):
    """Return {rollout_id: {metric: value}} from slime/Pacman log lines."""
    steps = {}
    for raw in lines:
        line = REPEATED.sub("", ANSI.sub("", raw.rstrip("\n")))
        step = metrics = None
        if match := EVALUATION.search(line):
            evidence = _literal(match.group(1))
            if evidence:
                metrics = dict(evidence.get("metrics", {}))
                step = metrics.pop("eval/step", evidence.get("rollout_id"))
        elif match := ROLLOUT.search(line):
            metrics = _literal(match.group(1))
            if metrics:
                step = metrics.pop("rollout/step", None)
        elif match := INDEXED.search(line):
            metrics = _literal(match.group(3))
            if metrics:
                step = int(match.group(2))
                metrics.pop("rollout/step", None)
                metrics.pop("train/step", None)
        numbers = _numbers(metrics or {})
        if numbers and step is not None:
            steps.setdefault(int(step), {}).update(numbers)
    return dict(sorted(steps.items()))


def write_jsonl(steps, path):
    with Path(path).open("w", encoding="utf-8", newline="\n") as out:
        for step, metrics in steps.items():
            out.write(json.dumps(dict(step=step, metrics=metrics), sort_keys=True) + "\n")


def write_swanlab(steps, logdir, project, name, config):
    import swanlab

    swanlab.init(project=project, experiment_name=name, mode="offline", logdir=str(logdir), config=config)
    try:
        for step, metrics in steps.items():
            swanlab.log(metrics, step=step)
    finally:
        swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="container log (docker logs output)")
    parser.add_argument("--output", type=Path, required=True, help="metrics.jsonl to write")
    parser.add_argument("--swanlab-dir", type=Path, help="write a SwanLab offline run under this directory")
    parser.add_argument("--project", default="pacman-slime")
    parser.add_argument("--name", required=True)
    parser.add_argument("--render", type=Path, help="render JSON recorded as the run config")
    args = parser.parse_args()
    with args.log.open(encoding="utf-8", errors="replace") as log:
        steps = parse_log(log)
    write_jsonl(steps, args.output)
    if args.swanlab_dir:
        config = dict(name=args.name, log=str(args.log))
        if args.render:
            render = json.loads(args.render.read_text(encoding="utf-8"))
            config.update(argv=" ".join(render.get("argv", [])), environment=render.get("environment", {}))
        write_swanlab(steps, args.swanlab_dir, args.project, args.name, config)
    keys = sorted({k for metrics in steps.values() for k in metrics})
    print(json.dumps(dict(steps=list(steps), metrics=len(keys), output=str(args.output))))


if __name__ == "__main__":
    main()
