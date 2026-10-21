"""Local-first tracking. A disposable SwanLab uploader never blocks training.

The append-only journal is authoritative; replay it with this module after an
outage. SDK startup/network hangs affect only the uploader subprocess.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .swanlab_export import parse_log, _literal, ANSI, REPEATED

SECRET = re.compile(r"api.?key|wandb.key|password|secret|access.?token|authorization", re.I)


def safe_argv(argv):
    result, hide = [], False
    for value in argv:
        if hide:
            result.append("[REDACTED]")
            hide = False
        elif value.startswith("--") and SECRET.search(value.split("=", 1)[0]):
            result.append(value.split("=", 1)[0] + "=[REDACTED]" if "=" in value else value)
            hide = "=" not in value
        else:
            result.append(value)
    return result


def metadata(config, environment, argv):
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for package in ("slime_pacman", "pacman_recipe", "pacman_env"):
        for path in sorted((root / package).rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = dict(config=config.as_dict(), argv=safe_argv(argv), source_sha256=sources,
               effective_clip_cov=environment.get("PACMAN_EXPERIMENTAL_CLIP_COV", "off"),
               total_lives=4 if config.episode_life_mode == "original_three_lives" else 1,
               reward="win: 1+life_bonus*(1-deaths/3)+speed_bonus*(1-steps/H); failure: 0; life bonus disabled for single_death",
               metric_step="source rollout index; not a completed-update counter")
    path = environment.get("PACMAN_FORMAL_CONFIG")
    if path:
        raw = Path(path).read_bytes()
        curriculum = json.loads(raw)
        out["curriculum"] = {k: v for k, v in curriculum.items() if k != "catalog"}
        out["curriculum_sha256"] = hashlib.sha256(raw).hexdigest()
        out["catalog"] = [{k: t[k] for k in ("task_id", "tier", "split", "layout_sha256", "ghosts") if k in t}
                          for t in curriculum.get("catalog", [])]
    return out


def metric_events(line):
    line = REPEATED.sub("", ANSI.sub("", line.rstrip()))
    for step, metrics in parse_log([line]).items():
        yield {"source_step": step, "metrics": metrics}
    if "Clip-Cov update selection: " in line:
        data = _literal(line.split("Clip-Cov update selection: ", 1)[1].strip())
        if data and all(k in data for k in ("ratio", "low", "high", "decisions", "eligible", "selected", "selected_share_of_positive_cov")):
            yield {"source_step": None, "metrics": {
                "clip_cov/" + k: data[k] for k in
                ("ratio", "low", "high", "decisions", "eligible", "selected", "selected_share_of_positive_cov")}}


def run_logged(command, config, environment, argv):
    root = Path(environment["PACMAN_RUN_DIR"]) / "tracking" / str(time.time_ns())
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(metadata(config, environment, argv), indent=2), encoding="utf-8")
    (root / "status.json").write_text(json.dumps({"state": "local_journal_ready"}))
    uploader = None
    try:
        uploader = subprocess.Popen([sys.executable, "-m", "slime_pacman.tracking", "--directory", str(root),
                                     "--mode", config.swanlab_mode, "--follow"],
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        (root / "status.json").write_text(json.dumps({"state": "local_only", "reason": "uploader_start_failed", "replay_required": True}))
    code = 1
    try:
        with (root / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as journal:
            with subprocess.Popen(command, env=dict(os.environ, **environment), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                  bufsize=1) as child:
                try:
                    for line in child.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        for event in metric_events(line):
                            event["time"] = time.time()
                            journal.write(json.dumps(event, allow_nan=False) + "\n")
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
    finally:
        (root / "DONE.json").write_text(json.dumps({"exit_code": code}))
        try:
            if uploader is not None:
                uploader.wait(timeout=5)
        except subprocess.TimeoutExpired:
            uploader.terminate()
            try:
                uploader.wait(timeout=2)
            except subprocess.TimeoutExpired:
                uploader.kill()
                uploader.wait()
            (root / "status.json").write_text(json.dumps({"state": "local_only", "reason": "uploader_timeout", "replay_required": True}))
    return code


def upload(directory, mode, follow=False):
    root = Path(directory)
    def status(state, **extra):
        (root / "status.json").write_text(json.dumps(dict(state=state, **extra)))
    try:
        import swanlab
    except ImportError:
        status("local_only", reason="swanlab_not_installed", replay_required=True)
        return
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    # No interactive login on a training server. A missing key falls back offline.
    selected = "cloud" if mode == "auto" and os.environ.get("SWANLAB_API_KEY") else "offline"
    kwargs = dict(project=cfg["config"]["swanlab_project"], experiment_name=root.parent.parent.name + "-" + root.name,
                  config=cfg, logdir=str(root / "swanlog"))
    try:
        swanlab.init(mode=selected, **kwargs)
    except Exception:
        # Do not store exception strings: SDK errors may include credentials/URLs.
        try:
            swanlab.finish()
            swanlab.init(mode="offline", **kwargs)
            selected = "offline"
        except Exception:
            status("local_only", reason="sdk_init_failed", replay_required=True)
            return
    status(selected, replay_required=selected != "cloud")
    try:
        path = root / "metrics.jsonl"
        while not path.exists():
            if not follow or (root / "DONE.json").exists():
                return
            time.sleep(0.2)
        with path.open(encoding="utf-8") as stream:
            event_index = 0
            while True:
                position = stream.tell()
                line = stream.readline()
                if line.endswith("\n"):
                    event = json.loads(line)
                    values = dict(event["metrics"])
                    if event["source_step"] is not None:
                        values["source/rollout_index"] = event["source_step"]
                    # Monotonic event axis avoids conflating pre-eval and post-update.
                    swanlab.log(values, step=event_index)
                    event_index += 1
                else:
                    stream.seek(position)
                    if not follow or (root / "DONE.json").exists():
                        break
                    time.sleep(0.2)
    except Exception:
        status("local_only", reason="sdk_log_failed", replay_required=True)
    finally:
        try:
            swanlab.finish()
        except Exception:
            status("local_only", reason="sdk_finish_failed", replay_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "offline"), default="auto")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    upload(args.directory, args.mode, args.follow)


if __name__ == "__main__":
    main()
