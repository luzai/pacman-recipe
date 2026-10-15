"""Fixed-sample paired native Decisions evaluation, with no retries or videos."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import subprocess
import time

import numpy as np

from .client import load_key
from .native import native_payload, run_one, runner_options
from pacman_recipe.level1.ascii_observation_spaced import space_ascii_map, unspace_ascii_map
from pacman_recipe.level1.prompts import prompt_contract_metadata
from pacman_recipe.level1.trajectories import audit_trajectory

MODES = ("primitive", "edward")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def source_hashes(repo):
    files = []
    for root in (repo / "pacman_recipe/level1", repo / "pacman_env",
                 repo / "baselines/decisions_vision", repo.parent / "pacman-python/pacman"):
        files.extend(p for p in root.rglob("*") if p.suffix in (".py", ".pyw", ".txt")
                     and "__pycache__" not in p.parts and not p.name.endswith(".local.txt"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def check_sources(frozen):
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in frozen.items()):
        raise RuntimeError("Frozen evaluation source changed")


def validate_artifacts(folder, observation):
    """Check the exact transmitted bodies, canonical spaced maps and game ledger."""
    decisions = json.loads((folder / "decisions.json").read_text(encoding="utf-8")) if (folder / "decisions.json").exists() else []
    maps = 0
    for d in decisions:
        stem = f"decision{d['index']:04d}"
        messages = json.loads((folder / f"{stem}_native_messages.json").read_text(encoding="utf-8"))
        body = json.loads((folder / f"{stem}_request.json").read_text(encoding="utf-8"))
        assert body == native_payload(messages, d["choices"])
        assert hashlib.sha256(json.dumps(body).encode()).hexdigest() == d["request_sha256"]
        if observation == "ascii":
            assert all(p["type"] == "input_text" for m in body["input"] for p in m["content"])
    if not (folder / "episode.json").exists():
        return dict(verified_requests=len(decisions), verified_maps=0, complete_trajectory=False)
    episode = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
    audit_trajectory(episode)
    for step in episode["trajectory"]:
        assert not step["score_components"]["fruit"]
        assert all(not a["fruit"]["active"] for a in step["atomic_substeps"])
        if observation == "ascii":
            maze = step["observation_ascii_map"]
            assert space_ascii_map(unspace_ascii_map(maze)) == maze
            assert maze == step["observation_context"]["ascii_map"]
            if step["model_called"]:
                assert maze in step["model_user_instruction"]
            maps += 1
    if observation == "ascii":
        assert not list(folder.glob("*.png"))
    return dict(verified_requests=len(decisions), verified_maps=maps,
                complete_trajectory=True, game_steps=len(episode["trajectory"]))


def wilson(wins, n):
    z = NormalDist().inv_cdf(.975)
    p = wins / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-half, center+half]


def mcnemar(edward_only, primitive_only):
    n = edward_only + primitive_only
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(edward_only, primitive_only)+1)) / 2**n) if n else 1.


def summarize(root, manifest, rows):
    n = len(manifest["seeds"])
    aggregates = {}
    for mode in MODES:
        all_rows = [r for r in rows if r["mode"] == mode]
        complete = [r for r in all_rows if r["status"] == "live"]
        wins = sum(r["won"] for r in complete)
        decisions = []
        for row in all_rows:
            path = root / f"seed{row['seed']}" / f"{mode}_seed{row['seed']}" / "decisions.json"
            if path.exists():
                decisions.extend(json.loads(path.read_text(encoding="utf-8")))
        latency = [d["request_seconds"] for d in decisions if d.get("source") == "live_api"]
        a = dict(attempts=n, completed=len(complete), failed=n-len(complete), wins=wins,
                 win_rate=wins/n, win_rate_wilson95=wilson(wins,n),
                 api_successes=len(latency), api_failures=sum(d.get("status")=="request_failed" for d in decisions),
                 forced_choices=sum(d.get("source")=="forced_singleton" for d in decisions))
        for field in ("score", "game_steps", "normal_pellet_clear_rate"):
            a["mean_"+field] = float(np.mean([r[field] for r in complete])) if complete else None
        a["terminal_reasons"] = {reason: sum(r["terminal_reason"] == reason for r in complete)
                                  for reason in sorted({r["terminal_reason"] for r in complete})}
        a["latency_mean_seconds"] = float(np.mean(latency)) if latency else None
        for percentile in (50,95,99):
            a[f"latency_p{percentile}_seconds"] = float(np.percentile(latency,percentile)) if latency else None
        aggregates[mode] = a
    pairs = [(next(r for r in rows if r["seed"]==s and r["mode"]=="primitive"),
              next(r for r in rows if r["seed"]==s and r["mode"]=="edward")) for s in manifest["seeds"]]
    def won(row):
        return row["status"] == "live" and row["won"]
    def discordant(ps):
        return (sum(won(e) and not won(p) for p,e in ps), sum(won(p) and not won(e) for p,e in ps))
    ew,pw = discordant(pairs)
    complete_pairs = [(p,e) for p,e in pairs if p["status"] == e["status"] == "live"]
    ce,cp = discordant(complete_pairs)
    rng = np.random.default_rng(20261015)
    def bootstrap(values):
        values = np.asarray(values)
        return np.quantile(values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1),[.025,.975]).tolist() if len(values) else None
    win_diff = [int(won(e))-int(won(p)) for p,e in pairs]
    score_diff = [e["score"]-p["score"] for p,e in complete_pairs]
    stats = dict(complete_pairs=len(complete_pairs), edward_only_wins=ew, primitive_only_wins=pw,
                 mcnemar_exact_two_sided_p=mcnemar(ew,pw), complete_case_mcnemar_p=mcnemar(ce,cp),
                 win_rate_difference=float(np.mean(win_diff)), win_difference_bootstrap95=bootstrap(win_diff),
                 mean_paired_score_difference=float(np.mean(score_diff)) if score_diff else None,
                 score_difference_bootstrap95=bootstrap(score_diff))
    report = dict(manifest=manifest, aggregates=aggregates, paired_statistics=stats, results=rows)
    write_json(root / "comparison.json", report)
    def number(value):
        return f"{value:.3f}" if value is not None else "N/A"
    lines = ["# Native Decisions paired comparison", "",
             f"Observation: {manifest['observation']}; {n} paired seeds; {manifest['max_steps']} game steps per episode.",
             "Edward risk_ranked; primitive refuse; normal ghosts, single death, fruit disabled. No videos or request retries.",
             "ASCII uses the existing spaced-map agent prompts verbatim. Structured state is retained; this is a harness comparison.", "",
             "| Mode | Clears / attempts | Complete / failed | Mean score (complete) | Mean steps | Pellet completion | API latency mean / p50 / p95 / p99 (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for mode,a in aggregates.items():
        rate = a["mean_normal_pellet_clear_rate"]
        latency = " / ".join(number(a[f"latency_{x}_seconds"]) for x in ("mean","p50","p95","p99"))
        lines.append(f"| {mode} | {a['wins']}/{n} | {a['completed']} / {a['failed']} | {number(a['mean_score'])} | {number(a['mean_game_steps'])} | {number(100*rate) if rate is not None else 'N/A'}% | {latency} |")
    lines += ["", f"Exact paired McNemar p={stats['mcnemar_exact_two_sided_p']:.6g}; complete pairs={len(complete_pairs)}.",
              f"Edward minus primitive win-rate difference: {stats['win_rate_difference']:.1%}; paired bootstrap 95% CI {stats['win_difference_bootstrap95']}.",
              f"Mean paired score difference (complete pairs): {stats['mean_paired_score_difference']}; bootstrap 95% CI {stats['score_difference_bootstrap95']}.",
              "Failed API attempts count as non-clears for the operational endpoint; missing complete-episode scores are not imputed. This rule was fixed before the batch.",
              "Latency pools successful API requests, excludes forced singleton choices and failed requests, and includes network work. Games pause during inference; concurrent load affects timing.",
              "Fixed layout and RNG seeds only: no unseen-map or trained-checkpoint generalization claim. Earlier image results used refuse and cannot isolate the effect of modality.", "",
              "| Seed | Primitive score | Steps | End | Edward score | Steps | End |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for p,e in pairs:
        lines.append(f"| {p['seed']} | {p.get('score','FAILED')} | {p.get('game_steps','-')} | {p.get('terminal_reason',p['status'])} | {e.get('score','FAILED')} | {e.get('game_steps','-')} | {e.get('terminal_reason',e['status'])} |")
    (root / "comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--observation", choices=("image","ascii"), default="ascii")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, choices=(32,256,512,2000), default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.workers <= 8:
        parser.error("episodes must be positive; workers must be 1..8")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["PACMAN_DISABLE_FRUIT"] = "1"
    repo = Path(__file__).resolve().parents[2]
    frozen = source_hashes(repo)
    manifest = dict(seeds=list(range(args.seed,args.seed+args.episodes)), modes=MODES,
        max_steps=args.max_steps, observation=args.observation, workers=args.workers,
        started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        git_head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip()),
        game_git_head=subprocess.check_output(["git","-C",str(repo.parent/"pacman-python"),"rev-parse","HEAD"],text=True).strip(),
        fruit_disabled=True, primary_endpoint="operational level clear, exact paired McNemar alpha=0.05",
        failure_rule="API failures are non-clears; no retry; complete-case scores and sensitivity reported separately",
        stopping_rule="all fixed seeds; no significance-based stopping", bootstrap_resamples=20000,
        mode_order="primitive first on even seeds; Edward first on odd seeds",
        options={m:runner_options(m,observation=args.observation) for m in MODES},
        prompts={m:prompt_contract_metadata(runner_options(m,observation=args.observation)["image_prompt_style"],
                  edward_options=m=="edward",fallback_mode="risk_ranked" if m=="edward" else "refuse") for m in MODES})
    write_json(root / "batch_manifest.json", manifest)
    key = load_key()

    def evaluate(seed):
        results = []
        for mode in (MODES if seed % 2 == 0 else MODES[::-1]):
            check_sources(frozen)
            folder = root / f"seed{seed}" / f"{mode}_seed{seed}"
            start = time.perf_counter()
            row = asyncio.run(run_one(mode,seed,args.max_steps,False,True,key,folder,observation=args.observation))
            row["episode_wall_seconds"] = time.perf_counter()-start
            try:
                row["validation"] = validate_artifacts(folder,args.observation)
            except Exception as exc:
                # Exact failure context is in local artifacts; no server/credential text is emitted.
                row.update(status="failed", error_type=type(exc).__name__,error="Local artifact validation failed")
            write_json(folder / "batch_summary.json",row)
            results.append(row)
            print(json.dumps(row),flush=True)
            check_sources(frozen)
        return results

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate,s) for s in manifest["seeds"]]
        for future in as_completed(futures):
            rows.extend(future.result())
            write_json(root / "progress.json",dict(completed_episodes=len(rows),planned_episodes=2*args.episodes,results=rows))
    check_sources(frozen)
    report = summarize(root,manifest,rows)
    write_json(root / "batch_complete.json",dict(episodes=len(rows),source_unchanged=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    key = None
    print(json.dumps(report["aggregates"]),flush=True)
    print(str(root / "comparison.md"),flush=True)


if __name__ == "__main__":
    main()
