"""Pass-rate statistics for start-state evaluations (dynamic-bank plan, section on evaluation).

Episode bootstrap stratified by seed (10,000 resamples, RNG seed 20261001): within every seed the
episodes are resampled with replacement and the pooled pass rate recomputed, so each seed keeps its
episode count. A "clear improvement" is a final-minus-baseline pass rate of at least 10 percentage
points whose 95% CI lower bound is above zero. Holdout seeds are reported separately.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261001
CLEAR_IMPROVEMENT = 0.10


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def by_seed(rows, group):
    seeds = {}
    for row in rows:
        if row["group"] == group:
            seeds.setdefault(int(row["seed"]), []).append(row)
    return dict(sorted(seeds.items()))


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _wins(rows):
    return np.array([float(r["terminal_reason"] == "all_normal_pellets") for r in rows])


def bootstrap_rates(seeds, rng, resamples=BOOTSTRAP_RESAMPLES):
    total = sum(len(rows) for rows in seeds.values())
    sums = np.zeros(resamples)
    for rows in seeds.values():
        wins = _wins(rows)
        draws = rng.integers(0, len(wins), size=(resamples, len(wins)))
        sums += wins[draws].sum(axis=1)
    return sums / total


def summarize(rows, group):
    seeds = by_seed(rows, group)
    out = dict(group=group, seeds={}, episodes=0, wins=0, terminals={})
    for seed, items in seeds.items():
        wins = int(_wins(items).sum())
        terminals = {}
        for r in items:
            terminals[r["terminal_reason"]] = terminals.get(r["terminal_reason"], 0) + 1
            out["terminals"][r["terminal_reason"]] = out["terminals"].get(r["terminal_reason"], 0) + 1
        out["seeds"][str(seed)] = dict(episodes=len(items), wins=wins, pass_rate=wins / len(items),
                                       wilson95=wilson(wins, len(items)), terminals=terminals)
        out["episodes"] += len(items)
        out["wins"] += wins
    if out["episodes"]:
        rates = bootstrap_rates(seeds, np.random.default_rng(BOOTSTRAP_SEED))
        out["pass_rate"] = out["wins"] / out["episodes"]
        out["bootstrap95"] = [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]
        out["death_rate"] = out["terminals"].get("death", 0) / out["episodes"]
        out["timeout_rate"] = out["terminals"].get("max_steps", 0) / out["episodes"]
    return out


def compare(baseline_rows, final_rows, group, seeds=None):
    base, final = by_seed(baseline_rows, group), by_seed(final_rows, group)
    if seeds is not None:
        base = {s: base[s] for s in seeds}
        final = {s: final[s] for s in seeds}
    if set(base) != set(final):
        raise ValueError("baseline and final evaluations cover different seeds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diff = bootstrap_rates(final, rng) - bootstrap_rates(base, rng)
    rate = lambda d: sum(_wins(r).sum() for r in d.values()) / sum(len(r) for r in d.values())
    point = rate(final) - rate(base)
    ci = [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
    per_seed = {str(s): float(_wins(final[s]).mean() - _wins(base[s]).mean()) for s in sorted(base)}
    return dict(group=group, seeds=sorted(base), baseline=rate(base), final=rate(final), difference=point,
                difference_bootstrap95=ci, clear_improvement=bool(point >= CLEAR_IMPROVEMENT and ci[0] > 0),
                # Reported alongside, not part of the pre-registered rule: a policy that collapses onto a few
                # seeds can raise the pooled rate while most seeds regress, and the within-seed bootstrap
                # above treats the seed set as fixed.
                per_seed_difference=per_seed,
                regressed_seeds=sorted(int(s) for s, d in per_seed.items() if d < 0),
                seed_cluster_bootstrap95=_seed_cluster_ci(base, final))


def _seed_cluster_ci(base, final, resamples=BOOTSTRAP_RESAMPLES):
    """Resample seeds (with their paired baseline/final episodes) and episodes within each seed.

    Rewards are binary, so resampling a seed's n episodes with replacement is a Binomial(n, p_hat) draw.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    seeds = sorted(base)
    stats = []
    for rows in (base, final):
        wins = [_wins(rows[s]) for s in seeds]
        if any(not np.isin(w, (0.0, 1.0)).all() for w in wins):
            raise ValueError("seed-cluster bootstrap expects binary rewards")
        stats.append((np.array([len(w) for w in wins]), np.array([w.mean() for w in wins])))
    picked = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    rates = []
    for n, p in stats:
        rates.append(rng.binomial(n[picked], p[picked]).sum(axis=1) / n[picked].sum(axis=1))
    diffs = rates[1] - rates[0]
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.results)
    report = dict(true_start=summarize(rows, "true_start"), holdout=summarize(rows, "holdout"))
    if args.baseline:
        base = load(args.baseline)
        report["vs_baseline"] = dict(true_start=compare(base, rows, "true_start"),
                                     holdout=compare(base, rows, "holdout"))
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
