#!/usr/bin/env python3
"""
iso_analysis.py — compare the arms at *equal cost*, not at nominally paired conditions.

The pre-registered pairs of protocol v2.0 assumed the paired conditions sat at the same
latency. Clean measurement showed they do not (e.g. R1280 6.61 ms vs S960 5.61 ms), so a
direct pair comparison silently gives one arm a larger budget.

This script instead builds an accuracy-vs-cost curve per arm, per seed, and compares the
arms only inside the cost range where both arms actually have measurements. No
extrapolation is performed; budgets outside the overlap are dropped and reported.

    python scripts/iso_analysis.py --cost latency
    python scripts/iso_analysis.py --cost energy

Outputs: output/figdata/iso_{cost}_curve.csv, iso_{cost}_comparison.csv
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.home() / "RAODpp"
OUT, FIG = ROOT / "output", ROOT / "output" / "figdata"

ARMS = {
    "resolution": ["E0", "R640", "R960", "R1280", "R1536"],  # E0 is VisDrone's 640
    # E0 is VisDrone's 640 baseline, R640 is UAVDT's; a dataset has one or the
    # other, so listing both lets the same arm definition serve both datasets
    "capacity":   ["E0", "R640", "S640", "M640", "L640"],     # capacity only, fixed 640 px
    "capacity_mixed": ["S640", "S960", "S1280"],      # yolov8s at 640-1280
    "p2head":     ["P640", "P960", "P1280"],          # yolov8n + P2 head at 640-1280
}
N_GRID = 25
PRACTICAL_FLOOR = 0.005   # protocol v1.1: smaller gains are reported as negligible


def load(metric, dataset, metric_cost='latency'):
    d = pd.read_csv(OUT / "runs.csv")
    if "dataset" in d.columns:
        d = d[d.dataset == dataset]
    known = {c for arm in ARMS.values() for c in arm}
    orphan = sorted(set(d.exp) - known)
    if orphan:
        print(f"WARNING: {orphan} appear in runs.csv for this dataset but belong to no "
              f"arm, so they take part in no comparison. Add them to ARMS or ignore "
              f"deliberately.")
    d = d[d.exp.isin(known)]
    need = ["exp", "seed", metric, "ms_mean", "power_w"]
    missing = [c for c in need if c not in d.columns]
    assert not missing, f"runs.csv lacks {missing}"
    d = d.dropna(subset=need)
    if metric_cost == "energy_marginal" and "idle_w" not in d.columns:
        raise SystemExit("no idle_w column: re-run measure_cost_only.py so that marginal "
                         "energy can be computed")
    d["energy_j"] = d.ms_mean * d.power_w / 1000.0
    # energy above the idle floor: NVML reports total board draw, so the total figure
    # charges every arm for the card merely being powered on
    if "idle_w" in d.columns and d.idle_w.notna().any():
        d["energy_marginal_j"] = (d.ms_mean * (d.power_w - d.idle_w) / 1000.0)
    return d


def arm_curve(d, arm, cost_col, metric, seed):
    """(cost, accuracy) points of one arm for one seed, sorted by cost."""
    sub = d[(d.exp.isin(ARMS[arm])) & (d.seed == seed)]
    sub = sub.groupby("exp")[[cost_col, metric]].mean().sort_values(cost_col)
    return sub[cost_col].values, sub[metric].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", choices=["latency", "energy", "energy_marginal"],
                    default="latency")
    ap.add_argument("--metric", default="mAP50_95")
    ap.add_argument("--dataset", default="visdrone")
    a = ap.parse_args()
    cost_col = {"latency": "ms_mean", "energy": "energy_j",
                "energy_marginal": "energy_marginal_j"}[a.cost]
    unit = "ms" if a.cost == "latency" else "J/frame"

    d = load(a.metric, a.dataset, a.cost)

    # an arm with no runs on this dataset is dropped rather than allowed to empty the
    # seed intersection: the UAVDT grid deliberately omits the P2 arm
    arms = {k: v for k, v in ARMS.items() if len(d[d.exp.isin(v)])}
    dropped = [k for k in ARMS if k not in arms]
    if dropped:
        print(f"arms without runs on {a.dataset}, skipped: {dropped}")
    assert len(arms) >= 2, f"need at least two arms with data, have {list(arms)}"
    seeds = sorted(set.intersection(*[set(d[d.exp.isin(v)].seed) for v in arms.values()]))
    assert len(seeds) >= 2, f"need >=2 shared seeds, found {seeds}"
    print(f"cost = {a.cost} ({unit}) | metric = {a.metric} | seeds = {seeds}\n")

    # per-arm coverage, per seed, then the intersection across seeds
    cov = {}
    for arm in arms:
        lo = max(arm_curve(d, arm, cost_col, a.metric, s)[0].min() for s in seeds)
        hi = min(arm_curve(d, arm, cost_col, a.metric, s)[0].max() for s in seeds)
        cov[arm] = (lo, hi)
        pts = arm_curve(d, arm, cost_col, a.metric, seeds[0])
        found = sorted(set(d[d.exp.isin(arms[arm])].exp))
        print(f"{arm:11s} covers {lo:.3f}-{hi:.3f} {unit}  "
              f"({len(pts[0])} points: {', '.join(f'{x:.2f}' for x in pts[0])})"
              f"  [{', '.join(found)}]")

    rows = []
    comp_rows = []          # one row per pair: the statistics, not the curve
    for x, y in itertools.combinations(arms, 2):
        lo, hi = max(cov[x][0], cov[y][0]), min(cov[x][1], cov[y][1])
        if hi <= lo:
            print(f"\n{x} vs {y}: no overlapping {a.cost} range — not comparable")
            rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "overlap": 0})
            comp_rows.append({"cost": a.cost, "arm_a": x, "arm_b": y,
                              "comparable": 0, "overlap_lo": float(lo),
                              "overlap_hi": float(hi), "n_seeds": len(seeds)})
            continue
        grid = np.linspace(lo, hi, N_GRID)
        per_seed = []
        for s in seeds:
            cx, vx = arm_curve(d, x, cost_col, a.metric, s)
            cy, vy = arm_curve(d, y, cost_col, a.metric, s)
            per_seed.append(np.interp(grid, cx, vx) - np.interp(grid, cy, vy))
        diff = np.array(per_seed)                      # seeds x grid
        mean_by_budget = diff.mean(axis=0)

        for g, m_, sd in zip(grid, mean_by_budget, diff.std(axis=0, ddof=1)):
            rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "budget": float(g),
                         "diff_mean": float(m_), "diff_std": float(sd), "overlap": 1})

        # one paired test on the seed-level average difference over the overlap
        per_seed_mean = diff.mean(axis=1)
        t = stats.ttest_rel(per_seed_mean, np.zeros_like(per_seed_mean))
        rng = np.random.default_rng(0)
        boot = rng.choice(per_seed_mean, size=(10000, len(per_seed_mean)),
                          replace=True).mean(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        win = "a" if per_seed_mean.mean() > 0 else "b"
        practical = abs(per_seed_mean.mean()) >= PRACTICAL_FLOOR
        sd = float(per_seed_mean.std(ddof=1))
        # paired Cohen's d: the mean difference in units of its own spread across seeds.
        # With three seeds the denominator is itself poorly determined, which is why the
        # manuscript reports d beside the interval rather than in place of it.
        cohen_d = float(per_seed_mean.mean() / sd) if sd > 0 else float("nan")
        comp_rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "comparable": 1,
                          "overlap_lo": float(lo), "overlap_hi": float(hi),
                          "n_seeds": len(seeds),
                          "diff_mean": float(per_seed_mean.mean()),
                          "diff_sd": sd, "cohen_d": cohen_d,
                          "ci_lo": ci[0], "ci_hi": ci[1],
                          "p_raw": float(t.pvalue), "practical": int(practical)})
        print(f"\n{x} vs {y}  over {lo:.3f}-{hi:.3f} {unit}")
        print(f"  mean difference ({x} - {y}) = {per_seed_mean.mean():+.4f} "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  p={float(t.pvalue):.4f}")
        print(f"  at the low end  ({lo:.3f}): {mean_by_budget[0]:+.4f}")
        print(f"  at the high end ({hi:.3f}): {mean_by_budget[-1]:+.4f}")
        print(f"  -> {'ARM ' + (x if win=='a' else y) + ' ahead'}"
              f"{' (practically relevant)' if practical else ' (below the 0.005 floor)'}")

    pd.DataFrame(rows).to_csv(FIG / f"iso_{a.dataset}_{a.cost}_curve.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(
        FIG / f"iso_{a.dataset}_{a.cost}_comparison.csv", index=False)

    # raw per-condition table so the curve can be redrawn
    tab = (d.groupby("exp")[[a.metric, "ms_mean", "power_w", "energy_j"]]
           .agg(["mean", "std"]).round(4))
    tab.to_csv(FIG / f"iso_{a.dataset}_{a.cost}_conditions.csv")
    print("\n=== conditions (mean over seeds) ===")
    cols = [a.metric, "ms_mean", "energy_j"] + (
        ["energy_marginal_j"] if "energy_marginal_j" in d.columns else [])
    print(d.groupby("exp")[cols].mean().round(4).sort_values("ms_mean").to_string())
    print(f"\nwritten: {FIG/f'iso_{a.dataset}_{a.cost}_curve.csv'}")


if __name__ == "__main__":
    main()
