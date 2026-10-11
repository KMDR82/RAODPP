#!/usr/bin/env python3
"""
dominance_table.py — the interpolation-free version of the paper's main claim.

A reviewer objected, correctly, that a mean difference computed on a 25-point grid between
arms with three or four measured points each is not 25 independent observations: it is a
numerical comparison of two sparse interpolated curves. Their question was whether the
result survives without the interpolation.

It can be answered with no new training and no curve fitting at all. A configuration
dominates another when it is BOTH more accurate AND cheaper. Dominance needs no
interpolation, no common grid, no iso-cost matching and no distributional assumption: it
is a direct comparison of two measured configurations, and if a resolution configuration
dominates a capacity configuration then no budget argument is needed to prefer it.

This script finds every such pair between the resolution arm and the isolated capacity
arm, on each cost axis, and reports two strengths of evidence:

    mean      the condition means dominate
    per-seed  dominance holds in every seed separately, which is the stronger claim
              because it cannot come from one lucky run

It prints a LaTeX table and writes output/figdata/dominance.csv.

    python scripts/dominance_table.py
    python scripts/dominance_table.py --dataset uavdt
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
OUT, FIGDATA = ROOT / "output", ROOT / "output" / "figdata"

RES_ARM = ["E0", "R640", "R960", "R1280", "R1536"]
CAP_ARM = ["E0", "R640", "S640", "M640", "L640"]
LABEL = {"E0": "n@640", "R640": "n@640", "R960": "n@960", "R1280": "n@1280",
         "R1536": "n@1536", "S640": "s@640", "M640": "m@640", "L640": "l@640"}
COSTS = [("ms_mean", "latency", "ms"), ("energy_j", "energy", "J"),
         ("energy_marginal_j", "marginal energy", "J")]
METRIC = "mAP50_95"


def load(dataset):
    d = pd.read_csv(OUT / "runs.csv")
    d = d[d.dataset == dataset] if "dataset" in d.columns else d
    d = d.dropna(subset=[METRIC, "ms_mean", "power_w"])
    d = d.assign(energy_j=d.ms_mean * d.power_w / 1000.0)
    if "idle_w" in d.columns:
        d["energy_marginal_j"] = d.ms_mean * (d.power_w - d.idle_w) / 1000.0
    return d


def run(dataset):
    """One dataset. Writes output/figdata/dominance_<dataset>.csv.

    One file per dataset, not one shared file: an earlier version wrote them all to
    dominance.csv, so running the second dataset silently erased the first and the
    graphical abstract came out with an empty panel.
    """
    d = load(dataset)

    present = set(d.exp)
    res = [c for c in RES_ARM if c in present]
    cap = [c for c in CAP_ARM if c in present]
    # the shared baseline belongs to both arms and cannot dominate itself
    pairs = [(r, c) for r in res for c in cap if r != c]
    if not pairs:
        raise SystemExit(f"no resolution/capacity pair present for {dataset}")

    seeds = sorted(set.intersection(*[set(d[d.exp == e].seed) for e in set(res) | set(cap)]))
    print(f"{dataset}: {len(res)} resolution condition(s), {len(cap)} capacity "
          f"condition(s), {len(seeds)} shared seed(s) {seeds}\n")

    mean = d.groupby("exp")[[METRIC] + [c for c, _, _ in COSTS
                                        if c in d.columns]].mean()
    rows = []
    for cost_col, cost_name, unit in COSTS:
        if cost_col not in d.columns:
            continue
        for r, c in pairs:
            dm = mean.loc[r, METRIC] - mean.loc[c, METRIC]
            dc = mean.loc[r, cost_col] - mean.loc[c, cost_col]
            if not (dm > 0 and dc < 0):
                continue                      # not dominance on the means
            per_seed = []
            for s in seeds:
                rr = d[(d.exp == r) & (d.seed == s)]
                cc = d[(d.exp == c) & (d.seed == s)]
                if rr.empty or cc.empty:
                    per_seed.append(False)
                    continue
                per_seed.append(
                    bool(rr[METRIC].mean() > cc[METRIC].mean()
                         and rr[cost_col].mean() < cc[cost_col].mean()))
            rows.append({
                "dataset": dataset, "cost": cost_name, "unit": unit,
                "better": r, "worse": c,
                "better_label": LABEL.get(r, r), "worse_label": LABEL.get(c, c),
                "map_better": round(mean.loc[r, METRIC], 4),
                "map_worse": round(mean.loc[c, METRIC], 4),
                "cost_better": round(mean.loc[r, cost_col], 3),
                "cost_worse": round(mean.loc[c, cost_col], 3),
                "d_map": round(dm, 4), "d_cost": round(dc, 3),
                "seeds_dominating": sum(per_seed), "n_seeds": len(seeds),
                "every_seed": int(all(per_seed) and len(per_seed) == len(seeds))})

    if not rows:
        print("No resolution configuration dominates a capacity configuration on any "
              "axis. The iso-cost comparison is then the only evidence, and the paper "
              "should say so.")
        return

    out = pd.DataFrame(rows)
    FIGDATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(FIGDATA / f"dominance_{dataset}.csv", index=False)

    strong = out[out.every_seed == 1]
    print(f"{len(out)} dominating pair(s); {len(strong)} hold in every seed\n")
    print(out[["cost", "better_label", "worse_label", "map_better", "map_worse",
               "cost_better", "cost_worse", "seeds_dominating", "n_seeds"]]
          .to_string(index=False))

    # which configurations dominate on every cost axis at once
    axes_n = out.groupby(["better", "worse"]).cost.nunique()
    allaxes = axes_n[axes_n == len([c for c, _, _ in COSTS if c in d.columns])]
    if len(allaxes):
        print("\ndominating on every cost axis measured:")
        for (r, c) in allaxes.index:
            ev = out[(out.better == r) & (out.worse == c)].every_seed.min()
            print(f"  {LABEL.get(r,r)} over {LABEL.get(c,c)}"
                  f"{'  (in every seed)' if ev else ''}")

    print("\n% ---------------- LaTeX, dominance table")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Axis & Pair & mAP$_{50\\text{-}95}$ & Cost & $\\Delta$ accuracy "
          "& Seeds \\\\")
    print("\\midrule")
    for r in out.itertuples():
        print(f"{r.cost.capitalize()} & {r.better_label} vs {r.worse_label} & "
              f"${r.map_better:.4f}$ vs ${r.map_worse:.4f}$ & "
              f"${r.cost_better:.3f}$ vs ${r.cost_worse:.3f}$\\,{r.unit} & "
              f"${r.d_map:+.4f}$ & ${r.seeds_dominating}/{r.n_seeds}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\nwritten: {FIGDATA / f'dominance_{dataset}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both",
                    choices=["visdrone", "uavdt", "both"],
                    help="default: both, each to its own CSV")
    a = ap.parse_args()
    for ds in (["visdrone", "uavdt"] if a.dataset == "both" else [a.dataset]):
        print(f"\n{'=' * 72}\n{ds}\n{'=' * 72}")
        try:
            run(ds)
        except SystemExit as e:
            print(f"  skipped: {e}")


if __name__ == "__main__":
    main()
