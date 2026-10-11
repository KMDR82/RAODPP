#!/usr/bin/env python3
"""
holm_table.py — apply Holm correction to the iso-cost comparisons and print the
manuscript's tables as LaTeX, from files rather than from memory.

Why. Table 4's p-column is labelled p_Holm and its caption says Holm corrects over the
four pre-registered comparisons, but the values in it are not a Holm correction of the
analysis's current p-values: two of them are the raw paired-t p-values and one matches
neither. Table 5's latency entry is 2x the raw p where the step-down procedure gives 1x.
The differences and the confidence intervals in both tables are correct; only the
correction is wrong, and it was wrong because nothing recomputed it when the arms changed.

This script removes the hand step. It reads the comparison CSVs that
patch_iso_comparison_csv.py makes iso_analysis.py write, applies Holm over the
pre-registered set for each dataset, marks everything else exploratory, and prints table
rows ready to paste. Run it after run_figures_v3.sh.

    python scripts/holm_table.py
    python scripts/holm_table.py --dataset uavdt

The pre-registered sets are fixed here because they were fixed in the protocol before the
results were seen. Do not add to them to make something significant: anything not listed
prints under "exploratory" and carries no corrected p-value, which is the honest place for
a comparison that became possible only after an arm definition was corrected.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "RAODpp"
FIGDATA = ROOT / "output" / "figdata"
COSTS = ["latency", "energy", "energy_marginal"]

# protocol v2.1 C4 for VisDrone; v2.3 for UAVDT, where the capacity arm available at
# pre-registration time was the mixed one and no P2 arm was run
PREREGISTERED = {
    "visdrone": [("latency", "resolution", "capacity"),
                 ("energy",  "resolution", "capacity"),
                 ("latency", "resolution", "p2head"),
                 ("energy",  "resolution", "p2head")],
    "uavdt":    [("latency", "resolution", "capacity_mixed"),
                 ("energy",  "resolution", "capacity_mixed")],
}
AXIS = {"latency": "Latency", "energy": "Energy", "energy_marginal": "Marginal"}
UNIT = {"latency": "ms", "energy": "J", "energy_marginal": "J"}
NAME = {"resolution": "resolution", "capacity": "capacity",
        "capacity_mixed": "mixed path", "p2head": "P2 head"}


def holm(pvals):
    """Holm-Bonferroni step-down, order preserved. Monotone by construction."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load(dataset):
    frames = []
    for cost in COSTS:
        f = FIGDATA / f"iso_{dataset}_{cost}_comparison.csv"
        if not f.exists():
            print(f"  missing: {f.name}")
            continue
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit(
            "no comparison CSV found. Run patch_iso_comparison_csv.py, then "
            "run_figures_v3.sh, then this script.")
    return pd.concat(frames, ignore_index=True)


def fmt_p(v):
    return "$<0.0001$" if v < 0.0001 else f"${v:.4f}$"


def row(r, p_col):
    lo, hi = r.overlap_lo, r.overlap_hi
    u = UNIT[r.cost]
    span = (f"${lo:.3f}$--${hi:.3f}$\\,{u}" if r.cost != "latency"
            else f"${lo:.3f}$--${hi:.3f}$\\,ms")
    d = (f"${r.cohen_d:+.2f}$" if hasattr(r, "cohen_d") and pd.notna(r.cohen_d)
         else "---")
    return (f"{AXIS[r.cost]} & {NAME.get(r.arm_a, r.arm_a)} $-$ "
            f"{NAME.get(r.arm_b, r.arm_b)} & {span} & ${r.diff_mean:+.4f}$ & "
            f"$[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]$ & {d} & {p_col} \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both",
                    choices=["visdrone", "uavdt", "both"])
    a = ap.parse_args()
    datasets = ["visdrone", "uavdt"] if a.dataset == "both" else [a.dataset]

    for ds in datasets:
        print(f"\n{'='*72}\n{ds}\n{'='*72}")
        d = load(ds)
        d["key"] = list(zip(d.cost, d.arm_a, d.arm_b))

        pre = PREREGISTERED[ds]
        sel = d[d.key.isin(pre) & (d.comparable == 1)].copy()
        absent = [k for k in pre if k not in set(d[d.comparable == 1].key)]
        if absent:
            print("pre-registered but not comparable on this dataset "
                  "(no overlapping budget):")
            for k in absent:
                print(f"  {k[0]}: {NAME.get(k[1],k[1])} - {NAME.get(k[2],k[2])}")
            print()

        if len(sel):
            sel["p_holm"] = holm(sel.p_raw.values)
            print(f"pre-registered, Holm over M={len(sel)} "
                  f"(the protocol fixed M={len(pre)}; "
                  f"{len(pre)-len(sel)} could not be computed)")
            cols = ["cost", "arm_a", "arm_b", "diff_mean"]
            cols += [c for c in ("diff_sd", "cohen_d") if c in sel.columns]
            cols += ["p_raw", "p_holm"]
            print(sel[cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
            print("\n% --- pre-registered rows")
            for r in sel.itertuples():
                print(row(r, fmt_p(r.p_holm)))

        ex = d[(~d.key.isin(pre)) & (d.comparable == 1)].copy()
        ex = ex.sort_values(["cost", "arm_a", "arm_b"])
        if len(ex):
            print("\n% --- exploratory, not pre-registered, no corrected p-value")
            for r in ex.itertuples():
                print(row(r, "---"))
            print(f"\n(raw p-values for the exploratory rows, for the caption only: "
                  + ", ".join(f"{NAME.get(r.arm_a,r.arm_a)}-{NAME.get(r.arm_b,r.arm_b)} "
                              f"{r.cost} p={r.p_raw:.4f}" for r in ex.itertuples()) + ")")

        bad = d[d.comparable == 0]
        if len(bad):
            print("\nnot comparable (no overlapping budget):")
            for r in bad.itertuples():
                print(f"  {r.cost:16s} {NAME.get(r.arm_a, r.arm_a)} - "
                      f"{NAME.get(r.arm_b, r.arm_b)}")


if __name__ == "__main__":
    main()
