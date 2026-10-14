#!/usr/bin/env python3
"""
verify_figures.py — check that the regenerated figures match the manuscript.

The figures and the manuscript are two places the same numbers live, and they have
already disagreed once: pareto.pdf and the iso_* figures were drawn before the capacity
arm was rebuilt and kept showing the mixed path as the capacity arm while the tables
showed the corrected one. This script compares the figure source CSVs against the values
printed in Table 4 and fails loudly on a mismatch, so the two cannot drift again silently.

    python scripts/verify_figures.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
FIG, FIGDATA = ROOT / "fig", ROOT / "output" / "figdata"

# Table 4 of the manuscript, and the conditions table above it. If the manuscript is
# edited these must be edited with it; that is the point of the file.
EXPECTED_ISO = {          # (cost axis, arm_a, arm_b): mean difference over the overlap
    ("latency",         "resolution", "capacity"): +0.0386,
    ("energy",          "resolution", "capacity"): +0.0550,
    ("energy_marginal", "resolution", "capacity"): +0.0582,
    ("latency",         "resolution", "p2head"):   +0.0337,
    ("energy",          "resolution", "p2head"):   +0.0378,
}
EXPECTED_MIXED = {
    ("latency", "capacity_mixed", "capacity"): +0.0666,
    ("energy",  "capacity_mixed", "capacity"): +0.0489,
}
EXPECTED_MAP = {          # condition -> mAP50-95 in the conditions table
    "E0": 0.1863, "S640": 0.2239, "R960": 0.2539, "S960": 0.2951,
    "P640": 0.2077, "P960": 0.2667, "M640": 0.2513, "R1280": 0.2985,
    "R1536": 0.3249, "S1280": 0.3360, "L640": 0.2672,
}
TOL = 0.0006              # the manuscript rounds to four places
FIGURES = ["pareto.pdf", "iso_latency.pdf", "iso_energy.pdf",
           "iso_energy_marginal.pdf", "iso_uavdt_latency.pdf",
           "iso_uavdt_energy.pdf", "iso_uavdt_energy_marginal.pdf",
           "iso_confound.pdf"]

fails, warns = [], []


def check_files():
    for f in FIGURES:
        p = FIG / f
        if not p.exists():
            fails.append(f"{f} was not produced")
        elif p.stat().st_size < 2000:
            fails.append(f"{f} is {p.stat().st_size} bytes, which is too small to be real")


def check_pareto():
    p = FIGDATA / "pareto_points.csv"
    if not p.exists():
        fails.append("pareto_points.csv missing: the patched fig_pareto did not run")
        return
    d = pd.read_csv(p)
    arms = set(d.arm)
    if "capacity_mixed" not in arms:
        fails.append("pareto has no capacity_mixed arm: the old arm map is still in use")
    cap = sorted(d[d.arm == "capacity"].condition.unique())
    if cap != ["E0", "L640", "M640", "S640"]:
        fails.append(f"pareto capacity arm is {cap}, expected ['E0','L640','M640','S640']")
    for cond, want in EXPECTED_MAP.items():
        got = d[d.condition == cond].map50_95.unique()
        if len(got) == 0:
            warns.append(f"{cond} absent from pareto_points.csv")
        elif abs(float(got[0]) - want) > TOL:
            fails.append(f"{cond}: figure says mAP {float(got[0]):.4f}, "
                         f"manuscript says {want:.4f}")


def iso_mean(dataset, cost, a, b):
    f = FIGDATA / f"iso_{dataset}_{cost}_curve.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[(d.overlap == 1)]
    sub = d[(d.arm_a == a) & (d.arm_b == b)]
    if len(sub):
        return float(sub.diff_mean.mean())
    sub = d[(d.arm_a == b) & (d.arm_b == a)]
    if len(sub):
        return -float(sub.diff_mean.mean())
    return None


def check_iso():
    for (cost, a, b), want in {**EXPECTED_ISO, **EXPECTED_MIXED}.items():
        got = iso_mean("visdrone", cost, a, b)
        if got is None:
            fails.append(f"{cost}: {a} - {b} absent from the curve CSV")
        elif abs(got - want) > 0.0015:
            fails.append(f"{cost}: {a} - {b} = {got:+.4f} in the figure data, "
                         f"{want:+.4f} in Table 4")
        else:
            print(f"  ok  {cost:<16} {a} - {b} = {got:+.4f}")


def main():
    print("figure files")
    check_files()
    print("pareto source")
    check_pareto()
    print("iso curves against Table 4")
    check_iso()

    print()
    for w in warns:
        print(f"WARN  {w}")
    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        print(f"\n{len(fails)} problem(s). The figures do not match the manuscript.")
        sys.exit(1)
    print("all figures agree with the manuscript"
          + (f" ({len(warns)} warning(s))" if warns else ""))


if __name__ == "__main__":
    main()
