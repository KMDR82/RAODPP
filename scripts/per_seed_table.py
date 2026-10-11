#!/usr/bin/env python3
"""
per_seed_table.py — the raw per-seed measurements, as an appendix table.

The paper currently says the per-seed values are in the released run table rather than
printed. With three seeds per condition that is a thin answer: the whole question a reader
has about a three-seed study is what the three numbers were. They fit on one page, so they
should be on it.

Prints a LaTeX table of every seed of every condition, accuracy and the two cost axes, and
writes output/figdata/per_seed_<dataset>.csv beside it.

    python scripts/per_seed_table.py
    python scripts/per_seed_table.py --dataset uavdt
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "RAODpp"
OUT, FIGDATA = ROOT / "output", ROOT / "output" / "figdata"

ORDER = ["E0", "R640", "S640", "R960", "S960", "P640", "P960",
         "M640", "R1280", "R1536", "S1280", "L640"]
LABEL = {"E0": "n@640", "R640": "n@640", "R960": "n@960", "R1280": "n@1280",
         "R1536": "n@1536", "S640": "s@640", "S960": "s@960", "S1280": "s@1280",
         "M640": "m@640", "L640": "l@640", "P640": "n-p2@640", "P960": "n-p2@960"}
ARM = {"E0": "resolution, capacity", "R640": "resolution, capacity",
       "R960": "resolution", "R1280": "resolution", "R1536": "resolution",
       "S640": "capacity, mixed", "M640": "capacity", "L640": "capacity",
       "S960": "mixed", "S1280": "mixed", "P640": "P2", "P960": "P2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone")
    a = ap.parse_args()

    d = pd.read_csv(OUT / "runs.csv")
    if "dataset" in d.columns:
        d = d[d.dataset == a.dataset]
    d = d.dropna(subset=["mAP50_95", "ms_mean", "power_w"])
    d = d[d.exp.isin(ORDER)]
    if d.empty:
        raise SystemExit(f"no rows for {a.dataset}")
    d = d.assign(energy_j=d.ms_mean * d.power_w / 1000.0)
    d["order"] = d.exp.map({c: i for i, c in enumerate(ORDER)})
    d = d.sort_values(["order", "seed"])

    keep = ["exp", "seed", "mAP50_95", "ms_mean", "energy_j"]
    FIGDATA.mkdir(parents=True, exist_ok=True)
    d[keep].to_csv(FIGDATA / f"per_seed_{a.dataset}.csv", index=False)

    print(f"{a.dataset}: {len(d)} runs over {d.exp.nunique()} conditions\n")
    print("% ---------------- LaTeX, per-seed appendix table")
    print("\\begin{tabular}{llrccc}")
    print("\\toprule")
    print("Code & Model @ size & Seed & mAP$_{50\\text{-}95}$ & Latency (ms) "
          "& J/frame \\\\")
    print("\\midrule")
    prev = None
    for r in d.itertuples():
        if prev is not None and r.exp != prev:
            print("\\addlinespace")
        code = r.exp if r.exp != prev else ""
        label = LABEL.get(r.exp, r.exp) if r.exp != prev else ""
        print(f"{code} & {label} & {int(r.seed)} & ${r.mAP50_95:.4f}$ & "
              f"${r.ms_mean:.3f}$ & ${r.energy_j:.3f}$ \\\\")
        prev = r.exp
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\n% per-condition spread, for the caption")
    g = d.groupby("exp").mAP50_95.agg(["count", "mean", "std"]).reindex(
        [c for c in ORDER if c in set(d.exp)])
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwritten: {FIGDATA / f'per_seed_{a.dataset}.csv'}")


if __name__ == "__main__":
    main()
