#!/usr/bin/env python3
"""
patch_iso_comparison_csv.py — make iso_analysis.py persist its comparison statistics.

iso_analysis.py computes the mean difference, the bootstrap interval and the paired-t
p-value for every pair of arms, prints them, and throws them away. Only the per-budget
curve is written to disk. Its own docstring promises an
iso_{cost}_comparison.csv that the code never writes.

That is how Table 4 drifted. Its differences and confidence intervals match the current
analysis, but its p-column does not match a Holm correction of the current p-values, and
no file exists that would have revealed the gap. A number that lives only in a terminal
scroll cannot be checked against anything later.

This patch adds one list and one to_csv, so each run leaves
output/figdata/iso_{dataset}_{cost}_comparison.csv behind with:

    cost, arm_a, arm_b, comparable, overlap_lo, overlap_hi, n_seeds,
    diff_mean, ci_lo, ci_hi, p_raw, practical

holm_table.py then reads those files, applies Holm over the pre-registered set, and
prints the LaTeX rows. Nothing in the analysis itself changes: the same numbers that were
printed before are printed now, and additionally saved.

Idempotent, verifies its anchors, writes one .bak.

    python scripts/patch_iso_comparison_csv.py
    python scripts/patch_iso_comparison_csv.py --check
"""
import argparse
import sys
from pathlib import Path

ISO = Path.home() / "RAODpp" / "scripts" / "iso_analysis.py"
MARKER = "comp_rows"

EDITS = [
    # 1. a second accumulator beside the curve rows
    ("""    rows = []
    for x, y in itertools.combinations(arms, 2):""",
     """    rows = []
    comp_rows = []          # one row per pair: the statistics, not the curve
    for x, y in itertools.combinations(arms, 2):"""),

    # 2. record the pairs that cannot be compared, rather than only printing it
    ("""            rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "overlap": 0})
            continue""",
     """            rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "overlap": 0})
            comp_rows.append({"cost": a.cost, "arm_a": x, "arm_b": y,
                              "comparable": 0, "overlap_lo": float(lo),
                              "overlap_hi": float(hi), "n_seeds": len(seeds)})
            continue"""),

    # 3. record the statistics of the pairs that can
    ("""        practical = abs(per_seed_mean.mean()) >= PRACTICAL_FLOOR""",
     """        practical = abs(per_seed_mean.mean()) >= PRACTICAL_FLOOR
        comp_rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "comparable": 1,
                          "overlap_lo": float(lo), "overlap_hi": float(hi),
                          "n_seeds": len(seeds),
                          "diff_mean": float(per_seed_mean.mean()),
                          "ci_lo": ci[0], "ci_hi": ci[1],
                          "p_raw": float(t.pvalue), "practical": int(practical)})"""),

    # 4. write it beside the curve
    ("""    pd.DataFrame(rows).to_csv(FIG / f"iso_{a.dataset}_{a.cost}_curve.csv", index=False)""",
     """    pd.DataFrame(rows).to_csv(FIG / f"iso_{a.dataset}_{a.cost}_curve.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(
        FIG / f"iso_{a.dataset}_{a.cost}_comparison.csv", index=False)"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not ISO.exists():
        print(f"FAIL: {ISO} not found")
        sys.exit(1)
    s = ISO.read_text(encoding="utf-8")

    if MARKER in s:
        print("iso_analysis.py: already writes the comparison CSV, nothing to do")
        return

    missing = [old.splitlines()[-1].strip()[:60] for old, _ in EDITS if old not in s]
    if missing:
        print("FAIL: anchors not found in iso_analysis.py:")
        for m in missing:
            print(f"  ...{m}")
        print("Send the current iso_analysis.py and the patch will be rebuilt against it.")
        sys.exit(1)

    if a.check:
        print("iso_analysis.py: all 4 anchors found, would add the comparison CSV "
              "(--check, nothing written)")
        return

    bak = ISO.with_suffix(".py.bak-comp")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
        print(f"backup -> {bak}")
    for old, new in EDITS:
        s = s.replace(old, new, 1)
    ISO.write_text(s, encoding="utf-8")
    print("iso_analysis.py: now writes iso_{dataset}_{cost}_comparison.csv")
    print("\nnext:  bash scripts/run_figures_v3.sh   then   python scripts/holm_table.py")


if __name__ == "__main__":
    main()
