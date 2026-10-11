#!/usr/bin/env python3
"""
patch_effect_size.py — make iso_analysis.py record the effect size it is supposed to report.

The method section says every comparison reports the mean difference, a bootstrap
interval, a p-value AND effect sizes. The p-value and the interval are there. The effect
sizes are not: no table and no sentence in the paper carries a single Cohen's d. The
method promises something the results never deliver, which is the kind of gap a careful
reader checks for.

The quantity is already in the code, one line from being saved: iso_analysis.py computes
the per-seed mean differences and their standard deviation is all Cohen's d needs. This
patch writes diff_sd and cohen_d into the comparison CSV, and holm_table.py then prints
them into the table rows.

Cliff's delta is deliberately not added. Over three paired observations a rank statistic
takes only a handful of values and is almost always +1 or -1, which looks like strong
evidence and is not. The manuscript text drops the claim instead of printing the number.

Idempotent, verifies its anchors, writes one .bak.

    python scripts/patch_effect_size.py [--check]
"""
import argparse
import sys
from pathlib import Path

ISO = Path.home() / "RAODpp" / "scripts" / "iso_analysis.py"
MARKER = "cohen_d"

OLD = '''        comp_rows.append({"cost": a.cost, "arm_a": x, "arm_b": y, "comparable": 1,
                          "overlap_lo": float(lo), "overlap_hi": float(hi),
                          "n_seeds": len(seeds),
                          "diff_mean": float(per_seed_mean.mean()),
                          "ci_lo": ci[0], "ci_hi": ci[1],
                          "p_raw": float(t.pvalue), "practical": int(practical)})'''

NEW = '''        sd = float(per_seed_mean.std(ddof=1))
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
                          "p_raw": float(t.pvalue), "practical": int(practical)})'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not ISO.exists():
        print(f"FAIL: {ISO} not found")
        sys.exit(1)
    s = ISO.read_text(encoding="utf-8")

    if MARKER in s:
        print("iso_analysis.py: already records the effect size, nothing to do")
        return
    if OLD not in s:
        print("FAIL: the comparison-row anchor was not found. Run "
              "patch_iso_comparison_csv.py first; if it has already run, send the file.")
        sys.exit(1)
    if a.check:
        print("iso_analysis.py: anchor found, would add diff_sd and cohen_d "
              "(--check, nothing written)")
        return

    bak = ISO.with_suffix(".py.bak-effect")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
        print(f"backup -> {bak}")
    ISO.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
    print("iso_analysis.py: comparison CSV now carries diff_sd and cohen_d")
    print("\nnext:  bash scripts/run_figures_v3.sh   then   python scripts/holm_table.py")


if __name__ == "__main__":
    main()
