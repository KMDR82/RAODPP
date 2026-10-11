#!/usr/bin/env python3
"""
analyze_results.py — RAOD++ F8 statistical analysis (protocol v1.1 A1).

    python scripts/analyze_results.py                 # primary metric mAP50-95
    python scripts/analyze_results.py --metric mAP50

Pre-registered plan, fixed before any comparison was computed:
  primary test        seed-matched paired t-test
  confirmation        Wilcoxon signed-rank (primary if normality is rejected)
  correction          Holm over the 4 pre-registered comparisons
  effect size         paired Cohen's d and Cliff's delta
  interval            bootstrap 95% CI of the mean difference, 10,000 resamples
  practical floor     a gain below 0.005 mAP50-95 is reported as detectable but negligible

Pre-registered comparisons: E5-E1, E5-E2, E5-E3, E4-E3.
Everything else printed here is exploratory and labelled as such.
"""
import argparse, itertools, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.home() / "RAODpp"
OUT, FIGDATA = ROOT / "output", ROOT / "output" / "figdata"
PRIMARY_SETS = {
    # v1: was the enrichment worth anything against equal-budget controls?
    "v1": [("E5", "E1"), ("E5", "E2"), ("E5", "E3"), ("E4", "E3")],
    # v2 (protocol v2.0 B5): does resolution beat capacity / a P2 head at equal latency?
    "v2": [("R960", "S640"), ("R1280", "S960"), ("R960", "P640"), ("R1280", "P960")],
}
COST_COLS = ["ms_mean", "fps", "power_w", "imgsz"]
# a study is analysed only over its own conditions; otherwise a condition that exists
# for one seed only (e.g. a single-seed sweep) collapses the paired table to n=1
STUDY_CONDS = {
    "v1": ["E0", "E1", "E2", "E3", "E4", "E5"],
    "v2": ["E0", "S640", "P640", "R960", "P960", "R1280", "S960"],
}
PRACTICAL_FLOOR = 0.005
N_BOOT = 10_000


def cliffs_delta(a, b):
    n = len(a) * len(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / n


def paired_cohens_d(d):
    return float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.nan


def boot_ci(d, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(idx):
        v = (len(pvals) - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(run, 1.0)
    return adj


def compare(a_vals, b_vals, a, b, metric):
    d = a_vals - b_vals
    n = len(d)
    sh_p = float(stats.shapiro(d).pvalue) if 3 <= n <= 5000 else np.nan
    t_p = float(stats.ttest_rel(a_vals, b_vals).pvalue)
    try:
        w_p = float(stats.wilcoxon(a_vals, b_vals).pvalue)
    except ValueError:
        w_p = np.nan
    lo, hi = boot_ci(d)
    normal = (sh_p >= 0.05) if not np.isnan(sh_p) else True
    return {
        "comparison": f"{a} - {b}", "metric": metric, "n_seeds": n,
        f"{a}_mean": float(a_vals.mean()), f"{b}_mean": float(b_vals.mean()),
        "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
        "ci_low": lo, "ci_high": hi,
        "shapiro_p": sh_p, "normality_ok": bool(normal),
        "p_ttest": t_p, "p_wilcoxon": w_p,
        "p_primary": t_p if normal else w_p,
        "primary_test": "paired t" if normal else "wilcoxon",
        "cohens_d": paired_cohens_d(d),
        "cliffs_delta": cliffs_delta(a_vals, b_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mAP50_95")
    ap.add_argument("--dataset", default="visdrone")
    ap.add_argument("--study", choices=list(PRIMARY_SETS), default="v2")
    a = ap.parse_args()
    m = a.metric
    PRIMARY = PRIMARY_SETS[a.study]

    df = pd.read_csv(OUT / "runs.csv")
    df = df[df.dataset == a.dataset]
    wide = df.pivot_table(index="seed", columns="exp", values=m)
    conds = [c for c in STUDY_CONDS[a.study] if c in wide.columns]
    missing = [c for c in STUDY_CONDS[a.study] if c not in wide.columns]
    if missing:
        print(f"WARNING: conditions absent from runs.csv: {missing}")
    wide = wide[conds].dropna()
    if len(wide) < 2:
        raise SystemExit(f"only {len(wide)} seed(s) have all of {conds}; "
                         "paired tests need at least 2 (protocol asks for 3+)")
    print(f"metric = {m} | complete seeds = {len(wide)} | conditions = {conds}\n")

    # ---------- descriptive table ----------
    desc = pd.DataFrame({
        "mean": wide.mean(), "std": wide.std(ddof=1), "min": wide.min(),
        "max": wide.max(), "n": wide.count(),
    }).round(4)
    have = [c for c in COST_COLS if c in df.columns]
    if have:
        cost = df.groupby("exp")[have].mean()
        desc = desc.join(cost.round(3))
        if {"ms_mean", "power_w"} <= set(desc.columns):
            desc["J_frame"] = (desc.ms_mean * desc.power_w / 1000).round(3)
    if "E0" in wide:
        desc["delta_vs_E0"] = (wide.mean() - wide["E0"].mean()).round(4)
    print("=== descriptive ===")
    print(desc.to_string(), "\n")
    desc.to_csv(FIGDATA / f"results_descriptive_{m}.csv")

    # ---------- pre-registered comparisons ----------
    rows = [compare(wide[x].values, wide[y].values, x, y, m)
            for x, y in PRIMARY if x in wide and y in wide]
    if not rows:
        print("no pre-registered comparison is available yet"); return
    res = pd.DataFrame(rows)
    if res.p_primary.isna().any():
        raise SystemExit("a p-value came back NaN (usually n<3 or zero variance); "
                         "refusing to report a verdict on that basis")
    res["p_holm"] = holm(res.p_primary.values)
    res["significant"] = res.p_holm < 0.05
    res["practically_relevant"] = res.diff_mean.abs() >= PRACTICAL_FLOOR
    res["verdict"] = np.where(
        ~res.significant, "no detectable difference",
        np.where(res.practically_relevant, "significant and practically relevant",
                 "detectable but practically negligible"))

    show = ["comparison", "diff_mean", "ci_low", "ci_high", "p_primary", "p_holm",
            "cohens_d", "cliffs_delta", "verdict"]
    print("=== pre-registered comparisons (Holm over 4) ===")
    print(res[show].round(4).to_string(index=False), "\n")
    res.round(6).to_csv(FIGDATA / f"stats_primary_{m}.csv", index=False)

    # ---------- exploratory pairs ----------
    extra = [(x, y) for x, y in itertools.combinations(conds, 2)
             if (y, x) not in PRIMARY and (x, y) not in PRIMARY]
    if extra:
        ex = pd.DataFrame([compare(wide[y].values, wide[x].values, y, x, m)
                           for x, y in extra])
        ex["note"] = "exploratory, not Holm-corrected"
        print("=== exploratory (uncorrected) ===")
        print(ex[["comparison", "diff_mean", "p_primary", "cohens_d"]]
              .round(4).to_string(index=False), "\n")
        ex.round(6).to_csv(FIGDATA / f"stats_exploratory_{m}.csv", index=False)

    # ---------- Pareto front on accuracy vs cost (protocol v2.0 B5) ----------
    if {"ms_mean"} <= set(desc.columns):
        pf = desc.dropna(subset=["ms_mean"]).copy()
        axes = ["ms_mean"] + (["J_frame"] if "J_frame" in pf.columns else [])
        dominated = {}
        for i in pf.index:
            for j in pf.index:
                if i == j:
                    continue
                better_or_equal = (pf.loc[j, "mean"] >= pf.loc[i, "mean"] and
                                   all(pf.loc[j, c] <= pf.loc[i, c] for c in axes))
                strictly_better = (pf.loc[j, "mean"] > pf.loc[i, "mean"] or
                                   any(pf.loc[j, c] < pf.loc[i, c] for c in axes))
                if better_or_equal and strictly_better:
                    dominated[i] = j
                    break
        pf["pareto"] = ["dominated by " + dominated[i] if i in dominated else "ON FRONT"
                        for i in pf.index]
        print("=== accuracy vs cost (Pareto over", ", ".join(["mean"] + axes), ") ===")
        cols = ["mean"] + axes + ["pareto"]
        print(pf[cols].sort_values("ms_mean").to_string(), "\n")
        pf.to_csv(FIGDATA / f"pareto_{m}.csv")

    # ---------- failure criteria (protocol F2 / K1) ----------
    treat, controls = (("E5", ["E1", "E2", "E3"]) if a.study == "v1"
                       else ("resolution arm", []))
    if a.study == "v2":
        print("=== H1: resolution beats capacity / P2 head at equal latency ===")
        for c in res.comparison:
            r = res[res.comparison == c].iloc[0]
            verdict = ("supports H1" if r.diff_mean > 0 and r.significant
                       else "does not support H1")
            print(f"  {c:20s} diff={r.diff_mean:+.4f}  p_holm={r.p_holm:.4f}  -> {verdict}")
        ok = all(res.diff_mean > 0) and all(res.significant)
        print(f"  overall: {'H1 SUPPORTED' if ok else 'H1 NOT FULLY SUPPORTED'}")
        json.dump({"metric": m, "study": "v2",
                   "per_comparison": {c: float(res[res.comparison == c].diff_mean.iloc[0])
                                      for c in res.comparison},
                   "overall": bool(ok)},
                  open(FIGDATA / f"h1_verdict_{m}.json", "w"), indent=2)
        return

    k1 = {}
    for ctrl in controls:
        r = res[res.comparison == f"E5 - {ctrl}"]
        if len(r):
            k1[ctrl] = bool(r.significant.iloc[0] and r.practically_relevant.iloc[0])
    if k1:
        print("=== K1 criterion: E5 beats every equal-budget control ===")
        for k, v in k1.items():
            print(f"  vs {k}: {'PASS' if v else 'FAIL'}")
        print(f"  overall: {'PASS' if all(k1.values()) else 'FAIL'}")
        json.dump({"metric": m, "per_control": k1, "overall": all(k1.values())},
                  open(FIGDATA / f"k1_verdict_{m}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
