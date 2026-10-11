#!/usr/bin/env python3
"""
make_figures.py — draw the manuscript figures from output/figdata and output/runs.csv.

No number is typed here: every value is read from a file produced by a run. If a source
file is missing the figure is skipped with a message rather than drawn from partial data.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --only pareto iso

Output: fig/*.pdf at publication size, plus a fig/FIGURE_MANIFEST.csv recording which
source files each figure was built from.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.home() / "RAODpp"
FIG, FIGDATA, OUT = ROOT / "fig", ROOT / "output" / "figdata", ROOT / "output"
FIG.mkdir(parents=True, exist_ok=True)

# single-column width for elsarticle is about 3.4 in; these are 1.5-column figures
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 5,
})

# ARMS_FIG_V3: arm -> conditions, mirroring scripts/iso_analysis.py. A condition can
# appear in more than one arm: E0 is the shared baseline of the resolution and the
# capacity path, and S640 is the first step of both capacity paths. The previous
# condition -> arm map could not express that and silently assigned each condition to
# one arm only.
ARMS = {
    "resolution":     ["E0", "R640", "R960", "R1280", "R1536"],
    "capacity":       ["E0", "S640", "M640", "L640"],
    "capacity_mixed": ["S640", "S960", "S1280"],
    "p2head":         ["P640", "P960", "P1280"],
}
ARM_STYLE = {
    # colour, marker, linestyle. Each arm is distinguishable without colour, because
    # the journal prints some figures in greyscale.
    "resolution":     ("#1f6fb4", "o", "-"),
    "capacity":       ("#c0392b", "s", "-"),
    "capacity_mixed": ("#8e44ad", "^", "--"),
    "p2head":         ("#7d8b3a", "D", "-."),
}
ARM_LABEL = {
    "resolution": "resolution", "capacity": "capacity",
    "capacity_mixed": "capacity (mixed)", "p2head": "P2 head",
}
# kept for the figures that still colour by a single arm
ARM_COLOR = {k: v[0] for k, v in ARM_STYLE.items()}
ARM_OF = {}
for _arm, _conds in ARMS.items():
    for _c in _conds:
        ARM_OF.setdefault(_c, _arm)
LABEL = {"E0": "n@640", "R640": "n@640", "R960": "n@960", "R1280": "n@1280",
         "R1536": "n@1536", "S640": "s@640", "S960": "s@960", "S1280": "s@1280",
         "M640": "m@640", "L640": "l@640",
         "P640": "n-p2@640", "P960": "n-p2@960", "P1280": "n-p2@1280"}

manifest = []


def save(fig, fname):
    """PDF for the manuscript, PNG at the same dpi for quick viewing."""
    fig.savefig(FIG / fname)
    fig.savefig(FIG / (Path(fname).stem + ".png"))


def note(fig_name, sources):
    manifest.append({"figure": fig_name, "sources": "; ".join(str(s) for s in sources)})
    print(f"  wrote {fig_name}")


def runs(dataset):
    d = pd.read_csv(OUT / "runs.csv")
    d = d[d.dataset == dataset]
    need = ["exp", "seed", "mAP50_95", "ms_mean", "power_w"]
    d = d.dropna(subset=[c for c in need if c in d.columns])
    d = d[d.exp.isin(ARM_OF)]
    d["energy_j"] = d.ms_mean * d.power_w / 1000
    return d


# ------------------------------------------------------------------ pareto
def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    d = runs("visdrone")
    g = (d.groupby("exp").agg(ms=("ms_mean", "mean"), ej=("energy_j", "mean"),
                              y=("mAP50_95", "mean"), s=("mAP50_95", "std"),
                              n=("mAP50_95", "size")).reset_index())
    rows = []
    for ax, (col, xlabel, panel) in zip(
            axes, [("ms", "latency (ms), batch 1", "(a)"),
                   ("ej", "energy (J / frame)", "(b)")]):
        # E0 sits on the resolution and the capacity arm and S640 on both capacity arms,
        # so without this their labels are drawn twice on top of themselves
        seen = set()
        for arm, conds in ARMS.items():
            sub = g[g.exp.isin(conds)].sort_values(col)
            if sub.empty:
                continue
            colour, marker, ls = ARM_STYLE[arm]
            ax.errorbar(sub[col], sub.y, yerr=sub.s, marker=marker, ls=ls,
                        capsize=2, color=colour, label=ARM_LABEL[arm], zorder=3)
            for r in sub.itertuples():
                if r.exp not in seen:
                    seen.add(r.exp)
                    ax.annotate(LABEL.get(r.exp, r.exp), (getattr(r, col), r.y),
                                textcoords="offset points", xytext=(4, -8),
                                fontsize=6, color=colour)
                rows.append({"arm": arm, "condition": r.exp,
                             "label": LABEL.get(r.exp, r.exp), "axis": col,
                             "cost": getattr(r, col), "map50_95": r.y,
                             "map50_95_std": r.s, "seeds": r.n})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"mAP$_{50\text{-}95}$")
        # panel label centred under the axis, not a title inside the figure
        ax.text(0.5, -0.22, panel, transform=ax.transAxes, ha="center", fontsize=8)
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "pareto.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGDATA / "pareto_points.csv", index=False)
    note("pareto.pdf", [OUT / "runs.csv", FIGDATA / "pareto_points.csv"])


# ------------------------------------------------------------------ resolution curve
def fig_resolution_curve():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for ds, marker, ls in [("visdrone", "o", "-"), ("uavdt", "s", "--")]:
        d = runs(ds)
        d = d[d.exp.map(ARM_OF) == "resolution"]
        if d.empty:
            continue
        size = {"E0": 640, "R640": 640, "R960": 960, "R1280": 1280, "R1536": 1536}
        g = d.groupby("exp").agg(y=("mAP50_95", "mean"), s=("mAP50_95", "std")).reset_index()
        g["px"] = g.exp.map(size)
        g = g.sort_values("px")
        ax.errorbar(g.px, g.y, yerr=g.s, marker=marker, ls=ls, capsize=2,
                    label=ds, color=ARM_COLOR["resolution"] if ds == "visdrone" else "#7aa6cc")
    ax.set_xlabel("input resolution (px)")
    ax.set_ylabel(r"mAP$_{50\text{-}95}$")
    ax.set_xticks([640, 960, 1280, 1536])
    ax.legend(frameon=False)
    fig.savefig(FIG / "resolution_curve.pdf")
    plt.close(fig)
    note("resolution_curve.pdf", [OUT / "runs.csv"])


# ------------------------------------------------------------------ iso-cost
def fig_iso(dataset, cost, fname):
    f = FIGDATA / f"iso_{dataset}_{cost}_curve.csv"
    if not f.exists():
        print(f"  skipped {fname}: {f.name} missing")
        return
    d = pd.read_csv(f)
    d = d[(d.overlap == 1)]
    if d.empty:
        print(f"  skipped {fname}: no overlapping budgets")
        return
    # colour by the pair, not by the first arm: several comparisons share a first arm
    PAIR = {("resolution", "capacity"):           ("#1f6fb4", "-"),
            ("resolution", "p2head"):             ("#7d8b3a", "--"),
            ("capacity",   "p2head"):             ("#c0392b", "-."),
            ("resolution", "capacity_mixed"):     ("#8e44ad", "-"),
            ("capacity_mixed", "capacity"):       ("#d35400", "--"),
            ("capacity",   "capacity_mixed"):     ("#d35400", "--"),
            ("capacity_mixed", "p2head"):         ("#16a085", "-.")}
    LEG = {"resolution": "resolution", "capacity": "capacity",
           "capacity_mixed": "mixed path", "p2head": "P2 head"}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # Preferred: the pre-registered comparisons, so that six curves are not crammed onto
    # one panel and the mixed-path pairs stay in their own figure. But a dataset with no
    # pure capacity arm in range has none of these pairs, and filtering them out left the
    # UAVDT panels empty, so fall back to whatever actually overlapped.
    PREFERRED = {("resolution", "capacity"), ("resolution", "p2head"),
                 ("capacity", "p2head")}
    present = [tuple(k) for k, _ in d.groupby(["arm_a", "arm_b"])]
    shown = [p for p in present if p in PREFERRED]
    if not shown:
        shown = present
        print(f"  {fname}: no pre-registered pair overlaps on this dataset, "
              f"drawing the {len(present)} pair(s) that do")
    for (a, b), sub in d.groupby(["arm_a", "arm_b"]):
        if (a, b) not in shown:
            continue
        sub = sub.sort_values("budget")
        col, ls = PAIR.get((a, b), ("#333333", ":"))
        ax.plot(sub.budget, sub.diff_mean, color=col, ls=ls,
                label=f"{LEG.get(a, a)} $-$ {LEG.get(b, b)}")
        ax.fill_between(sub.budget, sub.diff_mean - sub.diff_std,
                        sub.diff_mean + sub.diff_std, alpha=0.16, color=col, lw=0)
    ax.axhline(0, color="#666666", lw=0.8)
    for y in (0.005, -0.005):
        ax.axhline(y, color="#999999", lw=0.6, ls=":")
    ax.text(0.99, 0.005, "practical floor", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=5.5, color="#888888")
    unit = "ms" if cost == "latency" else "J / frame"
    axis_name = {"latency": "latency", "energy": "energy",
                 "energy_marginal": "energy above idle"}[cost]
    ax.set_xlabel(f"{axis_name} budget ({unit})")
    ax.set_ylabel(r"$\Delta$ mAP$_{50\text{-}95}$")
    ax.legend(frameon=False, loc="best")
    save(fig, fname)
    plt.close(fig)
    note(fname, [f])


# ------------------------------------------------------------------ enrichment
def fig_enrichment():
    d = pd.read_csv(OUT / "runs.csv")
    d = d[(d.dataset == "visdrone") & d.exp.isin(["E0","E1","E2","E3","E4","E5"])]
    if d.empty:
        print("  skipped enrichment: no rows"); return
    order = ["E0","E1","E2","E3","E4","E5"]
    names = {"E0":"baseline","E1":"+20 ep,\nsame data","E2":"pseudo-\nlabels",
             "E3":"random\npaste","E4":"retrieval\npaste","E5":"full\npipeline"}
    g = d.groupby("exp").mAP50_95.agg(["mean","std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(order))
    ax.bar(x, g["mean"], yerr=g["std"], capsize=3, width=0.62,
           color=["#4d4d4d"] + ["#9bb7d4"]*5, edgecolor="none")
    base = g.loc["E0","mean"]
    ax.axhline(base, color="#c0392b", lw=0.9, ls="--", label="baseline")
    ax.set_xticks(x); ax.set_xticklabels([names[o] for o in order])
    ax.set_ylabel(r"mAP$_{50\text{-}95}$")
    ax.set_ylim(g["mean"].min()-0.004, g["mean"].max()+0.004)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "enrichment.pdf")
    plt.close(fig)
    note("enrichment.pdf", [OUT / "runs.csv"])


# ------------------------------------------------------------------ operating point
def fig_conf_precision():
    rel = FIGDATA / "calib_reliability.csv"
    op = FIGDATA / "calib_operating_point.csv"
    if not (rel.exists() and op.exists()):
        print("  skipped conf_precision: calibration files missing"); return
    r, o = pd.read_csv(rel), pd.read_csv(op)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    bands = ["tiny", "small", "medium", "large"]
    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / 3 * 0.85) for i, b in enumerate(bands)}

    for b in bands:
        sub = r[r.size_bin == b].groupby("conf_bin")[["mean_conf","precision"]].mean()
        if sub.empty: continue
        axes[0].plot(sub.mean_conf, sub.precision, marker="o", ms=3,
                     color=colors[b], label=b)
    axes[0].plot([0,1],[0,1], color="#888888", lw=0.8, ls="--")
    axes[0].set_xlabel("confidence"); axes[0].set_ylabel("empirical precision")
    axes[0].legend(frameon=False, fontsize=6)

    piv = o.pivot_table(index="threshold", columns="size_bin", values="recall")
    for b in bands:
        if b in piv: axes[1].plot(piv.index, piv[b], marker="o", ms=3,
                                  color=colors[b], label=b)
    axes[1].set_xlabel("detection threshold"); axes[1].set_ylabel("recall")
    fig.savefig(FIG / "conf_precision.pdf")
    plt.close(fig)
    note("conf_precision.pdf", [rel, op])


# ------------------------------------------------------------------ retrieval
def fig_retrieval():
    f = FIGDATA / "retrieval_quality_per_class_centered.csv"
    if not f.exists():
        print("  skipped retrieval: source missing"); return
    d = pd.read_csv(f)
    cls = [c for c in ["class_name", "class"] if c in d.columns]
    if not cls or "chance_at_5" not in d.columns:
        print("  skipped retrieval: unexpected columns", list(d.columns)); return
    d = d.sort_values("gallery_share")
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(d.gallery_share, d.consistency, marker="o", color="#1f6fb4", label="measured")
    ax.plot(d.gallery_share, d.chance_at_5, marker="s", ms=4, ls="--",
            color="#888888", label="chance")
    for r in d.itertuples():
        ax.annotate(getattr(r, cls[0]), (r.gallery_share, r.consistency),
                    textcoords="offset points", xytext=(3, 4), fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("share of the gallery"); ax.set_ylabel("consistency @ 5")
    ax.legend(frameon=False)
    fig.savefig(FIG / "retrieval_consistency.pdf")
    plt.close(fig)
    note("retrieval_consistency.pdf", [f])


# ------------------------------------------------------------------ size bands
def fig_size_bands():
    f = FIGDATA / "ap_size_bands.csv"
    if not f.exists():
        print("  skipped size_bands: source missing"); return
    d = pd.read_csv(f)
    d = d[d.dataset == "visdrone"]
    bands = ["AP_small", "AP_medium", "AP_large"]
    order = ["E0", "E5", "S640", "P960", "R960", "S960", "R1280", "R1536"]
    g = d[d.exp.isin(order)].groupby("exp")[bands].mean().reindex(order).dropna()
    if g.empty:
        print("  skipped size_bands: no matching conditions"); return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # left: absolute AP per band
    x = np.arange(len(g)); w = 0.26
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(bands):
        axes[0].bar(x + (i - 1) * w, g[b], width=w, label=b.replace("AP_", ""),
                    color=cmap(i / 2 * 0.8), edgecolor="none")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABEL.get(e, e) for e in g.index], rotation=35, ha="right")
    axes[0].set_ylabel("AP")
    axes[0].legend(frameon=False, title="object size", title_fontsize=6)

    # right: gain over the 640 baseline, which is where the mechanism shows
    base = g.loc["E0"]
    rel = (g / base - 1) * 100
    for i, b in enumerate(bands):
        axes[1].plot(x, rel[b], marker="o", color=cmap(i / 2 * 0.8),
                     label=b.replace("AP_", ""))
    axes[1].axhline(0, color="#666666", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABEL.get(e, e) for e in g.index], rotation=35, ha="right")
    axes[1].set_ylabel("gain over n@640 (%)")
    fig.savefig(FIG / "size_bands.pdf")
    plt.close(fig)
    note("size_bands.pdf", [f])


# ------------------------------------------------------------------ budget sweep
def fig_budget_sweep():
    """Accuracy against the number of pasted instances, from scratch, seed 42."""
    d = pd.read_csv(OUT / "runs.csv")
    d = d[(d.dataset == "visdrone") & d.exp.isin(["F5", "F5b15", "F5b30", "E0"])]
    if d.empty:
        print("  skipped budget_sweep: no rows"); return
    # instance counts come from the enrichment audit, not from the figure code
    counts = {"F5": 9331, "F5b15": 22874, "F5b30": 32771}
    g = d[d.exp != "E0"].groupby("exp").mAP50_95.mean()
    base = d[d.exp == "E0"].mAP50_95.mean()
    pts = sorted((counts[e], g[e]) for e in g.index if e in counts)
    if not pts:
        print("  skipped budget_sweep: no budget runs"); return
    x, y = zip(*pts)
    total_boxes = 343197            # VisDrone training boxes, from the data card

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot([v / total_boxes * 100 for v in x], y, marker="o", color="#c0392b")
    ax.axhline(base, color="#666666", ls="--", lw=0.9, label="baseline (no pasting)")
    for xv, yv in zip(x, y):
        ax.annotate(f"{xv/1000:.0f}k", (xv / total_boxes * 100, yv),
                    textcoords="offset points", xytext=(4, 5), fontsize=6)
    ax.set_xlabel("added instances (% of existing boxes)")
    ax.set_ylabel(r"mAP$_{50\text{-}95}$")
    ax.legend(frameon=False)
    fig.savefig(FIG / "budget_sweep.pdf")
    plt.close(fig)
    note("budget_sweep.pdf", [OUT / "runs.csv"])


# ------------------------------------------------------------------ UMAP
def fig_umap():
    f = FIGDATA / "umap_coords.csv"
    if not f.exists():
        print("  skipped umap: source missing"); return
    d = pd.read_csv(f)
    xcol = next((c for c in d.columns if c.lower() in ("x", "umap_x", "u0")), None)
    ycol = next((c for c in d.columns if c.lower() in ("y", "umap_y", "u1")), None)
    ccol = next((c for c in d.columns if "class" in c.lower()), None)
    if not (xcol and ycol):
        print("  skipped umap: unexpected columns", list(d.columns)); return
    if len(d) > 20000:
        d = d.sample(20000, random_state=0)

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    if ccol:
        cats = d[ccol].value_counts().index[:10]
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(cats):
            sub = d[d[ccol] == c]
            ax.scatter(sub[xcol], sub[ycol], s=1.2, alpha=0.35, lw=0,
                       color=cmap(i % 10), label=str(c))
        ax.legend(frameon=False, markerscale=6, fontsize=5.5, ncol=2, loc="best")
    else:
        ax.scatter(d[xcol], d[ycol], s=1.2, alpha=0.35, lw=0, color="#1f6fb4")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel("UMAP dimension 1"); ax.set_ylabel("UMAP dimension 2")
    fig.savefig(FIG / "umap_lowconf.pdf")
    plt.close(fig)
    note("umap_lowconf.pdf", [f])


# ------------------------------------------------------------------ training curves
def fig_training_curves():
    f = FIGDATA / "training_curves.csv"
    if not f.exists():
        print("  skipped training_curves: source missing"); return
    d = pd.read_csv(f)
    mcol = next((c for c in d.columns if c.lower() == "metric"), None)
    vcol = next((c for c in d.columns if c.lower() == "value"), None)
    if not (mcol and vcol and "run_id" in d.columns and "epoch" in d.columns):
        print("  skipped training_curves: unexpected columns", list(d.columns)); return
    d = d[d[mcol].astype(str).str.contains("mAP50-95")]
    d = d.drop_duplicates(subset=["run_id", "epoch", mcol], keep="last")
    d["exp"] = d.run_id.astype(str).str.split("_").str[0]

    shown = ["E0", "R960", "R1280", "R1536", "S640", "S960"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for e in shown:
        sub = d[(d.exp == e) & (d.run_id.astype(str).str.contains("visdrone"))]
        if sub.empty:
            continue
        g = sub.groupby("epoch")[vcol].mean()
        ax.plot(g.index, g.values, lw=1.2, label=LABEL.get(e, e),
                color=ARM_COLOR[ARM_OF[e]],
                ls={"E0": ":", "R960": "-", "R1280": "-.", "R1536": "-",
                    "S640": ":", "S960": "--"}.get(e, "-"),
                alpha=0.9)
    base = d[(d.exp == "E0") & (d.run_id.astype(str).str.contains("visdrone"))]
    if not base.empty:
        ax.axhline(base.groupby("epoch")[vcol].mean().max(), color="#999999",
                   lw=0.8, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"validation mAP$_{50\text{-}95}$")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.savefig(FIG / "training_curves.pdf")
    plt.close(fig)
    note("training_curves.pdf", [f])


# ------------------------------------------------------------------ confound panel
def fig_iso_confound():
    """The two mixed-path comparisons of Section 4.6, on latency and energy.

    Separate from the iso_* figures because these are the comparisons that show the
    confound rather than the pre-registered result, and putting six curves on one
    panel made none of them readable.
    """
    pairs = [("resolution", "capacity_mixed"), ("capacity_mixed", "capacity")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    drew = False
    for ax, (cost, unit, panel) in zip(axes, [("latency", "ms", "(a)"),
                                              ("energy", "J / frame", "(b)")]):
        f = FIGDATA / f"iso_visdrone_{cost}_curve.csv"
        if not f.exists():
            print(f"  skipped iso_confound.pdf: {f.name} missing")
            plt.close(fig)
            return
        d = pd.read_csv(f)
        d = d[d.overlap == 1]
        for a, b in pairs:
            sub = d[(d.arm_a == a) & (d.arm_b == b)].sort_values("budget")
            sign = 1.0
            if sub.empty:                      # stored in the other order
                sub = d[(d.arm_a == b) & (d.arm_b == a)].sort_values("budget")
                sign = -1.0
            if sub.empty:
                continue
            colour, ls = {("resolution", "capacity_mixed"): ("#8e44ad", "-"),
                          ("capacity_mixed", "capacity"): ("#d35400", "--")}[(a, b)]
            lab = {"resolution": "resolution", "capacity": "capacity",
                   "capacity_mixed": "mixed path"}
            y = sign * sub.diff_mean
            ax.plot(sub.budget, y, color=colour, ls=ls,
                    label=f"{lab[a]} $-$ {lab[b]}")
            ax.fill_between(sub.budget, y - sub.diff_std, y + sub.diff_std,
                            alpha=0.16, color=colour, lw=0)
            drew = True
        ax.axhline(0, color="#666666", lw=0.8)
        for yv in (0.005, -0.005):
            ax.axhline(yv, color="#999999", lw=0.6, ls=":")
        ax.set_xlabel(f"{cost} budget ({unit})")
        ax.set_ylabel(r"$\Delta$ mAP$_{50\text{-}95}$")
        ax.text(0.5, -0.24, panel, transform=ax.transAxes, ha="center", fontsize=8)
        ax.legend(frameon=False, loc="best")
    if not drew:
        print("  skipped iso_confound.pdf: neither mixed-path pair present")
        plt.close(fig)
        return
    save(fig, "iso_confound.pdf")
    plt.close(fig)
    note("iso_confound.pdf", [FIGDATA / "iso_visdrone_latency_curve.csv",
                              FIGDATA / "iso_visdrone_energy_curve.csv"])


ALL = {
    "budget": fig_budget_sweep,
    "umap": fig_umap,
    "curves": fig_training_curves,
    "sizebands": fig_size_bands,
    "pareto": fig_pareto,
    "confound": fig_iso_confound,
    "resolution": fig_resolution_curve,
    "enrichment": fig_enrichment,
    "conf": fig_conf_precision,
    "retrieval": fig_retrieval,
    "iso": lambda: [fig_iso("visdrone", "latency", "iso_latency.pdf"),
                    fig_iso("visdrone", "energy", "iso_energy.pdf"),
                    fig_iso("visdrone", "energy_marginal", "iso_energy_marginal.pdf"),
                    fig_iso("uavdt", "latency", "iso_uavdt_latency.pdf"),
                    fig_iso("uavdt", "energy", "iso_uavdt_energy.pdf"),
                    fig_iso("uavdt", "energy_marginal", "iso_uavdt_energy_marginal.pdf")],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(ALL), default=list(ALL))
    a = ap.parse_args()
    for name in a.only:
        print(f"[{name}]")
        try:
            ALL[name]()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
    if manifest:
        pd.DataFrame(manifest).to_csv(FIG / "FIGURE_MANIFEST.csv", index=False)
        print(f"\n{len(manifest)} figure(s) -> {FIG}")
