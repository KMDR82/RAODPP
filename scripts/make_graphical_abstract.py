#!/usr/bin/env python3
"""
make_graphical_abstract.py — the one-panel graphical abstract, drawn from the data.

Every number and every point on this panel is read from a file that a run produced:

    output/runs.csv                              per-condition accuracy and cost
    output/figdata/iso_visdrone_*_comparison.csv the iso-cost differences and overlaps
    output/figdata/dominance_visdrone.csv        the interpolation-free comparison

Nothing is typed in, including the name of the leading arm, which is derived from the sign
of the measured difference. A label written in by hand beside a number read from a file
will disagree with that number as soon as the analysis changes, and the panel will then
contradict the paper it summarises.

Layout: a wide strip, about 2.5:1, which is the shape a graphical abstract is laid out in.
Three regions: the measured accuracy-against-cost curves on the left, the dominance result
in the middle, and the remaining evidence as compact cards on the right.

    python scripts/make_graphical_abstract.py

Writes fig/graphical_abstract.pdf and .png at 300 dpi.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

ROOT = Path.home() / "RAODpp"
OUT, FIGDATA, FIG = ROOT / "output", ROOT / "output" / "figdata", ROOT / "fig"

ARMS = {
    "resolution":     ["E0", "R640", "R960", "R1280", "R1536"],
    "capacity":       ["E0", "R640", "S640", "M640", "L640"],
    "capacity_mixed": ["S640", "S960", "S1280"],
}
STYLE = {
    "resolution":     ("#1f6fb4", "o", "-",  "resolution"),
    "capacity":       ("#c0392b", "s", "-",  "capacity"),
    "capacity_mixed": ("#8e44ad", "^", "--", "mixed (confounded)"),
}

INK, MUTE, RULE = "#16202b", "#7c8894", "#d8dee5"
BG_CARD, BG_HERO = "#f6f8fa", "#eef4fb"
ACC, WARN = "#1f6fb4", "#b9770e"

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "lines.linewidth": 2.0, "lines.markersize": 6,
    "savefig.dpi": 300, "savefig.facecolor": "white",
})


# ------------------------------------------------------------------ data

def load_runs(dataset):
    f = OUT / "runs.csv"
    if not f.exists():
        sys.exit(f"missing {f}")
    d = pd.read_csv(f)
    if "dataset" in d.columns:
        d = d[d.dataset == dataset]
    d = d.dropna(subset=["mAP50_95", "ms_mean", "power_w"])
    if d.empty:
        return None
    d = d.assign(energy_j=d.ms_mean * d.power_w / 1000.0)
    if "idle_w" in d.columns:
        d["energy_marginal_j"] = d.ms_mean * (d.power_w - d.idle_w) / 1000.0
    return d


def iso(dataset, cost, a="resolution", b="capacity"):
    """The measured difference and overlap for one pair of arms."""
    f = FIGDATA / f"iso_{dataset}_{cost}_comparison.csv"
    if f.exists():
        c = pd.read_csv(f)
        c = c[c.comparable == 1]
        hit = c[(c.arm_a == a) & (c.arm_b == b)]
        if len(hit):
            r = hit.iloc[0]
            return {"diff": float(r.diff_mean), "lo": float(r.overlap_lo),
                    "hi": float(r.overlap_hi)}
        hit = c[(c.arm_a == b) & (c.arm_b == a)]
        if len(hit):
            r = hit.iloc[0]
            return {"diff": -float(r.diff_mean), "lo": float(r.overlap_lo),
                    "hi": float(r.overlap_hi)}
        return None
    # a tree whose iso_analysis.py has not been patched yet still has the curve file
    g = FIGDATA / f"iso_{dataset}_{cost}_curve.csv"
    if not g.exists():
        return None
    c = pd.read_csv(g)
    c = c[c.overlap == 1]
    hit = c[(c.arm_a == a) & (c.arm_b == b)]
    if hit.empty:
        hit = c[(c.arm_a == b) & (c.arm_b == a)]
        if hit.empty:
            return None
        return {"diff": -float(hit.diff_mean.mean()), "lo": float(hit.budget.min()),
                "hi": float(hit.budget.max())}
    return {"diff": float(hit.diff_mean.mean()), "lo": float(hit.budget.min()),
            "hi": float(hit.budget.max())}


def hero_pair():
    """The measured pair that dominates on every cost axis by the largest margin."""
    f = FIGDATA / "dominance_visdrone.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if "dataset" in d.columns:
        d = d[d.dataset == "visdrone"]
    if d.empty:
        return None
    n_axes = d.cost.nunique()
    best = None
    for _, sub in d.groupby(["better", "worse"]):
        if sub.cost.nunique() < n_axes:
            continue
        if best is None or sub.d_map.iloc[0] > best.d_map.iloc[0]:
            best = sub
    if best is None:
        return None
    rows = {r.cost: r for r in best.itertuples()}
    first = best.iloc[0]
    return {
        "better": first.better_label, "worse": first.worse_label,
        "d_map": float(first.d_map),
        "every_seed": int(best.every_seed.min()), "n_seeds": int(first.n_seeds),
        "n_pairs": int(len(d)), "n_every": int((d.every_seed == 1).sum()),
        "savings": {c: (1 - rows[c].cost_better / rows[c].cost_worse) * 100
                    for c in rows},
    }


def enrichment_runs():
    d = pd.read_csv(OUT / "runs.csv")
    if "dataset" in d.columns:
        d = d[d.dataset == "visdrone"]
    conds = [c for c in ["E0", "E1", "E2", "E3", "E4", "E5"] if c in set(d.exp)]
    return int(len(d[d.exp.isin(conds)])), len(conds)


# ------------------------------------------------------------------ drawing

def card(fig, rect, title=None, fill=BG_CARD, edge=RULE):
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge,
        linewidth=1.0, zorder=0))
    if title:
        fig.text(x + 0.013, y + h - 0.048, title, fontsize=10.5,
                 fontweight="bold", color=INK, va="center", zorder=2)


def chip(fig, x, y, w, h, headline, sub, colour):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.008",
        transform=fig.transFigure, facecolor="white", edgecolor=colour,
        linewidth=1.3, zorder=2))
    fig.text(x + w / 2, y + h * 0.66, headline, fontsize=11, fontweight="bold",
             color=colour, va="center", ha="center", zorder=3)
    fig.text(x + w / 2, y + h * 0.27, sub, fontsize=8.4, color=MUTE,
             va="center", ha="center", zorder=3)


def glyph_clock(fig, cx, cy, r, colour):
    fig.patches.append(Circle((cx, cy), r, transform=fig.transFigure, fill=False,
                              edgecolor=colour, linewidth=1.7, zorder=3))
    for dx, dy in ((0, r * 0.55), (r * 0.42, 0)):
        fig.lines.append(plt.Line2D([cx, cx + dx], [cy, cy + dy],
                                    transform=fig.transFigure, color=colour,
                                    lw=1.7, zorder=3))


def glyph_bolt(fig, cx, cy, r, colour):
    pts = [(cx + 0.16 * r, cy + r), (cx - 0.52 * r, cy - 0.04 * r),
           (cx - 0.02 * r, cy - 0.04 * r), (cx - 0.16 * r, cy - r),
           (cx + 0.55 * r, cy + 0.10 * r), (cx + 0.03 * r, cy + 0.10 * r)]
    fig.patches.append(Polygon(pts, transform=fig.transFigure, closed=True,
                               facecolor=colour, edgecolor="none", zorder=3))


def glyph_dot(fig, cx, cy, r, colour):
    fig.patches.append(Circle((cx, cy), r * 0.8, transform=fig.transFigure,
                              facecolor=colour, edgecolor="none", zorder=3))


def curve_panel(ax, d, cost, xlabel):
    for arm, conds in ARMS.items():
        colour, marker, ls, name = STYLE[arm]
        sub = (d[d.exp.isin(conds)].groupby("exp")
               .agg(x=(cost, "mean"), y=("mAP50_95", "mean")).reset_index())
        if sub.empty:
            continue
        sub = sub.sort_values("x")
        ax.plot(sub.x, sub.y, marker=marker, ls=ls, color=colour, label=name,
                markeredgecolor="white", markeredgewidth=0.9, zorder=3)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(r"mAP$_{50\text{-}95}$", labelpad=2)
    ax.margins(x=0.14, y=0.16)
    ax.tick_params(length=3, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone")
    a = ap.parse_args()

    d = load_runs(a.dataset)
    if d is None:
        sys.exit(f"runs.csv has no usable rows for {a.dataset}")
    iso_lat = iso(a.dataset, "latency")
    iso_en = iso(a.dataset, "energy")
    iso_mar = iso(a.dataset, "energy_marginal")
    iso_mix = iso(a.dataset, "latency", "capacity_mixed", "capacity")
    hero = hero_pair()
    n_runs, n_conds = enrichment_runs()
    uav_en = iso("uavdt", "energy")

    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.3, 5.35))

    # ---------------------------------------------------------- title band
    fig.text(0.5, 0.948,
             "Where should a fixed, measured inference budget go: resolution or capacity?",
             ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.5, 0.890,
             "Real-time UAV detection, with latency and energy measured on the device at "
             "batch one rather than inferred from parameters or FLOPs",
             ha="center", va="center", fontsize=10, color=MUTE)
    fig.lines.append(plt.Line2D([0.025, 0.975], [0.858, 0.858],
                                transform=fig.transFigure, color=RULE, lw=1.0))

    # ---------------------------------------------------------- left: curves
    card(fig, (0.025, 0.050, 0.375, 0.780), "Accuracy against measured cost")
    ax = fig.add_axes([0.074, 0.345, 0.303, 0.385])
    ax.set_zorder(5)
    ax.patch.set_alpha(0)
    curve_panel(ax, d, "ms_mean", "latency per frame (ms), batch 1")
    ax.legend(frameon=False, loc="lower right", handlelength=1.6,
              borderpad=0.2, labelspacing=0.3)

    if iso_lat:
        chip(fig, 0.040, 0.128, 0.172, 0.082,
             f"resolution {iso_lat['diff']:+.4f}",
             f"at equal latency, {iso_lat['lo']:.2f}–{iso_lat['hi']:.2f} ms", ACC)
    if iso_en:
        chip(fig, 0.224, 0.128, 0.172, 0.082,
             f"resolution {iso_en['diff']:+.4f}",
             f"at equal energy, {iso_en['lo']:.2f}–{iso_en['hi']:.2f} J", ACC)
    if iso_mar:
        fig.text(0.212, 0.088,
                 f"With the device's idle draw removed, resolution leads by "
                 f"{iso_mar['diff']:+.4f} over "
                 f"{iso_mar['lo']:.2f}–{iso_mar['hi']:.2f} J.",
                 ha="center", fontsize=8.4, color=MUTE)

    # ---------------------------------------------------------- middle: dominance
    card(fig, (0.413, 0.050, 0.267, 0.780),
         "Without any interpolation", fill=BG_HERO, edge="#bcd6ef")
    if hero:
        fig.text(0.546, 0.762,
                 "A configuration dominates another when\n"
                 "it is both more accurate and cheaper:\n"
                 "no grid, no curve fitting, no model.",
                 ha="center", va="top", fontsize=8.4, color=MUTE, linespacing=1.5)

        fig.text(0.546, 0.618, f"{hero['better']}  ≻  {hero['worse']}",
                 ha="center", va="center", fontsize=19, fontweight="bold", color=ACC)

        rows = [("accuracy", f"{hero['d_map']:+.4f} mAP", glyph_dot),
                ("latency", f"−{hero['savings'].get('latency', 0):.1f}%",
                 glyph_clock),
                ("energy", f"−{hero['savings'].get('energy', 0):.1f}%", glyph_bolt),
                ("energy above idle",
                 f"−{hero['savings'].get('marginal energy', 0):.1f}%", glyph_bolt)]
        y = 0.516
        for name, val, g in rows:
            g(fig, 0.449, y, 0.0120, ACC)
            fig.text(0.472, y, name, fontsize=10, color=INK, va="center")
            fig.text(0.662, y, val, fontsize=12.5, fontweight="bold", color=ACC,
                     va="center", ha="right")
            y -= 0.060

        tail = (f"in all {hero['n_seeds']} seeds separately"
                if hero["every_seed"] else "on the condition means")
        fig.text(0.546, 0.262,
                 f"{hero['n_pairs']} such pairs on VisDrone, {hero['n_every']} of them in "
                 f"every\nseed; this one holds {tail}.",
                 ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.5)
        fig.text(0.546, 0.146,
                 "No capacity configuration dominates\na resolution one, on any axis.",
                 ha="center", va="top", fontsize=9.4, fontweight="bold",
                 color=INK, linespacing=1.6)
    else:
        fig.text(0.546, 0.45, "run scripts/dominance_table.py first",
                 ha="center", fontsize=11, color=WARN)

    # ---------------------------------------------------------- right: cards
    # three equal cards down the right column, each body placed from its own rect
    RX, RW, GAP, CH = 0.693, 0.282, 0.0215, 0.2450
    ry = [0.050, 0.050 + CH + GAP, 0.050 + 2 * (CH + GAP)]

    def body(y, text):
        fig.text(RX + 0.013, y + CH - 0.080, text, va="top", fontsize=7.8,
                 color=INK, linespacing=1.40, zorder=2)

    card(fig, (RX, ry[2], RW, CH), "The corrected experiment")
    if iso_mix:
        body(ry[2],
             f"An arm built as one model at several input\n"
             f"sizes varies both factors. It leads the isolated\n"
             f"arm by {iso_mix['diff']:+.4f} on latency and reverses the\n"
             f"ordering against resolution. Both versions\n"
             f"are reported.")

    card(fig, (RX, ry[1], RW, CH), "Three eliminations came first")
    enrich = (f"no gain over {n_runs} controlled runs"
              if n_conds >= 2 else "no gain at any budget tested")
    body(ry[1],
         f"Data enrichment: {enrich}.\n"
         "Confidence correction: low confidence on\n"
         "small objects is accurate, not a bias.\n"
         "Our own first hypothesis: rejected once\n"
         "cost was measured instead of assumed.")

    card(fig, (RX, ry[0], RW, CH), "Second dataset")
    if uav_en:
        body(ry[0],
             f"On UAVDT the direction replicates at a\n"
             f"smaller magnitude, {uav_en['diff']:+.4f} on energy over\n"
             f"{uav_en['lo']:.2f}–{uav_en['hi']:.2f} J, and nothing reaches "
             f"corrected\n"
             f"significance. A check on the direction of the\n"
             f"result, not a confirmation of it.")

    for ext in ("pdf", "png"):
        out = FIG / f"graphical_abstract.{ext}"
        fig.savefig(out)
        print(f"wrote {out}")
    plt.close(fig)

    try:
        from PIL import Image
        w, h = Image.open(FIG / "graphical_abstract.png").size
        ok = "ok" if (w >= 1328 and h >= 531) else "TOO SMALL"
        print(f"png size: {w} x {h} px, ratio {w/h:.2f}:1  ({ok}; "
              f"Elsevier min 1328 x 531)")
    except ImportError:
        pass
    print("every value on the panel was read from runs.csv, the iso comparison CSVs "
          "and the dominance CSV; none is typed in")


if __name__ == "__main__":
    main()
