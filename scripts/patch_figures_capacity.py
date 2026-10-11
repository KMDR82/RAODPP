#!/usr/bin/env python3
"""
patch_figures_capacity.py — bring make_figures.py in line with the rebuilt capacity arm.

Why this exists. When the capacity arm was rebuilt as n@640 -> s@640 -> m@640 -> l@640
at fixed resolution, iso_analysis.py was updated and the manuscript was rewritten, but
make_figures.py was not. Its arm map still assigns s@960 and s@1280 to "capacity" and
does not know m@640 or l@640 at all, so fig/pareto.pdf draws the mixed path under the
label "capacity" and omits the two conditions the corrected finding rests on. The iso_*
figures read CSVs from iso_analysis.py and only need those CSVs regenerated, but their
colour map has no entry for the mixed arm, so the new comparisons would all come out in
the same grey dotted fallback.

What it changes in scripts/make_figures.py:
  1. replaces the condition -> arm map with an arm -> conditions map, so E0 can belong to
     both the resolution and the capacity arm and S640 to both capacity arms, which a
     1:1 map cannot express
  2. rewrites fig_pareto to draw four arms, to write its own CSV, and to drop the figure
     title and use panel labels instead
  3. extends the iso pair colours to the mixed arm, restricts each iso figure to a
     declared pair list, and drops the in-figure title
  4. adds a save() helper writing PDF and PNG at 300 dpi
  5. adds fig_iso_confound, the two mixed-path comparisons, as "confound"

It verifies every anchor before touching anything, is idempotent (running it twice is a
no-op), and writes make_figures.py.bak once.

    python scripts/patch_figures_capacity.py            # patch
    python scripts/patch_figures_capacity.py --check    # report only, change nothing
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path.home() / "RAODpp"
TARGET = ROOT / "scripts" / "make_figures.py"
ISO = ROOT / "scripts" / "iso_analysis.py"
MARKER = "ARMS_FIG_V3"

# ---------------------------------------------------------------- new fragments

NEW_MAPS = '''# ARMS_FIG_V3: arm -> conditions, mirroring scripts/iso_analysis.py. A condition can
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
'''

NEW_SAVE = '''
def save(fig, fname):
    """PDF for the manuscript, PNG at the same dpi for quick viewing."""
    fig.savefig(FIG / fname)
    fig.savefig(FIG / (Path(fname).stem + ".png"))

'''

NEW_PARETO = '''def fig_pareto():
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
        ax.set_ylabel(r"mAP$_{50\\text{-}95}$")
        # panel label centred under the axis, not a title inside the figure
        ax.text(0.5, -0.22, panel, transform=ax.transAxes, ha="center", fontsize=8)
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "pareto.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGDATA / "pareto_points.csv", index=False)
    note("pareto.pdf", [OUT / "runs.csv", FIGDATA / "pareto_points.csv"])
'''

NEW_PAIR = '''    # colour by the pair, not by the first arm: several comparisons share a first arm
    PAIR = {("resolution", "capacity"):           ("#1f6fb4", "-"),
            ("resolution", "p2head"):             ("#7d8b3a", "--"),
            ("capacity",   "p2head"):             ("#c0392b", "-."),
            ("resolution", "capacity_mixed"):     ("#8e44ad", "-"),
            ("capacity_mixed", "capacity"):       ("#d35400", "--"),
            ("capacity",   "capacity_mixed"):     ("#d35400", "--"),
            ("capacity_mixed", "p2head"):         ("#16a085", "-.")}
    LEG = {"resolution": "resolution", "capacity": "capacity",
           "capacity_mixed": "mixed path", "p2head": "P2 head"}
'''

NEW_CONFOUND = '''
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
        ax.set_ylabel(r"$\\Delta$ mAP$_{50\\text{-}95}$")
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

'''

# ---------------------------------------------------------------- iso body edits

ISO_PAIR_ANCHOR = """    # colour by the pair, not by the first arm: two comparisons that both start with
    # "resolution" were otherwise drawn in the same colour
    PAIR = {("resolution", "capacity"): ("#1f6fb4", "-"),
            ("resolution", "p2head"):   ("#7d8b3a", "--"),
            ("capacity",   "p2head"):   ("#c0392b", "-.")}
"""

OLD_MAPS_RE = re.compile(
    r'ARM_COLOR = \{.*?LABEL = \{.*?"P640": "n-p2@640", "P960": "n-p2@960"\}\n',
    re.S)


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_iso():
    """iso_analysis.py must already carry the rebuilt arms; this script does not touch it."""
    if not ISO.exists():
        return "missing"
    s = ISO.read_text(encoding="utf-8")
    ok = ('"capacity_mixed"' in s and '"M640"' in s and '"L640"' in s)
    if ok:
        return "ok"
    return "stale"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change and exit")
    a = ap.parse_args()

    iso_state = check_iso()
    print(f"iso_analysis.py: {iso_state}")
    if iso_state == "stale":
        print("  its ARMS still lack M640/L640/capacity_mixed. The curve CSVs it writes")
        print("  would be the pre-correction ones, so the figures would still be wrong.")
        print("  Fix iso_analysis.py first; this script deliberately does not edit it.")
    elif iso_state == "missing":
        fail(f"{ISO} not found")

    if not TARGET.exists():
        fail(f"{TARGET} not found")
    s = TARGET.read_text(encoding="utf-8")

    if MARKER in s:
        print("make_figures.py: already patched, nothing to do")
        return

    # --- verify every anchor before writing anything
    problems = []
    if not OLD_MAPS_RE.search(s):
        problems.append("the ARM_COLOR / ARM_OF / LABEL block")
    if "def fig_pareto():" not in s:
        problems.append("def fig_pareto()")
    if ISO_PAIR_ANCHOR not in s:
        problems.append("the PAIR colour map inside fig_iso")
    if 'ax.set_title(f"{dataset}, {axis_name} axis", fontsize=8)' not in s:
        problems.append("the fig_iso title line")
    if '"iso": lambda: [fig_iso("visdrone", "latency", "iso_latency.pdf"),' not in s:
        problems.append("the ALL dispatch table")
    if problems:
        fail("anchors not found: " + "; ".join(problems)
             + ".\n      make_figures.py differs from what this patch expects. Do not"
               " force it;\n      send the current file and the patch will be rebuilt"
               " against it.")

    # --- 1. maps
    s = OLD_MAPS_RE.sub(NEW_MAPS.rstrip("\n") + "\n", s, count=1)

    # --- 2. save helper, inserted before note()
    s = s.replace("\ndef note(fig_name, sources):",
                  NEW_SAVE + "\ndef note(fig_name, sources):", 1)

    # --- 3. pareto, replaced whole
    i = s.index("def fig_pareto():")
    j = s.index("# ------------------------------------------------------------------ resolution curve")
    s = s[:i] + NEW_PARETO + "\n\n" + s[j:]

    # --- 4. iso: colours, pair filter, no title, save()
    s = s.replace(ISO_PAIR_ANCHOR, NEW_PAIR, 1)
    s = s.replace(
        '    for (a, b), sub in d.groupby(["arm_a", "arm_b"]):',
        '    # the pre-registered comparisons only; the mixed-path pairs have their own\n'
        '    # figure, and six curves on one panel were unreadable\n'
        '    SHOW = {("resolution", "capacity"), ("resolution", "p2head"),\n'
        '            ("capacity", "p2head")}\n'
        '    for (a, b), sub in d.groupby(["arm_a", "arm_b"]):\n'
        '        if (a, b) not in SHOW:\n'
        '            continue', 1)
    s = s.replace('        ax.plot(sub.budget, sub.diff_mean, color=col, ls=ls, label=f"{a} $-$ {b}")',
                  '        ax.plot(sub.budget, sub.diff_mean, color=col, ls=ls,\n'
                  '                label=f"{LEG.get(a, a)} $-$ {LEG.get(b, b)}")', 1)
    s = s.replace('    ax.set_title(f"{dataset}, {axis_name} axis", fontsize=8)\n', "", 1)
    s = s.replace("    fig.savefig(FIG / fname)\n    plt.close(fig)\n    note(fname, [f])",
                  "    save(fig, fname)\n    plt.close(fig)\n    note(fname, [f])", 1)

    # --- 5. confound figure + dispatch entry
    s = s.replace("\nALL = {", NEW_CONFOUND + "\nALL = {", 1)
    s = s.replace('    "pareto": fig_pareto,',
                  '    "pareto": fig_pareto,\n    "confound": fig_iso_confound,', 1)

    bak = TARGET.with_suffix(".py.bak")
    if a.check:
        print("make_figures.py: all anchors found, 5 edits would be applied (--check, "
              "nothing written)")
        return
    if not bak.exists():
        bak.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"backup -> {bak}")
    TARGET.write_text(s, encoding="utf-8")
    print("make_figures.py: patched (maps, pareto, iso colours and filter, save(), "
          "confound figure)")
    print("\nnext:  bash scripts/run_figures_v3.sh")


if __name__ == "__main__":
    main()
