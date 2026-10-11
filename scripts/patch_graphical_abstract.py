#!/usr/bin/env python3
"""
patch_graphical_abstract.py — the graphical abstract still carries the overturned result.

A reviewer found this and was right. The panel on page one of the PDF says the opposite
of the abstract below it, in three separate ways:

  1. CAP is {S640, S960, S1280}, the arm that varies capacity AND resolution. The panel
     draws the mixed path and labels it "capacity". This is the confound the paper is
     about, reproduced in the paper's own summary figure.
  2. The latency panel's heading hardcodes leader="capacity". The magnitude is read from
     the curve CSV and therefore updated itself when the arms were rebuilt, but the word
     did not, so the panel now reads "capacity leads by 0.0386" when the measurement says
     resolution leads by that amount. A number and a label moving independently is worse
     than either being stale.
  3. The takeaway reads "Latency-bound -> spend on capacity", which was the pre-correction
     verdict.

What this patch changes:
  - the capacity arm becomes the isolated one, n@640 -> s@640 -> m@640 -> l@640
  - the mixed path is drawn as a third, dashed line labelled as the confounded arm, so the
    figure makes the methodological point instead of falling for it
  - the leader word is derived from the sign of the measured difference
  - the takeaway states the corrected finding and the lesson

Run it, then regenerate: python scripts/make_graphical_abstract.py
Idempotent, verifies its anchors, writes one .bak.

    python scripts/patch_graphical_abstract.py [--check]
"""
import argparse
import sys
from pathlib import Path

TARGET = Path.home() / "RAODpp" / "scripts" / "make_graphical_abstract.py"
MARKER = "CAP_MIXED"

EDITS = [
    # 1. the arms
    ('''RES = {"E0": "n@640", "R640": "n@640", "R960": "n@960",
       "R1280": "n@1280", "R1536": "n@1536"}
CAP = {"S640": "s@640", "S960": "s@960", "S1280": "s@1280"}
C_RES, C_CAP, C_INK, C_MUTE = "#1f6fb4", "#c0392b", "#1b1b1b", "#8a8a8a"''',
     '''RES = {"E0": "n@640", "R640": "n@640", "R960": "n@960",
       "R1280": "n@1280", "R1536": "n@1536"}
# the isolated capacity path: resolution held at 640, capacity varied
CAP = {"E0": "n@640", "S640": "s@640", "M640": "m@640", "L640": "l@640"}
# the arm as first built, which varies both factors. Drawn, because the contrast between
# it and CAP is a result; labelled, so that it cannot be mistaken for the capacity arm
CAP_MIXED = {"S640": "s@640", "S960": "s@960", "S1280": "s@1280"}
C_RES, C_CAP, C_INK, C_MUTE = "#1f6fb4", "#c0392b", "#1b1b1b", "#8a8a8a"
C_MIX = "#8e44ad"'''),

    # 2. draw three arms, not two
    ('''    for mapping, colour, name in [(RES, C_RES, "resolution"), (CAP, C_CAP, "capacity")]:
        g = arm(d, mapping, cost)
        if g.empty:
            continue
        ax.plot(g.x, g.y, marker="o", color=colour, label=name,
                markeredgecolor="white", markeredgewidth=0.6)''',
     '''    for mapping, colour, name, ls in [
            (RES, C_RES, "resolution", "-"),
            (CAP, C_CAP, "capacity", "-"),
            (CAP_MIXED, C_MIX, "mixed (confounded)", "--")]:
        g = arm(d, mapping, cost)
        if g.empty:
            continue
        ax.plot(g.x, g.y, marker="o", ls=ls, color=colour, label=name,
                markeredgecolor="white", markeredgewidth=0.6)'''),

    # 3. the leader is measured, not asserted
    ('''    for i, (cost, iso_r, xlabel, leader, colour) in enumerate([
            ("ms_mean", iso_lat, "latency per frame (ms)", "capacity", C_CAP),
            ("energy_j", iso_en, "energy per frame (J)", "resolution", C_RES)]):''',
     '''    # the leading arm follows the sign of the measured difference, which is
    # resolution minus capacity, rather than being written in by hand
    def _leader(r):
        if not r:
            return "resolution", C_RES
        return (("resolution", C_RES) if r["diff"] > 0 else ("capacity", C_CAP))

    for i, (cost, iso_r, xlabel) in enumerate([
            ("ms_mean", iso_lat, "latency per frame (ms)"),
            ("energy_j", iso_en, "energy per frame (J)")]):
        leader, colour = _leader(iso_r)'''),

    # 4. the takeaway
    ('''             "Latency-bound  $\\\\rightarrow$  spend on capacity\\n"
             "Energy-bound   $\\\\rightarrow$  spend on resolution\\n\\n"
             "Report both axes, measured on the device.",''',
     '''             "Resolution leads on latency and on\\n"
             "energy, over the budgets where both\\n"
             "arms were measured.\\n\\n"
             "An arm that varies capacity and\\n"
             "resolution together reverses the\\n"
             "apparent answer on latency.\\n\\n"
             "Report both axes, measured on the device.",'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found")
        sys.exit(1)
    s = TARGET.read_text(encoding="utf-8")

    if MARKER in s:
        print("make_graphical_abstract.py: already patched, nothing to do")
        return

    missing = [i for i, (old, _) in enumerate(EDITS, 1) if old not in s]
    if missing:
        print(f"FAIL: anchors {missing} not found in make_graphical_abstract.py.")
        print("Send the current file and the patch will be rebuilt against it.")
        sys.exit(1)

    if a.check:
        print("make_graphical_abstract.py: all 4 anchors found (--check, nothing written)")
        return

    bak = TARGET.with_suffix(".py.bak-ga")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
        print(f"backup -> {bak}")
    for old, new in EDITS:
        s = s.replace(old, new, 1)
    TARGET.write_text(s, encoding="utf-8")
    print("make_graphical_abstract.py: capacity arm isolated, mixed path drawn as such, "
          "leader derived from the measurement, takeaway corrected")
    print("\nnext:  python scripts/make_graphical_abstract.py")


if __name__ == "__main__":
    main()
