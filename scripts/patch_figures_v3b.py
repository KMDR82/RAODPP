#!/usr/bin/env python3
"""
patch_figures_v3b.py — two corrections to the v3 figure pass.

1. The three UAVDT iso figures came out empty.

   The v3 patch restricted each iso figure to the pre-registered pairs
   (resolution-capacity, resolution-P2, capacity-P2) so that six curves would not be
   crammed onto one panel. On UAVDT none of those pairs overlaps: there is no P2 arm and
   the pure capacity arm does not reach the resolution arm's budget range, so the only
   comparison with any data is resolution against the mixed path. The filter removed it,
   matplotlib warned "No artists with labels found to put in legend" three times, and
   iso_uavdt_latency.pdf, iso_uavdt_energy.pdf and iso_uavdt_energy_marginal.pdf were
   written with axes and no curves. The filter now falls back to whatever did overlap
   when no preferred pair is available, and says so.

2. The UAVDT capacity arm was missing its baseline.

   ARMS["capacity"] starts at E0, which is VisDrone's 640-pixel nano baseline. UAVDT has
   no E0; its 640-pixel nano run is R640. The arm therefore collapsed to the single point
   S640, which printed as the nonsensical coverage "4.972-4.864" and could not be compared
   with anything. Adding R640 to the arm gives UAVDT a real, if short, capacity path at
   fixed resolution: n@640 -> s@640. VisDrone is unaffected because it has no R640 run.

   This does not change any VisDrone number, and verify_figures.py re-checks that.

Idempotent, verifies its anchors, writes one .bak per file.

    python scripts/patch_figures_v3b.py
    python scripts/patch_figures_v3b.py --check
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path.home() / "RAODpp"
FIGS = ROOT / "scripts" / "make_figures.py"
ISO = ROOT / "scripts" / "iso_analysis.py"

OLD_FILTER = '''    # the pre-registered comparisons only; the mixed-path pairs have their own
    # figure, and six curves on one panel were unreadable
    SHOW = {("resolution", "capacity"), ("resolution", "p2head"),
            ("capacity", "p2head")}
    for (a, b), sub in d.groupby(["arm_a", "arm_b"]):
        if (a, b) not in SHOW:
            continue
'''

NEW_FILTER = '''    # Preferred: the pre-registered comparisons, so that six curves are not crammed onto
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
'''

CAP_RE = re.compile(r'("capacity"\s*:\s*\[)([^\]]*)(\])')


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def patch_iso(check):
    if not ISO.exists():
        fail(f"{ISO} not found")
    s = ISO.read_text(encoding="utf-8")
    m = CAP_RE.search(s)
    if not m:
        fail("could not find the \"capacity\" arm in iso_analysis.py")
    body = m.group(2)
    if '"R640"' in body:
        print("iso_analysis.py: capacity arm already includes R640, nothing to do")
        return False
    if '"E0"' not in body:
        fail(f'the capacity arm is {body.strip()}, which does not start at E0; '
             "not editing it blindly")
    new_body = body.replace('"E0",', '"E0", "R640",', 1)
    new = s[:m.start(2)] + new_body + s[m.end(2):]
    # keep the reader honest about why two baselines are listed
    anchor = '"capacity"'
    if "# E0 is VisDrone's 640 baseline, R640 is UAVDT's" not in new:
        new = new.replace(
            anchor,
            '# E0 is VisDrone\'s 640 baseline, R640 is UAVDT\'s; a dataset has one or the\n'
            '    # other, so listing both lets the same arm definition serve both datasets\n'
            '    "capacity"', 1)
    if check:
        print(f"iso_analysis.py: would set the capacity arm to {new_body.strip()}")
        return True
    bak = ISO.with_suffix(".py.bak-v3b")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
        print(f"backup -> {bak}")
    ISO.write_text(new, encoding="utf-8")
    print(f"iso_analysis.py: capacity arm is now {new_body.strip()}")
    return True


def patch_figs(check):
    if not FIGS.exists():
        fail(f"{FIGS} not found")
    s = FIGS.read_text(encoding="utf-8")
    if "no pre-registered pair overlaps on this dataset" in s:
        print("make_figures.py: pair fallback already present, nothing to do")
        return False
    if OLD_FILTER not in s:
        fail("the v3 pair filter was not found in make_figures.py. Run "
             "patch_figures_capacity.py first, and if it has already run, send the file.")
    if check:
        print("make_figures.py: would replace the pair filter with the fallback version")
        return True
    bak = FIGS.with_suffix(".py.bak-v3b")
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
        print(f"backup -> {bak}")
    FIGS.write_text(s.replace(OLD_FILTER, NEW_FILTER, 1), encoding="utf-8")
    print("make_figures.py: iso pair filter now falls back when no preferred pair overlaps")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    changed = patch_iso(a.check) | patch_figs(a.check)
    if a.check:
        print("\n(--check: nothing written)")
    elif changed:
        print("\nnext:  bash scripts/run_figures_v3.sh")
    else:
        print("\nnothing to do")


if __name__ == "__main__":
    main()
