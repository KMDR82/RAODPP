#!/usr/bin/env python3
"""
patch_capacity_arm.py — add a capacity path that varies capacity only.

Review finding: the capacity arm as built (S640, S960, S1280) scales capacity *and*
resolution, so comparing it against the resolution arm compares two resolution-scaled
families rather than isolating the two axes. The claim the paper makes is stronger than
that design supports.

The fix is a path that leaves resolution fixed at 640 and raises capacity only:

    n@640 (E0)  ->  s@640 (S640)  ->  m@640 (new)  ->  l@640 (new, optional)

Both paths then start from the same configuration, E0, and differ in one factor each.
The existing S960/S1280 points are kept: they remain a legitimate mixed path and are
reported as such rather than deleted.

Idempotent; writes nothing if an anchor is missing or the result does not parse.

    python scripts/patch_capacity_arm.py
"""
import ast
from pathlib import Path

S = Path.home() / "RAODpp" / "scripts"
report = []


def edit(name, old, new, marker):
    p = S / name
    if not p.exists():
        report.append((name, "MISSING")); return
    s = p.read_text()
    if marker in s:
        report.append((name, "already patched")); return
    if old not in s:
        report.append((name, "ANCHOR NOT FOUND")); return
    s2 = s.replace(old, new, 1)
    try:
        ast.parse(s2)
    except SyntaxError as e:
        report.append((name, f"WOULD NOT PARSE: {e}")); return
    p.write_text(s2)
    report.append((name, "patched"))


# Batch sizes chosen to fit 24 GB at 640 px: yolov8m is 25.9 M parameters and yolov8l
# 43.7 M, against 11.2 M for yolov8s which trains at 32.
edit("train_resolution.py",
     '    "S640":  ("yolov8s.pt",     640, 32),',
     '    "S640":  ("yolov8s.pt",     640, 32),\n'
     '    # capacity-only path at fixed 640 px, so the capacity axis is isolated\n'
     '    "M640":  ("yolov8m.pt",     640, 16),\n'
     '    "L640":  ("yolov8l.pt",     640,  8),',
     '"M640"')

edit("iso_analysis.py",
     '"capacity":   ["S640", "S960", "S1280"],          # yolov8s at 640-1280',
     '"capacity":   ["E0", "S640", "M640", "L640"],     # capacity only, fixed 640 px\n'
     '    "capacity_mixed": ["S640", "S960", "S1280"],      # yolov8s at 640-1280',
     '"capacity_mixed"')

edit("finalize_runs.py",
     '"S640": 640, "S960": 960, "S1280": 1280,',
     '"S640": 640, "M640": 640, "L640": 640,\n'
     '         "S960": 960, "S1280": 1280,',
     '"M640"')

print("\n".join(f"{n:24s} {r}" for n, r in report))
raise SystemExit(1 if any(r.startswith(("ANCHOR", "WOULD", "MISSING"))
                          for _, r in report) else 0)
