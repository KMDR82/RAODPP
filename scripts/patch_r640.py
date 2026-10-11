#!/usr/bin/env python3
"""
patch_r640.py — add the R640 condition (yolov8n @640) to the grid.

On VisDrone the resolution arm's 640 point is E0, trained by train_e0.py before
train_resolution.py existed. UAVDT has no E0, so the arm needs its own 640 point or it
starts at 960 and the comparison loses its low end.

Idempotent; refuses to write if an anchor is missing or the result does not parse.

    python scripts/patch_r640.py
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


edit("train_resolution.py",
     'GRID = {\n    "R960":',
     'GRID = {\n'
     '    # R640 is the resolution arm\'s 640 point; on VisDrone that role was filled by\n'
     '    # E0, trained by train_e0.py before this script existed\n'
     '    "R640":  ("yolov8n.pt",     640, 64),\n'
     '    "R960":',
     '"R640"')

edit("iso_analysis.py",
     '"resolution": ["E0", "R960", "R1280", "R1536"],   # yolov8n at 640-1536',
     '"resolution": ["E0", "R640", "R960", "R1280", "R1536"],  # E0 is VisDrone\'s 640',
     '"R640"')

edit("finalize_runs.py",
     '"R960": 960, "R1280": 1280, "R1536": 1536,',
     '"R640": 640, "R960": 960, "R1280": 1280, "R1536": 1536,',
     '"R640"')

print("\n".join(f"{n:24s} {r}" for n, r in report))
raise SystemExit(1 if any(r.startswith(("ANCHOR", "WOULD", "MISSING"))
                          for _, r in report) else 0)
