#!/usr/bin/env python3
"""
patch_p1280.py — add P1280 (YOLOv8n-P2 @1280) to the grid and to the P2 arm.

Self-review item R2: the P2 arm has only two measured points, so its comparisons span
0.32 ms on the latency axis and the claim about it rests on that window. The arm is also
the one a reader of the UAV literature cares about most, since adding a P2 head is the
standard remedy there. A third point takes the arm to 640-1280 and makes the comparison
span a usable range.

Idempotent; writes nothing if an anchor is missing or the result does not parse.

    python scripts/patch_p1280.py
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


# batch 6: P2 at 960 needed 8, and the extra feature level at 1280 is the memory-heaviest
# configuration in the grid
edit("train_resolution.py",
     '    "S1280": ("yolov8s.pt",    1280,  8),',
     '    # self-review R2: a third P2 point so the arm spans a usable cost range\n'
     '    "P1280": ("yolov8-p2.yaml", 1280,  6),\n'
     '    "S1280": ("yolov8s.pt",    1280,  8),',
     '"P1280"')

edit("iso_analysis.py",
     '"p2head":     ["P640", "P960"],                   # yolov8n + P2 head at 640 / 960',
     '"p2head":     ["P640", "P960", "P1280"],          # yolov8n + P2 head at 640-1280',
     '"P1280"')

edit("finalize_runs.py",
     '"P640": 640, "P960": 960}',
     '"P640": 640, "P960": 960, "P1280": 1280}',
     '"P1280"')

print("\n".join(f"{n:24s} {r}" for n, r in report))
raise SystemExit(1 if any(r.startswith(("ANCHOR", "WOULD", "MISSING"))
                          for _, r in report) else 0)
