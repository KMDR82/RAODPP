#!/usr/bin/env python3
"""
patch_workers.py — expose the dataloader worker count on the command line.

A diagnostic on UAVDT with YOLOv8s, five epochs, two seeds:

    workers=0  ->  the two seeds diverge from the first epoch
    workers=8  ->  the two seeds are bit-identical

With multiple loader workers the sampling order stops depending on the seed, and for this
model on this dataset no other stochastic source remains, so three "independent" runs
produced the same trajectory to six decimals. The flag therefore silently removes the
variance the study is trying to measure, and it has to be selectable rather than fixed.

Idempotent; writes nothing if the anchor is missing or the result does not parse.

    python scripts/patch_workers.py
"""
import ast
from pathlib import Path

P = Path.home() / "RAODpp" / "scripts" / "train_resolution.py"
s = P.read_text()

if "--workers" in s:
    print("already patched")
    raise SystemExit(0)

edits = [
    ('    ap.add_argument("--deterministic", choices=["true", "false"], default="true",',
     '    ap.add_argument("--workers", type=int, default=None,\n'
     '                    help="dataloader workers; 0 restores the seed effect that "\n'
     '                         "multi-worker loading suppresses on UAVDT")\n'
     '    ap.add_argument("--deterministic", choices=["true", "false"], default="true",'),
    ('    CONFIG["DETERMINISTIC"] = (a.deterministic == "true")',
     '    CONFIG["DETERMINISTIC"] = (a.deterministic == "true")\n'
     '    if a.workers is not None:\n'
     '        CONFIG["WORKERS"] = a.workers'),
]

for old, new in edits:
    if old not in s:
        raise SystemExit(f"ANCHOR NOT FOUND, nothing written:\n  {old.strip()[:70]}")
    s = s.replace(old, new, 1)

try:
    ast.parse(s)
except SyntaxError as e:
    raise SystemExit(f"patched file would not parse, nothing written: {e}")

P.write_text(s)
print("patched train_resolution.py")
for tok in ["--workers", 'CONFIG["WORKERS"] = a.workers', 'workers=CONFIG["WORKERS"]']:
    print(f"  {tok:34s} {s.count(tok)} occurrence(s)")
