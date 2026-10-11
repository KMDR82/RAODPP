#!/usr/bin/env python3
"""
patch_deterministic.py — make the deterministic flag selectable per dataset.

A diagnostic showed that with `deterministic=True` the YOLOv8s runs on UAVDT follow an
identical trajectory across seeds: every loss, metric and learning-rate value matches to
six decimals, and only wall-clock time differs. With `deterministic=False` the same two
seeds diverge from the first epoch. The flag is therefore suppressing the seed effect for
that model on that dataset, so seed variance cannot be measured there.

The flag controls reproducibility, not the treatment under test, so it can differ between
datasets as long as it is constant within one. VisDrone keeps `deterministic=True`, under
which seed variance was already visible; UAVDT is re-run with `deterministic=False`.

Idempotent; refuses to write if the anchor is missing or the result does not parse.

    python scripts/patch_deterministic.py
"""
import ast
from pathlib import Path

P = Path.home() / "RAODpp" / "scripts" / "train_resolution.py"
s = P.read_text()

if "--deterministic" in s:
    print("already patched")
    raise SystemExit(0)

edits = [
    # 1. use the flag from CONFIG instead of a hard-coded True
    ("                         deterministic=True, cache=False, project=str(RUNS_DIR),",
     "                         deterministic=CONFIG[\"DETERMINISTIC\"], cache=False,\n"
     "                         project=str(RUNS_DIR),"),
    # 2. give it a default
    ('CONFIG = {"SEEDS"',
     'CONFIG = {"DETERMINISTIC": True, "SEEDS"'),
    # 3. expose it on the command line
    ('    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")',
     '    ap.add_argument("--dataset", choices=list(DATASETS), default="visdrone")\n'
     '    ap.add_argument("--deterministic", choices=["true", "false"], default="true",\n'
     '                    help="false lifts the seed suppression seen on UAVDT with '
     'yolov8s")'),
    # 4. apply it
    ('    CONFIG["DATASET"] = a.dataset',
     '    CONFIG["DATASET"] = a.dataset\n'
     '    CONFIG["DETERMINISTIC"] = (a.deterministic == "true")'),
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
for token in ["DETERMINISTIC", "--deterministic", 'a.deterministic == "true"']:
    print(f"  {token:26s} {s.count(token)} occurrence(s)")
