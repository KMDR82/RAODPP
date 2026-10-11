#!/usr/bin/env python3
"""Collect params / layers / GFLOPs for every finished run -> figdata/model_info.csv"""
import sys
from pathlib import Path

ROOT = Path.home() / "RAODpp"
sys.path.insert(0, str(ROOT / "scripts"))
import figdata

exp = sys.argv[1] if len(sys.argv) > 1 else "E0"
runs = sorted((ROOT / "model" / "runs").glob(f"{exp}_*_s*"))
print(f"found {len(runs)} runs for {exp}")

for rd in runs:
    w = rd / "weights" / "best.pt"
    if not w.exists():
        print(f"  {rd.name}: no best.pt, skipped")
        continue
    seed = int(rd.name.split("_s")[-1])
    row = figdata.capture_model_info(w, rd.name, exp, seed, ROOT / "output")
    print(f"  {rd.name}: params={row['params']:,} layers={row['layers']} gflops={row['gflops']}")

print("\nwrote:", ROOT / "output" / "figdata" / "model_info.csv")
