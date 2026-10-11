#!/usr/bin/env bash
# run_uavdt_v3.sh — UAVDT grid with workers=0.
#
# Why: a five-epoch diagnostic with YOLOv8s on UAVDT showed that with workers=8 two seeds
# produce bit-identical trajectories, while with workers=0 they diverge from the first
# epoch. Multi-worker loading removes the seed's influence on sampling order, and for this
# model on this dataset nothing else is stochastic, so seed variance could not be measured
# for the capacity arm. The whole UAVDT grid is re-run with workers=0 so one loader
# setting holds across all of its conditions.
#
# VisDrone is not touched: seed variance was visible there for every condition, so the
# flag did not suppress it on that dataset.
#
#   cd ~/RAODpp && nohup bash scripts/run_uavdt_v3.sh > output/logs/uavdt_v3.log 2>&1 &
set -u
ROOT="$HOME/RAODpp"
cd "$ROOT"; source .venv/bin/activate

# earlier runs are stashed, not deleted: the observation that motivated this re-run has to
# stay reproducible
STASH="$ROOT/model/runs_workers8_uavdt"
mkdir -p "$STASH"
for d in "$ROOT"/model/runs/*_uavdt_s*; do
  [ -d "$d" ] && mv "$d" "$STASH"/ 2>/dev/null
done
rm -rf "$ROOT"/model/runs/DIAG_*
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path.home()/"RAODpp"/"output"/"runs.csv"
d = pd.read_csv(p); n = len(d)
d = d[d.get("dataset", "") != "uavdt"]
d.to_csv(p, index=False)
print(f"runs.csv: {n} -> {len(d)} (uavdt rows moved out)", flush=True)
PY

for seed in 42 123 2024; do
  for c in R640 R960 R1280 S640 S960; do
    echo "=== $c seed $seed @ $(date) ===" 
    python -u scripts/train_resolution.py --dataset uavdt \
           --deterministic false --workers 0 \
           --conditions "$c" --seeds "$seed"
    sleep 20
  done
done
echo "=== ALL DONE $(date) ==="
