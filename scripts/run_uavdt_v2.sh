#!/usr/bin/env bash
# run_uavdt_v2.sh — UAVDT grid, deterministic mode off.
#
# Rationale (protocol note): with deterministic=True the YOLOv8s runs on UAVDT followed an
# identical trajectory across seeds — every loss, metric and learning rate matched to six
# decimals — so seed variance could not be measured. A diagnostic with deterministic=False
# showed the same two seeds diverging from the first epoch. The whole UAVDT grid is
# therefore re-run with the flag off, so that one setting holds for all its conditions.
# VisDrone keeps deterministic=True, under which seed variance was already visible.
#
#   cd ~/RAODpp && nohup bash scripts/run_uavdt_v2.sh > output/logs/uavdt_v2.log 2>&1 &
set -u
ROOT="$HOME/RAODpp"
cd "$ROOT"; source .venv/bin/activate

# the deterministic runs are moved aside rather than deleted, so the diagnostic that
# motivated this re-run stays reproducible
STASH="$ROOT/model/runs_deterministic_uavdt"
mkdir -p "$STASH"
for d in "$ROOT"/model/runs/*_uavdt_s*; do
  [ -d "$d" ] && mv "$d" "$STASH"/ 2>/dev/null
done
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path.home()/"RAODpp"/"output"/"runs.csv"
d = pd.read_csv(p); n = len(d)
d = d[d.get("dataset", "") != "uavdt"]
d.to_csv(p, index=False)
print(f"runs.csv: {n} -> {len(d)} (uavdt rows moved out)")
PY

for seed in 42 123 2024; do
  for c in R640 R960 R1280 S640 S960; do
    echo "=== $c seed $seed @ $(date) ==="
    python -u scripts/train_resolution.py --dataset uavdt --deterministic false \
           --conditions "$c" --seeds "$seed"
    sleep 20
  done
done
echo "=== ALL DONE $(date) ==="
