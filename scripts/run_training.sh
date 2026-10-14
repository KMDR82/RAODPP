#!/usr/bin/env bash
# run_training.sh - train the published grid.
#
# Needs the datasets in place and a GPU. The grid was run on a single NVIDIA RTX 4500 Ada
# (24 GB); the per-condition batch sizes in train_resolution.py are the largest that fit
# in that much memory at each input size.
#
# train_resolution.py is resumable: a condition and seed whose weights and runs.csv row
# already exist is skipped, so this can be interrupted and restarted.
#
#   bash scripts/run_training.sh visdrone
#   bash scripts/run_training.sh uavdt
#
set -u
DS="${1:-visdrone}"
cd "$HOME/RAODpp" || { echo "RAODpp not found in $HOME"; exit 1; }
[ -f .venv/bin/activate ] && source .venv/bin/activate

case "$DS" in
  visdrone) CONDS="R960 R1280 R1536 S640 S960 S1280 M640 L640 P640 P960" ;;
  uavdt)    CONDS="R640 R960 R1280 S640 S960" ;;
  *) echo "unknown dataset: $DS"; exit 1 ;;
esac
SEEDS="42 123 2024"

mkdir -p output/logs
LOG="output/logs/train_${DS}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== training $DS $(date -Is) ==="
echo "conditions: $CONDS"
echo "seeds:      $SEEDS"

if pgrep -fa "train_resolution.py" | grep -v "$$" | grep -q .; then
  echo "FAIL: training is already running. Two training processes on one GPU exhaust its"
  echo "memory and one run will die of it."
  exit 1
fi

# workers must be 0. With the dataloader at 8 workers the seed does not reach the training
# loop on UAVDT: three nominally independent runs produce bit-identical trajectories and a
# standard deviation of exactly zero.
python -u scripts/train_resolution.py \
  --dataset "$DS" --conditions $CONDS --seeds $SEEDS --workers 0
rc=$?
[ "$rc" -eq 0 ] || { echo "exited with $rc; rerun this script, finished runs are skipped"; exit "$rc"; }

echo
echo "--- seed variance check ---"
python - <<'PY'
import sys
import pandas as pd
from pathlib import Path
d = pd.read_csv(Path.home() / "RAODpp" / "output" / "runs.csv")
bad = []
for (ds, cond), sub in d.groupby(["dataset", "exp"]):
    if len(sub) < 2:
        continue
    sd = sub.mAP50_95.std(ddof=1)
    if sd == 0:
        bad.append(f"{ds}/{cond}")
    print(f"  {ds:<9} {cond:<7} {len(sub)} seed(s)  sd={sd:.6f}"
          + ("   <-- zero, the seed did not reach the training loop" if sd == 0 else ""))
if bad:
    sys.exit(f"\nFAIL: {bad} are identical across seeds. Do not use these results.")
PY
echo
echo "next: python scripts/measure_cost_only.py --dataset $DS"
echo "      bash scripts/run_analysis.sh"
