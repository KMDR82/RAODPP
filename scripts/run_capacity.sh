#!/usr/bin/env bash
# run_capacity.sh — the capacity-only path (review finding 1).
#
# The capacity arm as built varied capacity and resolution together, so it could not
# isolate the axis the paper's claim rests on. This adds m@640 and l@640, which together
# with the existing n@640 and s@640 give a path where only capacity changes.
#
# One condition per process: allocator fragmentation across consecutive runs caused three
# OOMs earlier in this study.
#
#   cd ~/RAODpp && nohup bash scripts/run_capacity.sh > output/logs/capacity640.log 2>&1 &
set -u
cd "$HOME/RAODpp"; source .venv/bin/activate
for seed in 42 123 2024; do
  for c in M640 L640; do
    echo "=== $c seed $seed @ $(date) ==="
    python -u scripts/train_resolution.py --dataset visdrone --conditions "$c" --seeds "$seed"
    sleep 20
  done
done
echo "=== ALL DONE $(date) ==="
