#!/usr/bin/env bash
# run_p1280.sh — third point for the P2 arm (self-review R2).
#
# The P2 arm had only 640 and 960, so its comparisons spanned 0.32 ms on the latency axis.
# Adding 1280 takes the arm to a usable range and lets the capacity-vs-P2 comparison, which
# the new cost measurements showed to split across the two axes, be tested over more than a
# sliver of budget.
#
# One condition per process: allocator fragmentation across consecutive runs caused three
# OOMs earlier in this study.
#
#   cd ~/RAODpp && nohup bash scripts/run_p1280.sh > output/logs/p1280.log 2>&1 &
set -u
ROOT="$HOME/RAODpp"
cd "$ROOT"; source .venv/bin/activate
for seed in 42 123 2024; do
  echo "=== P1280 seed $seed @ $(date) ==="
  python -u scripts/train_resolution.py --dataset visdrone \
         --conditions P1280 --seeds "$seed"
  sleep 20
done
echo "=== ALL DONE $(date) ==="
