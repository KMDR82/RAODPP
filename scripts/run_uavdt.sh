#!/usr/bin/env bash
set -u
cd "$HOME/RAODpp"; source .venv/bin/activate
for seed in 42 123 2024; do
  for c in R640 R960 R1280 S640 S960; do
    echo "=== $c seed $seed @ $(date) ==="
    python -u scripts/train_resolution.py --dataset uavdt --conditions "$c" --seeds "$seed"
    sleep 20
  done
done
echo "=== ALL DONE $(date) ==="
