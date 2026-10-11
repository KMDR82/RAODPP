#!/usr/bin/env bash
# Each condition/seed runs in its own process, so allocator fragmentation cannot
# accumulate across runs (three OOMs were caused by exactly that).
set -u
cd "$HOME/RAODpp"; source .venv/bin/activate
# expandable_segments removed: it corrupted the allocator state on resume
for spec in "S1280 42" "S1280 123" "S1280 2024" "R1536 123" "R1536 2024"; do
  set -- $spec
  echo "=== $1 seed $2 @ $(date) ==="
  python -u scripts/train_resolution.py --conditions "$1" --seeds "$2"
  sleep 20
done
echo "=== ALL DONE $(date) ==="
