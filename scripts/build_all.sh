#!/usr/bin/env bash
set -u
cd "$HOME/RAODpp"
source .venv/bin/activate
for m in E2 E3 E4 E5; do
  echo "=== $m $(date) ==="
  python -u scripts/upaste.py --mode "$m" --samples 0
done
echo "=== ALL DONE $(date) ==="
