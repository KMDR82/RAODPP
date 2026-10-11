#!/usr/bin/env bash
set -u
cd "$HOME/RAODpp"; source .venv/bin/activate
for B in 15 30; do
  python -u scripts/upaste.py --mode E5 --seeds 42 --samples 0 --budget "$B" --tag "b$B"
done
python -u scripts/train_scratch.py --conditions F5b15 F5b30 --seeds 42 \
       --source-map "F5b15:E5b15,F5b30:E5b30"
