#!/usr/bin/env bash
# run_treatments.sh — keeps the E1-E5 matrix alive across crashes and power cuts.
#
#   cd ~/RAODpp && nohup bash scripts/run_treatments.sh > /dev/null 2>&1 &
#   tail -f ~/RAODpp/output/logs/treat_latest.log
#
# Stop: touch ~/RAODpp/output/STOP_TREAT   (graceful)  |  pkill -f train_treatments.py

set -u
ROOT="$HOME/RAODpp"
LOGDIR="$ROOT/output/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/treat_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$LOGDIR/treat_latest.log"

attempt=1
echo "=== run_treatments.sh started $(date) ===" | tee -a "$LOG"

while [ "$attempt" -le 50 ]; do
  if [ -f "$ROOT/output/STOP_TREAT" ]; then
    echo "STOP file present — exiting." | tee -a "$LOG"; break
  fi
  echo "--- attempt $attempt at $(date) ---" | tee -a "$LOG"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  python -u "$ROOT/scripts/train_treatments.py" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "=== finished cleanly $(date) ===" | tee -a "$LOG"; break
  fi
  echo "!!! exited with status $status — retrying in 60 s" | tee -a "$LOG"
  sleep 60
  attempt=$((attempt + 1))
done
