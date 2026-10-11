#!/usr/bin/env bash
# run_scratch.sh — from-scratch comparison (F3 vs F5 vs E0), crash-resilient.
#   cd ~/RAODpp && nohup bash scripts/run_scratch.sh > /dev/null 2>&1 &
#   tail -f ~/RAODpp/output/logs/scratch_latest.log
set -u
ROOT="$HOME/RAODpp"; LOGDIR="$ROOT/output/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/scratch_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$LOGDIR/scratch_latest.log"
attempt=1
echo "=== run_scratch.sh started $(date) ===" | tee -a "$LOG"
while [ "$attempt" -le 50 ]; do
  [ -f "$ROOT/output/STOP_SCRATCH" ] && { echo "STOP file present" | tee -a "$LOG"; break; }
  echo "--- attempt $attempt $(date) ---" | tee -a "$LOG"
  source "$ROOT/.venv/bin/activate"
  python -u "$ROOT/scripts/train_scratch.py" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] && { echo "=== clean exit $(date) ===" | tee -a "$LOG"; break; }
  sleep 60; attempt=$((attempt+1))
done
