#!/usr/bin/env bash
# run_resilient.sh — keeps E0 alive across crashes, OOM kills and power cuts.
#
# Usage (detached, survives closing VS Code / the terminal):
#     cd ~/RAODpp && nohup bash scripts/run_resilient.sh > /dev/null 2>&1 &
#
# Watch progress:
#     tail -f ~/RAODpp/output/logs/e0_latest.log
#
# Stop it:
#     touch ~/RAODpp/output/STOP        # graceful, after the current attempt
#     pkill -f train_e0.py              # immediate

set -u
ROOT="$HOME/RAODpp"
LOGDIR="$ROOT/output/logs"
mkdir -p "$LOGDIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/e0_$STAMP.log"
ln -sf "$LOG" "$LOGDIR/e0_latest.log"

MAX_ATTEMPTS=50
attempt=1

echo "=== run_resilient.sh started $(date) ===" | tee -a "$LOG"

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if [ -f "$ROOT/output/STOP" ]; then
    echo "STOP file present — exiting." | tee -a "$LOG"
    break
  fi

  echo "--- attempt $attempt at $(date) ---" | tee -a "$LOG"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  python "$ROOT/scripts/train_e0.py" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}

  if [ "$status" -eq 0 ]; then
    echo "=== train_e0.py exited cleanly at $(date) ===" | tee -a "$LOG"
    break
  fi

  echo "!!! exited with status $status — retrying in 60 s" | tee -a "$LOG"
  sleep 60
  attempt=$((attempt + 1))
done

echo "=== run_resilient.sh finished $(date) ===" | tee -a "$LOG"
