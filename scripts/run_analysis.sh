#!/usr/bin/env bash
# run_analysis.sh - the full analysis pass over the recorded measurements.
#
# No training and no GPU. Everything below reads output/runs.csv and writes to
# output/figdata and fig. Takes a few minutes.
#
# Order matters: iso_analysis.py writes the curve and comparison CSVs that the figures,
# the tables and the graphical abstract all read, so it runs first.
#
#   bash scripts/run_analysis.sh
#
set -u
cd "$HOME/RAODpp" || { echo "RAODpp not found in $HOME"; exit 1; }
mkdir -p output/logs output/figdata fig

LOG="output/logs/analysis_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== analysis pass $(date -Is) ==="
echo "log: $LOG"

echo
echo "--- 1. iso-cost analysis ---"
fail=0
for ds in visdrone uavdt; do
  for cost in latency energy energy_marginal; do
    echo
    echo "  [$ds / $cost]"
    python scripts/iso_analysis.py --dataset "$ds" --cost "$cost" \
      || { echo "  FAILED: $ds / $cost"; fail=1; }
  done
done
[ "$fail" -eq 0 ] || { echo; echo "analysis failed, stopping before the figures"; exit 1; }

echo
echo "--- 2. tables ---"
python scripts/holm_table.py
python scripts/dominance_table.py
python scripts/per_seed_table.py

echo
echo "--- 3. figures ---"
python scripts/make_figures.py
python scripts/make_graphical_abstract.py

echo
echo "--- 4. verification ---"
python scripts/verify_figures.py
rc=$?

echo
[ "$rc" -eq 0 ] && echo "DONE. Figures agree with the manuscript." \
                || echo "DONE WITH PROBLEMS. Read section 4 above."
exit "$rc"
