#!/usr/bin/env bash
# run_figures_v3.sh — regenerate every figure that the capacity-arm correction invalidated.
#
# Background. pareto.pdf and the three VisDrone iso_*.pdf were drawn before the capacity
# arm was rebuilt as n@640 -> s@640 -> m@640 -> l@640 at fixed resolution. They still draw
# the mixed path (s@640/s@960/s@1280) under the label "capacity" and omit m@640 and l@640.
# The manuscript tables are correct; the figures are not.
#
# Order matters: iso_analysis.py writes the curve CSVs that the iso figures read, so the
# analysis must run before the figures. Nothing here trains or evaluates anything, so it
# costs minutes, not hours, and it touches no run directory.
#
#   bash scripts/run_figures_v3.sh
#
set -u
cd "$HOME/RAODpp" || { echo "RAODpp not found in $HOME"; exit 1; }
mkdir -p output/logs fig/_superseded

LOG="output/logs/figures_v3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== figure regeneration, capacity arm v3 — $(date -Is) ==="
echo "log: $LOG"

# ---------------------------------------------------------------- 0. patch the drawer
echo
echo "--- 0. patch make_figures.py ---"
python scripts/patch_figures_capacity.py || { echo "patch failed, stopping"; exit 1; }

# ---------------------------------------------------------------- 1. keep the old figures
echo
echo "--- 1. move the superseded figures aside ---"
# kept rather than overwritten: if a reviewer asks what the earlier figures showed, the
# answer should be a file and not a memory
for f in pareto.pdf iso_latency.pdf iso_energy.pdf iso_energy_marginal.pdf; do
  if [ -f "fig/$f" ] && [ ! -f "fig/_superseded/$f" ]; then
    cp "fig/$f" "fig/_superseded/$f"
    echo "  kept fig/_superseded/$f"
  fi
done

# ---------------------------------------------------------------- 2. re-run the analysis
echo
echo "--- 2. iso_analysis on the corrected arms ---"
fail=0
for ds in visdrone uavdt; do
  for cost in latency energy energy_marginal; do
    echo
    echo "  [$ds / $cost]"
    python scripts/iso_analysis.py --dataset "$ds" --cost "$cost" \
      || { echo "  FAILED: $ds / $cost"; fail=1; }
  done
done
[ "$fail" -eq 0 ] || { echo; echo "analysis failed, not drawing figures"; exit 1; }

# ---------------------------------------------------------------- 3. draw
echo
echo "--- 3. draw ---"
python scripts/make_figures.py --only pareto iso confound || { echo "drawing failed"; exit 1; }

# ---------------------------------------------------------------- 4. verify
echo
echo "--- 4. verify against the manuscript ---"
python scripts/verify_figures.py
rc=$?

echo
echo "--- files ---"
ls -la fig/pareto.pdf fig/iso_latency.pdf fig/iso_energy.pdf \
       fig/iso_energy_marginal.pdf fig/iso_confound.pdf 2>/dev/null
echo
if [ "$rc" -eq 0 ]; then
  echo "DONE. Figures agree with the manuscript."
  echo "Send these to the chat: fig/pareto.pdf fig/iso_latency.pdf fig/iso_energy.pdf"
  echo "fig/iso_energy_marginal.pdf fig/iso_confound.pdf fig/_superseded/pareto.pdf"
  echo "and output/figdata/pareto_points.csv"
else
  echo "DONE WITH PROBLEMS. Read section 4 above before using any of these figures."
fi
exit "$rc"
