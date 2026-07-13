#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="$ROOT_DIR/reports/stage_06"
REPORT="$REPORT_DIR/rtl_sim_results.txt"
mkdir -p "$REPORT_DIR"

{
  echo "Stage 6 final RTL simulation"
  echo "Runs Stage 6B, 6C, 6D, and 6E VCS regressions. Stage 6E includes the final projection_integrated_mha top and assertions."
  echo
} > "$REPORT"

failures=0
run_one() {
  local name="$1"
  local script="$2"
  echo "$ bash $script" | tee -a "$REPORT"
  if bash "$ROOT_DIR/$script" >> "$REPORT" 2>&1; then
    echo "${name}: PASS" | tee -a "$REPORT"
  else
    echo "${name}: FAIL" | tee -a "$REPORT"
    failures=$((failures + 1))
  fi
  echo >> "$REPORT"
}

run_one "stage6b-rtl-sim" "scripts/sim/run_stage6b_vcs.sh"
run_one "stage6c-rtl-sim" "scripts/sim/run_stage6c_vcs.sh"
run_one "stage6d-rtl-sim" "scripts/sim/run_stage6d_vcs.sh"
run_one "stage6e-rtl-sim" "scripts/sim/run_stage6e_vcs.sh"

if [[ "$failures" -ne 0 ]]; then
  echo "Stage 6 RTL simulation result: FAIL" | tee -a "$REPORT"
  exit 1
fi

echo "Stage 6 RTL simulation result: PASS" | tee -a "$REPORT"
