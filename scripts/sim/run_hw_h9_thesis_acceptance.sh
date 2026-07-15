#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_thesis_acceptance"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
REPORT="$REPORT_DIR/thesis_acceptance_regression.md"
PYTHON_BIN=${PYTHON:-python3}

mkdir -p "$BUILD_DIR" "$REPORT_DIR"

{
  echo "# Hardware Stage H9 Thesis Acceptance Regression"
  echo
  echo "Status: running."
  echo
  echo "| Item | Command | Result | Log | Acceptance relevance | Strict-only deferred item |"
  echo "|---|---|---|---|---|---|"
} > "$REPORT"

failures=0

run_step() {
  local item=$1
  local command=$2
  local relevance=$3
  local strict_only=$4
  local log="$BUILD_DIR/${item}.log"
  local result

  (cd "$ROOT_DIR" && bash -lc "$command") > "$log" 2>&1
  local code=$?
  if [ "$code" -eq 0 ]; then
    result=PASS
  else
    result=FAIL
    failures=$((failures + 1))
  fi
  echo "| $item | \`$command\` | $result | build/hw_h9_thesis_acceptance/${item}.log | $relevance | $strict_only |" >> "$REPORT"
}

run_step "01_h9_model_tests" \
  "$PYTHON_BIN scripts/sim/run_hw_h9_tests.py" \
  "H9 host/model tests, H9/H8 bit comparison, cycle model, py_compile, Stage8 host regression" \
  "No"

run_step "02_h9_rtl_bundle" \
  "bash scripts/sim/run_hw_h9_vcs.sh" \
  "H9 buffers, single-head RTL, matched staged/interleaved A/B, multi-head RTL, full-layer RTL, long-sequence/cache-full" \
  "No"

run_step "03_cycle_calibration" \
  "$PYTHON_BIN model/attention/paper_interleaved_cycle_model.py" \
  "Matched RTL total-cycle calibration for D_HEAD 8/16/64 and seq 1/2/8/16/32/64" \
  "No"

run_step "04_direct_reset" \
  "bash scripts/sim/run_hw_h9_reset_vcs.sh" \
  "Direct H9 interleaved datapath reset matrix, 64 injection labels" \
  "No"

run_step "05_direct_random" \
  "bash scripts/sim/run_hw_h9_random_backpressure_vcs.sh" \
  "Direct H9 datapath random backpressure, 20 fixed seeds" \
  "No"

run_step "06_multi_head_reset" \
  "bash scripts/sim/run_hw_h9_multi_head_reset_vcs.sh" \
  "Independent multi-head reset matrix on real multi_head_generation_engine hierarchy" \
  "No"

run_step "07_multi_head_random" \
  "bash scripts/sim/run_hw_h9_multi_head_random_backpressure_vcs.sh" \
  "Independent multi-head broad random backpressure on real multi_head_generation_engine hierarchy, 24 runs" \
  "No"

run_step "08_assertions" \
  "bash scripts/sim/run_hw_h9_assertion_vcs.sh" \
  "23 explicit H9 SVA properties, positive bind execution, 23/23 negative tests" \
  "No"

run_step "09_h9_lint" \
  "$PYTHON_BIN scripts/lint/run_hw_h9_lint.py" \
  "H9 vlogan/static hygiene/lint acceptance" \
  "No"

run_step "10_h9_dc_structural" \
  "$PYTHON_BIN scripts/synth/run_hw_h9_synth_check.py" \
  "H9 DC analyze/elaborate/link/check_design structural hierarchy check only" \
  "No"

run_step "11_stage8_regression" \
  "make PYTHON=$PYTHON_BIN stage8-test stage8-rtl-sim stage8-lint stage8-synth" \
  "Stage 8 accepted paper-array regression" \
  "No"

run_step "12_stage7_regression" \
  "make PYTHON=$PYTHON_BIN stage7a-test stage7b-test stage7b-rtl-sim stage7b-lint stage7b-synth stage7c-test stage7c-rtl-sim stage7c-lint stage7c-synth stage7d-test stage7d-rtl-sim stage7d-lint stage7d-synth" \
  "Stage 7 Pre-Norm transformer-layer regression" \
  "No"

run_step "13_stage6_regression" \
  "make PYTHON=$PYTHON_BIN stage6-test stage6-rtl-sim stage6-lint stage6-synth" \
  "Stage 6 projection-integrated MHA regression" \
  "No"

run_step "14_stage5_regression" \
  "make PYTHON=$PYTHON_BIN stage5-test stage5-rtl-sim stage5-lint stage5-synth" \
  "Stage 5 shared multi-head/current-token/cache semantics regression" \
  "No"

echo "| 15_full_layer_internal_reset_matrix | not run by thesis target | DEFERRED | reports/hw_h9/deferred_ip_verification.md | Deep internal transformer_layer reset injection is valuable for IP-grade verification but outside thesis acceptance scope | Yes |" >> "$REPORT"
echo "| 16_full_layer_internal_multi_endpoint_random | not run by thesis target | DEFERRED | reports/hw_h9/deferred_ip_verification.md | Deep internal transformer_layer multi-endpoint random backpressure is valuable for IP-grade verification but outside thesis acceptance scope | Yes |" >> "$REPORT"

{
  echo
  echo "## Result"
  echo
  if [ "$failures" -eq 0 ]; then
    echo "HARDWARE STAGE H9 PASS — UNDERGRADUATE THESIS SCOPE"
    echo
    echo "STRICT IP-GRADE H9 VERIFICATION NOT CLOSED"
    sed -i 's/Status: running./Status: PASS for undergraduate thesis scope./' "$REPORT"
  else
    echo "HW-H9 THESIS ACCEPTANCE FAILED"
    echo "failures=$failures"
    sed -i 's/Status: running./Status: FAIL./' "$REPORT"
  fi
} >> "$REPORT"

cat "$REPORT"
[ "$failures" -eq 0 ] && exit 0 || exit 1
