#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_assertions"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/assertion_negative_results.md"
MATRIX="$REPORT_DIR/assertion_execution_matrix.md"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$REPORT_DIR"

find_dw_sim_dir() {
  if [ -n "${DW_SIM_DIR:-}" ] && [ -d "$DW_SIM_DIR" ]; then
    echo "$DW_SIM_DIR"
    return 0
  fi
  for candidate in /usr/synopsys/*/dw/sim_ver /usr/synopsys/*/*/dw/sim_ver; do
    if [ -d "$candidate" ] && [ -f "$candidate/vcs/DW_exp2.v" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

DW_SIM_DIR_DETECTED=$(find_dw_sim_dir || true)
{
  echo "# Hardware Stage H9 Assertion Negative Results"
  echo
  echo "Positive tests bind \`h9_interleaved_assertions\` into \`paper_interleaved_attention_datapath\`."
  echo "Negative tests instantiate the same monitor in an isolated harness and intentionally violate one property per run."
  echo
} > "$SUMMARY"
{
  echo "# Hardware Stage H9 Assertion Execution Matrix"
  echo
  echo "| Item | Configuration | Testbench | Seed/Injection | Expected | Result | Log |"
  echo "|---|---|---|---|---|---|---|"
} > "$MATRIX"

if ! command -v vcs >/dev/null 2>&1; then
  echo "vcs: NOT FOUND" >> "$SUMMARY"
  echo "result=FAIL" >> "$SUMMARY"
  cat "$SUMMARY"
  exit 10
fi
if [ -z "$DW_SIM_DIR_DETECTED" ]; then
  echo "DesignWare sim dir: NOT FOUND" >> "$SUMMARY"
  echo "result=FAIL" >> "$SUMMARY"
  cat "$SUMMARY"
  exit 11
fi

DW_FILES=(
  "$DW_SIM_DIR_DETECTED/DW_fp_addsub.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_dp2.v"
  "$DW_SIM_DIR_DETECTED/DW_ifp_mult.v"
  "$DW_SIM_DIR_DETECTED/DW_ifp_addsub.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_ifp_conv.v"
  "$DW_SIM_DIR_DETECTED/DW_ifp_fp_conv.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_mult.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_add.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_mac.v"
  "$DW_SIM_DIR_DETECTED/DW_exp2.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_exp.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_div.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_sqrt.v"
)

COMMON_RTL=(
  "$ROOT_DIR/rtl/common/stream_reg.sv"
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_add_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_exp_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_recip_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_sqrt_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_to_fp16.sv"
  "$ROOT_DIR/rtl/pe/lane_mask_generator.sv"
  "$ROOT_DIR/rtl/pe/accumulator_bank.sv"
  "$ROOT_DIR/rtl/pe/pe_perf_counter.sv"
  "$ROOT_DIR/rtl/pe/pe_lane.sv"
  "$ROOT_DIR/rtl/pe/fp32_reduction_tree.sv"
  "$ROOT_DIR/rtl/pe/reconfigurable_pe_core.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_pe_cell.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_l1_reduction.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_l2_reduction.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_pe_group.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_array_8x8x2.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_score_packet_pkg.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_score_buffer.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_probability_fifo.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/h9_assertions.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_interleaved_attention_datapath.sv"
  "$ROOT_DIR/rtl/attention/attention_score_scaler.sv"
  "$ROOT_DIR/rtl/attention/softmax_reduction.sv"
  "$ROOT_DIR/rtl/attention/softmax_normalization.sv"
  "$ROOT_DIR/tb/rtl/hw_h9/h9_assertion_bind.sv"
)

failures=0
positive_simv="$BUILD_DIR/assertion_positive_simv"
positive_compile="$BUILD_DIR/assertion_positive_compile.log"
(cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
  +incdir+"$DW_SIM_DIR_DETECTED" \
  +define+HW_H9_D_HEAD=8 \
  -Mdir="$BUILD_DIR/assertion_positive_csrc" \
  -o "$positive_simv" \
  "${DW_FILES[@]}" \
  "${COMMON_RTL[@]}" \
  "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_single_head.sv" \
  -top tb_h9_single_head \
  -l "$positive_compile")
code=$?
if [ "$code" -ne 0 ]; then
  echo "positive bind compile failed exit_code=$code" >> "$SUMMARY"
  failures=$((failures + 1))
  positive_result=FAIL
else
  positive_log="$BUILD_DIR/assertion_positive_run.log"
  timeout 300s "$positive_simv" -l "$positive_log" >/dev/null 2>&1
  run_code=$?
  if [ "$run_code" -eq 0 ] && grep -q "HW_H9_SINGLE_HEAD_PASS" "$positive_log" &&
     ! grep -E "(h9_assertion .* failed|assert.*failed|Error:)" "$positive_log" >/dev/null 2>&1; then
    positive_result=PASS
  else
    positive_result=FAIL
    failures=$((failures + 1))
  fi
fi
echo "| positive_bind | H1/D8 seq8 | tb_h9_single_head + h9_assertion_bind | bind paper_interleaved_attention_datapath | all properties compile, bind, execute, and do not fire | $positive_result | build/hw_h9_assertions/assertion_positive_run.log |" >> "$MATRIX"

negative_simv="$BUILD_DIR/assertion_negative_simv"
negative_compile="$BUILD_DIR/assertion_negative_compile.log"
(cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
  -Mdir="$BUILD_DIR/assertion_negative_csrc" \
  -o "$negative_simv" \
  "$ROOT_DIR/rtl/attention/paper/interleaved/h9_assertions.sv" \
  "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_assertion_negative.sv" \
  -top tb_h9_assertion_negative \
  -l "$negative_compile")
code=$?
if [ "$code" -ne 0 ]; then
  echo "negative assertion harness compile failed exit_code=$code" >> "$SUMMARY"
  failures=$((failures + 1))
fi

NEGATIVE_CASES=(
  no_inner_and_outer_same_cycle
  no_mode_switch_with_inflight_operation
  no_outer_before_qk_retired
  no_outer_before_softmax_valid
  no_new_head_before_previous_retired
  score_count_conserved
  probability_count_conserved
  no_score_overflow
  no_score_underflow
  no_probability_overflow
  no_probability_underflow
  score_payload_stable_until_ready
  probability_payload_stable_until_ready
  probability_matches_v_index
  no_duplicate_sv_update
  no_missing_sv_update
  no_duplicate_head_done
  no_duplicate_cache_commit
  valid_seq_len_changes_only_by_commit
  reset_clears_interleaved_state
  no_unknown_control_when_active
  transaction_count_conserved
  progress_or_legal_stall
)

if [ -x "$negative_simv" ]; then
  for index in "${!NEGATIVE_CASES[@]}"; do
    case_name=${NEGATIVE_CASES[$index]}
    log="$BUILD_DIR/negative_${case_name}.log"
    timeout 120s "$negative_simv" +NEGATIVE_CASE="$case_name" -l "$log" >/dev/null 2>&1
    if grep -q "h9_assertion ${case_name} failed" "$log"; then
      result=PASS
      echo "negative_case=$case_name triggered_property=$case_name result=PASS log=build/hw_h9_assertions/$(basename "$log")" >> "$SUMMARY"
    else
      result=FAIL
      failures=$((failures + 1))
      echo "negative_case=$case_name result=FAIL log=build/hw_h9_assertions/$(basename "$log")" >> "$SUMMARY"
    fi
    echo "| $((index + 1)). $case_name | isolated monitor | tb_h9_assertion_negative | +NEGATIVE_CASE=$case_name | target property fires exactly as expected | $result | build/hw_h9_assertions/$(basename "$log") |" >> "$MATRIX"
  done
fi

{
  echo
  echo "explicit_sva_properties=${#NEGATIVE_CASES[@]}"
  echo "scoreboard_only_properties=0"
  echo "bind_target=paper_interleaved_attention_datapath"
  echo "positive_bind_result=$positive_result"
  echo "negative_failures=$failures"
  echo "matrix=reports/hw_h9/assertion_execution_matrix.md"
  if [ "$failures" -eq 0 ]; then
    echo "result=PASS"
  else
    echo "result=FAIL"
  fi
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
