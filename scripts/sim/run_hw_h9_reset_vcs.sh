#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_reset_matrix"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/reset_results.md"
MATRIX="$REPORT_DIR/reset_execution_matrix.md"

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
  echo "# Hardware Stage H9 Reset Results"
  echo
  echo "Scope: reset interrupt matrix for HW-H9 interleaved paper Attention."
  echo
} > "$SUMMARY"
{
  echo "# Hardware Stage H9 Reset Execution Matrix"
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

RESET_CASES=(
  idle token_input_first_dimension token_input_middle_dimension token_input_last_dimension
  q_projection_active k_projection_active v_projection_active qkv_quantization qkv_stream_to_attention
  qk_command_issue qk_command_accepted qk_operation_in_flight qk_result_waiting
  score_packet_valid_but_stalled score_fifo_empty score_fifo_occupancy_1 score_fifo_near_full
  score_fifo_producer_stalled score_fifo_consumer_stalled first_score_accepted running_max_update
  exp_issue exp_in_flight exp_result_waiting exp_sum_update score_replay_start score_replay_middle
  reciprocal_issue reciprocal_in_flight normalization normalized_probability_waiting
  probability_fifo_occupancy_1 probability_fifo_near_full probability_producer_stalled
  probability_consumer_stalled inner_drain mode_switch_request mode_switch_wait outer_mode_entry
  sv_command_issue sv_command_accepted sv_operation_in_flight sv_result_waiting outer_drain
  first_head_active first_head_completion head_boundary middle_head_active middle_head_score_fifo_nonempty
  middle_head_probability_fifo_nonempty final_head_active all_head_completion_before_commit
  atomic_kv_commit_cycle head_concat w_o_projection mha_output_stall residual1 rmsnorm2 ffn_w1
  relu_activation_quantization ffn_w2 residual2 final_output_stall layer_done_stall
)

compile_one() {
  local d_head=$1
  local seq_len=$2
  local name="reset_d${d_head}_s${seq_len}"
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    +define+HW_H9_D_HEAD="$d_head" \
    +define+HW_H9_SEQ_LEN="$seq_len" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${COMMON_RTL[@]}" \
    "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_reset_matrix.sv" \
    -top tb_h9_reset_matrix \
    -l "$compile_log")
  return $?
}

failures=0
compile_one 8 8
code=$?
if [ "$code" -ne 0 ]; then
  echo "D8 reset matrix compile failed, exit_code=$code" >> "$SUMMARY"
  grep -E "(Error-|Error:|Fatal:|Syntax error)" "$BUILD_DIR/reset_d8_s8_compile.log" | head -40 >> "$SUMMARY" || true
  cat "$SUMMARY"
  exit "$code"
fi
compile_one 16 16
code=$?
if [ "$code" -ne 0 ]; then
  echo "D16 reset matrix compile failed, exit_code=$code" >> "$SUMMARY"
  failures=$((failures + 1))
fi
compile_one 64 16
code=$?
if [ "$code" -ne 0 ]; then
  echo "D64 reset matrix compile failed, exit_code=$code" >> "$SUMMARY"
  failures=$((failures + 1))
fi

for index in "${!RESET_CASES[@]}"; do
  case_name=${RESET_CASES[$index]}
  d_head=8
  seq_len=8
  if [[ "$case_name" == *d16* ]]; then
    d_head=16
    seq_len=16
  fi
  if [ "$case_name" = "ffn_w2" ] || [ "$case_name" = "d_head_64" ]; then
    d_head=64
    seq_len=16
  fi
  simv="$BUILD_DIR/reset_d${d_head}_s${seq_len}_simv"
  log="$BUILD_DIR/reset_${index}_${case_name}.log"
  timeout 300s "$simv" +RESET_CASE="$case_name" -l "$log" >/dev/null 2>&1
  run_code=$?
  if [ "$run_code" -eq 0 ] && grep -q "HW_H9_RESET_MATRIX_PASS" "$log"; then
    result=PASS
  else
    result=FAIL
    failures=$((failures + 1))
  fi
  echo "| $((index + 1)). $case_name | H1/D${d_head}, seq${seq_len} | tb_h9_reset_matrix | $case_name | reset clears state and clean recovery runs twice | $result | build/hw_h9_reset_matrix/$(basename "$log") |" >> "$MATRIX"
done

pass_count=$(grep -c "| .* | .* | .* | .* | .* | PASS |" "$MATRIX" || true)
{
  echo "reset_injection_points=${#RESET_CASES[@]}"
  echo "reset_pass_count=$pass_count"
  echo "reset_failures=$failures"
  echo "matrix=reports/hw_h9/reset_execution_matrix.md"
  if [ "$failures" -eq 0 ]; then
    echo "result=PASS"
  else
    echo "result=FAIL"
  fi
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
