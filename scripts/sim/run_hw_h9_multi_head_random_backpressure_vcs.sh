#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_multi_head_random_backpressure"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/multi_head_random_backpressure_results.md"
MATRIX="$REPORT_DIR/multi_head_random_backpressure_matrix.md"

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
  echo "# Hardware Stage H9 Multi-Head Random Backpressure Results"
  echo
  echo "Scope: broad fixed-seed random stress on the real multi_head_generation_engine hierarchy."
  echo "Watchdog formula in testbench: N_HEAD * token_count * (D_HEAD + token_count) * 250 + 50000."
  echo "Endpoint mask: token_valid_gap, multi_head_output_ready, multi_head_done_ready, real_head_boundary_pressure, real_commit_near_pressure."
  echo
} > "$SUMMARY"
{
  echo "# Hardware Stage H9 Multi-Head Random Backpressure Matrix"
  echo
  echo "| Seed | DUT | Config | Endpoint Mask | Stall Pattern | Cycles | Watchdog | Result | Log |"
  echo "|---:|---|---|---|---|---:|---:|---|---|"
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
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_interleaved_attention_datapath.sv"
  "$ROOT_DIR/rtl/attention/attention_score_scaler.sv"
  "$ROOT_DIR/rtl/attention/softmax_reduction.sv"
  "$ROOT_DIR/rtl/attention/softmax_normalization.sv"
  "$ROOT_DIR/rtl/memory/sram_2p_wrapper.sv"
  "$ROOT_DIR/rtl/attention/score_buffer.sv"
  "$ROOT_DIR/rtl/attention/paper/paper_attention_adapter.sv"
  "$ROOT_DIR/rtl/attention/single_head_attention_controller.sv"
  "$ROOT_DIR/rtl/attention/single_head_attention.sv"
)

STAGE5_RTL=(
  "$ROOT_DIR/rtl/cache/multi_head_kv_cache_manager.sv"
  "$ROOT_DIR/rtl/cache/multi_head_generation_controller.sv"
  "$ROOT_DIR/rtl/attention/multi_head_generation_engine.sv"
)

compile_one() {
  local n_head=$1
  local d_head=$2
  local max_seq_len=$3
  local name="mh_random_h${n_head}_d${d_head}_s${max_seq_len}"
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  if [ -x "$simv" ]; then
    return 0
  fi
  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    +define+STAGE5_N_HEAD="$n_head" \
    +define+STAGE5_D_HEAD="$d_head" \
    +define+STAGE5_MAX_SEQ_LEN="$max_seq_len" \
    +define+STAGE5_ATTENTION_PE_ARCH=1 \
    +define+STAGE5_ATTENTION_SCHEDULE=1 \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${COMMON_RTL[@]}" \
    "${STAGE5_RTL[@]}" \
    "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_multi_head_random_backpressure.sv" \
    -top tb_h9_multi_head_random_backpressure \
    -l "$compile_log")
}

run_one() {
  local seed=$1
  local n_head=$2
  local d_head=$3
  local max_seq_len=$4
  local token_count=$5
  local pattern=$6
  local endpoint_mask="token_valid_gap,multi_head_output_ready,multi_head_done_ready,real_head_boundary_pressure,real_commit_near_pressure"
  local name="mh_random_h${n_head}_d${d_head}_s${max_seq_len}"
  local simv="$BUILD_DIR/${name}_simv"
  local log="$BUILD_DIR/random_seed_${seed}_h${n_head}_d${d_head}_seq${token_count}.log"
  compile_one "$n_head" "$d_head" "$max_seq_len"
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "compile_failed seed=$seed config=H${n_head}/D${d_head}/seq${token_count} exit_code=$compile_code" >> "$SUMMARY"
    echo "| $seed | multi_head_generation_engine | H${n_head}/D${d_head}, seq${token_count} | $endpoint_mask | pattern=$pattern | 0 | 0 | FAIL | build/hw_h9_multi_head_random_backpressure/${name}_compile.log |" >> "$MATRIX"
    failures=$((failures + 1))
    return
  fi
  timeout 1200s "$simv" +SEED="$seed" +TOKEN_COUNT="$token_count" +PATTERN="$pattern" -l "$log" >/dev/null 2>&1
  local run_code=$?
  local pass_line
  pass_line=$(grep "HW_H9_MULTI_HEAD_RANDOM_PASS" "$log" | tail -1 || true)
  if [ "$run_code" -eq 0 ] && [ -n "$pass_line" ]; then
    result=PASS
    cycles=$(echo "$pass_line" | sed -n 's/.* cycles=\([0-9][0-9]*\).*/\1/p')
    watchdog=$(echo "$pass_line" | sed -n 's/.* watchdog=\([0-9][0-9]*\).*/\1/p')
    echo "$pass_line" >> "$SUMMARY"
  else
    result=FAIL
    cycles=0
    watchdog=$((n_head * token_count * (d_head + token_count) * 250 + 50000))
    failures=$((failures + 1))
  fi
  echo "| $seed | multi_head_generation_engine | H${n_head}/D${d_head}, seq${token_count} | $endpoint_mask | pattern=$pattern | ${cycles:-0} | ${watchdog:-0} | $result | build/hw_h9_multi_head_random_backpressure/$(basename "$log") |" >> "$MATRIX"
}

failures=0
RUNS=(
  "101|2|8|8|2|1"
  "211|2|8|8|2|2"
  "307|2|8|8|2|3"
  "401|2|8|8|2|4"
  "503|2|8|8|8|5"
  "601|2|8|8|8|2"
  "701|2|8|8|8|3"
  "809|2|8|8|8|4"
  "907|2|8|16|16|5"
  "1009|2|8|16|16|2"
  "1103|2|8|16|16|3"
  "1201|2|8|16|16|4"
  "1301|4|8|8|8|5"
  "1409|4|8|8|8|2"
  "1511|4|8|8|8|3"
  "1601|4|8|8|8|4"
  "1709|2|16|8|8|5"
  "1801|2|16|8|8|2"
  "1907|2|16|8|8|3"
  "2003|2|16|8|8|4"
  "2111|1|64|8|8|5"
  "2203|1|64|8|8|2"
  "2309|1|64|8|8|3"
  "2411|1|64|8|8|4"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r seed n_head d_head max_seq token_count pattern <<< "$entry"
  run_one "$seed" "$n_head" "$d_head" "$max_seq" "$token_count" "$pattern"
done

pass_count=$(grep -c "| PASS |" "$MATRIX" || true)
{
  echo
  echo "run_count=${#RUNS[@]}"
  echo "pass_count=$pass_count"
  echo "failures=$failures"
  echo "matrix=reports/hw_h9/multi_head_random_backpressure_matrix.md"
  if [ "$failures" -eq 0 ]; then
    echo "result=PASS"
  else
    echo "result=FAIL"
  fi
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
