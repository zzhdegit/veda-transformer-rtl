#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_random_backpressure"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/random_backpressure_results.md"
MATRIX="$REPORT_DIR/random_backpressure_matrix.md"

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
  echo "# Hardware Stage H9 Random Backpressure Results"
  echo
  echo "Watchdog formula: calibrated_cycles + load_ops*20 + seq_len*80 + output_tiles*80 + 2000."
  echo
} > "$SUMMARY"
{
  echo "# Hardware Stage H9 Random Backpressure Matrix"
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

compile_one() {
  local d_head=$1
  local seq_len=$2
  local name="random_d${d_head}_s${seq_len}"
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  if [ -x "$simv" ]; then
    return 0
  fi
  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    +define+HW_H9_D_HEAD="$d_head" \
    +define+HW_H9_SEQ_LEN="$seq_len" \
    +define+HW_H9_TOKEN_COUNT=2 \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${COMMON_RTL[@]}" \
    "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_random_backpressure.sv" \
    -top tb_h9_random_backpressure \
    -l "$compile_log")
  return $?
}

SEEDS=(101 211 307 401 503 601 701 809 907 1009 1103 1201 1301 1409 1511 1601 1709 1801 1907 2003)
D_HEADS=(8 8 8 8 16 16 64 64 8 16 64 8 16 64 8 16 64 8 16 64)
SEQS=(1 8 16 32 16 32 16 32 8 8 8 16 16 16 32 32 32 1 2 8)
PATTERNS=(0 1 2 3 4 5 3 4 5 2 1 4 5 3 2 1 5 0 3 4)

failures=0
for index in "${!SEEDS[@]}"; do
  seed=${SEEDS[$index]}
  d_head=${D_HEADS[$index]}
  seq_len=${SEQS[$index]}
  pattern=${PATTERNS[$index]}
  compile_one "$d_head" "$seq_len"
  code=$?
  if [ "$code" -ne 0 ]; then
    echo "compile failed seed=$seed d_head=$d_head seq=$seq_len exit_code=$code" >> "$SUMMARY"
    failures=$((failures + 1))
    echo "| $((index + 1)) | H1/D${d_head}, seq${seq_len} | tb_h9_random_backpressure | seed=$seed pattern=$pattern | no deadlock, stable payload, bit-exact output | FAIL | build/hw_h9_random_backpressure/random_d${d_head}_s${seq_len}_compile.log |" >> "$MATRIX"
    continue
  fi
  simv="$BUILD_DIR/random_d${d_head}_s${seq_len}_simv"
  log="$BUILD_DIR/random_seed_${seed}_d${d_head}_s${seq_len}.log"
  timeout 600s "$simv" +SEED="$seed" +PATTERN="$pattern" -l "$log" >/dev/null 2>&1
  run_code=$?
  if [ "$run_code" -eq 0 ] && grep -q "HW_H9_RANDOM_BACKPRESSURE_PASS" "$log"; then
    result=PASS
  else
    result=FAIL
    failures=$((failures + 1))
  fi
  pass_line=$(grep "HW_H9_RANDOM_BACKPRESSURE_PASS" "$log" | tail -1 || true)
  echo "| $((index + 1)) | H1/D${d_head}, seq${seq_len} | tb_h9_random_backpressure | seed=$seed pattern=$pattern | no deadlock, stable payload, bit-exact output | $result | build/hw_h9_random_backpressure/$(basename "$log") |" >> "$MATRIX"
  if [ -n "$pass_line" ]; then
    echo "$pass_line" >> "$SUMMARY"
  fi
done

{
  echo
  echo "seed_count=${#SEEDS[@]}"
  echo "seeds=${SEEDS[*]}"
  echo "failures=$failures"
  echo "matrix=reports/hw_h9/random_backpressure_matrix.md"
  if [ "$failures" -eq 0 ]; then
    echo "result=PASS"
  else
    echo "result=FAIL"
  fi
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
