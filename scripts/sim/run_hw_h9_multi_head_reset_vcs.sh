#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_multi_head_reset"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/multi_head_reset_results.md"
MATRIX="$REPORT_DIR/multi_head_reset_matrix.md"

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
  echo "# Hardware Stage H9 Multi-Head Reset Results"
  echo
  echo "Scope: independent reset injection on the real multi_head_generation_engine hierarchy."
  echo "Proxy or Independent: all listed acceptance rows must be Independent."
  echo
} > "$SUMMARY"
{
  echo "# Hardware Stage H9 Multi-Head Reset Matrix"
  echo
  echo "| Injection | Real DUT | Real State/Handshake | Config | Proxy or Independent | Recovery | Result | Log |"
  echo "|---|---|---|---|---|---|---|---|"
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
  local name="mh_reset_h${n_head}_d${d_head}_s${max_seq_len}"
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
    "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_multi_head_reset_matrix.sv" \
    -top tb_h9_multi_head_reset_matrix \
    -l "$compile_log")
}

run_case() {
  local index=$1
  local case_name=$2
  local n_head=$3
  local d_head=$4
  local max_seq_len=$5
  local real_condition=$6
  local name="mh_reset_h${n_head}_d${d_head}_s${max_seq_len}"
  local simv="$BUILD_DIR/${name}_simv"
  local log="$BUILD_DIR/reset_${index}_${case_name}_h${n_head}_d${d_head}.log"
  compile_one "$n_head" "$d_head" "$max_seq_len"
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "compile_failed case=$case_name config=H${n_head}/D${d_head} exit_code=$compile_code" >> "$SUMMARY"
    echo "| $case_name | multi_head_generation_engine | $real_condition | H${n_head}/D${d_head}, seq${max_seq_len} | Independent | not run | FAIL | build/hw_h9_multi_head_reset/${name}_compile.log |" >> "$MATRIX"
    failures=$((failures + 1))
    return
  fi
  timeout 900s "$simv" +RESET_CASE="$case_name" -l "$log" >/dev/null 2>&1
  local run_code=$?
  if [ "$run_code" -eq 0 ] && grep -q "HW_H9_MULTI_HEAD_RESET_PASS" "$log"; then
    result=PASS
  else
    result=FAIL
    failures=$((failures + 1))
  fi
  echo "| $case_name | multi_head_generation_engine | $real_condition | H${n_head}/D${d_head}, seq${max_seq_len} | Independent | two clean tokens, one commit/output/done each | $result | build/hw_h9_multi_head_reset/$(basename "$log") |" >> "$MATRIX"
}

failures=0
index=0

CASES=(
  "first_head_qk_active|2|8|8|active_head_index=0 and child datapath PH_QK/QK active"
  "first_head_score_fifo_nonempty|2|8|8|active_head_index=0 and score_fifo_occupancy>0"
  "first_head_sfu_active|2|8|8|active_head_index=0 and SFU reduction/normalization busy"
  "first_head_probability_fifo_nonempty|2|8|8|active_head_index=0 and probability_fifo_occupancy>0"
  "first_head_sv_active|2|8|8|active_head_index=0 and child datapath PH_OUTER/sV active"
  "first_head_result_waiting|2|8|8|active_head_index=0 and child output_valid is stalled"
  "first_head_done_next_not_started|2|8|8|controller ST_HEAD_SWITCH before next head"
  "head_boundary|2|8|8|controller ST_HEAD_SWITCH real boundary"
  "middle_head_qk_active|4|8|8|active_head_index neither first nor final and child QK active"
  "middle_head_sfu_active|4|8|8|active_head_index neither first nor final and SFU active"
  "middle_head_probability_fifo_nonempty|4|8|8|active_head_index neither first nor final and probability FIFO nonempty"
  "middle_head_sv_active|4|8|8|active_head_index neither first nor final and sV active"
  "final_head_active|2|16|8|active_head_index=final and controller ST_ATTENTION_RUN"
  "all_heads_computed_before_commit|2|16|8|all head_done_seen bits set before commit"
  "atomic_kv_commit_cycle|2|16|8|cache_commit_valid && cache_commit_ready"
  "commit_after_before_output|2|16|8|controller ST_COMMIT_CURRENT_TOKEN after commit before top output"
  "multi_head_output_stalled|2|8|8|output_valid && !output_ready"
  "multi_head_done_stalled|2|8|8|done_valid && !done_ready"
  "first_head_qk_active|1|64|8|D64 compatibility: child datapath PH_QK/QK active"
  "first_head_probability_fifo_nonempty|1|64|8|D64 compatibility: probability FIFO nonempty"
)

for entry in "${CASES[@]}"; do
  IFS='|' read -r case_name n_head d_head max_seq real_condition <<< "$entry"
  index=$((index + 1))
  run_case "$index" "$case_name" "$n_head" "$d_head" "$max_seq" "$real_condition"
done

pass_count=$(grep -c "| PASS |" "$MATRIX" || true)
{
  echo
  echo "independent_reset_rows=$index"
  echo "pass_count=$pass_count"
  echo "failures=$failures"
  echo "matrix=reports/hw_h9/multi_head_reset_matrix.md"
  if [ "$failures" -eq 0 ]; then
    echo "result=PASS"
  else
    echo "result=FAIL"
  fi
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
