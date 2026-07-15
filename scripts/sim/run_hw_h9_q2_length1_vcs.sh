#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_q2_length1"
REPORT_DIR="$ROOT_DIR/reports/hw_h9_numeric_repair"
SUMMARY="$REPORT_DIR/q2_length1_vcs.txt"
M3_ARTIFACT_ROOT=${M3_ARTIFACT_ROOT:-/workspace/VEDA_artifacts/ml_m3}
VECTOR_FILE="$M3_ARTIFACT_ROOT/vectors/len_1/case_len_1.mem"

mkdir -p "$BUILD_DIR" "$REPORT_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$REPORT_DIR"

find_dw_sim_dir() {
  if [ -n "${DW_SIM_DIR:-}" ] && [ -d "$DW_SIM_DIR" ]; then
    echo "$DW_SIM_DIR"
    return 0
  fi
  for candidate in /usr/synopsys/*/dw/sim_ver /usr/synopsys/*/*/dw/sim_ver; do
    if [ -d "$candidate" ] && [ -f "$candidate/DW_fp_add.v" ] && [ -f "$candidate/DW_fp_exp.v" ]; then
      echo "$candidate"
      return 0
    fi
  done
  for candidate in /usr/synopsys/*/dw/sim_ver /usr/synopsys/*/*/dw/sim_ver; do
    if [ -d "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

DW_SIM_DIR_DETECTED=$(find_dw_sim_dir || true)

{
  echo "HW-H9-N1 Q2 length1 transformer_layer RTL simulation"
  echo "Build dir: build/hw_h9_q2_length1"
  echo "Artifact root: $M3_ARTIFACT_ROOT"
  echo "Vector file: $VECTOR_FILE"
  if command -v vcs >/dev/null 2>&1; then
    echo "vcs: FOUND"
    vcs -ID 2>&1 | sed -n '1p'
  else
    echo "vcs: NOT FOUND"
    echo "result=FAIL"
    exit 10
  fi
  if [ -n "$DW_SIM_DIR_DETECTED" ]; then
    echo "DesignWare sim dir: detected"
  else
    echo "DesignWare sim dir: NOT FOUND"
    echo "result=FAIL"
    exit 11
  fi
  if [ -r "$VECTOR_FILE" ]; then
    echo "M3 vector file: readable"
  else
    echo "M3 vector file: NOT READABLE"
    echo "result=FAIL"
    exit 12
  fi
  echo
} > "$SUMMARY"

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
  "$ROOT_DIR/rtl/memory/sram_2p_wrapper.sv"
)

ARITH_RTL=(
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_add_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_exp_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_recip_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_sqrt_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_to_fp16.sv"
)

PE_RTL=(
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
  "$ROOT_DIR/rtl/attention/paper/paper_attention_adapter.sv"
  "$ROOT_DIR/rtl/attention/paper/interleaved/paper_interleaved_attention_datapath.sv"
)

ATTN_RTL=(
  "$ROOT_DIR/rtl/attention/attention_score_scaler.sv"
  "$ROOT_DIR/rtl/attention/score_buffer.sv"
  "$ROOT_DIR/rtl/attention/softmax_reduction.sv"
  "$ROOT_DIR/rtl/attention/softmax_normalization.sv"
  "$ROOT_DIR/rtl/attention/single_head_attention_controller.sv"
  "$ROOT_DIR/rtl/attention/single_head_attention.sv"
  "$ROOT_DIR/rtl/cache/multi_head_kv_cache_manager.sv"
  "$ROOT_DIR/rtl/cache/multi_head_generation_controller.sv"
  "$ROOT_DIR/rtl/attention/multi_head_generation_engine.sv"
)

PROJ_RTL=(
  "$ROOT_DIR/rtl/projection/projection_input_buffer.sv"
  "$ROOT_DIR/rtl/projection/projection_weight_buffer.sv"
  "$ROOT_DIR/rtl/projection/shared_gemv_projection_core.sv"
  "$ROOT_DIR/rtl/projection/projection_controller.sv"
  "$ROOT_DIR/rtl/projection/qkv_staging_buffer.sv"
  "$ROOT_DIR/rtl/projection/concat_fp16_buffer.sv"
  "$ROOT_DIR/rtl/projection/head_concat_quantizer.sv"
  "$ROOT_DIR/rtl/projection/output_projection_controller.sv"
  "$ROOT_DIR/rtl/attention/projection_integrated_mha.sv"
)

TRANSFORMER_RTL=(
  "$ROOT_DIR/rtl/transformer/rmsnorm_engine.sv"
  "$ROOT_DIR/rtl/transformer/residual_add_engine.sv"
  "$ROOT_DIR/rtl/transformer/ffn_engine.sv"
  "$ROOT_DIR/rtl/transformer/transformer_layer.sv"
)

compile_schedule() {
  local schedule=$1
  local name=$2
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +define+STAGE7_N_HEAD=8 \
    +define+STAGE7_D_HEAD=8 \
    +define+STAGE7_MAX_SEQ_LEN=128 \
    +define+STAGE7_MAX_WEIGHT_LINES=60000 \
    +define+STAGE7_MAX_TOKENS=2 \
    +define+STAGE7_ATTENTION_PE_ARCH=1 \
    +define+STAGE7_ATTENTION_SCHEDULE="$schedule" \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" "${COMMON_RTL[@]}" "${ARITH_RTL[@]}" "${PE_RTL[@]}" "${ATTN_RTL[@]}" "${PROJ_RTL[@]}" "${TRANSFORMER_RTL[@]}" \
    "$ROOT_DIR/tb/rtl/stage7/tb_stage7d_transformer_layer.sv" \
    -top tb_stage7d_transformer_layer \
    -l "$compile_log") >/dev/null 2>&1
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    return "$compile_code"
  fi
  echo "$simv"
  return 0
}

run_case() {
  local simv=$1
  local case_name=$2
  local stall_mode=$3
  local run_log="$BUILD_DIR/${case_name}_run.log"

  timeout 3600s "$simv" \
    +STAGE7D_VECTOR_FILE="$VECTOR_FILE" \
    +STAGE7D_OUTPUT_STALL="$stall_mode" \
    -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE7D_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$case_name result=PASS run_exit_code=$run_code" >> "$SUMMARY"
    grep -E "(STAGE7D_LAYER_TOKEN_DONE|STAGE7D_TRANSFORMER_LAYER_PASS)" "$run_log" >> "$SUMMARY" || true
    return 0
  fi
  echo "$case_name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
  grep -E "(STAGE7D_|CHECK_FAIL|Fatal:|Error:|assert.*failed)" "$run_log" | head -80 >> "$SUMMARY" || true
  return 1
}

failures=0

simv_staged=$(compile_schedule 0 h8_staged)
compile_code=$?
if [ "$compile_code" -ne 0 ]; then
  failures=$((failures + 1))
else
  run_case "$simv_staged" h8_staged_no_stall 0 || failures=$((failures + 1))
  run_case "$simv_staged" h8_staged_output_stall 1 || failures=$((failures + 1))
fi

simv_interleaved=$(compile_schedule 1 h9_interleaved)
compile_code=$?
if [ "$compile_code" -ne 0 ]; then
  failures=$((failures + 1))
else
  run_case "$simv_interleaved" h9_interleaved_no_stall 0 || failures=$((failures + 1))
  run_case "$simv_interleaved" h9_interleaved_output_stall 1 || failures=$((failures + 1))
fi

{
  echo
  echo "Run markers:"
  for log in "$BUILD_DIR"/*_run.log; do
    grep -E "(STAGE7D_LAYER_TOKEN_DONE|STAGE7D_TRANSFORMER_LAYER_PASS|CHECK_FAIL)" "$log" || true
  done
  echo "result=$([ "$failures" -eq 0 ] && echo PASS || echo FAIL)"
  echo "Full logs: build/hw_h9_q2_length1/*_compile.log and *_run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
exit "$failures"
