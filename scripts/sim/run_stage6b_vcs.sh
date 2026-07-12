#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage6b_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/stage_06"
SUMMARY="$REPORT_DIR/phase_6b_vcs_rtl_sim.txt"
VECTOR_DIR="$BUILD_DIR/vectors"

mkdir -p "$BUILD_DIR" "$REPORT_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$REPORT_DIR" "$VECTOR_DIR"

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
  echo "Stage 6B VCS RTL simulation"
  echo "Build dir: build/stage6b_rtl_sim"
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
  echo
} > "$SUMMARY"

PYTHON_BIN=python3
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage6b_vectors.py" "$VECTOR_DIR" > "$BUILD_DIR/vector_gen.log" 2>&1
vector_code=$?
if [ "$vector_code" -ne 0 ]; then
  {
    echo "vector_gen_exit_code=$vector_code"
    echo "result=FAIL"
    cat "$BUILD_DIR/vector_gen.log"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$vector_code"
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
)

COMMON_RTL=(
  "$ROOT_DIR/rtl/common/stream_reg.sv"
)

ARITH_RTL=(
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_add_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_to_fp16.sv"
)

PE_RTL=(
  "$ROOT_DIR/rtl/pe/lane_mask_generator.sv"
  "$ROOT_DIR/rtl/pe/accumulator_bank.sv"
  "$ROOT_DIR/rtl/pe/pe_perf_counter.sv"
  "$ROOT_DIR/rtl/pe/pe_lane.sv"
  "$ROOT_DIR/rtl/pe/fp32_reduction_tree.sv"
  "$ROOT_DIR/rtl/pe/reconfigurable_pe_core.sv"
)

PROJ_RTL=(
  "$ROOT_DIR/rtl/projection/projection_input_buffer.sv"
  "$ROOT_DIR/rtl/projection/projection_weight_buffer.sv"
  "$ROOT_DIR/rtl/projection/shared_gemv_projection_core.sv"
  "$ROOT_DIR/rtl/projection/projection_controller.sv"
)

compile_and_run() {
  local name=$1
  local top=$2
  local vector_plusarg=$3
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"
  shift 3
  local rtl_files=("$@")

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${rtl_files[@]}" \
    "$ROOT_DIR/tb/rtl/stage6/${top}.sv" \
    -top "$top" \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    return "$compile_code"
  fi

  timeout 300s "$simv" $vector_plusarg -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE6B_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code" >> "$SUMMARY"
    return 0
  fi
  echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
  return 1
}

failures=0

compile_and_run fp32_to_fp16 tb_fp32_to_fp16 \
  "+FP32_TO_FP16_VECTOR_FILE=$VECTOR_DIR/stage6b_fp32_to_fp16.mem" \
  "${COMMON_RTL[@]}" "$ROOT_DIR/rtl/arithmetic/fp32_to_fp16.sv" || failures=$((failures + 1))

compile_and_run shared_gemv_projection_core tb_shared_gemv_projection_core \
  "+GEMV_VECTOR_FILE=$VECTOR_DIR/stage6b_gemv.mem" \
  "${DW_FILES[@]}" "${COMMON_RTL[@]}" "${ARITH_RTL[@]}" "${PE_RTL[@]}" "${PROJ_RTL[@]}" || failures=$((failures + 1))

{
  echo
  echo "Run markers:"
  for log in "$BUILD_DIR"/*_run.log; do
    grep -E "(STAGE6B_|CHECK_FAIL)" "$log" || true
  done
  echo "result=$([ "$failures" -eq 0 ] && echo PASS || echo FAIL)"
  echo "Full logs: build/stage6b_rtl_sim/*_compile.log and *_run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
exit "$failures"
