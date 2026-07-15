#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_numeric_repair"
REPORT_DIR="$ROOT_DIR/reports/hw_h9_numeric_repair"
SUMMARY="$REPORT_DIR/numeric_repair_vcs.txt"
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
    if [ -d "$candidate" ] && [ -f "$candidate/DW_fp_add.v" ]; then
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
  echo "HW-H9-N1 numeric repair RTL simulation"
  echo "Build dir: build/hw_h9_numeric_repair"
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

"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_hw_h9_numeric_repair_vectors.py" "$VECTOR_DIR" > "$BUILD_DIR/vector_gen.log" 2>&1
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

RTL_FILES=(
  "$ROOT_DIR/rtl/common/stream_reg.sv"
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_add_wrapper.sv"
  "$ROOT_DIR/rtl/pe/lane_mask_generator.sv"
  "$ROOT_DIR/rtl/pe/accumulator_bank.sv"
  "$ROOT_DIR/rtl/pe/pe_perf_counter.sv"
  "$ROOT_DIR/rtl/pe/pe_lane.sv"
  "$ROOT_DIR/rtl/pe/fp32_reduction_tree.sv"
  "$ROOT_DIR/rtl/pe/reconfigurable_pe_core.sv"
)

SIMV="$BUILD_DIR/hw_h9_numeric_repair_simv"
COMPILE_LOG="$BUILD_DIR/compile.log"
RUN_LOG="$BUILD_DIR/run.log"

(cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
  +incdir+"$DW_SIM_DIR_DETECTED" \
  -Mdir="$BUILD_DIR/csrc" \
  -o "$SIMV" \
  "${DW_FILES[@]}" "${RTL_FILES[@]}" \
  "$ROOT_DIR/tb/rtl/hw_h9_numeric/tb_hw_h9_numeric_repair.sv" \
  -top tb_hw_h9_numeric_repair \
  -l "$COMPILE_LOG")
compile_code=$?
if [ "$compile_code" -ne 0 ]; then
  {
    echo "compile_exit_code=$compile_code"
    echo "result=FAIL"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$compile_code"
fi

timeout 600s "$SIMV" \
  +ADD_VECTOR_FILE="$VECTOR_DIR/hw_h9_numeric_add.mem" \
  +REDUCTION_VECTOR_FILE="$VECTOR_DIR/hw_h9_numeric_reduction.mem" \
  +CORE_VECTOR_FILE="$VECTOR_DIR/hw_h9_numeric_core.mem" \
  -l "$RUN_LOG"
run_code=$?
log_errors=0
if grep -E "(HW_H9_NUMERIC_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$RUN_LOG" >/dev/null 2>&1; then
  log_errors=1
fi

{
  echo "compile_exit_code=$compile_code"
  echo "run_exit_code=$run_code"
  echo "assertion_markers=$log_errors"
  echo
  echo "Run markers:"
  grep -E "(HW_H9_NUMERIC_|CHECK_FAIL)" "$RUN_LOG" || true
  echo "Full logs: build/hw_h9_numeric_repair/compile.log and run.log"
} >> "$SUMMARY"

if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
  echo "result=PASS" >> "$SUMMARY"
  cat "$SUMMARY"
  exit 0
fi

echo "result=FAIL" >> "$SUMMARY"
cat "$SUMMARY"
exit 1
