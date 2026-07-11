#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage1_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/stage_01"
SUMMARY="$REPORT_DIR/vcs_rtl_sim.txt"

mkdir -p "$BUILD_DIR" "$REPORT_DIR"

RTL_FILES=(
  "$ROOT_DIR/rtl/common/stream_reg.sv"
  "$ROOT_DIR/rtl/common/skid_buffer.sv"
  "$ROOT_DIR/rtl/memory/sync_fifo.sv"
  "$ROOT_DIR/rtl/memory/sram_1p_wrapper.sv"
  "$ROOT_DIR/rtl/memory/sram_2p_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/mul_unit.sv"
  "$ROOT_DIR/rtl/arithmetic/add_unit.sv"
  "$ROOT_DIR/rtl/arithmetic/mac_unit.sv"
  "$ROOT_DIR/rtl/arithmetic/compare_max.sv"
  "$ROOT_DIR/rtl/arithmetic/round_sat.sv"
)

TB_FILE="$ROOT_DIR/tb/rtl/stage1/tb_stage1_all.sv"

{
  echo "Stage 1 VCS RTL simulation"
  echo "Root: $ROOT_DIR"
  echo "Build dir: build/stage1_rtl_sim"
  if command -v vcs >/dev/null 2>&1; then
    echo "vcs: FOUND"
    vcs -ID 2>&1 | sed -n '1p'
  else
    echo "vcs: NOT FOUND"
    exit 10
  fi
  echo
} > "$SUMMARY"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 2

vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
  -Mdir="$BUILD_DIR/csrc" \
  -o "$BUILD_DIR/simv" \
  "${RTL_FILES[@]}" "$TB_FILE" \
  -top tb_stage1_all \
  -l "$BUILD_DIR/compile.log"
compile_code=$?

if [ "$compile_code" -ne 0 ]; then
  {
    echo "compile_exit_code=$compile_code"
    echo "result=FAIL"
  } >> "$SUMMARY"
  exit "$compile_code"
fi

timeout 120s "$BUILD_DIR/simv" -l "$BUILD_DIR/run.log"
run_code=$?

assert_errors=0
if grep -E "(^Error:|STAGE1_TB_FAIL|CHECK_FAIL|Fatal:|assert.*failed)" "$BUILD_DIR/run.log" >/dev/null 2>&1; then
  assert_errors=1
fi

if grep -q "STAGE1_RTL_SIM_PASS" "$BUILD_DIR/run.log" && [ "$run_code" -eq 0 ] && [ "$assert_errors" -eq 0 ]; then
  result=PASS
  exit_code=0
else
  result=FAIL
  exit_code=1
fi

{
  echo "compile_exit_code=$compile_code"
  echo "run_exit_code=$run_code"
  echo "assertion_or_tb_errors=$assert_errors"
  echo "pass_marker=$(grep -c STAGE1_RTL_SIM_PASS "$BUILD_DIR/run.log" || true)"
  echo "result=$result"
  echo
  echo "Run markers:"
  grep -E "^(TEST |STAGE1_RTL_SIM_PASS)" "$BUILD_DIR/run.log" || true
  echo "Full logs: build/stage1_rtl_sim/compile.log and build/stage1_rtl_sim/run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
exit "$exit_code"
