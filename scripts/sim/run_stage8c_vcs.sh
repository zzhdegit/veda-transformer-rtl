#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage8c_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/stage_08"
SUMMARY="$REPORT_DIR/phase_8c_vcs_rtl_sim.txt"

mkdir -p "$BUILD_DIR" "$REPORT_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$REPORT_DIR"

find_dw_sim_dir() {
  if [ -n "${DW_SIM_DIR:-}" ] && [ -d "$DW_SIM_DIR" ]; then
    echo "$DW_SIM_DIR"
    return 0
  fi
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
  echo "Stage 8C independent paper array RTL simulation"
  echo "Build dir: build/stage8c_rtl_sim"
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
  "$ROOT_DIR/rtl/pe/paper/paper_pe_cell.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_l1_reduction.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_l2_reduction.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_pe_group.sv"
  "$ROOT_DIR/rtl/pe/paper/paper_array_8x8x2.sv"
)

compile_log="$BUILD_DIR/paper_array_compile.log"
run_log="$BUILD_DIR/paper_array_run.log"
simv="$BUILD_DIR/paper_array_simv"

(cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
  +incdir+"$DW_SIM_DIR_DETECTED" \
  -Mdir="$BUILD_DIR/paper_array_csrc" \
  -o "$simv" \
  "${DW_FILES[@]}" \
  "${RTL_FILES[@]}" \
  "$ROOT_DIR/tb/rtl/stage8/tb_paper_array_inner.sv" \
  -top tb_paper_array_inner \
  -l "$compile_log")
compile_code=$?
if [ "$compile_code" -ne 0 ]; then
  {
    echo "paper_array compile_exit_code=$compile_code"
    echo "result=FAIL"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$compile_code"
fi

timeout 240s "$simv" -l "$run_log"
run_code=$?
log_errors=0
if grep -E "(STAGE8C_.*FAIL|CHECK_FAIL|Fatal:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
  log_errors=1
fi

if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
  result=PASS
else
  result=FAIL
fi

{
  echo "paper_array result=$result run_exit_code=$run_code assertion_markers=$log_errors"
  echo
  echo "Run markers:"
  grep -E "(STAGE8C_|CHECK_FAIL)" "$run_log" || true
  echo "result=$result"
  echo "Full logs: build/stage8c_rtl_sim/*_compile.log and *_run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
[ "$result" = "PASS" ] && exit 0 || exit 1
