#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage1b_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/stage_01b"
SUMMARY="$REPORT_DIR/vcs_rtl_sim.txt"
VECTOR_FILE="$BUILD_DIR/fp32_mac_vectors.mem"

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
  echo "Stage 1B VCS RTL simulation"
  echo "Build dir: build/stage1b_rtl_sim"
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

"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage1b_vectors.py" "$VECTOR_FILE" > "$BUILD_DIR/vector_gen.log" 2>&1
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

compile_and_run() {
  local name=$1
  local top=$2
  local expect_fail=$3
  shift 3
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "$@" \
    -top "$top" \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    return "$compile_code"
  fi

  timeout 180s "$simv" +VECTOR_FILE="$VECTOR_FILE" +EXPECT_FUSED="${EXPECT_FUSED:-1}" -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE1B_.*FAIL|CHECK_FAIL|Fatal:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi

  if [ "$expect_fail" = "1" ]; then
    if [ "$run_code" -ne 0 ] || [ "$log_errors" -ne 0 ]; then
      echo "$name result=PASS_EXPECTED_FAILURE run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
      return 0
    fi
    echo "$name result=FAIL_EXPECTED_ASSERTION_NOT_SEEN run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    return 1
  fi

  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code" >> "$SUMMARY"
    return 0
  fi
  echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
  return 1
}

failures=0

compile_and_run fp16_to_fp32 tb_fp16_to_fp32 0 \
  "$ROOT_DIR/rtl/common/stream_reg.sv" \
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv" \
  "$ROOT_DIR/tb/rtl/stage1b/tb_fp16_to_fp32.sv" || failures=$((failures + 1))

compile_and_run fp16_to_fp32_invalid tb_fp16_to_fp32_invalid_assert 1 \
  "$ROOT_DIR/rtl/common/stream_reg.sv" \
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv" \
  "$ROOT_DIR/tb/rtl/stage1b/tb_fp16_to_fp32_invalid_assert.sv" || failures=$((failures + 1))

compile_and_run dw_fp_mac_semantics tb_dw_fp_mac_semantics 0 \
  "${DW_FILES[@]}" \
  "$ROOT_DIR/tb/rtl/stage1b/tb_dw_fp_mac_semantics.sv" || failures=$((failures + 1))

semantics_log="$BUILD_DIR/dw_fp_mac_semantics_run.log"
if grep -q "DW_FP_MAC_SEMANTICS_FUSED" "$semantics_log" 2>/dev/null; then
  EXPECT_FUSED=1
  MAC_SEMANTICS="fused"
elif grep -q "DW_FP_MAC_SEMANTICS_NON_FUSED" "$semantics_log" 2>/dev/null; then
  EXPECT_FUSED=0
  MAC_SEMANTICS="non_fused"
else
  EXPECT_FUSED=1
  MAC_SEMANTICS="unknown"
  failures=$((failures + 1))
fi
echo "dw_fp_mac_semantics=$MAC_SEMANTICS" >> "$SUMMARY"

RNE_RND=$(grep -E "DW_FP_MAC_RNE_RND=" "$semantics_log" 2>/dev/null | sed -E 's/.*DW_FP_MAC_RNE_RND=([0-9]+).*/\1/' | head -n 1)
if [ "$RNE_RND" = "4" ]; then
  echo "dw_fp_mac_rne_rnd=4" >> "$SUMMARY"
else
  echo "dw_fp_mac_rne_rnd=${RNE_RND:-NOT_FOUND}" >> "$SUMMARY"
  failures=$((failures + 1))
fi

compile_and_run fp32_mac_wrapper tb_fp32_mac_wrapper 0 \
  "${DW_FILES[@]}" \
  "$ROOT_DIR/rtl/common/stream_reg.sv" \
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv" \
  "$ROOT_DIR/tb/rtl/stage1b/tb_fp32_mac_wrapper.sv" || failures=$((failures + 1))

compile_and_run fp32_mac_invalid tb_fp32_mac_invalid_assert 1 \
  "${DW_FILES[@]}" \
  "$ROOT_DIR/rtl/common/stream_reg.sv" \
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv" \
  "$ROOT_DIR/tb/rtl/stage1b/tb_fp32_mac_invalid_assert.sv" || failures=$((failures + 1))

{
  echo
  echo "Run markers:"
  for log in "$BUILD_DIR"/*_run.log; do
    grep -E "(STAGE1B_|DW_FP_MAC_SEMANTICS_|DW_FP_MAC_RNE_RND=)" "$log" || true
  done
  echo "vector_count=$(wc -l < "$VECTOR_FILE")"
  echo "result=$([ "$failures" -eq 0 ] && echo PASS || echo FAIL)"
  echo "Full logs: build/stage1b_rtl_sim/*_compile.log and *_run.log"
} >> "$SUMMARY"

cat "$SUMMARY"
exit "$failures"
