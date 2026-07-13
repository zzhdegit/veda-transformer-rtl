#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/stage8d_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/stage_08"
SUMMARY="$REPORT_DIR/phase_8d_vcs_rtl_sim.txt"
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
  echo "Stage 8D paper-array attention RTL simulation"
  echo "Build dir: build/stage8d_rtl_sim"
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

"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage3_vectors.py" "$VECTOR_DIR" > "$BUILD_DIR/vector_gen.log" 2>&1
vector_code=$?
if [ "$vector_code" -eq 0 ]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage5_vectors.py" "$VECTOR_DIR" >> "$BUILD_DIR/vector_gen.log" 2>&1
  vector_code=$?
fi
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
  "$DW_SIM_DIR_DETECTED/DW_exp2.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_exp.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_div.v"
)

RTL_FILES=(
  "$ROOT_DIR/rtl/common/stream_reg.sv"
  "$ROOT_DIR/rtl/memory/sram_2p_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp16_to_fp32.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_mac_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_add_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_exp_wrapper.sv"
  "$ROOT_DIR/rtl/arithmetic/fp32_recip_wrapper.sv"
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

failures=0

compile_and_run_attention() {
  local d_head=$1
  local vector_file=$2
  local name="paper_single_head_d${d_head}"
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +define+STAGE3_ATTENTION_PE_ARCH=1 \
    +define+STAGE3_D_HEAD="$d_head" \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${RTL_FILES[@]}" \
    "$ROOT_DIR/tb/rtl/stage3/tb_single_head_attention.sv" \
    -top tb_single_head_attention \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    failures=$((failures + 1))
    return
  fi

  timeout 240s "$simv" +ATTENTION_VECTOR_FILE="$VECTOR_DIR/$vector_file" -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE3_ATTENTION_TB_FAIL|CHECK_FAIL|Fatal:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "STAGE3_ATTENTION_PERF" "$run_log" >> "$SUMMARY" || true
  else
    echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE3_ATTENTION_TB_FAIL|CHECK_FAIL|Fatal:|assert.*failed)" "$run_log" | head -20 >> "$SUMMARY" || true
    failures=$((failures + 1))
  fi
}

compile_and_run_attention 8 stage3_attention_d8.mem
compile_and_run_attention 16 stage3_attention_d16.mem

compile_and_run_generation() {
  local n_head=$1
  local d_head=$2
  local vector_file=$3
  local name="paper_multi_head_h${n_head}_d${d_head}"
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +define+STAGE5_ATTENTION_PE_ARCH=1 \
    +define+STAGE5_N_HEAD="$n_head" \
    +define+STAGE5_D_HEAD="$d_head" \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${RTL_FILES[@]}" \
    "$ROOT_DIR/tb/rtl/stage5/tb_multi_head_generation_engine.sv" \
    -top tb_multi_head_generation_engine \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    failures=$((failures + 1))
    return
  fi

  timeout 300s "$simv" +GENERATION_VECTOR_FILE="$VECTOR_DIR/$vector_file" -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE5_GENERATION_TB_FAIL|CHECK_FAIL|Fatal:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE5_GENERATION_PERF|STAGE5_SHARED_MULTIHEAD_PASS)" "$run_log" >> "$SUMMARY" || true
  else
    echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE5_GENERATION_TB_FAIL|CHECK_FAIL|Fatal:|assert.*failed)" "$run_log" | head -20 >> "$SUMMARY" || true
    failures=$((failures + 1))
  fi
}

compile_and_run_generation 1 8 stage5_generation_h1_d8.mem
compile_and_run_generation 2 8 stage5_generation_h2_d8.mem
compile_and_run_generation 4 8 stage5_generation_h4_d8.mem
compile_and_run_generation 2 16 stage5_generation_h2_d16.mem

if [ "$failures" -eq 0 ]; then
  echo "result=PASS" >> "$SUMMARY"
else
  echo "result=FAIL failures=$failures" >> "$SUMMARY"
fi

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
