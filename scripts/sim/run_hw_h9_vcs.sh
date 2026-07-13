#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="$ROOT_DIR/build/hw_h9_rtl_sim"
REPORT_DIR="$ROOT_DIR/reports/hw_h9"
SUMMARY="$REPORT_DIR/rtl_sim.txt"
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
  echo "Hardware Stage H9 RTL simulation"
  echo "Build dir: build/hw_h9_rtl_sim"
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
  "$DW_SIM_DIR_DETECTED/DW_exp2.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_exp.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_div.v"
  "$DW_SIM_DIR_DETECTED/DW_fp_sqrt.v"
)

PYTHON_BIN=python3
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

mkdir -p "$VECTOR_DIR/stage5_seq8" "$VECTOR_DIR/stage5_seq32" "$VECTOR_DIR/stage7d"
"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage5_vectors.py" "$VECTOR_DIR/stage5_seq8" \
  --max-seq-len 8 --configs 1x8,2x8,4x8,2x16,1x64 > "$BUILD_DIR/stage5_seq8_vector_gen.log" 2>&1
vector_code=$?
if [ "$vector_code" -ne 0 ]; then
  {
    echo "stage5_seq8_vector_gen_exit_code=$vector_code"
    echo "result=FAIL"
    cat "$BUILD_DIR/stage5_seq8_vector_gen.log"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$vector_code"
fi
"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage5_vectors.py" "$VECTOR_DIR/stage5_seq32" \
  --max-seq-len 32 --configs 1x8 > "$BUILD_DIR/stage5_seq32_vector_gen.log" 2>&1
vector_code=$?
if [ "$vector_code" -ne 0 ]; then
  {
    echo "stage5_seq32_vector_gen_exit_code=$vector_code"
    echo "result=FAIL"
    cat "$BUILD_DIR/stage5_seq32_vector_gen.log"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$vector_code"
fi
"$PYTHON_BIN" "$ROOT_DIR/scripts/sim/gen_stage7d_vectors.py" "$VECTOR_DIR/stage7d" > "$BUILD_DIR/stage7d_vector_gen.log" 2>&1
vector_code=$?
if [ "$vector_code" -ne 0 ]; then
  {
    echo "stage7d_vector_gen_exit_code=$vector_code"
    echo "result=FAIL"
    cat "$BUILD_DIR/stage7d_vector_gen.log"
  } >> "$SUMMARY"
  cat "$SUMMARY"
  exit "$vector_code"
fi

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

LAYER_RTL=(
  "$ROOT_DIR/rtl/projection/projection_input_buffer.sv"
  "$ROOT_DIR/rtl/projection/projection_weight_buffer.sv"
  "$ROOT_DIR/rtl/projection/shared_gemv_projection_core.sv"
  "$ROOT_DIR/rtl/projection/projection_controller.sv"
  "$ROOT_DIR/rtl/projection/qkv_staging_buffer.sv"
  "$ROOT_DIR/rtl/projection/concat_fp16_buffer.sv"
  "$ROOT_DIR/rtl/projection/head_concat_quantizer.sv"
  "$ROOT_DIR/rtl/projection/output_projection_controller.sv"
  "$ROOT_DIR/rtl/attention/projection_integrated_mha.sv"
  "$ROOT_DIR/rtl/transformer/rmsnorm_engine.sv"
  "$ROOT_DIR/rtl/transformer/residual_add_engine.sv"
  "$ROOT_DIR/rtl/transformer/ffn_engine.sv"
  "$ROOT_DIR/rtl/transformer/transformer_layer.sv"
)

failures=0

compile_and_run() {
  local name=$1
  local top=$2
  local tb_file=$3
  shift 3
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    "$@" \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${COMMON_RTL[@]}" \
    "$tb_file" \
    -top "$top" \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    grep -E "(Error-|Error:|Fatal:|Syntax error|Parsing design file)" "$compile_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
    return
  fi

  timeout 300s "$simv" -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(HW_H9_.*_FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "HW_H9_.*_PASS" "$run_log" >> "$SUMMARY" || true
  else
    echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(HW_H9_.*_FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed)" "$run_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
  fi
}

compile_and_run_generation() {
  local name=$1
  local n_head=$2
  local d_head=$3
  local max_seq_len=$4
  local vector_file=$5
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

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
    "$ROOT_DIR/tb/rtl/stage5/tb_multi_head_generation_engine.sv" \
    -top tb_multi_head_generation_engine \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    grep -E "(Error-|Error:|Fatal:|Syntax error|Parsing design file)" "$compile_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
    return
  fi

  timeout 1800s "$simv" \
    +GENERATION_VECTOR_FILE="$vector_file" \
    -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE5_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE5_SHARED_MULTIHEAD_PASS|STAGE5_GENERATION_PERF)" "$run_log" | tail -12 >> "$SUMMARY" || true
  else
    echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE5_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed)" "$run_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
  fi
}

compile_and_run_layer() {
  local name=$1
  local n_head=$2
  local d_head=$3
  local vector_file=$4
  local simv="$BUILD_DIR/${name}_simv"
  local compile_log="$BUILD_DIR/${name}_compile.log"
  local run_log="$BUILD_DIR/${name}_run.log"

  (cd "$BUILD_DIR" && vcs -full64 -sverilog -debug_access+pp -assert svaext -timescale=1ns/1ps \
    +incdir+"$DW_SIM_DIR_DETECTED" \
    +define+STAGE7_N_HEAD="$n_head" \
    +define+STAGE7_D_HEAD="$d_head" \
    +define+STAGE7_ATTENTION_PE_ARCH=1 \
    +define+STAGE7_ATTENTION_SCHEDULE=1 \
    -Mdir="$BUILD_DIR/${name}_csrc" \
    -o "$simv" \
    "${DW_FILES[@]}" \
    "${COMMON_RTL[@]}" \
    "${STAGE5_RTL[@]}" \
    "${LAYER_RTL[@]}" \
    "$ROOT_DIR/tb/rtl/stage7/tb_stage7d_transformer_layer.sv" \
    -top tb_stage7d_transformer_layer \
    -l "$compile_log")
  local compile_code=$?
  if [ "$compile_code" -ne 0 ]; then
    echo "$name compile_exit_code=$compile_code" >> "$SUMMARY"
    grep -E "(Error-|Error:|Fatal:|Syntax error|Parsing design file)" "$compile_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
    return
  fi

  timeout 1800s "$simv" \
    +STAGE7D_VECTOR_FILE="$vector_file" \
    -l "$run_log"
  local run_code=$?
  local log_errors=0
  if grep -E "(STAGE7D_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed|unsupported .* assertion failed)" "$run_log" >/dev/null 2>&1; then
    log_errors=1
  fi
  if [ "$run_code" -eq 0 ] && [ "$log_errors" -eq 0 ]; then
    echo "$name result=PASS run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "STAGE7D_TRANSFORMER_LAYER_PASS" "$run_log" >> "$SUMMARY" || true
  else
    echo "$name result=FAIL run_exit_code=$run_code assertion_markers=$log_errors" >> "$SUMMARY"
    grep -E "(STAGE7D_.*FAIL|CHECK_FAIL|Fatal:|Error:|assert.*failed)" "$run_log" | head -40 >> "$SUMMARY" || true
    failures=$((failures + 1))
  fi
}

compile_and_run "score_buffer" "tb_h9_score_buffer" "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_score_buffer.sv"
compile_and_run "probability_fifo" "tb_h9_probability_fifo" "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_probability_fifo.sv"
compile_and_run "single_head_d8" "tb_h9_single_head" "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_single_head.sv" +define+HW_H9_D_HEAD=8
compile_and_run "single_head_d16" "tb_h9_single_head" "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_single_head.sv" +define+HW_H9_D_HEAD=16
compile_and_run "single_head_d64" "tb_h9_single_head" "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_single_head.sv" +define+HW_H9_D_HEAD=64

for d_head in 8 16 64; do
  for seq_len in 1 2 8 16 32 64; do
    compile_and_run "matched_ab_staged_d${d_head}_s${seq_len}" \
      "tb_h9_matched_ab_single_head" \
      "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_matched_ab_single_head.sv" \
      +define+HW_H9_D_HEAD=${d_head} \
      +define+HW_H9_SEQ_LEN=${seq_len} \
      +define+HW_H9_SCHEDULE=0
    compile_and_run "matched_ab_interleaved_d${d_head}_s${seq_len}" \
      "tb_h9_matched_ab_single_head" \
      "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_matched_ab_single_head.sv" \
      +define+HW_H9_D_HEAD=${d_head} \
      +define+HW_H9_SEQ_LEN=${seq_len} \
      +define+HW_H9_SCHEDULE=1 \
      +define+HW_H9_INTERLEAVED
  done
done

compile_and_run_generation "h9_multi_head_h1_d8_seq8" 1 8 8 "$VECTOR_DIR/stage5_seq8/stage5_generation_h1_d8.mem"
compile_and_run_generation "h9_multi_head_h2_d8_seq8" 2 8 8 "$VECTOR_DIR/stage5_seq8/stage5_generation_h2_d8.mem"
compile_and_run_generation "h9_multi_head_h4_d8_seq8" 4 8 8 "$VECTOR_DIR/stage5_seq8/stage5_generation_h4_d8.mem"
compile_and_run_generation "h9_multi_head_h2_d16_seq8" 2 16 8 "$VECTOR_DIR/stage5_seq8/stage5_generation_h2_d16.mem"
compile_and_run_generation "h9_multi_head_h1_d64_seq8" 1 64 8 "$VECTOR_DIR/stage5_seq8/stage5_generation_h1_d64.mem"
compile_and_run_generation "h9_sequence_cache_full_h1_d8_seq32" 1 8 32 "$VECTOR_DIR/stage5_seq32/stage5_generation_h1_d8.mem"

compile_and_run_layer "h9_layer_h1_d8" 1 8 "$VECTOR_DIR/stage7d/stage7d_h1_d8.mem"
compile_and_run_layer "h9_layer_h2_d8" 2 8 "$VECTOR_DIR/stage7d/stage7d_h2_d8.mem"
compile_and_run_layer "h9_layer_h2_d8_two_token" 2 8 "$VECTOR_DIR/stage7d/stage7d_h2_d8_two_token.mem"
compile_and_run_layer "h9_layer_h4_d8" 4 8 "$VECTOR_DIR/stage7d/stage7d_h4_d8.mem"
compile_and_run_layer "h9_layer_h2_d16" 2 16 "$VECTOR_DIR/stage7d/stage7d_h2_d16.mem"

for d_head in 8 16 64; do
  for seq_len in 16 32; do
    compile_and_run "matched_ab_staged_bp_d${d_head}_s${seq_len}" \
      "tb_h9_matched_ab_single_head" \
      "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_matched_ab_single_head.sv" \
      +define+HW_H9_D_HEAD=${d_head} \
      +define+HW_H9_SEQ_LEN=${seq_len} \
      +define+HW_H9_SCHEDULE=0 \
      +define+HW_H9_DETERMINISTIC_BP
    compile_and_run "matched_ab_interleaved_bp_d${d_head}_s${seq_len}" \
      "tb_h9_matched_ab_single_head" \
      "$ROOT_DIR/tb/rtl/hw_h9/tb_h9_matched_ab_single_head.sv" \
      +define+HW_H9_D_HEAD=${d_head} \
      +define+HW_H9_SEQ_LEN=${seq_len} \
      +define+HW_H9_SCHEDULE=1 \
      +define+HW_H9_INTERLEAVED \
      +define+HW_H9_DETERMINISTIC_BP
  done
done

if [ "$failures" -eq 0 ]; then
  echo "result=PASS" >> "$SUMMARY"
else
  echo "result=FAIL failures=$failures" >> "$SUMMARY"
fi

cat "$SUMMARY"
[ "$failures" -eq 0 ] && exit 0 || exit 1
