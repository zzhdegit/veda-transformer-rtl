set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_08"
set build_dir "$root_dir/build/stage8d_dc"
file mkdir $report_dir
file mkdir $build_dir

if {![info exists ::env(DW_FOUNDATION_SLDB)] || $::env(DW_FOUNDATION_SLDB) eq ""} {
  puts "ERROR: DW_FOUNDATION_SLDB environment variable is required."
  exit 10
}

set dw_sldb $::env(DW_FOUNDATION_SLDB)
if {![file exists $dw_sldb]} {
  puts "ERROR: DW_FOUNDATION_SLDB does not name a readable file."
  exit 11
}

set synthetic_library [list $dw_sldb]
set link_library [concat "*" $synthetic_library]

set_app_var hdlin_enable_vpp true
define_design_lib WORK -path "$build_dir/WORK"
cd $build_dir

set rtl_files [list \
  "$root_dir/rtl/common/stream_reg.sv" \
  "$root_dir/rtl/memory/sram_2p_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp16_to_fp32.sv" \
  "$root_dir/rtl/arithmetic/fp32_mac_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_add_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_exp_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_recip_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_sqrt_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_to_fp16.sv" \
  "$root_dir/rtl/pe/lane_mask_generator.sv" \
  "$root_dir/rtl/pe/accumulator_bank.sv" \
  "$root_dir/rtl/pe/pe_perf_counter.sv" \
  "$root_dir/rtl/pe/pe_lane.sv" \
  "$root_dir/rtl/pe/fp32_reduction_tree.sv" \
  "$root_dir/rtl/pe/reconfigurable_pe_core.sv" \
  "$root_dir/rtl/pe/paper/paper_pe_cell.sv" \
  "$root_dir/rtl/pe/paper/paper_l1_reduction.sv" \
  "$root_dir/rtl/pe/paper/paper_l2_reduction.sv" \
  "$root_dir/rtl/pe/paper/paper_pe_group.sv" \
  "$root_dir/rtl/pe/paper/paper_array_8x8x2.sv" \
  "$root_dir/rtl/attention/paper/paper_attention_adapter.sv" \
  "$root_dir/rtl/attention/attention_score_scaler.sv" \
  "$root_dir/rtl/attention/score_buffer.sv" \
  "$root_dir/rtl/attention/softmax_reduction.sv" \
  "$root_dir/rtl/attention/softmax_normalization.sv" \
  "$root_dir/rtl/attention/single_head_attention_controller.sv" \
  "$root_dir/rtl/attention/single_head_attention.sv" \
  "$root_dir/rtl/cache/multi_head_kv_cache_manager.sv" \
  "$root_dir/rtl/cache/multi_head_generation_controller.sv" \
  "$root_dir/rtl/attention/multi_head_generation_engine.sv" \
  "$root_dir/rtl/projection/projection_input_buffer.sv" \
  "$root_dir/rtl/projection/projection_weight_buffer.sv" \
  "$root_dir/rtl/projection/shared_gemv_projection_core.sv" \
  "$root_dir/rtl/projection/projection_controller.sv" \
  "$root_dir/rtl/projection/qkv_staging_buffer.sv" \
  "$root_dir/rtl/projection/concat_fp16_buffer.sv" \
  "$root_dir/rtl/projection/head_concat_quantizer.sv" \
  "$root_dir/rtl/projection/output_projection_controller.sv" \
  "$root_dir/rtl/attention/projection_integrated_mha.sv" \
  "$root_dir/rtl/transformer/rmsnorm_engine.sv" \
  "$root_dir/rtl/transformer/residual_add_engine.sv" \
  "$root_dir/rtl/transformer/ffn_engine.sv" \
  "$root_dir/rtl/transformer/transformer_layer.sv" \
]

puts "STAGE8D_DC_ELABORATE: analyze RTL with SYNTHESIS defined; this is not PPA."
analyze -format sverilog -define SYNTHESIS $rtl_files

proc check_one {label top params report_path hierarchy_path} {
  puts "STAGE8D_DC_ELABORATE_TOP: $label"
  elaborate $top -parameters $params
  current_design $top
  link
  redirect -file $report_path { check_design }
  if {$hierarchy_path ne ""} {
    redirect -file $hierarchy_path { report_hierarchy }
  }
  remove_design -designs
}

set common "PE_NUM=8,MAX_SEQ_LEN=8,META_W=16,COUNTER_W=64"

check_one "single_head_attention_D8_LEGACY" single_head_attention "D_HEAD=8,$common,ATTENTION_PE_ARCH=0" "$report_dir/dc_check_stage8d_single_head_d8_legacy.rpt" ""
check_one "single_head_attention_D8_PAPER" single_head_attention "D_HEAD=8,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_single_head_d8_paper.rpt" "$report_dir/dc_hierarchy_stage8d_single_head_d8_paper.rpt"
check_one "multi_head_generation_H2_D8_LEGACY" multi_head_generation_engine "N_HEAD=2,D_HEAD=8,$common,ATTENTION_PE_ARCH=0" "$report_dir/dc_check_stage8d_generation_h2_d8_legacy.rpt" ""
check_one "multi_head_generation_H2_D8_PAPER" multi_head_generation_engine "N_HEAD=2,D_HEAD=8,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_generation_h2_d8_paper.rpt" "$report_dir/dc_hierarchy_stage8d_generation_h2_d8_paper.rpt"
check_one "transformer_H1_D8_PAPER" transformer_layer "N_HEAD=1,D_HEAD=8,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_transformer_h1_d8_paper.rpt" ""
check_one "transformer_H2_D8_LEGACY" transformer_layer "N_HEAD=2,D_HEAD=8,$common,ATTENTION_PE_ARCH=0" "$report_dir/dc_check_stage8d_transformer_h2_d8_legacy.rpt" ""
check_one "transformer_H2_D8_PAPER" transformer_layer "N_HEAD=2,D_HEAD=8,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_transformer_h2_d8_paper.rpt" "$report_dir/dc_hierarchy_stage8d_transformer_h2_d8_paper.rpt"
check_one "transformer_H4_D8_PAPER" transformer_layer "N_HEAD=4,D_HEAD=8,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_transformer_h4_d8_paper.rpt" ""
check_one "transformer_H2_D16_PAPER" transformer_layer "N_HEAD=2,D_HEAD=16,$common,ATTENTION_PE_ARCH=1" "$report_dir/dc_check_stage8d_transformer_h2_d16_paper.rpt" ""

puts "STAGE8D_DC_ELABORATE_PASS"
exit 0
