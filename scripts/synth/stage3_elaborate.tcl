set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_03"
set build_dir "$root_dir/build/stage3_dc"
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
  "$root_dir/rtl/pe/lane_mask_generator.sv" \
  "$root_dir/rtl/pe/accumulator_bank.sv" \
  "$root_dir/rtl/pe/pe_perf_counter.sv" \
  "$root_dir/rtl/pe/pe_lane.sv" \
  "$root_dir/rtl/pe/fp32_reduction_tree.sv" \
  "$root_dir/rtl/pe/reconfigurable_pe_core.sv" \
  "$root_dir/rtl/attention/attention_score_scaler.sv" \
  "$root_dir/rtl/attention/score_buffer.sv" \
  "$root_dir/rtl/attention/softmax_reduction.sv" \
  "$root_dir/rtl/attention/softmax_normalization.sv" \
  "$root_dir/rtl/attention/single_head_attention_controller.sv" \
  "$root_dir/rtl/attention/single_head_attention.sv" \
]

puts "STAGE3_DC_ELABORATE: analyze RTL with SYNTHESIS defined; this is not PPA."
analyze -format sverilog -define SYNTHESIS $rtl_files

set tops [list \
  fp32_exp_wrapper \
  fp32_recip_wrapper \
  attention_score_scaler \
  score_buffer \
  softmax_reduction \
  softmax_normalization \
  single_head_attention \
]

foreach top $tops {
  puts "STAGE3_DC_ELABORATE_TOP: $top"
  elaborate $top
  current_design $top
  link
  redirect -file "$report_dir/dc_check_${top}.rpt" { check_design }
  remove_design -designs
}

puts "STAGE3_DC_ELABORATE_TOP: score_buffer_DEPTH4096"
elaborate score_buffer -parameters {DEPTH=4096}
current_design score_buffer
link
redirect -file "$report_dir/dc_check_score_buffer_depth4096.rpt" { check_design }
remove_design -designs

puts "STAGE3_DC_ELABORATE_TOP: single_head_attention_D16"
elaborate single_head_attention -parameters {PE_NUM=8 D_HEAD=16 MAX_SEQ_LEN=32 META_W=16}
current_design single_head_attention
link
redirect -file "$report_dir/dc_check_single_head_attention_d16.rpt" { check_design }
remove_design -designs

puts "STAGE3_DC_ELABORATE_PASS"
exit 0

