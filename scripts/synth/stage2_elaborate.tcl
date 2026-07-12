set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_02"
set build_dir "$root_dir/build/stage2_dc"
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
  "$root_dir/rtl/arithmetic/fp16_to_fp32.sv" \
  "$root_dir/rtl/arithmetic/fp32_mac_wrapper.sv" \
  "$root_dir/rtl/arithmetic/fp32_add_wrapper.sv" \
  "$root_dir/rtl/pe/lane_mask_generator.sv" \
  "$root_dir/rtl/pe/accumulator_bank.sv" \
  "$root_dir/rtl/pe/pe_perf_counter.sv" \
  "$root_dir/rtl/pe/pe_lane.sv" \
  "$root_dir/rtl/pe/fp32_reduction_tree.sv" \
  "$root_dir/rtl/pe/reconfigurable_pe_core.sv" \
]

puts "STAGE2_DC_ELABORATE: analyze RTL with SYNTHESIS defined; this is not PPA."
analyze -format sverilog -define SYNTHESIS $rtl_files

set tops [list \
  fp32_add_wrapper \
  pe_lane \
  fp32_reduction_tree \
  accumulator_bank \
  lane_mask_generator \
  pe_perf_counter \
  reconfigurable_pe_core \
]

foreach top $tops {
  puts "STAGE2_DC_ELABORATE_TOP: $top"
  elaborate $top
  current_design $top
  link
  redirect -file "$report_dir/dc_check_${top}.rpt" { check_design }
  remove_design -designs
}

puts "STAGE2_DC_ELABORATE_TOP: reconfigurable_pe_core_PE128"
elaborate reconfigurable_pe_core -parameters {PE_NUM=128 META_W=16}
current_design reconfigurable_pe_core
link
redirect -file "$report_dir/dc_check_reconfigurable_pe_core_pe128.rpt" { check_design }
remove_design -designs

puts "STAGE2_DC_ELABORATE_PASS"
exit 0
