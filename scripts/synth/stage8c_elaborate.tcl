set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_08"
set build_dir "$root_dir/build/stage8c_dc"
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
  "$root_dir/rtl/pe/paper/paper_pe_cell.sv" \
  "$root_dir/rtl/pe/paper/paper_l1_reduction.sv" \
  "$root_dir/rtl/pe/paper/paper_l2_reduction.sv" \
  "$root_dir/rtl/pe/paper/paper_pe_group.sv" \
  "$root_dir/rtl/pe/paper/paper_array_8x8x2.sv" \
]

puts "STAGE8C_DC_ELABORATE: analyze RTL with SYNTHESIS defined; this is not PPA."
analyze -format sverilog -define SYNTHESIS $rtl_files

puts "STAGE8C_DC_ELABORATE_TOP: paper_array_8x8x2"
elaborate paper_array_8x8x2
current_design paper_array_8x8x2
link
redirect -file "$report_dir/dc_check_stage8c_paper_array_8x8x2.rpt" { check_design }
redirect -file "$report_dir/dc_hierarchy_stage8c_paper_array_8x8x2.rpt" { report_hierarchy }
remove_design -designs

puts "STAGE8C_DC_ELABORATE_PASS"
exit 0
