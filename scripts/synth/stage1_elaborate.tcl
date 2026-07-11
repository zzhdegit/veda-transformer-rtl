set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_01"
set build_dir "$root_dir/build/stage1_dc"
file mkdir $report_dir
file mkdir $build_dir

set_app_var hdlin_enable_vpp true
define_design_lib WORK -path "$build_dir/WORK"
cd $build_dir

set rtl_files [list \
  "$root_dir/rtl/common/stream_reg.sv" \
  "$root_dir/rtl/common/skid_buffer.sv" \
  "$root_dir/rtl/memory/sync_fifo.sv" \
  "$root_dir/rtl/memory/sram_1p_wrapper.sv" \
  "$root_dir/rtl/memory/sram_2p_wrapper.sv" \
  "$root_dir/rtl/arithmetic/mul_unit.sv" \
  "$root_dir/rtl/arithmetic/add_unit.sv" \
  "$root_dir/rtl/arithmetic/mac_unit.sv" \
  "$root_dir/rtl/arithmetic/compare_max.sv" \
  "$root_dir/rtl/arithmetic/round_sat.sv" \
]

puts "STAGE1_DC_ELABORATE: analyze RTL with SYNTHESIS defined; this is not PPA."
analyze -format sverilog -define SYNTHESIS $rtl_files

set tops [list \
  stream_reg \
  skid_buffer \
  sync_fifo \
  sram_1p_wrapper \
  sram_2p_wrapper \
  mul_unit \
  add_unit \
  mac_unit \
  compare_max \
  round_sat \
]

foreach top $tops {
  puts "STAGE1_DC_ELABORATE_TOP: $top"
  elaborate $top
  current_design $top
  link
  check_design > "$report_dir/dc_check_${top}.rpt"
  remove_design -designs
}

puts "STAGE1_DC_ELABORATE_PASS"
exit 0
