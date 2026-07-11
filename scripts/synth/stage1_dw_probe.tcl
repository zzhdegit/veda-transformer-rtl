set script_path [file normalize [info script]]
set script_dir [file dirname $script_path]
set root_dir [file dirname [file dirname $script_dir]]
set report_dir "$root_dir/reports/stage_01"
set build_dir "$root_dir/build/stage1_dw_probe"
file mkdir $report_dir
file mkdir $build_dir
cd $build_dir

if {![info exists ::env(DW_FOUNDATION_SLDB)] || $::env(DW_FOUNDATION_SLDB) eq ""} {
  puts "ERROR: DW_FOUNDATION_SLDB environment variable is required."
  exit 10
}

set dw_sldb $::env(DW_FOUNDATION_SLDB)
if {[file exists $dw_sldb]} {
  set synthetic_library [list $dw_sldb]
  set link_library [concat "*" $synthetic_library]
} else {
  puts "ERROR: DW_FOUNDATION_SLDB does not name a readable file."
  exit 11
}

set probe_rtl "$build_dir/dw_probe_synth.v"
set fp [open $probe_rtl "w"]
puts $fp {module dw_probe_synth (
  input  [15:0] h_a,
  input  [15:0] h_b,
  input  [31:0] f_a,
  input  [31:0] f_b,
  input  [31:0] f_c,
  input  [2:0]  rnd,
  output [15:0] h_mul_z,
  output [31:0] f_add_z,
  output [31:0] f_mac_z,
  output [31:0] exp_z,
  output [31:0] div_z,
  output [31:0] recip_z,
  output [31:0] sqrt_z,
  output [31:0] invsqrt_z,
  output [7:0]  status_or
);
  wire [7:0] h_mul_status;
  wire [7:0] f_add_status;
  wire [7:0] f_mac_status;
  wire [7:0] exp_status;
  wire [7:0] div_status;
  wire [7:0] recip_status;
  wire [7:0] sqrt_status;
  wire [7:0] invsqrt_status;

  DW_fp_mult #(10, 5, 1) u_fp16_mult (.a(h_a), .b(h_b), .rnd(rnd), .z(h_mul_z), .status(h_mul_status));
  DW_fp_add #(23, 8, 1) u_fp32_add (.a(f_a), .b(f_b), .rnd(rnd), .z(f_add_z), .status(f_add_status));
  DW_fp_mac #(23, 8, 1) u_fp32_mac (.a(f_a), .b(f_b), .c(f_c), .rnd(rnd), .z(f_mac_z), .status(f_mac_status));
  DW_fp_exp #(23, 8, 1, 2) u_fp32_exp (.a(f_a), .z(exp_z), .status(exp_status));
  DW_fp_div #(23, 8, 1, 0) u_fp32_div (.a(f_a), .b(f_b), .rnd(rnd), .z(div_z), .status(div_status));
  DW_fp_recip #(23, 8, 1, 0) u_fp32_recip (.a(f_a), .rnd(rnd), .z(recip_z), .status(recip_status));
  DW_fp_sqrt #(23, 8, 1) u_fp32_sqrt (.a(f_a), .rnd(rnd), .z(sqrt_z), .status(sqrt_status));
  DW_fp_invsqrt #(23, 8, 1) u_fp32_invsqrt (.a(f_a), .rnd(rnd), .z(invsqrt_z), .status(invsqrt_status));
  assign status_or = h_mul_status | f_add_status | f_mac_status | exp_status | div_status |
                     recip_status | sqrt_status | invsqrt_status;
endmodule}
close $fp

puts "STAGE1_DW_DC_PROBE: analyze/elaborate only; no target PDK, no PPA."
analyze -format verilog $probe_rtl
elaborate dw_probe_synth
current_design dw_probe_synth
link
check_design > "$report_dir/dw_probe_check_design.rpt"
puts "STAGE1_DW_DC_PROBE_PASS"
exit 0
