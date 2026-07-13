if {![info exists ::env(DW_FOUNDATION_SLDB)]} {
    puts "Error: DW_FOUNDATION_SLDB is not set"
    exit 2
}

set dw_sldb $::env(DW_FOUNDATION_SLDB)
set synthetic_library [list $dw_sldb]
set link_library [concat "*" $synthetic_library]

set root [file normalize [file join [pwd] "../.."]]
set report_dir [file join $root "reports/stage_07"]
file mkdir $report_dir

set rtl_files [list \
    [file join $root "rtl/common/stream_reg.sv"] \
    [file join $root "rtl/arithmetic/fp16_to_fp32.sv"] \
    [file join $root "rtl/arithmetic/fp32_mac_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_add_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_to_fp16.sv"] \
    [file join $root "rtl/pe/lane_mask_generator.sv"] \
    [file join $root "rtl/pe/accumulator_bank.sv"] \
    [file join $root "rtl/pe/pe_perf_counter.sv"] \
    [file join $root "rtl/pe/pe_lane.sv"] \
    [file join $root "rtl/pe/fp32_reduction_tree.sv"] \
    [file join $root "rtl/pe/reconfigurable_pe_core.sv"] \
    [file join $root "rtl/transformer/ffn_engine.sv"] \
]

define_design_lib WORK -path ./WORK
analyze -format sverilog -define SYNTHESIS $rtl_files

proc check_one {top params report_path} {
    elaborate $top -parameters $params
    link
    check_design > $report_path
    remove_design -designs
}

check_one ffn_engine "D_MODEL=8,PE_NUM=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7c_ffn_d8.rpt"]
check_one ffn_engine "D_MODEL=16,PE_NUM=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7c_ffn_d16.rpt"]

exit 0
