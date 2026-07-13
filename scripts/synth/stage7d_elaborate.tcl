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
    [file join $root "rtl/memory/sram_2p_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp16_to_fp32.sv"] \
    [file join $root "rtl/arithmetic/fp32_mac_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_add_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_exp_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_recip_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_sqrt_wrapper.sv"] \
    [file join $root "rtl/arithmetic/fp32_to_fp16.sv"] \
    [file join $root "rtl/pe/lane_mask_generator.sv"] \
    [file join $root "rtl/pe/accumulator_bank.sv"] \
    [file join $root "rtl/pe/pe_perf_counter.sv"] \
    [file join $root "rtl/pe/pe_lane.sv"] \
    [file join $root "rtl/pe/fp32_reduction_tree.sv"] \
    [file join $root "rtl/pe/reconfigurable_pe_core.sv"] \
    [file join $root "rtl/pe/paper/paper_pe_cell.sv"] \
    [file join $root "rtl/pe/paper/paper_l1_reduction.sv"] \
    [file join $root "rtl/pe/paper/paper_l2_reduction.sv"] \
    [file join $root "rtl/pe/paper/paper_pe_group.sv"] \
    [file join $root "rtl/pe/paper/paper_array_8x8x2.sv"] \
    [file join $root "rtl/attention/paper/paper_attention_adapter.sv"] \
    [file join $root "rtl/attention/paper/interleaved/paper_interleaved_attention_datapath.sv"] \
    [file join $root "rtl/attention/attention_score_scaler.sv"] \
    [file join $root "rtl/attention/score_buffer.sv"] \
    [file join $root "rtl/attention/softmax_reduction.sv"] \
    [file join $root "rtl/attention/softmax_normalization.sv"] \
    [file join $root "rtl/attention/single_head_attention_controller.sv"] \
    [file join $root "rtl/attention/single_head_attention.sv"] \
    [file join $root "rtl/cache/multi_head_kv_cache_manager.sv"] \
    [file join $root "rtl/cache/multi_head_generation_controller.sv"] \
    [file join $root "rtl/attention/multi_head_generation_engine.sv"] \
    [file join $root "rtl/projection/projection_input_buffer.sv"] \
    [file join $root "rtl/projection/projection_weight_buffer.sv"] \
    [file join $root "rtl/projection/shared_gemv_projection_core.sv"] \
    [file join $root "rtl/projection/projection_controller.sv"] \
    [file join $root "rtl/projection/qkv_staging_buffer.sv"] \
    [file join $root "rtl/projection/concat_fp16_buffer.sv"] \
    [file join $root "rtl/projection/head_concat_quantizer.sv"] \
    [file join $root "rtl/projection/output_projection_controller.sv"] \
    [file join $root "rtl/attention/projection_integrated_mha.sv"] \
    [file join $root "rtl/transformer/rmsnorm_engine.sv"] \
    [file join $root "rtl/transformer/residual_add_engine.sv"] \
    [file join $root "rtl/transformer/ffn_engine.sv"] \
    [file join $root "rtl/transformer/transformer_layer.sv"] \
]

define_design_lib WORK -path ./WORK
analyze -format sverilog -define SYNTHESIS $rtl_files

proc check_one {top params report_path} {
    elaborate $top -parameters $params
    link
    check_design > $report_path
    remove_design -designs
}

check_one transformer_layer "N_HEAD=1,D_HEAD=8,PE_NUM=8,MAX_SEQ_LEN=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7d_transformer_h1_d8.rpt"]
check_one transformer_layer "N_HEAD=2,D_HEAD=8,PE_NUM=8,MAX_SEQ_LEN=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7d_transformer_h2_d8.rpt"]
check_one transformer_layer "N_HEAD=4,D_HEAD=8,PE_NUM=8,MAX_SEQ_LEN=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7d_transformer_h4_d8.rpt"]
check_one transformer_layer "N_HEAD=2,D_HEAD=16,PE_NUM=8,MAX_SEQ_LEN=8,META_W=16,COUNTER_W=64" [file join $report_dir "dc_check_stage7d_transformer_h2_d16.rpt"]

exit 0
