`default_nettype none

module multi_head_generation_engine #(
    parameter int N_HEAD = 2,
    parameter int PE_NUM = 8,
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter int ATTENTION_PE_ARCH = 0,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [HEAD_W-1:0]            token_head,
    input  logic [DIM_W-1:0]             token_dim,
    input  logic [15:0]                  token_q_fp16,
    input  logic [15:0]                  token_k_fp16,
    input  logic [15:0]                  token_v_fp16,
    input  logic                         token_last_dim,
    input  logic                         token_last_head,
    input  logic [META_W-1:0]            token_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [HEAD_W-1:0]            output_head,
    output logic [DIM_W-1:0]             output_base_dim,
    output logic [PE_NUM*32-1:0]         output_vector_fp32,
    output logic [PE_NUM-1:0]            output_lane_mask,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last_tile,
    output logic                         output_last_head,
    output logic                         output_last_token,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,
    output logic [SEQ_LEN_W-1:0]         done_valid_seq_len,
    output logic [SEQ_LEN_W-1:0]         current_valid_seq_len,

    output logic [COUNTER_W-1:0]         perf_generation_steps,
    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_per_head_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_head_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_provisional_write_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_read_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_write_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_commit_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_paper_array_active_cycles,
    output logic [COUNTER_W-1:0]         perf_paper_array_idle_cycles,
    output logic [COUNTER_W-1:0]         perf_inner_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_outer_mode_cycles,
    output logic [COUNTER_W-1:0]         perf_group0_active_cycles,
    output logic [COUNTER_W-1:0]         perf_group1_active_cycles,
    output logic [COUNTER_W-1:0]         perf_tail_masked_pe_cycles,
    output logic [COUNTER_W-1:0]         perf_mode_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_array_input_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_array_output_stall_cycles,
    output logic [SEQ_LEN_W-1:0]         perf_peak_valid_seq_len
);
    logic cache_rd_valid;
    logic cache_rd_ready;
    logic [HEAD_W-1:0] cache_rd_head;
    logic [TOKEN_W-1:0] cache_rd_token;
    logic [DIM_W-1:0] cache_rd_dim;
    logic cache_provisional_read_enable;
    logic cache_rd_rsp_valid;
    logic cache_rd_rsp_ready;
    logic [HEAD_W-1:0] cache_rd_rsp_head;
    logic [TOKEN_W-1:0] cache_rd_rsp_token;
    logic [DIM_W-1:0] cache_rd_rsp_dim;
    logic [15:0] cache_rd_rsp_k_fp16;
    logic [15:0] cache_rd_rsp_v_fp16;

    logic cache_append_valid;
    logic cache_append_ready;
    logic [HEAD_W-1:0] cache_append_head;
    logic [TOKEN_W-1:0] cache_append_token_index;
    logic [DIM_W-1:0] cache_append_dim;
    logic [15:0] cache_append_k_fp16;
    logic [15:0] cache_append_v_fp16;
    logic cache_append_last_dim;
    logic cache_append_last_head;
    logic cache_append_complete;
    logic cache_commit_valid;
    logic cache_commit_ready;
    logic [TOKEN_W-1:0] cache_commit_token_index;
    logic cache_abort_valid;
    logic cache_append_incomplete;
    logic cache_provisional_valid;
    logic [N_HEAD-1:0] cache_provisional_head_valid;
    logic [TOKEN_W-1:0] cache_provisional_token_index;
    logic cache_full;
    logic cache_error_valid;
    logic cache_error_ready;
    logic [7:0] cache_error_code;

    logic sha_load_valid;
    logic sha_load_ready;
    logic [1:0] sha_load_kind;
    logic [TOKEN_W-1:0] sha_load_token;
    logic [DIM_W-1:0] sha_load_dim;
    logic [15:0] sha_load_data;
    logic sha_start_valid;
    logic sha_start_ready;
    logic [SEQ_LEN_W-1:0] sha_start_seq_len;
    logic [META_W-1:0] sha_start_meta;
    logic sha_output_valid;
    logic sha_output_ready;
    logic [DIM_W-1:0] sha_output_base_dim;
    logic [PE_NUM*32-1:0] sha_output_vector_fp32;
    logic [PE_NUM-1:0] sha_output_lane_mask;
    logic [7:0] sha_output_status;
    logic sha_output_invalid;
    logic [META_W-1:0] sha_output_meta;
    logic sha_output_last;
    logic sha_done_valid;
    logic sha_done_ready;
    logic [7:0] sha_done_status;
    logic sha_done_invalid;
    logic [META_W-1:0] sha_done_meta;

    logic [COUNTER_W-1:0] sha_perf_total_attention_cycles;
    logic [COUNTER_W-1:0] sha_perf_qk_cycles;
    logic [COUNTER_W-1:0] sha_perf_qk_pe_busy_cycles;
    logic [COUNTER_W-1:0] sha_perf_scale_cycles;
    logic [COUNTER_W-1:0] sha_perf_reduction_cycles;
    logic [COUNTER_W-1:0] sha_perf_reduction_finalize_cycles;
    logic [COUNTER_W-1:0] sha_perf_normalization_cycles;
    logic [COUNTER_W-1:0] sha_perf_sv_cycles;
    logic [COUNTER_W-1:0] sha_perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] sha_perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] sha_perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] sha_perf_output_stall_cycles;
    logic [COUNTER_W-1:0] sha_perf_score_buffer_peak_occupancy;
    logic [COUNTER_W-1:0] sha_perf_paper_array_active_cycles;
    logic [COUNTER_W-1:0] sha_perf_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] sha_perf_inner_mode_cycles;
    logic [COUNTER_W-1:0] sha_perf_outer_mode_cycles;
    logic [COUNTER_W-1:0] sha_perf_group0_active_cycles;
    logic [COUNTER_W-1:0] sha_perf_group1_active_cycles;
    logic [COUNTER_W-1:0] sha_perf_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] sha_perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] sha_perf_array_input_stall_cycles;
    logic [COUNTER_W-1:0] sha_perf_array_output_stall_cycles;

    multi_head_kv_cache_manager #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .COUNTER_W(COUNTER_W)
    ) u_multi_head_kv_cache_manager (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .rd_valid                  (cache_rd_valid),
        .rd_ready                  (cache_rd_ready),
        .rd_head                   (cache_rd_head),
        .rd_token                  (cache_rd_token),
        .rd_dim                    (cache_rd_dim),
        .provisional_read_enable   (cache_provisional_read_enable),
        .rd_rsp_valid              (cache_rd_rsp_valid),
        .rd_rsp_ready              (cache_rd_rsp_ready),
        .rd_rsp_head               (cache_rd_rsp_head),
        .rd_rsp_token              (cache_rd_rsp_token),
        .rd_rsp_dim                (cache_rd_rsp_dim),
        .rd_rsp_k_fp16             (cache_rd_rsp_k_fp16),
        .rd_rsp_v_fp16             (cache_rd_rsp_v_fp16),
        .append_valid              (cache_append_valid),
        .append_ready              (cache_append_ready),
        .append_head               (cache_append_head),
        .append_token_index        (cache_append_token_index),
        .append_dim                (cache_append_dim),
        .append_k_fp16             (cache_append_k_fp16),
        .append_v_fp16             (cache_append_v_fp16),
        .append_last_dim           (cache_append_last_dim),
        .append_last_head          (cache_append_last_head),
        .append_complete           (cache_append_complete),
        .commit_valid              (cache_commit_valid),
        .commit_ready              (cache_commit_ready),
        .commit_token_index        (cache_commit_token_index),
        .abort_valid               (cache_abort_valid),
        .valid_seq_len             (current_valid_seq_len),
        .append_incomplete         (cache_append_incomplete),
        .provisional_valid         (cache_provisional_valid),
        .provisional_head_valid    (cache_provisional_head_valid),
        .provisional_token_index   (cache_provisional_token_index),
        .cache_full                (cache_full),
        .error_valid               (cache_error_valid),
        .error_ready               (cache_error_ready),
        .error_code                (cache_error_code),
        .perf_cache_read_cycles    (perf_cache_read_cycles),
        .perf_cache_write_cycles   (perf_cache_write_cycles),
        .perf_cache_stall_cycles   (perf_cache_stall_cycles),
        .perf_peak_valid_seq_len   (perf_peak_valid_seq_len)
    );

    single_head_attention #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(ATTENTION_PE_ARCH),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_shared_single_head_attention (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .load_valid                       (sha_load_valid),
        .load_ready                       (sha_load_ready),
        .load_kind                        (sha_load_kind),
        .load_token                       (sha_load_token),
        .load_dim                         (sha_load_dim),
        .load_data                        (sha_load_data),
        .start_valid                      (sha_start_valid),
        .start_ready                      (sha_start_ready),
        .start_seq_len                    (sha_start_seq_len),
        .start_meta                       (sha_start_meta),
        .output_valid                     (sha_output_valid),
        .output_ready                     (sha_output_ready),
        .output_base_dim                  (sha_output_base_dim),
        .output_vector_fp32               (sha_output_vector_fp32),
        .output_lane_mask                 (sha_output_lane_mask),
        .output_status                    (sha_output_status),
        .output_invalid                   (sha_output_invalid),
        .output_meta                      (sha_output_meta),
        .output_last                      (sha_output_last),
        .done_valid                       (sha_done_valid),
        .done_ready                       (sha_done_ready),
        .done_status                      (sha_done_status),
        .done_invalid                     (sha_done_invalid),
        .done_meta                        (sha_done_meta),
        .perf_total_attention_cycles      (sha_perf_total_attention_cycles),
        .perf_qk_cycles                   (sha_perf_qk_cycles),
        .perf_qk_pe_busy_cycles           (sha_perf_qk_pe_busy_cycles),
        .perf_scale_cycles                (sha_perf_scale_cycles),
        .perf_reduction_cycles            (sha_perf_reduction_cycles),
        .perf_reduction_finalize_cycles   (sha_perf_reduction_finalize_cycles),
        .perf_normalization_cycles        (sha_perf_normalization_cycles),
        .perf_sv_cycles                   (sha_perf_sv_cycles),
        .perf_pe_stall_cycles             (sha_perf_pe_stall_cycles),
        .perf_sfu_stall_cycles            (sha_perf_sfu_stall_cycles),
        .perf_buffer_stall_cycles         (sha_perf_buffer_stall_cycles),
        .perf_output_stall_cycles         (sha_perf_output_stall_cycles),
        .perf_score_buffer_peak_occupancy (sha_perf_score_buffer_peak_occupancy),
        .perf_paper_array_active_cycles   (sha_perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles     (sha_perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles           (sha_perf_inner_mode_cycles),
        .perf_outer_mode_cycles           (sha_perf_outer_mode_cycles),
        .perf_group0_active_cycles        (sha_perf_group0_active_cycles),
        .perf_group1_active_cycles        (sha_perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles       (sha_perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles          (sha_perf_mode_switch_cycles),
        .perf_array_input_stall_cycles    (sha_perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles   (sha_perf_array_output_stall_cycles)
    );

    assign perf_paper_array_active_cycles = sha_perf_paper_array_active_cycles;
    assign perf_paper_array_idle_cycles = sha_perf_paper_array_idle_cycles;
    assign perf_inner_mode_cycles = sha_perf_inner_mode_cycles;
    assign perf_outer_mode_cycles = sha_perf_outer_mode_cycles;
    assign perf_group0_active_cycles = sha_perf_group0_active_cycles;
    assign perf_group1_active_cycles = sha_perf_group1_active_cycles;
    assign perf_tail_masked_pe_cycles = sha_perf_tail_masked_pe_cycles;
    assign perf_mode_switch_cycles = sha_perf_mode_switch_cycles;
    assign perf_array_input_stall_cycles = sha_perf_array_input_stall_cycles;
    assign perf_array_output_stall_cycles = sha_perf_array_output_stall_cycles;

    multi_head_generation_controller #(
        .N_HEAD(N_HEAD),
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_multi_head_generation_controller (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .token_valid                      (token_valid),
        .token_ready                      (token_ready),
        .token_head                       (token_head),
        .token_dim                        (token_dim),
        .token_q_fp16                     (token_q_fp16),
        .token_k_fp16                     (token_k_fp16),
        .token_v_fp16                     (token_v_fp16),
        .token_last_dim                   (token_last_dim),
        .token_last_head                  (token_last_head),
        .token_meta                       (token_meta),
        .output_valid                     (output_valid),
        .output_ready                     (output_ready),
        .output_head                      (output_head),
        .output_base_dim                  (output_base_dim),
        .output_vector_fp32               (output_vector_fp32),
        .output_lane_mask                 (output_lane_mask),
        .output_status                    (output_status),
        .output_invalid                   (output_invalid),
        .output_meta                      (output_meta),
        .output_last_tile                 (output_last_tile),
        .output_last_head                 (output_last_head),
        .output_last_token                (output_last_token),
        .done_valid                       (done_valid),
        .done_ready                       (done_ready),
        .done_status                      (done_status),
        .done_invalid                     (done_invalid),
        .done_meta                        (done_meta),
        .done_valid_seq_len               (done_valid_seq_len),
        .cache_rd_valid                   (cache_rd_valid),
        .cache_rd_ready                   (cache_rd_ready),
        .cache_rd_head                    (cache_rd_head),
        .cache_rd_token                   (cache_rd_token),
        .cache_rd_dim                     (cache_rd_dim),
        .cache_provisional_read_enable    (cache_provisional_read_enable),
        .cache_rd_rsp_valid               (cache_rd_rsp_valid),
        .cache_rd_rsp_ready               (cache_rd_rsp_ready),
        .cache_rd_rsp_head                (cache_rd_rsp_head),
        .cache_rd_rsp_token               (cache_rd_rsp_token),
        .cache_rd_rsp_dim                 (cache_rd_rsp_dim),
        .cache_rd_rsp_k_fp16              (cache_rd_rsp_k_fp16),
        .cache_rd_rsp_v_fp16              (cache_rd_rsp_v_fp16),
        .cache_append_valid               (cache_append_valid),
        .cache_append_ready               (cache_append_ready),
        .cache_append_head                (cache_append_head),
        .cache_append_token_index         (cache_append_token_index),
        .cache_append_dim                 (cache_append_dim),
        .cache_append_k_fp16              (cache_append_k_fp16),
        .cache_append_v_fp16              (cache_append_v_fp16),
        .cache_append_last_dim            (cache_append_last_dim),
        .cache_append_last_head           (cache_append_last_head),
        .cache_append_complete            (cache_append_complete),
        .cache_commit_valid               (cache_commit_valid),
        .cache_commit_ready               (cache_commit_ready),
        .cache_commit_token_index         (cache_commit_token_index),
        .cache_abort_valid                (cache_abort_valid),
        .cache_valid_seq_len              (current_valid_seq_len),
        .cache_append_incomplete          (cache_append_incomplete),
        .cache_provisional_valid          (cache_provisional_valid),
        .cache_provisional_head_valid     (cache_provisional_head_valid),
        .cache_provisional_token_index    (cache_provisional_token_index),
        .cache_full                       (cache_full),
        .cache_error_valid                (cache_error_valid),
        .cache_error_ready                (cache_error_ready),
        .cache_error_code                 (cache_error_code),
        .sha_load_valid                   (sha_load_valid),
        .sha_load_ready                   (sha_load_ready),
        .sha_load_kind                    (sha_load_kind),
        .sha_load_token                   (sha_load_token),
        .sha_load_dim                     (sha_load_dim),
        .sha_load_data                    (sha_load_data),
        .sha_start_valid                  (sha_start_valid),
        .sha_start_ready                  (sha_start_ready),
        .sha_start_seq_len                (sha_start_seq_len),
        .sha_start_meta                   (sha_start_meta),
        .sha_output_valid                 (sha_output_valid),
        .sha_output_ready                 (sha_output_ready),
        .sha_output_base_dim              (sha_output_base_dim),
        .sha_output_vector_fp32           (sha_output_vector_fp32),
        .sha_output_lane_mask             (sha_output_lane_mask),
        .sha_output_status                (sha_output_status),
        .sha_output_invalid               (sha_output_invalid),
        .sha_output_meta                  (sha_output_meta),
        .sha_output_last                  (sha_output_last),
        .sha_done_valid                   (sha_done_valid),
        .sha_done_ready                   (sha_done_ready),
        .sha_done_status                  (sha_done_status),
        .sha_done_invalid                 (sha_done_invalid),
        .sha_done_meta                    (sha_done_meta),
        .sha_perf_total_attention_cycles  (sha_perf_total_attention_cycles),
        .sha_perf_pe_stall_cycles         (sha_perf_pe_stall_cycles),
        .sha_perf_sfu_stall_cycles        (sha_perf_sfu_stall_cycles),
        .perf_generation_steps            (perf_generation_steps),
        .perf_total_cycles                (perf_total_cycles),
        .perf_per_head_attention_cycles   (perf_per_head_attention_cycles),
        .perf_head_switch_cycles          (perf_head_switch_cycles),
        .perf_provisional_write_cycles    (perf_provisional_write_cycles),
        .perf_commit_cycles               (perf_commit_cycles),
        .perf_pe_stall_cycles             (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles            (perf_sfu_stall_cycles),
        .perf_output_stall_cycles         (perf_output_stall_cycles)
    );

`ifndef SYNTHESIS
    initial begin
        if (N_HEAD <= 0) begin
            $fatal(1, "multi_head_generation_engine N_HEAD must be positive");
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(output_valid && $isunknown({output_head, output_base_dim, output_vector_fp32,
                                                  output_lane_mask, output_status, output_invalid,
                                                  output_meta, output_last_tile, output_last_head,
                                                  output_last_token})))
                else $error("multi_head_generation_engine no unknown output when valid failed");
            assert (!(cache_commit_valid && (cache_provisional_head_valid != {N_HEAD{1'b1}})))
                else $error("multi_head_generation_engine no_partial_head_commit failed");
        end
    end
`endif
endmodule

`default_nettype wire
