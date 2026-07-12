`default_nettype none

module generation_attention_engine #(
    parameter int PE_NUM = 8,
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [DIM_W-1:0]             token_dim,
    input  logic [15:0]                  token_q_fp16,
    input  logic [15:0]                  token_k_fp16,
    input  logic [15:0]                  token_v_fp16,
    input  logic                         token_last_dim,
    input  logic [META_W-1:0]            token_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [DIM_W-1:0]             output_base_dim,
    output logic [PE_NUM*32-1:0]         output_vector_fp32,
    output logic [PE_NUM-1:0]            output_lane_mask,
    output logic [7:0]                   output_status,
    output logic                         output_invalid,
    output logic [META_W-1:0]            output_meta,
    output logic                         output_last,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,
    output logic [SEQ_LEN_W-1:0]         done_valid_seq_len,
    output logic [SEQ_LEN_W-1:0]         current_valid_seq_len,

    output logic [COUNTER_W-1:0]         perf_generation_steps,
    output logic [COUNTER_W-1:0]         perf_provisional_append_cycles,
    output logic [COUNTER_W-1:0]         perf_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_commit_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_read_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_write_cycles,
    output logic [COUNTER_W-1:0]         perf_cache_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [SEQ_LEN_W-1:0]         perf_peak_valid_seq_len
);
    logic cache_rd_valid;
    logic cache_rd_ready;
    logic [TOKEN_W-1:0] cache_rd_token;
    logic [DIM_W-1:0] cache_rd_dim;
    logic cache_provisional_read_enable;
    logic cache_rd_rsp_valid;
    logic cache_rd_rsp_ready;
    logic [TOKEN_W-1:0] cache_rd_rsp_token;
    logic [DIM_W-1:0] cache_rd_rsp_dim;
    logic [15:0] cache_rd_rsp_k_fp16;
    logic [15:0] cache_rd_rsp_v_fp16;

    logic cache_append_valid;
    logic cache_append_ready;
    logic [TOKEN_W-1:0] cache_append_token_index;
    logic [DIM_W-1:0] cache_append_dim;
    logic [15:0] cache_append_k_fp16;
    logic [15:0] cache_append_v_fp16;
    logic cache_append_last_dim;
    logic cache_append_complete;
    logic cache_commit_valid;
    logic cache_commit_ready;
    logic [TOKEN_W-1:0] cache_commit_token_index;
    logic cache_abort_valid;
    logic cache_append_incomplete;
    logic cache_provisional_valid;
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

    kv_cache_manager #(
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .COUNTER_W(COUNTER_W)
    ) u_kv_cache_manager (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .rd_valid                (cache_rd_valid),
        .rd_ready                (cache_rd_ready),
        .rd_token                (cache_rd_token),
        .rd_dim                  (cache_rd_dim),
        .provisional_read_enable (cache_provisional_read_enable),
        .rd_rsp_valid            (cache_rd_rsp_valid),
        .rd_rsp_ready            (cache_rd_rsp_ready),
        .rd_rsp_token            (cache_rd_rsp_token),
        .rd_rsp_dim              (cache_rd_rsp_dim),
        .rd_rsp_k_fp16           (cache_rd_rsp_k_fp16),
        .rd_rsp_v_fp16           (cache_rd_rsp_v_fp16),
        .append_valid            (cache_append_valid),
        .append_ready            (cache_append_ready),
        .append_token_index      (cache_append_token_index),
        .append_dim              (cache_append_dim),
        .append_k_fp16           (cache_append_k_fp16),
        .append_v_fp16           (cache_append_v_fp16),
        .append_last_dim         (cache_append_last_dim),
        .append_complete         (cache_append_complete),
        .commit_valid            (cache_commit_valid),
        .commit_ready            (cache_commit_ready),
        .commit_token_index      (cache_commit_token_index),
        .abort_valid             (cache_abort_valid),
        .valid_seq_len           (current_valid_seq_len),
        .append_incomplete       (cache_append_incomplete),
        .provisional_valid       (cache_provisional_valid),
        .provisional_token_index (cache_provisional_token_index),
        .cache_full              (cache_full),
        .error_valid             (cache_error_valid),
        .error_ready             (cache_error_ready),
        .error_code              (cache_error_code),
        .perf_cache_read_cycles  (perf_cache_read_cycles),
        .perf_cache_write_cycles (perf_cache_write_cycles),
        .perf_cache_stall_cycles (perf_cache_stall_cycles),
        .perf_peak_valid_seq_len (perf_peak_valid_seq_len)
    );

    single_head_attention #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_single_head_attention (
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
        .perf_score_buffer_peak_occupancy (sha_perf_score_buffer_peak_occupancy)
    );

    generation_attention_controller #(
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_generation_controller (
        .clk                             (clk),
        .rst_n                           (rst_n),
        .token_valid                     (token_valid),
        .token_ready                     (token_ready),
        .token_dim                       (token_dim),
        .token_q_fp16                    (token_q_fp16),
        .token_k_fp16                    (token_k_fp16),
        .token_v_fp16                    (token_v_fp16),
        .token_last_dim                  (token_last_dim),
        .token_meta                      (token_meta),
        .output_valid                    (output_valid),
        .output_ready                    (output_ready),
        .output_base_dim                 (output_base_dim),
        .output_vector_fp32              (output_vector_fp32),
        .output_lane_mask                (output_lane_mask),
        .output_status                   (output_status),
        .output_invalid                  (output_invalid),
        .output_meta                     (output_meta),
        .output_last                     (output_last),
        .done_valid                      (done_valid),
        .done_ready                      (done_ready),
        .done_status                     (done_status),
        .done_invalid                    (done_invalid),
        .done_meta                       (done_meta),
        .done_valid_seq_len              (done_valid_seq_len),
        .cache_rd_valid                  (cache_rd_valid),
        .cache_rd_ready                  (cache_rd_ready),
        .cache_rd_token                  (cache_rd_token),
        .cache_rd_dim                    (cache_rd_dim),
        .cache_provisional_read_enable   (cache_provisional_read_enable),
        .cache_rd_rsp_valid              (cache_rd_rsp_valid),
        .cache_rd_rsp_ready              (cache_rd_rsp_ready),
        .cache_rd_rsp_token              (cache_rd_rsp_token),
        .cache_rd_rsp_dim                (cache_rd_rsp_dim),
        .cache_rd_rsp_k_fp16             (cache_rd_rsp_k_fp16),
        .cache_rd_rsp_v_fp16             (cache_rd_rsp_v_fp16),
        .cache_append_valid              (cache_append_valid),
        .cache_append_ready              (cache_append_ready),
        .cache_append_token_index        (cache_append_token_index),
        .cache_append_dim                (cache_append_dim),
        .cache_append_k_fp16             (cache_append_k_fp16),
        .cache_append_v_fp16             (cache_append_v_fp16),
        .cache_append_last_dim           (cache_append_last_dim),
        .cache_append_complete           (cache_append_complete),
        .cache_commit_valid              (cache_commit_valid),
        .cache_commit_ready              (cache_commit_ready),
        .cache_commit_token_index        (cache_commit_token_index),
        .cache_abort_valid               (cache_abort_valid),
        .cache_valid_seq_len             (current_valid_seq_len),
        .cache_append_incomplete         (cache_append_incomplete),
        .cache_provisional_valid         (cache_provisional_valid),
        .cache_provisional_token_index   (cache_provisional_token_index),
        .cache_full                      (cache_full),
        .cache_error_valid               (cache_error_valid),
        .cache_error_ready               (cache_error_ready),
        .cache_error_code                (cache_error_code),
        .sha_load_valid                  (sha_load_valid),
        .sha_load_ready                  (sha_load_ready),
        .sha_load_kind                   (sha_load_kind),
        .sha_load_token                  (sha_load_token),
        .sha_load_dim                    (sha_load_dim),
        .sha_load_data                   (sha_load_data),
        .sha_start_valid                 (sha_start_valid),
        .sha_start_ready                 (sha_start_ready),
        .sha_start_seq_len               (sha_start_seq_len),
        .sha_start_meta                  (sha_start_meta),
        .sha_output_valid                (sha_output_valid),
        .sha_output_ready                (sha_output_ready),
        .sha_output_base_dim             (sha_output_base_dim),
        .sha_output_vector_fp32          (sha_output_vector_fp32),
        .sha_output_lane_mask            (sha_output_lane_mask),
        .sha_output_status               (sha_output_status),
        .sha_output_invalid              (sha_output_invalid),
        .sha_output_meta                 (sha_output_meta),
        .sha_output_last                 (sha_output_last),
        .sha_done_valid                  (sha_done_valid),
        .sha_done_ready                  (sha_done_ready),
        .sha_done_status                 (sha_done_status),
        .sha_done_invalid                (sha_done_invalid),
        .sha_done_meta                   (sha_done_meta),
        .sha_perf_total_attention_cycles (sha_perf_total_attention_cycles),
        .sha_perf_pe_stall_cycles        (sha_perf_pe_stall_cycles),
        .sha_perf_sfu_stall_cycles       (sha_perf_sfu_stall_cycles),
        .perf_generation_steps           (perf_generation_steps),
        .perf_provisional_append_cycles  (perf_provisional_append_cycles),
        .perf_attention_cycles           (perf_attention_cycles),
        .perf_commit_cycles              (perf_commit_cycles),
        .perf_pe_stall_cycles            (perf_pe_stall_cycles),
        .perf_sfu_stall_cycles           (perf_sfu_stall_cycles),
        .perf_total_cycles               (perf_total_cycles)
    );
endmodule

`default_nettype wire
