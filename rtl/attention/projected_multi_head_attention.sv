`default_nettype none

module projected_multi_head_attention #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    parameter int PE_NUM = 8,
    parameter int MAX_SEQ_LEN = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         hidden_valid,
    output logic                         hidden_ready,
    input  logic [MODEL_W-1:0]           hidden_dim,
    input  logic [15:0]                  hidden_data_fp16,
    input  logic                         hidden_last,
    input  logic [META_W-1:0]            hidden_meta,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [1:0]                   weight_kind,
    input  logic [MODEL_W-1:0]           weight_output_index,
    input  logic [MODEL_W-1:0]           weight_input_index,
    input  logic [15:0]                  weight_data_fp16,
    input  logic                         weight_last,
    input  logic                         weight_commit,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [META_W-1:0]            start_meta,

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

    output logic [COUNTER_W-1:0]         perf_q_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_k_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_v_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_qkv_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_generation_steps,
    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_weight_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_buffer_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles,
    output logic [SEQ_LEN_W-1:0]         perf_peak_valid_seq_len
);
    logic qkv_valid;
    logic qkv_ready;
    logic qkv_hidden_ready;
    logic qkv_weight_ready;
    logic qkv_start_ready;
    logic [HEAD_W-1:0] qkv_head;
    logic [DIM_W-1:0] qkv_dim;
    logic [15:0] qkv_q_fp16;
    logic [15:0] qkv_k_fp16;
    logic [15:0] qkv_v_fp16;
    logic qkv_last_dim;
    logic qkv_last_head;
    logic [META_W-1:0] qkv_meta;

    logic qkv_done_valid;
    logic [7:0] qkv_done_status;
    logic qkv_done_invalid;
    logic [META_W-1:0] qkv_done_meta;
    logic [COUNTER_W-1:0] qkv_pe_stall_cycles;
    logic [COUNTER_W-1:0] qkv_output_stall_cycles;
    logic [COUNTER_W-1:0] stage5_total_cycles;
    logic [COUNTER_W-1:0] stage5_per_head_attention_cycles;
    logic [COUNTER_W-1:0] stage5_cache_stall_cycles;
    logic [COUNTER_W-1:0] stage5_pe_stall_cycles;
    logic [COUNTER_W-1:0] stage5_sfu_stall_cycles;
    logic [COUNTER_W-1:0] stage5_output_stall_cycles;
    logic [COUNTER_W-1:0] unused_cache_read_cycles;
    logic [COUNTER_W-1:0] unused_cache_write_cycles;
    logic [COUNTER_W-1:0] unused_head_switch_cycles;
    logic [COUNTER_W-1:0] unused_provisional_write_cycles;
    logic [COUNTER_W-1:0] unused_commit_cycles;
    logic active_q;

    wire start_fire = start_valid && start_ready;
    wire done_fire = done_valid && done_ready;

    initial begin
        if (D_MODEL != N_HEAD * D_HEAD) begin
            $fatal(1, "projected_multi_head_attention d_model_equals_n_head_times_d_head failed");
        end
    end

    assign perf_attention_cycles = stage5_per_head_attention_cycles;
    assign perf_total_cycles = stage5_total_cycles + perf_q_projection_cycles +
                               perf_k_projection_cycles + perf_v_projection_cycles;
    assign perf_pe_stall_cycles = qkv_pe_stall_cycles + stage5_pe_stall_cycles;
    assign perf_sfu_stall_cycles = stage5_sfu_stall_cycles;
    assign perf_buffer_stall_cycles = stage5_cache_stall_cycles;
    assign perf_output_stall_cycles = qkv_output_stall_cycles + stage5_output_stall_cycles;
    assign hidden_ready = !active_q && qkv_hidden_ready;
    assign weight_ready = !active_q && qkv_weight_ready;
    assign start_ready = !active_q && qkv_start_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_q <= 1'b0;
        end else begin
            if (start_fire) begin
                active_q <= 1'b1;
            end
            if (done_fire) begin
                active_q <= 1'b0;
            end
        end
    end

    qkv_projection_engine #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_qkv_projection_engine (
        .clk                           (clk),
        .rst_n                         (rst_n),
        .input_valid                   (hidden_valid && !active_q),
        .input_ready                   (qkv_hidden_ready),
        .input_dim                     (hidden_dim),
        .input_data_fp16               (hidden_data_fp16),
        .input_last                    (hidden_last),
        .input_meta                    (hidden_meta),
        .weight_valid                  (weight_valid && !active_q),
        .weight_ready                  (qkv_weight_ready),
        .weight_kind                   (weight_kind),
        .weight_output_index           (weight_output_index),
        .weight_input_index            (weight_input_index),
        .weight_data_fp16              (weight_data_fp16),
        .weight_last                   (weight_last),
        .weight_commit                 (weight_commit),
        .start_valid                   (start_valid && !active_q),
        .start_ready                   (qkv_start_ready),
        .start_meta                    (start_meta),
        .qkv_valid                     (qkv_valid),
        .qkv_ready                     (qkv_ready),
        .qkv_head                      (qkv_head),
        .qkv_dim                       (qkv_dim),
        .qkv_q_fp16                    (qkv_q_fp16),
        .qkv_k_fp16                    (qkv_k_fp16),
        .qkv_v_fp16                    (qkv_v_fp16),
        .qkv_last_dim                  (qkv_last_dim),
        .qkv_last_head                 (qkv_last_head),
        .qkv_meta                      (qkv_meta),
        .done_valid                    (qkv_done_valid),
        .done_ready                    (1'b1),
        .done_status                   (qkv_done_status),
        .done_invalid                  (qkv_done_invalid),
        .done_meta                     (qkv_done_meta),
        .perf_q_projection_cycles      (perf_q_projection_cycles),
        .perf_k_projection_cycles      (perf_k_projection_cycles),
        .perf_v_projection_cycles      (perf_v_projection_cycles),
        .perf_qkv_quantization_cycles  (perf_qkv_quantization_cycles),
        .perf_weight_stall_cycles      (perf_weight_stall_cycles),
        .perf_pe_stall_cycles          (qkv_pe_stall_cycles),
        .perf_output_stall_cycles      (qkv_output_stall_cycles)
    );

    multi_head_generation_engine #(
        .N_HEAD(N_HEAD),
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_multi_head_generation_engine (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .token_valid                    (qkv_valid),
        .token_ready                    (qkv_ready),
        .token_head                     (qkv_head),
        .token_dim                      (qkv_dim),
        .token_q_fp16                   (qkv_q_fp16),
        .token_k_fp16                   (qkv_k_fp16),
        .token_v_fp16                   (qkv_v_fp16),
        .token_last_dim                 (qkv_last_dim),
        .token_last_head                (qkv_last_head),
        .token_meta                     (qkv_meta),
        .output_valid                   (output_valid),
        .output_ready                   (output_ready),
        .output_head                    (output_head),
        .output_base_dim                (output_base_dim),
        .output_vector_fp32             (output_vector_fp32),
        .output_lane_mask               (output_lane_mask),
        .output_status                  (output_status),
        .output_invalid                 (output_invalid),
        .output_meta                    (output_meta),
        .output_last_tile               (output_last_tile),
        .output_last_head               (output_last_head),
        .output_last_token              (output_last_token),
        .done_valid                     (done_valid),
        .done_ready                     (done_ready),
        .done_status                    (done_status),
        .done_invalid                   (done_invalid),
        .done_meta                      (done_meta),
        .done_valid_seq_len             (done_valid_seq_len),
        .current_valid_seq_len          (current_valid_seq_len),
        .perf_generation_steps          (perf_generation_steps),
        .perf_total_cycles              (stage5_total_cycles),
        .perf_per_head_attention_cycles (stage5_per_head_attention_cycles),
        .perf_head_switch_cycles        (unused_head_switch_cycles),
        .perf_provisional_write_cycles  (unused_provisional_write_cycles),
        .perf_cache_read_cycles         (unused_cache_read_cycles),
        .perf_cache_write_cycles        (unused_cache_write_cycles),
        .perf_cache_stall_cycles        (stage5_cache_stall_cycles),
        .perf_commit_cycles             (unused_commit_cycles),
        .perf_pe_stall_cycles           (stage5_pe_stall_cycles),
        .perf_sfu_stall_cycles          (stage5_sfu_stall_cycles),
        .perf_output_stall_cycles       (stage5_output_stall_cycles),
        .perf_peak_valid_seq_len        (perf_peak_valid_seq_len)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(qkv_valid && !qkv_ready && $past(qkv_valid && !qkv_ready) &&
                      !$stable({qkv_head, qkv_dim, qkv_q_fp16, qkv_k_fp16, qkv_v_fp16,
                                qkv_last_dim, qkv_last_head, qkv_meta})))
                else $error("projected_multi_head_attention qkv stream stable under backpressure failed");
            assert (!(output_valid && $isunknown({output_head, output_base_dim, output_vector_fp32,
                                                  output_lane_mask, output_status, output_invalid,
                                                  output_meta, output_last_tile, output_last_head,
                                                  output_last_token})))
                else $error("projected_multi_head_attention no_unknown_output_when_valid failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta, done_valid_seq_len})))
                else $error("projected_multi_head_attention no_unknown_done_when_valid failed");
        end
    end
`endif
endmodule

`default_nettype wire
