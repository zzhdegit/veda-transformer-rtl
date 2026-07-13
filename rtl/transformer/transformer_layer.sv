`default_nettype none

module transformer_layer #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    parameter int PE_NUM = 8,
    parameter int MAX_SEQ_LEN = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter int ATTENTION_PE_ARCH = 0,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int D_FFN = 4 * D_MODEL,
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int FFN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1),
    localparam int CONV_META_W = META_W + MODEL_W + 1,
    localparam int LANE_W = (PE_NUM <= 1) ? 1 : $clog2(PE_NUM)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [2:0]                   weight_kind,
    input  logic [FFN_W-1:0]             weight_output_index,
    input  logic [FFN_W-1:0]             weight_input_index,
    input  logic [15:0]                  weight_data_fp16,
    input  logic                         weight_last,
    input  logic                         weight_commit,

    input  logic                         token_valid,
    output logic                         token_ready,
    input  logic [MODEL_W-1:0]           token_dim,
    input  logic [15:0]                  token_hidden_fp16,
    input  logic                         token_last_dim,
    input  logic [META_W-1:0]            token_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [MODEL_W-1:0]           output_base_dim,
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
    output logic [COUNTER_W-1:0]         perf_total_layer_cycles,
    output logic [COUNTER_W-1:0]         perf_input_load_cycles,
    output logic [COUNTER_W-1:0]         perf_norm1_reduce_cycles,
    output logic [COUNTER_W-1:0]         perf_norm1_apply_cycles,
    output logic [COUNTER_W-1:0]         perf_mha_cycles,
    output logic [COUNTER_W-1:0]         perf_residual1_cycles,
    output logic [COUNTER_W-1:0]         perf_norm2_reduce_cycles,
    output logic [COUNTER_W-1:0]         perf_norm2_apply_cycles,
    output logic [COUNTER_W-1:0]         perf_ffn1_cycles,
    output logic [COUNTER_W-1:0]         perf_relu_cycles,
    output logic [COUNTER_W-1:0]         perf_activation_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_ffn2_cycles,
    output logic [COUNTER_W-1:0]         perf_residual2_cycles,
    output logic [COUNTER_W-1:0]         perf_final_output_cycles,
    output logic [COUNTER_W-1:0]         perf_norm_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_mha_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_ffn_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_weight_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_buffer_stall_cycles,
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
    localparam logic [2:0] KIND_WQ = 3'd0;
    localparam logic [2:0] KIND_WK = 3'd1;
    localparam logic [2:0] KIND_WV = 3'd2;
    localparam logic [2:0] KIND_WO = 3'd3;
    localparam logic [2:0] KIND_NORM1_GAMMA = 3'd4;
    localparam logic [2:0] KIND_NORM2_GAMMA = 3'd5;
    localparam logic [2:0] KIND_FFN_W1 = 3'd6;
    localparam logic [2:0] KIND_FFN_W2 = 3'd7;
    localparam logic [7:0] STATUS_OK = 8'h00;

    typedef enum logic [4:0] {
        ST_LOAD_INPUT,
        ST_START_NORM1,
        ST_NORM1_RUN,
        ST_MHA_RUN,
        ST_RES1_LOAD,
        ST_START_RES1,
        ST_RES1_RUN,
        ST_START_NORM2,
        ST_NORM2_RUN,
        ST_START_FFN,
        ST_FFN_RUN,
        ST_START_RES2,
        ST_RES2_RUN,
        ST_FINAL_DONE
    } state_e;

    state_e state_q = ST_LOAD_INPUT;

    logic [31:0] x_mem [0:D_MODEL-1];
    logic [31:0] mha_mem [0:D_MODEL-1];
    logic [31:0] res1_mem [0:D_MODEL-1];
    logic [31:0] ffn_mem [0:D_MODEL-1];

    logic [MODEL_W-1:0] load_index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [SEQ_LEN_W-1:0] seq_len_q;
    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;
    logic [SEQ_LEN_W-1:0] done_seq_len_q;

    logic final_tile_valid_q;
    logic [MODEL_W-1:0] final_tile_base_q;
    logic [PE_NUM*32-1:0] final_tile_vector_q;
    logic [PE_NUM-1:0] final_tile_mask_q;
    logic [7:0] final_tile_status_q;
    logic final_tile_invalid_q;
    logic [META_W-1:0] final_tile_meta_q;
    logic final_tile_last_q;
    logic [31:0] final_tile_lane_q [0:PE_NUM-1];
    logic [PE_NUM-1:0] final_tile_mask_work;
    logic [PE_NUM*32-1:0] final_tile_vector_work;
    logic res2_done_seen_q;
    logic final_output_seen_q;

    logic conv_in_valid;
    logic conv_in_ready;
    logic [CONV_META_W-1:0] conv_in_meta;
    logic conv_out_valid;
    logic conv_out_ready;
    logic [31:0] conv_out_data;
    logic [CONV_META_W-1:0] conv_out_meta;
    logic conv_out_last;
    logic conv_out_invalid;
    logic conv_out_underflow_or_ftz;
    logic conv_out_last_dim;
    logic [MODEL_W-1:0] conv_out_dim;
    logic [META_W-1:0] conv_out_user_meta;

    logic norm1_gamma_ready;
    logic norm1_input_ready;
    logic norm1_start_ready;
    logic norm1_output_valid;
    logic norm1_output_ready;
    logic [MODEL_W-1:0] norm1_output_dim;
    logic [15:0] norm1_output_data;
    logic [7:0] norm1_output_status;
    logic norm1_output_invalid;
    logic [META_W-1:0] norm1_output_meta;
    logic norm1_output_last;
    logic norm1_done_valid;
    logic norm1_done_ready;
    logic [7:0] norm1_done_status;
    logic norm1_done_invalid;
    logic [META_W-1:0] norm1_done_meta;
    logic [31:0] norm1_debug_sum_sq;
    logic [31:0] norm1_debug_inv_rms;
    logic [COUNTER_W-1:0] norm1_reduce_cycles;
    logic [COUNTER_W-1:0] norm1_apply_cycles;
    logic [COUNTER_W-1:0] norm1_sfu_stall_cycles;
    logic [COUNTER_W-1:0] norm1_output_stall_cycles;

    logic norm2_gamma_ready;
    logic norm2_input_valid;
    logic norm2_input_ready;
    logic [MODEL_W-1:0] norm2_input_dim;
    logic [31:0] norm2_input_data;
    logic norm2_input_last;
    logic [META_W-1:0] norm2_input_meta;
    logic norm2_input_commit;
    logic norm2_start_ready;
    logic norm2_output_valid;
    logic norm2_output_ready;
    logic [MODEL_W-1:0] norm2_output_dim;
    logic [15:0] norm2_output_data;
    logic [7:0] norm2_output_status;
    logic norm2_output_invalid;
    logic [META_W-1:0] norm2_output_meta;
    logic norm2_output_last;
    logic norm2_done_valid;
    logic norm2_done_ready;
    logic [7:0] norm2_done_status;
    logic norm2_done_invalid;
    logic [META_W-1:0] norm2_done_meta;
    logic [31:0] norm2_debug_sum_sq;
    logic [31:0] norm2_debug_inv_rms;
    logic [COUNTER_W-1:0] norm2_reduce_cycles;
    logic [COUNTER_W-1:0] norm2_apply_cycles;
    logic [COUNTER_W-1:0] norm2_sfu_stall_cycles;
    logic [COUNTER_W-1:0] norm2_output_stall_cycles;

    logic stage6_weight_ready;
    logic stage6_token_valid;
    logic stage6_token_ready;
    logic stage6_output_valid;
    logic stage6_output_ready;
    logic [MODEL_W-1:0] stage6_output_base_dim;
    logic [PE_NUM*32-1:0] stage6_output_vector;
    logic [PE_NUM-1:0] stage6_output_mask;
    logic [7:0] stage6_output_status;
    logic stage6_output_invalid;
    logic [META_W-1:0] stage6_output_meta;
    logic stage6_output_last;
    logic stage6_done_valid;
    logic stage6_done_ready;
    logic [7:0] stage6_done_status;
    logic stage6_done_invalid;
    logic [META_W-1:0] stage6_done_meta;
    logic [SEQ_LEN_W-1:0] stage6_done_seq_len;
    logic [SEQ_LEN_W-1:0] stage6_current_seq_len;
    logic [COUNTER_W-1:0] stage6_perf_generation_steps;
    logic [COUNTER_W-1:0] stage6_perf_total_cycles;
    logic [COUNTER_W-1:0] stage6_perf_q_projection_cycles;
    logic [COUNTER_W-1:0] stage6_perf_k_projection_cycles;
    logic [COUNTER_W-1:0] stage6_perf_v_projection_cycles;
    logic [COUNTER_W-1:0] stage6_perf_qkv_quantization_cycles;
    logic [COUNTER_W-1:0] stage6_perf_attention_cycles;
    logic [COUNTER_W-1:0] stage6_perf_concat_quantization_cycles;
    logic [COUNTER_W-1:0] stage6_perf_output_projection_cycles;
    logic [COUNTER_W-1:0] stage6_perf_projection_pe_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_attention_pe_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_sfu_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_output_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_paper_array_active_cycles;
    logic [COUNTER_W-1:0] stage6_perf_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] stage6_perf_inner_mode_cycles;
    logic [COUNTER_W-1:0] stage6_perf_outer_mode_cycles;
    logic [COUNTER_W-1:0] stage6_perf_group0_active_cycles;
    logic [COUNTER_W-1:0] stage6_perf_group1_active_cycles;
    logic [COUNTER_W-1:0] stage6_perf_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] stage6_perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] stage6_perf_array_input_stall_cycles;
    logic [COUNTER_W-1:0] stage6_perf_array_output_stall_cycles;
    logic [SEQ_LEN_W-1:0] stage6_perf_peak_valid_seq_len;

    logic res1_input_valid;
    logic res1_input_ready;
    logic res1_start_ready;
    logic res1_output_valid;
    logic res1_output_ready;
    logic [MODEL_W-1:0] res1_output_dim;
    logic [31:0] res1_output_data;
    logic [7:0] res1_output_status;
    logic res1_output_invalid;
    logic [META_W-1:0] res1_output_meta;
    logic res1_output_last;
    logic res1_done_valid;
    logic res1_done_ready;
    logic [7:0] res1_done_status;
    logic res1_done_invalid;
    logic [META_W-1:0] res1_done_meta;
    logic [COUNTER_W-1:0] res1_add_cycles;
    logic [COUNTER_W-1:0] res1_output_stall_cycles;

    logic ffn_weight_ready;
    logic ffn_input_valid;
    logic ffn_input_ready;
    logic ffn_start_ready;
    logic ffn_output_valid;
    logic ffn_output_ready;
    logic [MODEL_W-1:0] ffn_output_dim;
    logic [31:0] ffn_output_data;
    logic [7:0] ffn_output_status;
    logic ffn_output_invalid;
    logic [META_W-1:0] ffn_output_meta;
    logic ffn_output_last;
    logic ffn_done_valid;
    logic ffn_done_ready;
    logic [7:0] ffn_done_status;
    logic ffn_done_invalid;
    logic [META_W-1:0] ffn_done_meta;
    logic [COUNTER_W-1:0] ffn1_cycles;
    logic [COUNTER_W-1:0] relu_cycles;
    logic [COUNTER_W-1:0] activation_quant_cycles;
    logic [COUNTER_W-1:0] ffn2_cycles;
    logic [COUNTER_W-1:0] ffn_pe_stall_cycles;
    logic [COUNTER_W-1:0] ffn_output_stall_cycles;

    logic res2_input_valid;
    logic res2_input_ready;
    logic res2_start_ready;
    logic res2_output_valid;
    logic res2_output_ready;
    logic [MODEL_W-1:0] res2_output_dim;
    logic [31:0] res2_output_data;
    logic [7:0] res2_output_status;
    logic res2_output_invalid;
    logic [META_W-1:0] res2_output_meta;
    logic res2_output_last;
    logic res2_done_valid;
    logic res2_done_ready;
    logic [7:0] res2_done_status;
    logic res2_done_invalid;
    logic [META_W-1:0] res2_done_meta;
    logic [COUNTER_W-1:0] res2_add_cycles;
    logic [COUNTER_W-1:0] res2_output_stall_cycles;

    wire weight_is_stage6 = weight_kind <= KIND_WO;
    wire weight_is_norm1 = weight_kind == KIND_NORM1_GAMMA;
    wire weight_is_norm2 = weight_kind == KIND_NORM2_GAMMA;
    wire weight_is_ffn = weight_kind == KIND_FFN_W1 || weight_kind == KIND_FFN_W2;
    wire done_fire = done_valid && done_ready;
    wire output_fire = output_valid && output_ready;
    wire conv_out_fire = conv_out_valid && conv_out_ready;
    wire stage6_output_fire = stage6_output_valid && stage6_output_ready;
    wire stage6_done_fire = stage6_done_valid && stage6_done_ready;
    wire res1_input_fire = res1_input_valid && res1_input_ready;
    wire res1_output_fire = res1_output_valid && res1_output_ready;
    wire res1_done_fire = res1_done_valid && res1_done_ready;
    wire norm2_output_fire = norm2_output_valid && norm2_output_ready;
    wire norm2_done_fire = norm2_done_valid && norm2_done_ready;
    wire ffn_output_fire = ffn_output_valid && ffn_output_ready;
    wire ffn_done_fire = ffn_done_valid && ffn_done_ready;
    wire res2_output_fire = res2_output_valid && res2_output_ready;
    wire res2_done_fire = res2_done_valid && res2_done_ready;
    wire [LANE_W-1:0] final_lane = LANE_W'(int'(res2_output_dim) % PE_NUM);
    wire final_tile_first_lane = (int'(res2_output_dim) % PE_NUM) == 0;
    wire final_tile_last_lane = (int'(res2_output_dim) % PE_NUM) == (PE_NUM - 1);
    wire final_scalar_last = res2_output_dim == MODEL_W'(D_MODEL - 1);
    wire final_tile_complete = final_tile_last_lane || final_scalar_last;

    initial begin
        if (D_MODEL <= 0 || D_FFN <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "transformer_layer parameters must be positive");
        end
        if (D_MODEL != N_HEAD * D_HEAD) begin
            $fatal(1, "transformer_layer D_MODEL relation failed");
        end
    end

    assign weight_ready =
        (state_q == ST_LOAD_INPUT) &&
        ((weight_is_stage6 && stage6_weight_ready) ||
         (weight_is_norm1 && norm1_gamma_ready) ||
         (weight_is_norm2 && norm2_gamma_ready) ||
         (weight_is_ffn && ffn_weight_ready));

    assign token_ready = (state_q == ST_LOAD_INPUT) && conv_in_ready && !done_valid_q;
    assign conv_in_valid = (state_q == ST_LOAD_INPUT) && token_valid && token_ready;
    assign conv_in_meta = {token_last_dim, token_dim, token_meta};
    assign conv_out_ready = (state_q == ST_LOAD_INPUT) && norm1_input_ready;
    assign conv_out_last_dim = conv_out_meta[CONV_META_W-1];
    assign conv_out_dim = conv_out_meta[META_W +: MODEL_W];
    assign conv_out_user_meta = conv_out_meta[META_W-1:0];

    assign stage6_token_valid = (state_q == ST_NORM1_RUN) && norm1_output_valid;
    assign norm1_output_ready = (state_q == ST_NORM1_RUN) && stage6_token_ready;
    assign norm1_done_ready = state_q == ST_NORM1_RUN;
    assign stage6_output_ready = state_q == ST_MHA_RUN;
    assign stage6_done_ready = state_q == ST_MHA_RUN;

    assign res1_input_valid = state_q == ST_RES1_LOAD;
    assign res1_output_ready = (state_q == ST_RES1_RUN) && norm2_input_ready;
    assign res1_done_ready = state_q == ST_RES1_RUN;
    assign norm2_input_valid = (state_q == ST_RES1_RUN) && res1_output_valid;
    assign norm2_input_dim = res1_output_dim;
    assign norm2_input_data = res1_output_data;
    assign norm2_input_last = res1_output_last;
    assign norm2_input_meta = res1_output_meta;
    assign norm2_input_commit = res1_output_fire && res1_output_last;

    assign norm2_output_ready = (state_q == ST_NORM2_RUN) && ffn_input_ready;
    assign norm2_done_ready = state_q == ST_NORM2_RUN;
    assign ffn_input_valid = (state_q == ST_NORM2_RUN) && norm2_output_valid;

    assign ffn_output_ready = (state_q == ST_FFN_RUN) && res2_input_ready;
    assign ffn_done_ready = state_q == ST_FFN_RUN;
    assign res2_input_valid = (state_q == ST_FFN_RUN) && ffn_output_valid;

    assign res2_output_ready = (state_q == ST_RES2_RUN) && !final_tile_valid_q;
    assign res2_done_ready = state_q == ST_RES2_RUN;

    assign output_valid = final_tile_valid_q;
    assign output_base_dim = final_tile_base_q;
    assign output_vector_fp32 = final_tile_vector_q;
    assign output_lane_mask = final_tile_mask_q;
    assign output_status = final_tile_status_q;
    assign output_invalid = final_tile_invalid_q;
    assign output_meta = final_tile_meta_q;
    assign output_last = final_tile_last_q;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;
    assign done_valid_seq_len = done_seq_len_q;
    assign current_valid_seq_len = stage6_current_seq_len;
    assign perf_peak_valid_seq_len = stage6_perf_peak_valid_seq_len;

    assign perf_norm1_reduce_cycles = norm1_reduce_cycles;
    assign perf_norm1_apply_cycles = norm1_apply_cycles;
    assign perf_mha_cycles = stage6_perf_total_cycles;
    assign perf_residual1_cycles = res1_add_cycles;
    assign perf_norm2_reduce_cycles = norm2_reduce_cycles;
    assign perf_norm2_apply_cycles = norm2_apply_cycles;
    assign perf_ffn1_cycles = ffn1_cycles;
    assign perf_relu_cycles = relu_cycles;
    assign perf_activation_quantization_cycles = activation_quant_cycles;
    assign perf_ffn2_cycles = ffn2_cycles;
    assign perf_residual2_cycles = res2_add_cycles;
    assign perf_norm_stall_cycles = norm1_sfu_stall_cycles + norm2_sfu_stall_cycles;
    assign perf_mha_stall_cycles = stage6_perf_attention_pe_stall_cycles;
    assign perf_ffn_pe_stall_cycles = ffn_pe_stall_cycles;
    assign perf_weight_stall_cycles = stage6_perf_weight_stall_cycles;
    assign perf_buffer_stall_cycles = stage6_perf_buffer_stall_cycles;
    assign perf_output_stall_cycles = stage6_perf_output_stall_cycles + norm1_output_stall_cycles +
                                      norm2_output_stall_cycles + res1_output_stall_cycles +
                                      res2_output_stall_cycles + ffn_output_stall_cycles;
    assign perf_paper_array_active_cycles = stage6_perf_paper_array_active_cycles;
    assign perf_paper_array_idle_cycles = stage6_perf_paper_array_idle_cycles;
    assign perf_inner_mode_cycles = stage6_perf_inner_mode_cycles;
    assign perf_outer_mode_cycles = stage6_perf_outer_mode_cycles;
    assign perf_group0_active_cycles = stage6_perf_group0_active_cycles;
    assign perf_group1_active_cycles = stage6_perf_group1_active_cycles;
    assign perf_tail_masked_pe_cycles = stage6_perf_tail_masked_pe_cycles;
    assign perf_mode_switch_cycles = stage6_perf_mode_switch_cycles;
    assign perf_array_input_stall_cycles = stage6_perf_array_input_stall_cycles;
    assign perf_array_output_stall_cycles = stage6_perf_array_output_stall_cycles;

    always_comb begin
        final_tile_vector_work = '0;
        final_tile_mask_work = final_tile_mask_q;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            final_tile_vector_work[lane*32 +: 32] = final_tile_lane_q[lane];
        end
        final_tile_vector_work[int'(final_lane)*32 +: 32] = res2_output_data;
        final_tile_mask_work[int'(final_lane)] = 1'b1;
    end

    fp16_to_fp32 #(
        .META_W(CONV_META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_input_convert (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (conv_in_valid),
        .in_ready             (conv_in_ready),
        .in_data              (token_hidden_fp16),
        .in_meta              (conv_in_meta),
        .in_last              (token_last_dim),
        .out_valid            (conv_out_valid),
        .out_ready            (conv_out_ready),
        .out_data             (conv_out_data),
        .out_meta             (conv_out_meta),
        .out_last             (conv_out_last),
        .out_invalid          (conv_out_invalid),
        .out_underflow_or_ftz (conv_out_underflow_or_ftz)
    );

    rmsnorm_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_norm1 (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .clear                    (1'b0),
        .gamma_valid              (weight_valid && weight_ready && weight_is_norm1),
        .gamma_ready              (norm1_gamma_ready),
        .gamma_dim                (MODEL_W'(weight_output_index)),
        .gamma_data_fp16          (weight_data_fp16),
        .gamma_commit             (weight_commit),
        .input_valid              (conv_out_valid && conv_out_ready),
        .input_ready              (norm1_input_ready),
        .input_dim                (conv_out_dim),
        .input_data_fp32          (conv_out_data),
        .input_last               (conv_out_last_dim),
        .input_meta               (conv_out_user_meta),
        .input_commit             (conv_out_fire && conv_out_last_dim),
        .start_valid              (state_q == ST_START_NORM1),
        .start_ready              (norm1_start_ready),
        .start_meta               (meta_q),
        .output_valid             (norm1_output_valid),
        .output_ready             (norm1_output_ready),
        .output_dim               (norm1_output_dim),
        .output_data_fp16         (norm1_output_data),
        .output_status            (norm1_output_status),
        .output_invalid           (norm1_output_invalid),
        .output_meta              (norm1_output_meta),
        .output_last              (norm1_output_last),
        .done_valid               (norm1_done_valid),
        .done_ready               (norm1_done_ready),
        .done_status              (norm1_done_status),
        .done_invalid             (norm1_done_invalid),
        .done_meta                (norm1_done_meta),
        .debug_sum_sq             (norm1_debug_sum_sq),
        .debug_inv_rms            (norm1_debug_inv_rms),
        .perf_reduce_cycles       (norm1_reduce_cycles),
        .perf_apply_cycles        (norm1_apply_cycles),
        .perf_sfu_stall_cycles    (norm1_sfu_stall_cycles),
        .perf_output_stall_cycles (norm1_output_stall_cycles)
    );

    projection_integrated_mha #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(ATTENTION_PE_ARCH),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_mha (
        .clk                                  (clk),
        .rst_n                                (rst_n),
        .weight_valid                         (weight_valid && weight_ready && weight_is_stage6),
        .weight_ready                         (stage6_weight_ready),
        .weight_kind                          (weight_kind[1:0]),
        .weight_output_index                  (MODEL_W'(weight_output_index)),
        .weight_input_index                   (MODEL_W'(weight_input_index)),
        .weight_data_fp16                     (weight_data_fp16),
        .weight_last                          (weight_last),
        .weight_commit                        (weight_commit),
        .token_valid                          (stage6_token_valid),
        .token_ready                          (stage6_token_ready),
        .token_dim                            (norm1_output_dim),
        .token_hidden_fp16                    (norm1_output_data),
        .token_last_dim                       (norm1_output_last),
        .token_meta                           (norm1_output_meta),
        .output_valid                         (stage6_output_valid),
        .output_ready                         (stage6_output_ready),
        .output_base_dim                      (stage6_output_base_dim),
        .output_vector_fp32                   (stage6_output_vector),
        .output_lane_mask                     (stage6_output_mask),
        .output_status                        (stage6_output_status),
        .output_invalid                       (stage6_output_invalid),
        .output_meta                          (stage6_output_meta),
        .output_last                          (stage6_output_last),
        .done_valid                           (stage6_done_valid),
        .done_ready                           (stage6_done_ready),
        .done_status                          (stage6_done_status),
        .done_invalid                         (stage6_done_invalid),
        .done_meta                            (stage6_done_meta),
        .done_valid_seq_len                   (stage6_done_seq_len),
        .current_valid_seq_len                (stage6_current_seq_len),
        .perf_generation_steps                (stage6_perf_generation_steps),
        .perf_total_cycles                    (stage6_perf_total_cycles),
        .perf_q_projection_cycles             (stage6_perf_q_projection_cycles),
        .perf_k_projection_cycles             (stage6_perf_k_projection_cycles),
        .perf_v_projection_cycles             (stage6_perf_v_projection_cycles),
        .perf_qkv_quantization_cycles         (stage6_perf_qkv_quantization_cycles),
        .perf_attention_cycles                (stage6_perf_attention_cycles),
        .perf_concat_quantization_cycles      (stage6_perf_concat_quantization_cycles),
        .perf_output_projection_cycles        (stage6_perf_output_projection_cycles),
        .perf_projection_pe_stall_cycles      (stage6_perf_projection_pe_stall_cycles),
        .perf_attention_pe_stall_cycles       (stage6_perf_attention_pe_stall_cycles),
        .perf_sfu_stall_cycles                (stage6_perf_sfu_stall_cycles),
        .perf_weight_stall_cycles             (stage6_perf_weight_stall_cycles),
        .perf_buffer_stall_cycles             (stage6_perf_buffer_stall_cycles),
        .perf_output_stall_cycles             (stage6_perf_output_stall_cycles),
        .perf_paper_array_active_cycles       (stage6_perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles         (stage6_perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles               (stage6_perf_inner_mode_cycles),
        .perf_outer_mode_cycles               (stage6_perf_outer_mode_cycles),
        .perf_group0_active_cycles            (stage6_perf_group0_active_cycles),
        .perf_group1_active_cycles            (stage6_perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles           (stage6_perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles              (stage6_perf_mode_switch_cycles),
        .perf_array_input_stall_cycles        (stage6_perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles       (stage6_perf_array_output_stall_cycles),
        .perf_peak_valid_seq_len              (stage6_perf_peak_valid_seq_len)
    );

    residual_add_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_residual1 (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .clear                    (1'b0),
        .input_valid              (res1_input_valid),
        .input_ready              (res1_input_ready),
        .input_dim                (load_index_q),
        .input_lhs_fp32           (x_mem[int'(load_index_q)]),
        .input_rhs_fp32           (mha_mem[int'(load_index_q)]),
        .input_last               (load_index_q == MODEL_W'(D_MODEL - 1)),
        .input_meta               (meta_q),
        .input_commit             (res1_input_fire && load_index_q == MODEL_W'(D_MODEL - 1)),
        .start_valid              (state_q == ST_START_RES1),
        .start_ready              (res1_start_ready),
        .start_meta               (meta_q),
        .output_valid             (res1_output_valid),
        .output_ready             (res1_output_ready),
        .output_dim               (res1_output_dim),
        .output_data_fp32         (res1_output_data),
        .output_status            (res1_output_status),
        .output_invalid           (res1_output_invalid),
        .output_meta              (res1_output_meta),
        .output_last              (res1_output_last),
        .done_valid               (res1_done_valid),
        .done_ready               (res1_done_ready),
        .done_status              (res1_done_status),
        .done_invalid             (res1_done_invalid),
        .done_meta                (res1_done_meta),
        .perf_add_cycles          (res1_add_cycles),
        .perf_output_stall_cycles (res1_output_stall_cycles)
    );

    rmsnorm_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_norm2 (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .clear                    (1'b0),
        .gamma_valid              (weight_valid && weight_ready && weight_is_norm2),
        .gamma_ready              (norm2_gamma_ready),
        .gamma_dim                (MODEL_W'(weight_output_index)),
        .gamma_data_fp16          (weight_data_fp16),
        .gamma_commit             (weight_commit),
        .input_valid              (norm2_input_valid),
        .input_ready              (norm2_input_ready),
        .input_dim                (norm2_input_dim),
        .input_data_fp32          (norm2_input_data),
        .input_last               (norm2_input_last),
        .input_meta               (norm2_input_meta),
        .input_commit             (norm2_input_commit),
        .start_valid              (state_q == ST_START_NORM2),
        .start_ready              (norm2_start_ready),
        .start_meta               (meta_q),
        .output_valid             (norm2_output_valid),
        .output_ready             (norm2_output_ready),
        .output_dim               (norm2_output_dim),
        .output_data_fp16         (norm2_output_data),
        .output_status            (norm2_output_status),
        .output_invalid           (norm2_output_invalid),
        .output_meta              (norm2_output_meta),
        .output_last              (norm2_output_last),
        .done_valid               (norm2_done_valid),
        .done_ready               (norm2_done_ready),
        .done_status              (norm2_done_status),
        .done_invalid             (norm2_done_invalid),
        .done_meta                (norm2_done_meta),
        .debug_sum_sq             (norm2_debug_sum_sq),
        .debug_inv_rms            (norm2_debug_inv_rms),
        .perf_reduce_cycles       (norm2_reduce_cycles),
        .perf_apply_cycles        (norm2_apply_cycles),
        .perf_sfu_stall_cycles    (norm2_sfu_stall_cycles),
        .perf_output_stall_cycles (norm2_output_stall_cycles)
    );

    ffn_engine #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_ffn (
        .clk                                  (clk),
        .rst_n                                (rst_n),
        .clear                                (1'b0),
        .weight_valid                         (weight_valid && weight_ready && weight_is_ffn),
        .weight_ready                         (ffn_weight_ready),
        .weight_kind                          (weight_kind == KIND_FFN_W2),
        .weight_output_index                  (weight_output_index),
        .weight_input_index                   (weight_input_index),
        .weight_data_fp16                     (weight_data_fp16),
        .weight_commit                        (weight_commit),
        .input_valid                          (ffn_input_valid),
        .input_ready                          (ffn_input_ready),
        .input_dim                            (norm2_output_dim),
        .input_data_fp16                      (norm2_output_data),
        .input_last                           (norm2_output_last),
        .input_meta                           (norm2_output_meta),
        .input_commit                         (norm2_output_fire && norm2_output_last),
        .start_valid                          (state_q == ST_START_FFN),
        .start_ready                          (ffn_start_ready),
        .start_meta                           (meta_q),
        .output_valid                         (ffn_output_valid),
        .output_ready                         (ffn_output_ready),
        .output_dim                           (ffn_output_dim),
        .output_data_fp32                     (ffn_output_data),
        .output_status                        (ffn_output_status),
        .output_invalid                       (ffn_output_invalid),
        .output_meta                          (ffn_output_meta),
        .output_last                          (ffn_output_last),
        .done_valid                           (ffn_done_valid),
        .done_ready                           (ffn_done_ready),
        .done_status                          (ffn_done_status),
        .done_invalid                         (ffn_done_invalid),
        .done_meta                            (ffn_done_meta),
        .perf_ffn1_cycles                     (ffn1_cycles),
        .perf_relu_cycles                     (relu_cycles),
        .perf_activation_quantization_cycles  (activation_quant_cycles),
        .perf_ffn2_cycles                     (ffn2_cycles),
        .perf_pe_stall_cycles                 (ffn_pe_stall_cycles),
        .perf_output_stall_cycles             (ffn_output_stall_cycles)
    );

    residual_add_engine #(
        .D_MODEL(D_MODEL),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_residual2 (
        .clk                      (clk),
        .rst_n                    (rst_n),
        .clear                    (1'b0),
        .input_valid              (res2_input_valid),
        .input_ready              (res2_input_ready),
        .input_dim                (ffn_output_dim),
        .input_lhs_fp32           (res1_mem[int'(ffn_output_dim)]),
        .input_rhs_fp32           (ffn_output_data),
        .input_last               (ffn_output_last),
        .input_meta               (ffn_output_meta),
        .input_commit             (ffn_output_fire && ffn_output_last),
        .start_valid              (state_q == ST_START_RES2),
        .start_ready              (res2_start_ready),
        .start_meta               (meta_q),
        .output_valid             (res2_output_valid),
        .output_ready             (res2_output_ready),
        .output_dim               (res2_output_dim),
        .output_data_fp32         (res2_output_data),
        .output_status            (res2_output_status),
        .output_invalid           (res2_output_invalid),
        .output_meta              (res2_output_meta),
        .output_last              (res2_output_last),
        .done_valid               (res2_done_valid),
        .done_ready               (res2_done_ready),
        .done_status              (res2_done_status),
        .done_invalid             (res2_done_invalid),
        .done_meta                (res2_done_meta),
        .perf_add_cycles          (res2_add_cycles),
        .perf_output_stall_cycles (res2_output_stall_cycles)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_LOAD_INPUT;
            load_index_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            seq_len_q <= '0;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            done_seq_len_q <= '0;
            final_tile_valid_q <= 1'b0;
            final_tile_base_q <= '0;
            final_tile_vector_q <= '0;
            final_tile_mask_q <= '0;
            final_tile_status_q <= STATUS_OK;
            final_tile_invalid_q <= 1'b0;
            final_tile_meta_q <= '0;
            final_tile_last_q <= 1'b0;
            res2_done_seen_q <= 1'b0;
            final_output_seen_q <= 1'b0;
            perf_generation_steps <= '0;
            perf_total_layer_cycles <= '0;
            perf_input_load_cycles <= '0;
            perf_final_output_cycles <= '0;
            for (int dim = 0; dim < D_MODEL; dim++) begin
                x_mem[dim] <= 32'd0;
                mha_mem[dim] <= 32'd0;
                res1_mem[dim] <= 32'd0;
                ffn_mem[dim] <= 32'd0;
            end
            for (int lane = 0; lane < PE_NUM; lane++) begin
                final_tile_lane_q[lane] <= 32'd0;
            end
        end else begin
            if (done_fire) begin
                done_valid_q <= 1'b0;
                done_status_q <= STATUS_OK;
                done_invalid_q <= 1'b0;
                done_meta_q <= '0;
                state_q <= ST_LOAD_INPUT;
            end

            if (output_fire) begin
                final_tile_valid_q <= 1'b0;
                final_tile_mask_q <= '0;
                if (final_tile_last_q) begin
                    final_output_seen_q <= 1'b1;
                end
            end

            if (conv_out_fire) begin
                x_mem[int'(conv_out_dim)] <= conv_out_data;
                meta_q <= conv_out_user_meta;
                status_q <= STATUS_OK;
                invalid_q <= conv_out_invalid;
                if (conv_out_last_dim) begin
                    state_q <= ST_START_NORM1;
                end
            end

            if (stage6_output_fire) begin
                for (int lane = 0; lane < PE_NUM; lane++) begin
                    if (stage6_output_mask[lane] &&
                        (int'(stage6_output_base_dim) + lane) < D_MODEL) begin
                        mha_mem[int'(stage6_output_base_dim) + lane] <= stage6_output_vector[lane*32 +: 32];
                    end
                end
            end

            if (res1_output_fire) begin
                res1_mem[int'(res1_output_dim)] <= res1_output_data;
                status_q <= status_q | res1_output_status;
                invalid_q <= invalid_q | res1_output_invalid;
            end

            if (ffn_output_fire) begin
                ffn_mem[int'(ffn_output_dim)] <= ffn_output_data;
            end

            if (res2_output_fire) begin
                if (final_tile_first_lane) begin
                    final_tile_base_q <= res2_output_dim;
                    final_tile_mask_q <= '0;
                end
                final_tile_lane_q[int'(final_lane)] <= res2_output_data;
                final_tile_mask_q[int'(final_lane)] <= 1'b1;
                status_q <= status_q | res2_output_status;
                invalid_q <= invalid_q | res2_output_invalid;
                if (final_tile_complete) begin
                    final_tile_valid_q <= 1'b1;
                    final_tile_vector_q <= final_tile_vector_work;
                    final_tile_mask_q <= final_tile_mask_work;
                    final_tile_status_q <= status_q | res2_output_status;
                    final_tile_invalid_q <= invalid_q | res2_output_invalid;
                    final_tile_meta_q <= res2_output_meta;
                    final_tile_last_q <= final_scalar_last;
                end
            end

            if (res2_done_fire) begin
                res2_done_seen_q <= 1'b1;
                status_q <= status_q | res2_done_status;
                invalid_q <= invalid_q | res2_done_invalid;
            end

            unique case (state_q)
                ST_LOAD_INPUT: begin
                    if (token_valid && token_ready) begin
                        perf_input_load_cycles <= perf_input_load_cycles + COUNTER_W'(1);
                    end
                end

                ST_START_NORM1: begin
                    if (norm1_start_ready) begin
                        state_q <= ST_NORM1_RUN;
                    end
                end

                ST_NORM1_RUN: begin
                    if (norm1_done_valid && norm1_done_ready) begin
                        status_q <= status_q | norm1_done_status;
                        invalid_q <= invalid_q | norm1_done_invalid;
                        state_q <= ST_MHA_RUN;
                    end
                end

                ST_MHA_RUN: begin
                    if (stage6_done_fire) begin
                        status_q <= status_q | stage6_done_status;
                        invalid_q <= invalid_q | stage6_done_invalid;
                        meta_q <= stage6_done_meta;
                        seq_len_q <= stage6_done_seq_len;
                        if (stage6_done_invalid) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= status_q | stage6_done_status;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= stage6_done_meta;
                            done_seq_len_q <= stage6_done_seq_len;
                            state_q <= ST_FINAL_DONE;
                        end else begin
                            load_index_q <= '0;
                            state_q <= ST_RES1_LOAD;
                        end
                    end
                end

                ST_RES1_LOAD: begin
                    if (res1_input_fire) begin
                        if (load_index_q == MODEL_W'(D_MODEL - 1)) begin
                            state_q <= ST_START_RES1;
                        end else begin
                            load_index_q <= load_index_q + MODEL_W'(1);
                        end
                    end
                end

                ST_START_RES1: begin
                    if (res1_start_ready) begin
                        state_q <= ST_RES1_RUN;
                    end
                end

                ST_RES1_RUN: begin
                    if (res1_done_fire) begin
                        status_q <= status_q | res1_done_status;
                        invalid_q <= invalid_q | res1_done_invalid;
                        state_q <= ST_START_NORM2;
                    end
                end

                ST_START_NORM2: begin
                    if (norm2_start_ready) begin
                        state_q <= ST_NORM2_RUN;
                    end
                end

                ST_NORM2_RUN: begin
                    if (norm2_done_fire) begin
                        status_q <= status_q | norm2_done_status;
                        invalid_q <= invalid_q | norm2_done_invalid;
                        state_q <= ST_START_FFN;
                    end
                end

                ST_START_FFN: begin
                    if (ffn_start_ready) begin
                        state_q <= ST_FFN_RUN;
                    end
                end

                ST_FFN_RUN: begin
                    if (ffn_done_fire) begin
                        status_q <= status_q | ffn_done_status;
                        invalid_q <= invalid_q | ffn_done_invalid;
                        state_q <= ST_START_RES2;
                    end
                end

                ST_START_RES2: begin
                    if (res2_start_ready) begin
                        res2_done_seen_q <= 1'b0;
                        final_output_seen_q <= 1'b0;
                        final_tile_valid_q <= 1'b0;
                        final_tile_mask_q <= '0;
                        state_q <= ST_RES2_RUN;
                    end
                end

                ST_RES2_RUN: begin
                    if (output_fire) begin
                        perf_final_output_cycles <= perf_final_output_cycles + COUNTER_W'(1);
                    end
                    if ((output_fire && final_tile_last_q && (res2_done_seen_q || res2_done_fire)) ||
                        (res2_done_fire && final_output_seen_q)) begin
                        done_valid_q <= 1'b1;
                        done_status_q <= status_q | res2_done_status;
                        done_invalid_q <= invalid_q | res2_done_invalid;
                        done_meta_q <= meta_q;
                        done_seq_len_q <= seq_len_q;
                        perf_generation_steps <= perf_generation_steps + COUNTER_W'(1);
                        state_q <= ST_FINAL_DONE;
                    end
                end

                ST_FINAL_DONE: begin
                    if (!done_valid_q) begin
                        state_q <= ST_LOAD_INPUT;
                    end
                end

                default: state_q <= ST_LOAD_INPUT;
            endcase

            if (state_q != ST_LOAD_INPUT || token_valid) begin
                perf_total_layer_cycles <= perf_total_layer_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (D_MODEL == N_HEAD * D_HEAD)
                else $error("transformer_layer d_model_equals_n_head_times_d_head failed");
            if (output_valid) begin
                assert (!$isunknown({output_base_dim, output_vector_fp32, output_lane_mask,
                                     output_status, output_invalid, output_meta, output_last}))
                    else $error("transformer_layer no_unknown_output_when_valid failed");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("transformer_layer output valid dropped under backpressure");
                assert ($stable({output_base_dim, output_vector_fp32, output_lane_mask,
                                 output_status, output_invalid, output_meta, output_last}))
                    else $error("transformer_layer output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
