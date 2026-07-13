`default_nettype none

module paper_interleaved_attention_datapath #(
    parameter int PE_NUM = 8,
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int LANE_COUNT_W = $clog2(PE_NUM + 1),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1),
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN),
    localparam int D_ADDR_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int TILE_COUNT = (D_HEAD + PE_NUM - 1) / PE_NUM,
    localparam int TILE_W = (TILE_COUNT <= 1) ? 1 : $clog2(TILE_COUNT),
    localparam int KV_DEPTH = MAX_SEQ_LEN * D_HEAD
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         load_valid,
    output logic                         load_ready,
    input  logic [1:0]                   load_kind,
    input  logic [TOKEN_W-1:0]           load_token,
    input  logic [D_ADDR_W-1:0]          load_dim,
    input  logic [15:0]                  load_data,

    input  logic                         start_valid,
    output logic                         start_ready,
    input  logic [SEQ_LEN_W-1:0]         start_seq_len,
    input  logic [META_W-1:0]            start_meta,

    output logic                         output_valid,
    input  logic                         output_ready,
    output logic [D_ADDR_W-1:0]          output_base_dim,
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

    output logic [COUNTER_W-1:0]         perf_total_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_qk_cycles,
    output logic [COUNTER_W-1:0]         perf_qk_pe_busy_cycles,
    output logic [COUNTER_W-1:0]         perf_scale_cycles,
    output logic [COUNTER_W-1:0]         perf_reduction_cycles,
    output logic [COUNTER_W-1:0]         perf_reduction_finalize_cycles,
    output logic [COUNTER_W-1:0]         perf_normalization_cycles,
    output logic [COUNTER_W-1:0]         perf_sv_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_buffer_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_score_buffer_peak_occupancy,
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

    output logic [COUNTER_W-1:0]         perf_qk_sfu_overlap_cycles,
    output logic [COUNTER_W-1:0]         perf_qk_only_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_during_qk_cycles,
    output logic [COUNTER_W-1:0]         perf_score_fifo_full_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_score_fifo_empty_cycles,
    output logic [COUNTER_W-1:0]         perf_score_fifo_peak_occupancy,
    output logic [COUNTER_W-1:0]         perf_sfu_sv_overlap_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_only_cycles,
    output logic [COUNTER_W-1:0]         perf_sv_only_cycles,
    output logic [COUNTER_W-1:0]         perf_probability_fifo_full_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_probability_fifo_empty_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_probability_fifo_peak_occupancy,
    output logic [COUNTER_W-1:0]         perf_inner_to_outer_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_pipeline_bubble_cycles
);
    localparam logic [1:0] LOAD_Q = 2'd0;
    localparam logic [1:0] LOAD_K = 2'd1;
    localparam logic [1:0] LOAD_V = 2'd2;
    localparam logic [1:0] MODE_INNER_PRODUCT = 2'd1;
    localparam logic [1:0] MODE_OUTER_PRODUCT = 2'd2;

    typedef enum logic [2:0] {
        PH_IDLE,
        PH_QK,
        PH_NORM_START,
        PH_OUTER,
        PH_OUTPUT,
        PH_DONE,
        PH_ERROR
    } phase_e;

    typedef enum logic [2:0] {
        QK_IDLE,
        QK_SEND_ARRAY,
        QK_WAIT_ARRAY,
        QK_SCALE_SEND,
        QK_SCALE_WAIT,
        QK_DONE
    } qk_state_e;

    typedef enum logic [1:0] {
        SV_IDLE,
        SV_SEND_ARRAY,
        SV_WAIT_ARRAY,
        SV_DONE
    } sv_state_e;

    phase_e phase_q;
    qk_state_e qk_state_q;
    sv_state_e sv_state_q;

    logic [SEQ_LEN_W-1:0] seq_len_q;
    logic [SEQ_LEN_W-1:0] qk_token_q;
    logic [SEQ_LEN_W-1:0] reduce_token_q;
    logic [SEQ_LEN_W-1:0] norm_token_q;
    logic [SEQ_LEN_W-1:0] sv_token_q;
    logic [SEQ_LEN_W-1:0] score_valid_count_q;
    logic [SEQ_LEN_W-1:0] prob_valid_count_q;
    logic [TILE_W-1:0] output_tile_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic active_q;
    logic softmax_final_seen_q;
    logic [31:0] final_max_q;
    logic [31:0] final_exp_sum_q;
    logic array_result_seen_q;
    logic array_done_seen_q;
    logic [31:0] qk_raw_score_q;
    logic [128*32-1:0] full_output_vector_q;

    logic output_valid_q;
    logic [D_ADDR_W-1:0] output_base_dim_q;
    logic [PE_NUM*32-1:0] output_vector_q;
    logic [PE_NUM-1:0] output_lane_mask_q;
    logic [7:0] output_status_q;
    logic output_invalid_q;
    logic [META_W-1:0] output_meta_q;
    logic output_last_q;
    logic done_valid_q;

    logic [15:0] q_mem [0:D_HEAD-1];
    logic [15:0] k_mem [0:KV_DEPTH-1];
    logic [15:0] v_mem [0:KV_DEPTH-1];
    logic [31:0] score_mem [0:MAX_SEQ_LEN-1];
    logic [31:0] prob_mem [0:MAX_SEQ_LEN-1];

    logic [127:0] native_lane_mask_comb;
    logic [1:0] native_group_mask_comb;
    logic [128*16-1:0] qk_operand_a_comb;
    logic [128*16-1:0] qk_operand_b_comb;
    logic [128*16-1:0] sv_operand_b_comb;
    logic [PE_NUM*32-1:0] output_vector_comb;
    logic [PE_NUM-1:0] output_lane_mask_comb;

    logic array_cmd_valid;
    logic array_cmd_ready;
    logic [1:0] array_cmd_mode;
    logic array_cmd_clear_acc;
    logic array_cmd_tile_last;
    logic [31:0] array_cmd_scalar;
    logic [128*16-1:0] array_cmd_operand_a;
    logic [128*16-1:0] array_cmd_operand_b;
    logic array_cmd_last;
    logic array_result_valid;
    logic array_result_ready;
    logic [1:0] array_result_mode;
    logic [15:0] array_result_tile_id;
    logic [31:0] array_result_scalar_fp32;
    logic [128*32-1:0] array_result_vector_fp32;
    logic [127:0] array_result_lane_mask;
    logic [7:0] array_result_status;
    logic array_result_invalid;
    logic [META_W-1:0] array_result_meta;
    logic array_result_last;
    logic array_done_valid;
    logic array_done_ready;
    logic [7:0] array_done_status;
    logic array_done_invalid;
    logic [META_W-1:0] array_done_meta;

    logic scaler_in_valid;
    logic scaler_in_ready;
    logic scaler_out_valid;
    logic scaler_out_ready;
    logic [31:0] scaler_out_score;
    logic [7:0] scaler_out_status;
    logic scaler_out_invalid;

    logic reduction_in_valid;
    logic reduction_in_ready;
    logic reduction_final_valid;
    logic reduction_final_ready;
    logic [31:0] reduction_final_max;
    logic [31:0] reduction_final_exp_sum;
    logic [7:0] reduction_final_status;
    logic reduction_final_invalid;
    logic reduction_busy;
    logic [31:0] reduction_processed_count;

    logic norm_start_valid;
    logic norm_start_ready;
    logic norm_score_valid;
    logic norm_score_ready;
    logic norm_prob_valid;
    logic norm_prob_ready;
    logic [31:0] norm_prob_value;
    logic [TOKEN_W-1:0] norm_prob_index;
    logic norm_prob_last;
    logic [7:0] norm_prob_status;
    logic norm_prob_invalid;
    logic norm_busy;

    logic [SEQ_LEN_W-1:0] score_fifo_occupancy;
    logic [SEQ_LEN_W-1:0] prob_fifo_occupancy;

    wire load_fire = load_valid && load_ready;
    wire start_fire = start_valid && start_ready;
    wire output_fire = output_valid && output_ready;
    wire done_fire = done_valid && done_ready;
    wire array_cmd_fire = array_cmd_valid && array_cmd_ready;
    wire array_result_fire = array_result_valid && array_result_ready;
    wire array_done_fire = array_done_valid && array_done_ready;
    wire scaler_in_fire = scaler_in_valid && scaler_in_ready;
    wire scaler_out_fire = scaler_out_valid && scaler_out_ready;
    wire reduction_in_fire = reduction_in_valid && reduction_in_ready;
    wire reduction_final_fire = reduction_final_valid && reduction_final_ready;
    wire norm_start_fire = norm_start_valid && norm_start_ready;
    wire norm_score_fire = norm_score_valid && norm_score_ready;
    wire norm_prob_fire = norm_prob_valid && norm_prob_ready;
    wire qk_active = (phase_q == PH_QK) && (qk_state_q != QK_IDLE) && (qk_state_q != QK_DONE);
    wire sfu_during_qk = (phase_q == PH_QK) && (reduction_busy || reduction_in_valid || reduction_final_valid);
    wire norm_active = (phase_q == PH_OUTER) && (norm_busy || norm_score_valid || norm_prob_valid);
    wire sv_active = (phase_q == PH_OUTER) && (sv_state_q inside {SV_SEND_ARRAY, SV_WAIT_ARRAY});

    initial begin
        if (PE_NUM <= 0 || D_HEAD <= 0 || D_HEAD > 128 || MAX_SEQ_LEN <= 0 ||
            META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "paper_interleaved_attention_datapath parameter range failed");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "paper_interleaved_attention_datapath PE_NUM must be a power of two");
        end
    end

    function automatic int h9_cell_index(input int dim);
        int group;
        int local_dim_index;
        int row;
        int column;
        begin
            group = dim % 2;
            local_dim_index = dim / 2;
            row = local_dim_index % 8;
            column = local_dim_index / 8;
            h9_cell_index = group * 64 + row * 8 + column;
        end
    endfunction

    always_comb begin
        native_lane_mask_comb = 128'd0;
        native_group_mask_comb = 2'd0;
        qk_operand_a_comb = '0;
        qk_operand_b_comb = '0;
        sv_operand_b_comb = '0;
        for (int dim = 0; dim < D_HEAD; dim++) begin
            int cell_index;
            int kv_index;
            cell_index = h9_cell_index(dim);
            native_lane_mask_comb[cell_index] = 1'b1;
            native_group_mask_comb[cell_index / 64] = 1'b1;
            qk_operand_a_comb[cell_index*16 +: 16] = q_mem[dim];
            kv_index = int'(qk_token_q) * D_HEAD + dim;
            qk_operand_b_comb[cell_index*16 +: 16] = k_mem[kv_index];
            kv_index = int'(sv_token_q) * D_HEAD + dim;
            sv_operand_b_comb[cell_index*16 +: 16] = v_mem[kv_index];
        end
    end

    always_comb begin
        output_vector_comb = '0;
        output_lane_mask_comb = '0;
        for (int lane = 0; lane < PE_NUM; lane++) begin
            int dim;
            int cell_index;
            dim = int'(output_tile_q) * PE_NUM + lane;
            if (dim < D_HEAD) begin
                cell_index = h9_cell_index(dim);
                output_vector_comb[lane*32 +: 32] = full_output_vector_q[cell_index*32 +: 32];
                output_lane_mask_comb[lane] = 1'b1;
            end
        end
    end

    assign load_ready = (phase_q == PH_IDLE) && !output_valid_q && !done_valid_q;
    assign start_ready = (phase_q == PH_IDLE) && !output_valid_q && !done_valid_q;
    assign output_valid = output_valid_q;
    assign output_base_dim = output_base_dim_q;
    assign output_vector_fp32 = output_vector_q;
    assign output_lane_mask = output_lane_mask_q;
    assign output_status = output_status_q;
    assign output_invalid = output_invalid_q;
    assign output_meta = output_meta_q;
    assign output_last = output_last_q;
    assign done_valid = done_valid_q;
    assign done_status = status_q;
    assign done_invalid = invalid_q;
    assign done_meta = meta_q;

    assign array_cmd_valid =
        ((phase_q == PH_QK) && (qk_state_q == QK_SEND_ARRAY)) ||
        ((phase_q == PH_OUTER) && (sv_state_q == SV_SEND_ARRAY) && (prob_valid_count_q > sv_token_q));
    assign array_cmd_mode = (phase_q == PH_OUTER) ? MODE_OUTER_PRODUCT : MODE_INNER_PRODUCT;
    assign array_cmd_clear_acc = (phase_q == PH_OUTER) ? (sv_token_q == '0) : 1'b1;
    assign array_cmd_tile_last = (phase_q == PH_OUTER) ? (sv_token_q == (seq_len_q - SEQ_LEN_W'(1))) : 1'b1;
    assign array_cmd_scalar = (phase_q == PH_OUTER) ? prob_mem[int'(sv_token_q)] : 32'd0;
    assign array_cmd_operand_a = (phase_q == PH_OUTER) ? '0 : qk_operand_a_comb;
    assign array_cmd_operand_b = (phase_q == PH_OUTER) ? sv_operand_b_comb : qk_operand_b_comb;
    assign array_cmd_last = (phase_q == PH_OUTER) ?
        (sv_token_q == (seq_len_q - SEQ_LEN_W'(1))) :
        (qk_token_q == (seq_len_q - SEQ_LEN_W'(1)));
    assign array_result_ready = ((phase_q == PH_QK) && (qk_state_q == QK_WAIT_ARRAY)) ||
                                ((phase_q == PH_OUTER) && (sv_state_q == SV_WAIT_ARRAY));
    assign array_done_ready = array_result_ready;

    paper_array_8x8x2 #(
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_paper_array_8x8x2 (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .cmd_valid                      (array_cmd_valid),
        .cmd_ready                      (array_cmd_ready),
        .cmd_mode                       (array_cmd_mode),
        .cmd_k_size                     (16'(D_HEAD)),
        .cmd_m_size                     (16'd1),
        .cmd_n_size                     (16'd1),
        .cmd_tile_id                    (16'(qk_token_q)),
        .cmd_meta                       (meta_q),
        .cmd_clear_acc                  (array_cmd_clear_acc),
        .cmd_tile_last                  (array_cmd_tile_last),
        .cmd_group_mask                 (native_group_mask_comb),
        .cmd_lane_mask                  (native_lane_mask_comb),
        .cmd_scalar_fp32                (array_cmd_scalar),
        .cmd_operand_a_fp16             (array_cmd_operand_a),
        .cmd_operand_b_fp16             (array_cmd_operand_b),
        .cmd_last                       (array_cmd_last),
        .result_valid                   (array_result_valid),
        .result_ready                   (array_result_ready),
        .result_mode                    (array_result_mode),
        .result_tile_id                 (array_result_tile_id),
        .result_scalar_fp32             (array_result_scalar_fp32),
        .result_vector_fp32             (array_result_vector_fp32),
        .result_lane_mask               (array_result_lane_mask),
        .result_status                  (array_result_status),
        .result_invalid                 (array_result_invalid),
        .result_meta                    (array_result_meta),
        .result_last                    (array_result_last),
        .done_valid                     (array_done_valid),
        .done_ready                     (array_done_ready),
        .done_status                    (array_done_status),
        .done_invalid                   (array_done_invalid),
        .done_meta                      (array_done_meta),
        .perf_paper_array_active_cycles (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles   (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles         (perf_inner_mode_cycles),
        .perf_outer_mode_cycles         (perf_outer_mode_cycles),
        .perf_group0_active_cycles      (perf_group0_active_cycles),
        .perf_group1_active_cycles      (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles     (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles        (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles  (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles (perf_array_output_stall_cycles)
    );

    assign scaler_in_valid = (phase_q == PH_QK) && (qk_state_q == QK_SCALE_SEND);
    assign scaler_out_ready = (phase_q == PH_QK) && (qk_state_q == QK_SCALE_WAIT);

    attention_score_scaler #(
        .D_HEAD(D_HEAD),
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_score_scaler (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (scaler_in_valid),
        .in_ready    (scaler_in_ready),
        .in_score    (qk_raw_score_q),
        .in_meta     (meta_q),
        .in_last     (qk_token_q == (seq_len_q - SEQ_LEN_W'(1))),
        .out_valid   (scaler_out_valid),
        .out_ready   (scaler_out_ready),
        .out_score   (scaler_out_score),
        .out_status  (scaler_out_status),
        .out_invalid (scaler_out_invalid),
        .out_meta    (),
        .out_last    ()
    );

    assign reduction_in_valid = (phase_q == PH_QK) && (reduce_token_q < score_valid_count_q) &&
                                (reduce_token_q < seq_len_q) && !softmax_final_seen_q;
    assign reduction_final_ready = (phase_q == PH_QK) && (qk_state_q == QK_DONE);

    softmax_reduction #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_softmax_reduction (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (start_fire),
        .in_valid        (reduction_in_valid),
        .in_ready        (reduction_in_ready),
        .in_score        (score_mem[int'(reduce_token_q)]),
        .in_meta         (meta_q),
        .in_last         (reduce_token_q == (seq_len_q - SEQ_LEN_W'(1))),
        .final_valid     (reduction_final_valid),
        .final_ready     (reduction_final_ready),
        .final_max       (reduction_final_max),
        .final_exp_sum   (reduction_final_exp_sum),
        .final_status    (reduction_final_status),
        .final_invalid   (reduction_final_invalid),
        .final_meta      (),
        .busy            (reduction_busy),
        .processed_count (reduction_processed_count)
    );

    assign norm_start_valid = (phase_q == PH_NORM_START);
    assign norm_score_valid = (phase_q == PH_OUTER) && (norm_token_q < seq_len_q);
    assign norm_prob_ready = (phase_q == PH_OUTER) && (prob_valid_count_q < seq_len_q);

    softmax_normalization #(
        .META_W(META_W),
        .TOKEN_W(TOKEN_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_softmax_normalization (
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (start_fire),
        .start_valid    (norm_start_valid),
        .start_ready    (norm_start_ready),
        .start_max      (final_max_q),
        .start_exp_sum  (final_exp_sum_q),
        .start_meta     (meta_q),
        .score_valid    (norm_score_valid),
        .score_ready    (norm_score_ready),
        .score_value    (score_mem[int'(norm_token_q)]),
        .score_index    (norm_token_q[TOKEN_W-1:0]),
        .score_last     (norm_token_q == (seq_len_q - SEQ_LEN_W'(1))),
        .prob_valid     (norm_prob_valid),
        .prob_ready     (norm_prob_ready),
        .prob_value     (norm_prob_value),
        .prob_index     (norm_prob_index),
        .prob_last      (norm_prob_last),
        .prob_status    (norm_prob_status),
        .prob_invalid   (norm_prob_invalid),
        .prob_meta      (),
        .busy           (norm_busy)
    );

    assign score_fifo_occupancy = (score_valid_count_q >= reduce_token_q) ? (score_valid_count_q - reduce_token_q) : '0;
    assign prob_fifo_occupancy = (prob_valid_count_q >= sv_token_q) ? (prob_valid_count_q - sv_token_q) : '0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_q <= PH_IDLE;
            qk_state_q <= QK_IDLE;
            sv_state_q <= SV_IDLE;
            seq_len_q <= '0;
            qk_token_q <= '0;
            reduce_token_q <= '0;
            norm_token_q <= '0;
            sv_token_q <= '0;
            score_valid_count_q <= '0;
            prob_valid_count_q <= '0;
            output_tile_q <= '0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            active_q <= 1'b0;
            softmax_final_seen_q <= 1'b0;
            final_max_q <= 32'd0;
            final_exp_sum_q <= 32'd0;
            array_result_seen_q <= 1'b0;
            array_done_seen_q <= 1'b0;
            qk_raw_score_q <= 32'd0;
            full_output_vector_q <= '0;
            output_valid_q <= 1'b0;
            output_base_dim_q <= '0;
            output_vector_q <= '0;
            output_lane_mask_q <= '0;
            output_status_q <= 8'd0;
            output_invalid_q <= 1'b0;
            output_meta_q <= '0;
            output_last_q <= 1'b0;
            done_valid_q <= 1'b0;
        end else begin
            if (load_fire) begin
                if (load_kind == LOAD_Q) begin
                    q_mem[int'(load_dim)] <= load_data;
                end else if (load_kind == LOAD_K) begin
                    k_mem[int'(load_token) * D_HEAD + int'(load_dim)] <= load_data;
                end else if (load_kind == LOAD_V) begin
                    v_mem[int'(load_token) * D_HEAD + int'(load_dim)] <= load_data;
                end
            end

            if (output_fire) begin
                output_valid_q <= 1'b0;
            end
            if (done_fire) begin
                done_valid_q <= 1'b0;
                active_q <= 1'b0;
                phase_q <= PH_IDLE;
            end

            if (reduction_in_fire) begin
                reduce_token_q <= reduce_token_q + SEQ_LEN_W'(1);
            end
            if (reduction_final_fire) begin
                final_max_q <= reduction_final_max;
                final_exp_sum_q <= reduction_final_exp_sum;
                status_q <= status_q | reduction_final_status;
                invalid_q <= invalid_q | reduction_final_invalid;
                softmax_final_seen_q <= 1'b1;
            end
            if (norm_score_fire) begin
                norm_token_q <= norm_token_q + SEQ_LEN_W'(1);
            end
            if (norm_prob_fire) begin
                prob_mem[int'(norm_prob_index)] <= norm_prob_value;
                prob_valid_count_q <= prob_valid_count_q + SEQ_LEN_W'(1);
                status_q <= status_q | norm_prob_status;
                invalid_q <= invalid_q | norm_prob_invalid;
            end

            unique case (phase_q)
                PH_IDLE: begin
                    if (start_fire) begin
                        seq_len_q <= start_seq_len;
                        qk_token_q <= '0;
                        reduce_token_q <= '0;
                        norm_token_q <= '0;
                        sv_token_q <= '0;
                        score_valid_count_q <= '0;
                        prob_valid_count_q <= '0;
                        output_tile_q <= '0;
                        meta_q <= start_meta;
                        status_q <= 8'd0;
                        invalid_q <= 1'b0;
                        active_q <= 1'b1;
                        softmax_final_seen_q <= 1'b0;
                        final_max_q <= 32'd0;
                        final_exp_sum_q <= 32'd0;
                        array_result_seen_q <= 1'b0;
                        array_done_seen_q <= 1'b0;
                        full_output_vector_q <= '0;
                        if ((start_seq_len == '0) || (start_seq_len > SEQ_LEN_W'(MAX_SEQ_LEN))) begin
                            status_q <= 8'h80;
                            invalid_q <= 1'b1;
                            done_valid_q <= 1'b1;
                            phase_q <= PH_ERROR;
                        end else begin
                            qk_state_q <= QK_SEND_ARRAY;
                            sv_state_q <= SV_IDLE;
                            phase_q <= PH_QK;
                        end
                    end
                end

                PH_QK: begin
                    unique case (qk_state_q)
                        QK_SEND_ARRAY: begin
                            if (array_cmd_fire) begin
                                array_result_seen_q <= 1'b0;
                                array_done_seen_q <= 1'b0;
                                qk_state_q <= QK_WAIT_ARRAY;
                            end
                        end
                        QK_WAIT_ARRAY: begin
                            if (array_result_fire) begin
                                qk_raw_score_q <= array_result_scalar_fp32;
                                status_q <= status_q | array_result_status;
                                invalid_q <= invalid_q | array_result_invalid;
                                array_result_seen_q <= 1'b1;
                            end
                            if (array_done_fire) begin
                                status_q <= status_q | array_done_status;
                                invalid_q <= invalid_q | array_done_invalid;
                                array_done_seen_q <= 1'b1;
                            end
                            if ((array_result_seen_q || array_result_fire) &&
                                (array_done_seen_q || array_done_fire)) begin
                                qk_state_q <= QK_SCALE_SEND;
                            end
                        end
                        QK_SCALE_SEND: begin
                            if (scaler_in_fire) begin
                                qk_state_q <= QK_SCALE_WAIT;
                            end
                        end
                        QK_SCALE_WAIT: begin
                            if (scaler_out_fire) begin
                                score_mem[int'(qk_token_q)] <= scaler_out_score;
                                score_valid_count_q <= score_valid_count_q + SEQ_LEN_W'(1);
                                status_q <= status_q | scaler_out_status;
                                invalid_q <= invalid_q | scaler_out_invalid;
                                if (qk_token_q == (seq_len_q - SEQ_LEN_W'(1))) begin
                                    qk_state_q <= QK_DONE;
                                end else begin
                                    qk_token_q <= qk_token_q + SEQ_LEN_W'(1);
                                    qk_state_q <= QK_SEND_ARRAY;
                                end
                            end
                        end
                        default: qk_state_q <= QK_SEND_ARRAY;
                    endcase

                    if ((qk_state_q == QK_DONE) && softmax_final_seen_q) begin
                        phase_q <= PH_NORM_START;
                    end
                end

                PH_NORM_START: begin
                    if (norm_start_fire) begin
                        sv_state_q <= SV_SEND_ARRAY;
                        phase_q <= PH_OUTER;
                    end
                end

                PH_OUTER: begin
                    unique case (sv_state_q)
                        SV_SEND_ARRAY: begin
                            if (array_cmd_fire) begin
                                array_result_seen_q <= 1'b0;
                                array_done_seen_q <= 1'b0;
                                sv_state_q <= SV_WAIT_ARRAY;
                            end
                        end
                        SV_WAIT_ARRAY: begin
                            if (array_result_fire) begin
                                full_output_vector_q <= array_result_vector_fp32;
                                status_q <= status_q | array_result_status;
                                invalid_q <= invalid_q | array_result_invalid;
                                array_result_seen_q <= 1'b1;
                            end
                            if (array_done_fire) begin
                                status_q <= status_q | array_done_status;
                                invalid_q <= invalid_q | array_done_invalid;
                                array_done_seen_q <= 1'b1;
                            end
                            if (array_done_seen_q || array_done_fire) begin
                                if (sv_token_q == (seq_len_q - SEQ_LEN_W'(1))) begin
                                    if (array_result_seen_q || array_result_fire) begin
                                        sv_state_q <= SV_DONE;
                                    end
                                end else begin
                                    sv_token_q <= sv_token_q + SEQ_LEN_W'(1);
                                    sv_state_q <= SV_SEND_ARRAY;
                                end
                            end
                        end
                        default: sv_state_q <= SV_SEND_ARRAY;
                    endcase

                    if ((sv_state_q == SV_DONE) && (prob_valid_count_q == seq_len_q) && !norm_busy && !norm_prob_valid) begin
                        output_tile_q <= '0;
                        phase_q <= PH_OUTPUT;
                    end
                end

                PH_OUTPUT: begin
                    if (!output_valid_q) begin
                        output_valid_q <= 1'b1;
                        output_base_dim_q <= D_ADDR_W'(int'(output_tile_q) * PE_NUM);
                        output_vector_q <= output_vector_comb;
                        output_lane_mask_q <= output_lane_mask_comb;
                        output_status_q <= status_q;
                        output_invalid_q <= invalid_q;
                        output_meta_q <= meta_q;
                        output_last_q <= (output_tile_q == TILE_W'(TILE_COUNT - 1));
                    end else if (output_fire) begin
                        if (output_tile_q == TILE_W'(TILE_COUNT - 1)) begin
                            done_valid_q <= 1'b1;
                            phase_q <= PH_DONE;
                        end else begin
                            output_tile_q <= output_tile_q + TILE_W'(1);
                        end
                    end
                end

                PH_DONE: begin
                    if (done_fire) begin
                        phase_q <= PH_IDLE;
                    end
                end

                PH_ERROR: begin
                    if (done_fire) begin
                        phase_q <= PH_IDLE;
                    end
                end

                default: phase_q <= PH_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_attention_cycles <= '0;
            perf_qk_cycles <= '0;
            perf_qk_pe_busy_cycles <= '0;
            perf_scale_cycles <= '0;
            perf_reduction_cycles <= '0;
            perf_reduction_finalize_cycles <= '0;
            perf_normalization_cycles <= '0;
            perf_sv_cycles <= '0;
            perf_pe_stall_cycles <= '0;
            perf_sfu_stall_cycles <= '0;
            perf_buffer_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
            perf_score_buffer_peak_occupancy <= '0;
            perf_qk_sfu_overlap_cycles <= '0;
            perf_qk_only_cycles <= '0;
            perf_sfu_during_qk_cycles <= '0;
            perf_score_fifo_full_stall_cycles <= '0;
            perf_score_fifo_empty_cycles <= '0;
            perf_score_fifo_peak_occupancy <= '0;
            perf_sfu_sv_overlap_cycles <= '0;
            perf_sfu_only_cycles <= '0;
            perf_sv_only_cycles <= '0;
            perf_probability_fifo_full_stall_cycles <= '0;
            perf_probability_fifo_empty_stall_cycles <= '0;
            perf_probability_fifo_peak_occupancy <= '0;
            perf_inner_to_outer_switch_cycles <= '0;
            perf_pipeline_bubble_cycles <= '0;
        end else begin
            if (start_fire) begin
                perf_total_attention_cycles <= '0;
                perf_qk_cycles <= '0;
                perf_qk_pe_busy_cycles <= '0;
                perf_scale_cycles <= '0;
                perf_reduction_cycles <= '0;
                perf_reduction_finalize_cycles <= '0;
                perf_normalization_cycles <= '0;
                perf_sv_cycles <= '0;
                perf_pe_stall_cycles <= '0;
                perf_sfu_stall_cycles <= '0;
                perf_buffer_stall_cycles <= '0;
                perf_output_stall_cycles <= '0;
                perf_score_buffer_peak_occupancy <= '0;
                perf_qk_sfu_overlap_cycles <= '0;
                perf_qk_only_cycles <= '0;
                perf_sfu_during_qk_cycles <= '0;
                perf_score_fifo_full_stall_cycles <= '0;
                perf_score_fifo_empty_cycles <= '0;
                perf_score_fifo_peak_occupancy <= '0;
                perf_sfu_sv_overlap_cycles <= '0;
                perf_sfu_only_cycles <= '0;
                perf_sv_only_cycles <= '0;
                perf_probability_fifo_full_stall_cycles <= '0;
                perf_probability_fifo_empty_stall_cycles <= '0;
                perf_probability_fifo_peak_occupancy <= '0;
                perf_inner_to_outer_switch_cycles <= '0;
                perf_pipeline_bubble_cycles <= '0;
            end else begin
                if (active_q) begin
                    perf_total_attention_cycles <= perf_total_attention_cycles + COUNTER_W'(1);
                end
                if (qk_active) begin
                    perf_qk_cycles <= perf_qk_cycles + COUNTER_W'(1);
                    perf_qk_pe_busy_cycles <= perf_qk_pe_busy_cycles + COUNTER_W'(1);
                end
                if (qk_state_q inside {QK_SCALE_SEND, QK_SCALE_WAIT}) begin
                    perf_scale_cycles <= perf_scale_cycles + COUNTER_W'(1);
                end
                if (reduction_busy || reduction_in_valid) begin
                    perf_reduction_cycles <= perf_reduction_cycles + COUNTER_W'(1);
                end
                if ((phase_q == PH_QK) && (qk_state_q == QK_DONE) && !softmax_final_seen_q) begin
                    perf_reduction_finalize_cycles <= perf_reduction_finalize_cycles + COUNTER_W'(1);
                end
                if (norm_active) begin
                    perf_normalization_cycles <= perf_normalization_cycles + COUNTER_W'(1);
                end
                if (sv_active || phase_q == PH_OUTPUT) begin
                    perf_sv_cycles <= perf_sv_cycles + COUNTER_W'(1);
                end
                if (array_cmd_valid && !array_cmd_ready) begin
                    perf_pe_stall_cycles <= perf_pe_stall_cycles + COUNTER_W'(1);
                end
                if ((scaler_in_valid && !scaler_in_ready) || (scaler_out_ready && !scaler_out_valid) ||
                    (reduction_in_valid && !reduction_in_ready) ||
                    (reduction_final_ready && !reduction_final_valid) ||
                    (norm_start_valid && !norm_start_ready) || (norm_score_valid && !norm_score_ready) ||
                    (norm_prob_ready && !norm_prob_valid && norm_busy)) begin
                    perf_sfu_stall_cycles <= perf_sfu_stall_cycles + COUNTER_W'(1);
                end
                if ((output_valid && !output_ready) || (done_valid && !done_ready)) begin
                    perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
                end
                if (score_fifo_occupancy > perf_score_buffer_peak_occupancy[SEQ_LEN_W-1:0]) begin
                    perf_score_buffer_peak_occupancy <= COUNTER_W'(score_fifo_occupancy);
                end
                if (score_fifo_occupancy > perf_score_fifo_peak_occupancy[SEQ_LEN_W-1:0]) begin
                    perf_score_fifo_peak_occupancy <= COUNTER_W'(score_fifo_occupancy);
                end
                if (prob_fifo_occupancy > perf_probability_fifo_peak_occupancy[SEQ_LEN_W-1:0]) begin
                    perf_probability_fifo_peak_occupancy <= COUNTER_W'(prob_fifo_occupancy);
                end
                if ((phase_q == PH_QK) && (score_fifo_occupancy == '0) && reduction_in_valid && !reduction_in_ready) begin
                    perf_score_fifo_empty_cycles <= perf_score_fifo_empty_cycles + COUNTER_W'(1);
                end
                if ((phase_q == PH_OUTER) && (prob_valid_count_q <= sv_token_q) && (sv_state_q == SV_SEND_ARRAY)) begin
                    perf_probability_fifo_empty_stall_cycles <= perf_probability_fifo_empty_stall_cycles + COUNTER_W'(1);
                end
                if (qk_active && sfu_during_qk) begin
                    perf_qk_sfu_overlap_cycles <= perf_qk_sfu_overlap_cycles + COUNTER_W'(1);
                    perf_sfu_during_qk_cycles <= perf_sfu_during_qk_cycles + COUNTER_W'(1);
                end else if (qk_active) begin
                    perf_qk_only_cycles <= perf_qk_only_cycles + COUNTER_W'(1);
                end
                if (norm_active && sv_active) begin
                    perf_sfu_sv_overlap_cycles <= perf_sfu_sv_overlap_cycles + COUNTER_W'(1);
                end else if (norm_active) begin
                    perf_sfu_only_cycles <= perf_sfu_only_cycles + COUNTER_W'(1);
                end else if (sv_active) begin
                    perf_sv_only_cycles <= perf_sv_only_cycles + COUNTER_W'(1);
                end
                if (phase_q == PH_NORM_START) begin
                    perf_inner_to_outer_switch_cycles <= perf_inner_to_outer_switch_cycles + COUNTER_W'(1);
                end
                if (active_q && !qk_active && !sfu_during_qk && !norm_active && !sv_active &&
                    (phase_q != PH_OUTPUT) && (phase_q != PH_DONE)) begin
                    perf_pipeline_bubble_cycles <= perf_pipeline_bubble_cycles + COUNTER_W'(1);
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(qk_active && sv_active))
                else $error("paper_interleaved_attention no_inner_and_outer_same_cycle failed");
            assert (!(array_cmd_valid && array_cmd_ready && array_cmd_mode == MODE_OUTER_PRODUCT &&
                      phase_q != PH_OUTER))
                else $error("paper_interleaved_attention no_outer_before_qk_retired failed");
            assert (!(phase_q == PH_OUTER && !softmax_final_seen_q))
                else $error("paper_interleaved_attention no_outer_before_softmax_state_valid failed");
            assert (!(array_cmd_valid && array_cmd_ready && (array_cmd_mode != MODE_INNER_PRODUCT) &&
                      (array_cmd_mode != MODE_OUTER_PRODUCT)))
                else $error("paper_interleaved_attention illegal_mode failed");
            assert (!(norm_prob_fire && (norm_prob_index != prob_valid_count_q[TOKEN_W-1:0])))
                else $error("paper_interleaved_attention probability_index_monotonic failed");
            assert (!(sv_state_q == SV_SEND_ARRAY && prob_valid_count_q > sv_token_q &&
                      sv_token_q >= seq_len_q))
                else $error("paper_interleaved_attention probability_matches_v_index failed");
            assert (!(output_valid && $isunknown({output_base_dim, output_vector_fp32, output_lane_mask,
                                                  output_status, output_invalid, output_meta, output_last})))
                else $error("paper_interleaved_attention no_unknown_output_when_valid failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta})))
                else $error("paper_interleaved_attention no_unknown_done_when_valid failed");
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("paper_interleaved_attention output valid dropped under backpressure");
                assert ($stable({output_base_dim, output_vector_fp32, output_lane_mask,
                                 output_status, output_invalid, output_meta, output_last}))
                    else $error("paper_interleaved_attention output stable until ready failed");
            end
            if ($past(rst_n) && $past(done_valid && !done_ready)) begin
                assert (done_valid)
                    else $error("paper_interleaved_attention done valid dropped under backpressure");
                assert ($stable({done_status, done_invalid, done_meta}))
                    else $error("paper_interleaved_attention done stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
