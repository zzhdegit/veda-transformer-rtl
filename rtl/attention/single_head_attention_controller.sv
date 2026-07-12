`default_nettype none

module single_head_attention_controller #(
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
    input  logic [1:0]                   load_kind,    // 0: q, 1: K, 2: V
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
    output logic [COUNTER_W-1:0]         perf_score_buffer_peak_occupancy
);
    localparam logic [1:0] LOAD_Q = 2'd0;
    localparam logic [1:0] LOAD_K = 2'd1;
    localparam logic [1:0] LOAD_V = 2'd2;
    localparam logic [1:0] MODE_QK_INNER = 2'd1;
    localparam logic [1:0] MODE_SV_OUTER = 2'd2;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_QK_SEND_TILE,
        ST_QK_WAIT_RESULT,
        ST_SCALE_SEND,
        ST_SCALE_WAIT,
        ST_SCORE_STORE,
        ST_REDUCE_SEND,
        ST_WAIT_REDUCTION,
        ST_NORM_START,
        ST_NORM_READ_REQ,
        ST_NORM_READ_RSP,
        ST_NORM_SEND_SCORE,
        ST_NORM_WAIT_PROB,
        ST_SV_TILE_START,
        ST_SV_PROB_READ_REQ,
        ST_SV_PROB_READ_RSP,
        ST_SV_SEND_TILE,
        ST_SV_WAIT_RESULT,
        ST_OUTPUT_TILE,
        ST_DONE,
        ST_ERROR
    } state_e;

    state_e state_q;
    logic [SEQ_LEN_W-1:0] seq_len_q;
    logic [SEQ_LEN_W-1:0] qk_token_q;
    logic [TILE_W-1:0] qk_tile_q;
    logic [SEQ_LEN_W-1:0] norm_token_q;
    logic [SEQ_LEN_W-1:0] sv_token_q;
    logic [TILE_W-1:0] sv_tile_q;
    logic [31:0] raw_score_q;
    logic [31:0] scaled_score_q;
    logic [31:0] final_max_q;
    logic [31:0] final_exp_sum_q;
    logic [31:0] prob_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic active_q;

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

    logic [PE_NUM-1:0] qk_lane_mask_comb;
    logic [PE_NUM-1:0] sv_lane_mask_comb;
    logic [PE_NUM*16-1:0] q_vector_comb;
    logic [PE_NUM*16-1:0] k_vector_comb;
    logic [PE_NUM*16-1:0] v_vector_comb;
    logic [D_ADDR_W-1:0] sv_base_dim_comb;

    logic pe_in_valid;
    logic pe_in_ready;
    logic [1:0] pe_in_mode;
    logic pe_in_clear;
    logic pe_in_tile_first;
    logic pe_in_tile_last;
    logic [PE_NUM-1:0] pe_in_lane_mask;
    logic [31:0] pe_in_scalar_fp32;
    logic [PE_NUM*16-1:0] pe_in_vector_a_fp16;
    logic [PE_NUM*16-1:0] pe_in_vector_b_fp16;
    logic pe_in_last;
    logic pe_out_valid;
    logic pe_out_ready;
    logic [1:0] pe_out_mode;
    logic [31:0] pe_out_scalar_fp32;
    logic [PE_NUM*32-1:0] pe_out_vector_fp32;
    logic [PE_NUM-1:0] pe_out_lane_mask;
    logic [7:0] pe_out_status;
    logic pe_out_invalid;
    logic [META_W-1:0] pe_out_meta;
    logic pe_out_last;

    logic scaler_in_valid;
    logic scaler_in_ready;
    logic scaler_out_valid;
    logic scaler_out_ready;
    logic [31:0] scaler_out_score;
    logic [7:0] scaler_out_status;
    logic scaler_out_invalid;

    logic score_clear;
    logic score_wr_valid;
    logic score_wr_ready;
    logic score_rd_valid;
    logic score_rd_ready;
    logic score_rsp_valid;
    logic score_rsp_ready;
    logic [TOKEN_W-1:0] score_rsp_addr;
    logic [31:0] score_rsp_score;
    logic [SEQ_LEN_W-1:0] score_valid_length;
    logic [SEQ_LEN_W-1:0] score_occupancy;
    logic [SEQ_LEN_W-1:0] score_peak_occupancy;

    logic prob_clear;
    logic prob_read_rewind;
    logic prob_wr_valid;
    logic prob_wr_ready;
    logic prob_rd_valid;
    logic prob_rd_ready;
    logic prob_rsp_valid;
    logic prob_rsp_ready;
    logic [TOKEN_W-1:0] prob_rsp_addr;
    logic [31:0] prob_rsp_score;
    logic [SEQ_LEN_W-1:0] prob_valid_length;
    logic [SEQ_LEN_W-1:0] prob_occupancy;
    logic [SEQ_LEN_W-1:0] prob_peak_occupancy;

    logic reduction_clear;
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

    logic norm_clear;
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

    wire load_fire = load_valid && load_ready;
    wire start_fire = start_valid && start_ready;
    wire output_fire = output_valid && output_ready;
    wire done_fire = done_valid && done_ready;
    wire pe_in_fire = pe_in_valid && pe_in_ready;
    wire pe_out_fire = pe_out_valid && pe_out_ready;
    wire scaler_in_fire = scaler_in_valid && scaler_in_ready;
    wire scaler_out_fire = scaler_out_valid && scaler_out_ready;
    wire score_wr_fire = score_wr_valid && score_wr_ready;
    wire score_rd_fire = score_rd_valid && score_rd_ready;
    wire score_rsp_fire = score_rsp_valid && score_rsp_ready;
    wire prob_wr_fire = prob_wr_valid && prob_wr_ready;
    wire prob_rd_fire = prob_rd_valid && prob_rd_ready;
    wire prob_rsp_fire = prob_rsp_valid && prob_rsp_ready;
    wire reduction_in_fire = reduction_in_valid && reduction_in_ready;
    wire reduction_final_fire = reduction_final_valid && reduction_final_ready;
    wire norm_start_fire = norm_start_valid && norm_start_ready;
    wire norm_score_fire = norm_score_valid && norm_score_ready;
    wire norm_prob_fire = norm_prob_valid && norm_prob_ready;

    initial begin
        if (PE_NUM <= 0 || D_HEAD <= 0 || MAX_SEQ_LEN <= 0) begin
            $fatal(1, "single_head_attention_controller parameters must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "single_head_attention_controller PE_NUM must be a power of two");
        end
        if (META_W <= 0) begin
            $fatal(1, "single_head_attention_controller META_W must be positive");
        end
    end

    function automatic logic [PE_NUM-1:0] tile_mask(input logic [TILE_W-1:0] tile_idx);
        logic [PE_NUM-1:0] mask;
        int base;
        begin
            mask = '0;
            base = int'(tile_idx) * PE_NUM;
            for (int lane = 0; lane < PE_NUM; lane++) begin
                if ((base + lane) < D_HEAD) begin
                    mask[lane] = 1'b1;
                end
            end
            tile_mask = mask;
        end
    endfunction

    always_comb begin
        q_vector_comb = '0;
        k_vector_comb = '0;
        v_vector_comb = '0;
        qk_lane_mask_comb = tile_mask(qk_tile_q);
        sv_lane_mask_comb = tile_mask(sv_tile_q);
        sv_base_dim_comb = D_ADDR_W'(int'(sv_tile_q) * PE_NUM);

        for (int lane = 0; lane < PE_NUM; lane++) begin
            int qk_dim;
            int sv_dim;
            int k_index;
            int v_index;
            qk_dim = int'(qk_tile_q) * PE_NUM + lane;
            sv_dim = int'(sv_tile_q) * PE_NUM + lane;
            if (qk_dim < D_HEAD) begin
                k_index = int'(qk_token_q) * D_HEAD + qk_dim;
                q_vector_comb[lane*16 +: 16] = q_mem[qk_dim];
                k_vector_comb[lane*16 +: 16] = k_mem[k_index];
            end
            if (sv_dim < D_HEAD) begin
                v_index = int'(sv_token_q) * D_HEAD + sv_dim;
                v_vector_comb[lane*16 +: 16] = v_mem[v_index];
            end
        end
    end

    assign load_ready = (state_q == ST_IDLE) && !output_valid_q && !done_valid_q;
    assign start_ready = (state_q == ST_IDLE) && !output_valid_q && !done_valid_q;

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

    assign pe_in_valid = (state_q == ST_QK_SEND_TILE) || (state_q == ST_SV_SEND_TILE);
    assign pe_in_mode = (state_q == ST_SV_SEND_TILE) ? MODE_SV_OUTER : MODE_QK_INNER;
    assign pe_in_clear = (state_q == ST_SV_SEND_TILE) ? (sv_token_q == '0) : (qk_tile_q == '0);
    assign pe_in_tile_first = pe_in_clear;
    assign pe_in_tile_last = (state_q == ST_SV_SEND_TILE) ? (sv_token_q == (seq_len_q - SEQ_LEN_W'(1))) : (qk_tile_q == TILE_W'(TILE_COUNT - 1));
    assign pe_in_lane_mask = (state_q == ST_SV_SEND_TILE) ? sv_lane_mask_comb : qk_lane_mask_comb;
    assign pe_in_scalar_fp32 = (state_q == ST_SV_SEND_TILE) ? prob_q : 32'd0;
    assign pe_in_vector_a_fp16 = (state_q == ST_SV_SEND_TILE) ? '0 : q_vector_comb;
    assign pe_in_vector_b_fp16 = (state_q == ST_SV_SEND_TILE) ? v_vector_comb : k_vector_comb;
    assign pe_in_last = (state_q == ST_SV_SEND_TILE) ?
        ((sv_tile_q == TILE_W'(TILE_COUNT - 1)) && (sv_token_q == (seq_len_q - SEQ_LEN_W'(1)))) :
        (qk_token_q == (seq_len_q - SEQ_LEN_W'(1)));
    assign pe_out_ready = (state_q == ST_QK_WAIT_RESULT) || (state_q == ST_SV_WAIT_RESULT);

    reconfigurable_pe_core #(
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_pe_core (
        .clk                         (clk),
        .rst_n                       (rst_n),
        .in_valid                    (pe_in_valid),
        .in_ready                    (pe_in_ready),
        .in_mode                     (pe_in_mode),
        .in_clear                    (pe_in_clear),
        .in_tile_first               (pe_in_tile_first),
        .in_tile_last                (pe_in_tile_last),
        .in_use_explicit_mask        (1'b1),
        .in_active_lanes             ('0),
        .in_lane_mask                (pe_in_lane_mask),
        .in_scalar_fp32              (pe_in_scalar_fp32),
        .in_vector_a_fp16            (pe_in_vector_a_fp16),
        .in_vector_b_fp16            (pe_in_vector_b_fp16),
        .in_meta                     (meta_q),
        .in_last                     (pe_in_last),
        .out_valid                   (pe_out_valid),
        .out_ready                   (pe_out_ready),
        .out_mode                    (pe_out_mode),
        .out_scalar_fp32             (pe_out_scalar_fp32),
        .out_vector_fp32             (pe_out_vector_fp32),
        .out_lane_mask               (pe_out_lane_mask),
        .out_status                  (pe_out_status),
        .out_invalid                 (pe_out_invalid),
        .out_meta                    (pe_out_meta),
        .out_last                    (pe_out_last),
        .perf_total_cycles           (),
        .perf_busy_cycles            (),
        .perf_active_lane_cycles     (),
        .perf_available_lane_cycles  (),
        .perf_input_stall_cycles     (),
        .perf_output_stall_cycles    (),
        .perf_mode_switch_cycles     (),
        .perf_tile_count             (),
        .perf_operation_count        (),
        .perf_invalid_count          ()
    );

    assign scaler_in_valid = (state_q == ST_SCALE_SEND);
    assign scaler_out_ready = (state_q == ST_SCALE_WAIT);

    attention_score_scaler #(
        .D_HEAD(D_HEAD),
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_score_scaler (
        .clk         (clk),
        .rst_n       (rst_n),
        .in_valid    (scaler_in_valid),
        .in_ready    (scaler_in_ready),
        .in_score    (raw_score_q),
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

    assign score_clear = start_fire;
    assign score_wr_valid = (state_q == ST_SCORE_STORE);
    assign score_rd_valid = (state_q == ST_NORM_READ_REQ);
    assign score_rsp_ready = (state_q == ST_NORM_READ_RSP);

    score_buffer #(
        .DEPTH(MAX_SEQ_LEN),
        .COUNTER_W(COUNTER_W),
        .READ_LATENCY(1)
    ) u_score_buffer (
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (score_clear),
        .read_rewind    (1'b0),
        .wr_valid       (score_wr_valid),
        .wr_ready       (score_wr_ready),
        .wr_addr        (qk_token_q[TOKEN_W-1:0]),
        .wr_score       (scaled_score_q),
        .rd_valid       (score_rd_valid),
        .rd_ready       (score_rd_ready),
        .rd_addr        (norm_token_q[TOKEN_W-1:0]),
        .rsp_valid      (score_rsp_valid),
        .rsp_ready      (score_rsp_ready),
        .rsp_addr       (score_rsp_addr),
        .rsp_score      (score_rsp_score),
        .valid_length   (score_valid_length),
        .occupancy      (score_occupancy),
        .peak_occupancy (score_peak_occupancy)
    );

    assign prob_clear = start_fire;
    assign prob_read_rewind = (state_q == ST_SV_TILE_START);
    assign prob_wr_valid = (state_q == ST_NORM_WAIT_PROB) && norm_prob_valid;
    assign norm_prob_ready = (state_q == ST_NORM_WAIT_PROB) && prob_wr_ready;
    assign prob_rd_valid = (state_q == ST_SV_PROB_READ_REQ);
    assign prob_rsp_ready = (state_q == ST_SV_PROB_READ_RSP);

    score_buffer #(
        .DEPTH(MAX_SEQ_LEN),
        .COUNTER_W(COUNTER_W),
        .READ_LATENCY(1)
    ) u_probability_buffer (
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (prob_clear),
        .read_rewind    (prob_read_rewind),
        .wr_valid       (prob_wr_valid),
        .wr_ready       (prob_wr_ready),
        .wr_addr        (norm_prob_index),
        .wr_score       (norm_prob_value),
        .rd_valid       (prob_rd_valid),
        .rd_ready       (prob_rd_ready),
        .rd_addr        (sv_token_q[TOKEN_W-1:0]),
        .rsp_valid      (prob_rsp_valid),
        .rsp_ready      (prob_rsp_ready),
        .rsp_addr       (prob_rsp_addr),
        .rsp_score      (prob_rsp_score),
        .valid_length   (prob_valid_length),
        .occupancy      (prob_occupancy),
        .peak_occupancy (prob_peak_occupancy)
    );

    assign reduction_clear = start_fire;
    assign reduction_in_valid = (state_q == ST_REDUCE_SEND);
    assign reduction_final_ready = (state_q == ST_WAIT_REDUCTION);

    softmax_reduction #(
        .META_W(META_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_softmax_reduction (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (reduction_clear),
        .in_valid        (reduction_in_valid),
        .in_ready        (reduction_in_ready),
        .in_score        (scaled_score_q),
        .in_meta         (meta_q),
        .in_last         (qk_token_q == (seq_len_q - SEQ_LEN_W'(1))),
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

    assign norm_clear = start_fire;
    assign norm_start_valid = (state_q == ST_NORM_START);
    assign norm_score_valid = (state_q == ST_NORM_SEND_SCORE);

    softmax_normalization #(
        .META_W(META_W),
        .TOKEN_W(TOKEN_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_softmax_normalization (
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (norm_clear),
        .start_valid    (norm_start_valid),
        .start_ready    (norm_start_ready),
        .start_max      (final_max_q),
        .start_exp_sum  (final_exp_sum_q),
        .start_meta     (meta_q),
        .score_valid    (norm_score_valid),
        .score_ready    (norm_score_ready),
        .score_value    (score_rsp_score),
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            seq_len_q <= '0;
            qk_token_q <= '0;
            qk_tile_q <= '0;
            norm_token_q <= '0;
            sv_token_q <= '0;
            sv_tile_q <= '0;
            raw_score_q <= 32'd0;
            scaled_score_q <= 32'd0;
            final_max_q <= 32'd0;
            final_exp_sum_q <= 32'd0;
            prob_q <= 32'd0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            active_q <= 1'b0;
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
            end

            unique case (state_q)
                ST_IDLE: begin
                    if (start_fire) begin
                        seq_len_q <= start_seq_len;
                        qk_token_q <= '0;
                        qk_tile_q <= '0;
                        norm_token_q <= '0;
                        sv_token_q <= '0;
                        sv_tile_q <= '0;
                        raw_score_q <= 32'd0;
                        scaled_score_q <= 32'd0;
                        final_max_q <= 32'd0;
                        final_exp_sum_q <= 32'd0;
                        prob_q <= 32'd0;
                        meta_q <= start_meta;
                        status_q <= 8'd0;
                        invalid_q <= 1'b0;
                        active_q <= 1'b1;
                        if ((start_seq_len == '0) || (start_seq_len > SEQ_LEN_W'(MAX_SEQ_LEN))) begin
                            status_q <= 8'h80;
                            invalid_q <= 1'b1;
                            done_valid_q <= 1'b1;
                            state_q <= ST_ERROR;
                        end else begin
                            state_q <= ST_QK_SEND_TILE;
                        end
                    end
                end

                ST_QK_SEND_TILE: begin
                    if (pe_in_fire) begin
                        if (qk_tile_q == TILE_W'(TILE_COUNT - 1)) begin
                            state_q <= ST_QK_WAIT_RESULT;
                        end else begin
                            qk_tile_q <= qk_tile_q + TILE_W'(1);
                        end
                    end
                end

                ST_QK_WAIT_RESULT: begin
                    if (pe_out_fire) begin
                        raw_score_q <= pe_out_scalar_fp32;
                        status_q <= status_q | pe_out_status;
                        invalid_q <= invalid_q | pe_out_invalid;
                        state_q <= ST_SCALE_SEND;
                    end
                end

                ST_SCALE_SEND: begin
                    if (scaler_in_fire) begin
                        state_q <= ST_SCALE_WAIT;
                    end
                end

                ST_SCALE_WAIT: begin
                    if (scaler_out_fire) begin
                        scaled_score_q <= scaler_out_score;
                        status_q <= status_q | scaler_out_status;
                        invalid_q <= invalid_q | scaler_out_invalid;
                        state_q <= ST_SCORE_STORE;
                    end
                end

                ST_SCORE_STORE: begin
                    if (score_wr_fire) begin
                        state_q <= ST_REDUCE_SEND;
                    end
                end

                ST_REDUCE_SEND: begin
                    if (reduction_in_fire) begin
                        if (qk_token_q == (seq_len_q - SEQ_LEN_W'(1))) begin
                            state_q <= ST_WAIT_REDUCTION;
                        end else begin
                            qk_token_q <= qk_token_q + SEQ_LEN_W'(1);
                            qk_tile_q <= '0;
                            state_q <= ST_QK_SEND_TILE;
                        end
                    end
                end

                ST_WAIT_REDUCTION: begin
                    if (reduction_final_fire) begin
                        final_max_q <= reduction_final_max;
                        final_exp_sum_q <= reduction_final_exp_sum;
                        status_q <= status_q | reduction_final_status;
                        invalid_q <= invalid_q | reduction_final_invalid;
                        norm_token_q <= '0;
                        state_q <= ST_NORM_START;
                    end
                end

                ST_NORM_START: begin
                    if (norm_start_fire) begin
                        state_q <= ST_NORM_READ_REQ;
                    end
                end

                ST_NORM_READ_REQ: begin
                    if (score_rd_fire) begin
                        state_q <= ST_NORM_READ_RSP;
                    end
                end

                ST_NORM_READ_RSP: begin
                    if (score_rsp_fire) begin
                        state_q <= ST_NORM_SEND_SCORE;
                    end
                end

                ST_NORM_SEND_SCORE: begin
                    if (norm_score_fire) begin
                        state_q <= ST_NORM_WAIT_PROB;
                    end
                end

                ST_NORM_WAIT_PROB: begin
                    if (norm_prob_fire) begin
                        status_q <= status_q | norm_prob_status;
                        invalid_q <= invalid_q | norm_prob_invalid;
                        if (norm_prob_last) begin
                            sv_tile_q <= '0;
                            sv_token_q <= '0;
                            state_q <= ST_SV_TILE_START;
                        end else begin
                            norm_token_q <= norm_token_q + SEQ_LEN_W'(1);
                            state_q <= ST_NORM_READ_REQ;
                        end
                    end
                end

                ST_SV_TILE_START: begin
                    sv_token_q <= '0;
                    state_q <= ST_SV_PROB_READ_REQ;
                end

                ST_SV_PROB_READ_REQ: begin
                    if (prob_rd_fire) begin
                        state_q <= ST_SV_PROB_READ_RSP;
                    end
                end

                ST_SV_PROB_READ_RSP: begin
                    if (prob_rsp_fire) begin
                        prob_q <= prob_rsp_score;
                        state_q <= ST_SV_SEND_TILE;
                    end
                end

                ST_SV_SEND_TILE: begin
                    if (pe_in_fire) begin
                        if (sv_token_q == (seq_len_q - SEQ_LEN_W'(1))) begin
                            state_q <= ST_SV_WAIT_RESULT;
                        end else begin
                            sv_token_q <= sv_token_q + SEQ_LEN_W'(1);
                            state_q <= ST_SV_PROB_READ_REQ;
                        end
                    end
                end

                ST_SV_WAIT_RESULT: begin
                    if (pe_out_fire) begin
                        output_vector_q <= pe_out_vector_fp32;
                        output_lane_mask_q <= pe_out_lane_mask;
                        output_base_dim_q <= sv_base_dim_comb;
                        output_status_q <= status_q | pe_out_status;
                        output_invalid_q <= invalid_q | pe_out_invalid;
                        output_meta_q <= meta_q;
                        output_last_q <= (sv_tile_q == TILE_W'(TILE_COUNT - 1));
                        output_valid_q <= 1'b1;
                        status_q <= status_q | pe_out_status;
                        invalid_q <= invalid_q | pe_out_invalid;
                        state_q <= ST_OUTPUT_TILE;
                    end
                end

                ST_OUTPUT_TILE: begin
                    if (output_fire) begin
                        if (sv_tile_q == TILE_W'(TILE_COUNT - 1)) begin
                            done_valid_q <= 1'b1;
                            state_q <= ST_DONE;
                        end else begin
                            sv_tile_q <= sv_tile_q + TILE_W'(1);
                            state_q <= ST_SV_TILE_START;
                        end
                    end
                end

                ST_DONE: begin
                    if (done_fire) begin
                        state_q <= ST_IDLE;
                    end
                end

                ST_ERROR: begin
                    if (done_fire) begin
                        state_q <= ST_IDLE;
                    end
                end

                default: state_q <= ST_IDLE;
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
            end else begin
                if (active_q) begin
                    perf_total_attention_cycles <= perf_total_attention_cycles + COUNTER_W'(1);
                end
                if ((state_q == ST_QK_SEND_TILE) || (state_q == ST_QK_WAIT_RESULT)) begin
                    perf_qk_cycles <= perf_qk_cycles + COUNTER_W'(1);
                    perf_qk_pe_busy_cycles <= perf_qk_pe_busy_cycles + COUNTER_W'(1);
                end
                if ((state_q == ST_SCALE_SEND) || (state_q == ST_SCALE_WAIT)) begin
                    perf_scale_cycles <= perf_scale_cycles + COUNTER_W'(1);
                end
                if ((state_q == ST_REDUCE_SEND) || reduction_busy) begin
                    perf_reduction_cycles <= perf_reduction_cycles + COUNTER_W'(1);
                end
                if (state_q == ST_WAIT_REDUCTION) begin
                    perf_reduction_finalize_cycles <= perf_reduction_finalize_cycles + COUNTER_W'(1);
                end
                if ((state_q inside {ST_NORM_START, ST_NORM_READ_REQ, ST_NORM_READ_RSP, ST_NORM_SEND_SCORE, ST_NORM_WAIT_PROB}) || norm_busy) begin
                    perf_normalization_cycles <= perf_normalization_cycles + COUNTER_W'(1);
                end
                if (state_q inside {ST_SV_TILE_START, ST_SV_PROB_READ_REQ, ST_SV_PROB_READ_RSP, ST_SV_SEND_TILE, ST_SV_WAIT_RESULT, ST_OUTPUT_TILE}) begin
                    perf_sv_cycles <= perf_sv_cycles + COUNTER_W'(1);
                end
                if ((pe_in_valid && !pe_in_ready) || ((state_q == ST_QK_WAIT_RESULT || state_q == ST_SV_WAIT_RESULT) && !pe_out_valid)) begin
                    perf_pe_stall_cycles <= perf_pe_stall_cycles + COUNTER_W'(1);
                end
                if ((scaler_in_valid && !scaler_in_ready) || ((state_q == ST_SCALE_WAIT) && !scaler_out_valid) ||
                    (reduction_in_valid && !reduction_in_ready) || ((state_q == ST_WAIT_REDUCTION) && !reduction_final_valid) ||
                    (norm_start_valid && !norm_start_ready) || (norm_score_valid && !norm_score_ready) ||
                    ((state_q == ST_NORM_WAIT_PROB) && !norm_prob_valid)) begin
                    perf_sfu_stall_cycles <= perf_sfu_stall_cycles + COUNTER_W'(1);
                end
                if ((score_wr_valid && !score_wr_ready) || (score_rd_valid && !score_rd_ready) ||
                    ((state_q == ST_NORM_READ_RSP) && !score_rsp_valid) ||
                    (prob_wr_valid && !prob_wr_ready) || (prob_rd_valid && !prob_rd_ready) ||
                    ((state_q == ST_SV_PROB_READ_RSP) && !prob_rsp_valid)) begin
                    perf_buffer_stall_cycles <= perf_buffer_stall_cycles + COUNTER_W'(1);
                end
                if ((output_valid && !output_ready) || (done_valid && !done_ready)) begin
                    perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
                end
                perf_score_buffer_peak_occupancy <= COUNTER_W'(score_peak_occupancy);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(start_valid && !start_ready && state_q != ST_IDLE))
                else $error("single_head_attention no_start_while_busy failed");
            assert (!(score_valid_length > seq_len_q && active_q))
                else $error("single_head_attention score_write_count <= seq_len failed");
            assert (!(reduction_processed_count > {26'd0, seq_len_q}))
                else $error("single_head_attention reduction_count <= seq_len failed");
            assert (!((state_q inside {ST_SV_PROB_READ_REQ, ST_SV_PROB_READ_RSP, ST_SV_SEND_TILE}) && (prob_valid_length < seq_len_q)))
                else $error("single_head_attention no SV update before probability valid failed");
            assert (!(output_valid && $isunknown({output_base_dim, output_vector_fp32, output_lane_mask, output_status, output_invalid, output_meta, output_last})))
                else $error("single_head_attention no unknown output when valid failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta})))
                else $error("single_head_attention no unknown done when valid failed");
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("single_head_attention output valid dropped under backpressure");
                assert ($stable({output_base_dim, output_vector_fp32, output_lane_mask, output_status, output_invalid, output_meta, output_last}))
                    else $error("single_head_attention output stable until ready failed");
            end
            if ($past(rst_n) && $past(done_valid && !done_ready)) begin
                assert (done_valid)
                    else $error("single_head_attention done valid dropped under backpressure");
                assert ($stable({done_status, done_invalid, done_meta}))
                    else $error("single_head_attention metadata stable failed");
            end
            if ($past(rst_n) && $past(load_valid && !load_ready)) begin
                assert (load_valid)
                    else $error("single_head_attention load valid dropped under backpressure");
                assert ($stable({load_kind, load_token, load_dim, load_data}))
                    else $error("single_head_attention load payload stable until ready failed");
            end
            assert (!(norm_prob_fire && (norm_prob_index != norm_token_q[TOKEN_W-1:0])))
                else $error("single_head_attention score index and V index match failed");
        end
    end
`endif
endmodule

`default_nettype wire

