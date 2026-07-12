`default_nettype none

module kv_cache_manager #(
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    parameter int COUNTER_W = 64,
    localparam int TOKEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int DEPTH = MAX_SEQ_LEN * D_HEAD,
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  rd_valid,
    output logic                  rd_ready,
    input  logic [TOKEN_W-1:0]    rd_token,
    input  logic [DIM_W-1:0]      rd_dim,
    input  logic                  provisional_read_enable,
    output logic                  rd_rsp_valid,
    input  logic                  rd_rsp_ready,
    output logic [TOKEN_W-1:0]    rd_rsp_token,
    output logic [DIM_W-1:0]      rd_rsp_dim,
    output logic [15:0]           rd_rsp_k_fp16,
    output logic [15:0]           rd_rsp_v_fp16,

    input  logic                  append_valid,
    output logic                  append_ready,
    input  logic [TOKEN_W-1:0]    append_token_index,
    input  logic [DIM_W-1:0]      append_dim,
    input  logic [15:0]           append_k_fp16,
    input  logic [15:0]           append_v_fp16,
    input  logic                  append_last_dim,
    input  logic                  append_complete,

    input  logic                  commit_valid,
    output logic                  commit_ready,
    input  logic [TOKEN_W-1:0]    commit_token_index,
    input  logic                  abort_valid,

    output logic [SEQ_LEN_W-1:0]  valid_seq_len,
    output logic                  append_incomplete,
    output logic                  provisional_valid,
    output logic [TOKEN_W-1:0]    provisional_token_index,
    output logic                  cache_full,

    output logic                  error_valid,
    input  logic                  error_ready,
    output logic [7:0]            error_code,

    output logic [COUNTER_W-1:0]  perf_cache_read_cycles,
    output logic [COUNTER_W-1:0]  perf_cache_write_cycles,
    output logic [COUNTER_W-1:0]  perf_cache_stall_cycles,
    output logic [SEQ_LEN_W-1:0]  perf_peak_valid_seq_len
);
    localparam logic [7:0] STATUS_READ_RANGE      = 8'h91;
    localparam logic [7:0] STATUS_APPEND_RANGE    = 8'h92;
    localparam logic [7:0] STATUS_APPEND_ORDER    = 8'h93;
    localparam logic [7:0] STATUS_APPEND_COMPLETE = 8'h94;
    localparam logic [7:0] STATUS_CACHE_FULL      = 8'h95;
    localparam logic [7:0] STATUS_COMMIT          = 8'h96;

    logic [15:0] k_mem [0:DEPTH-1];
    logic [15:0] v_mem [0:DEPTH-1];

    logic [SEQ_LEN_W-1:0] valid_seq_len_q;
    logic append_in_progress_q;
    logic provisional_complete_q;
    logic [TOKEN_W-1:0] provisional_token_q;
    logic [DIM_W-1:0] expected_dim_q;

    logic rd_rsp_valid_q;
    logic [TOKEN_W-1:0] rd_rsp_token_q;
    logic [DIM_W-1:0] rd_rsp_dim_q;
    logic [15:0] rd_rsp_k_q;
    logic [15:0] rd_rsp_v_q;

    logic error_valid_q;
    logic [7:0] error_code_q;

    logic [ADDR_W-1:0] rd_address;
    logic rd_token_in_range;
    logic rd_dim_in_range;
    logic rd_address_in_range;

    logic [ADDR_W-1:0] append_address;
    logic append_token_in_range;
    logic append_dim_in_range;
    logic append_address_in_range;

    wire rd_fire = rd_valid && rd_ready;
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;
    wire append_fire = append_valid && append_ready;
    wire commit_fire = commit_valid && commit_ready;
    wire error_fire = error_valid && error_ready;
    wire final_append_dim = append_dim == DIM_W'(D_HEAD - 1);

    kv_address_generator #(
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) u_rd_addr (
        .token_index      (rd_token),
        .dimension        (rd_dim),
        .address          (rd_address),
        .token_in_range   (rd_token_in_range),
        .dim_in_range     (rd_dim_in_range),
        .address_in_range (rd_address_in_range)
    );

    kv_address_generator #(
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) u_append_addr (
        .token_index      (append_token_index),
        .dimension        (append_dim),
        .address          (append_address),
        .token_in_range   (append_token_in_range),
        .dim_in_range     (append_dim_in_range),
        .address_in_range (append_address_in_range)
    );

    initial begin
        if (D_HEAD <= 0 || MAX_SEQ_LEN <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "kv_cache_manager parameters must be positive");
        end
    end

    assign valid_seq_len = valid_seq_len_q;
    assign append_incomplete = append_in_progress_q;
    assign provisional_valid = provisional_complete_q;
    assign provisional_token_index = provisional_token_q;
    assign cache_full = (valid_seq_len_q == SEQ_LEN_W'(MAX_SEQ_LEN));
    assign rd_ready = !append_in_progress_q && !rd_rsp_valid_q;
    assign append_ready = !rd_rsp_valid_q && !cache_full && !provisional_complete_q;
    assign commit_ready = provisional_complete_q && !rd_rsp_valid_q;

    assign rd_rsp_valid = rd_rsp_valid_q;
    assign rd_rsp_token = rd_rsp_token_q;
    assign rd_rsp_dim = rd_rsp_dim_q;
    assign rd_rsp_k_fp16 = rd_rsp_k_q;
    assign rd_rsp_v_fp16 = rd_rsp_v_q;
    assign error_valid = error_valid_q;
    assign error_code = error_code_q;

    function automatic logic current_provisional_read;
        begin
            current_provisional_read =
                (SEQ_LEN_W'(rd_token) == valid_seq_len_q) &&
                provisional_complete_q &&
                provisional_read_enable;
        end
    endfunction

    function automatic logic read_allowed;
        begin
            if (!rd_address_in_range || !rd_dim_in_range || !rd_token_in_range) begin
                read_allowed = 1'b0;
            end else begin
                read_allowed =
                    (SEQ_LEN_W'(rd_token) < valid_seq_len_q) ||
                    current_provisional_read();
            end
        end
    endfunction

    function automatic logic append_order_legal;
        begin
            if (!append_token_in_range || !append_dim_in_range || !append_address_in_range) begin
                append_order_legal = 1'b0;
            end else if (!append_in_progress_q) begin
                append_order_legal =
                    !provisional_complete_q &&
                    (append_token_index == valid_seq_len_q[TOKEN_W-1:0]) &&
                    (append_dim == '0);
            end else begin
                append_order_legal =
                    (append_token_index == provisional_token_q) &&
                    (append_dim == expected_dim_q);
            end
        end
    endfunction

    function automatic logic append_complete_legal;
        begin
            if (final_append_dim) begin
                append_complete_legal = append_last_dim && append_complete;
            end else begin
                append_complete_legal = !append_last_dim && !append_complete;
            end
        end
    endfunction

    function automatic logic commit_legal;
        begin
            commit_legal =
                provisional_complete_q &&
                (commit_token_index == provisional_token_q) &&
                (SEQ_LEN_W'(commit_token_index) == valid_seq_len_q) &&
                !cache_full;
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_seq_len_q <= '0;
            append_in_progress_q <= 1'b0;
            provisional_complete_q <= 1'b0;
            provisional_token_q <= '0;
            expected_dim_q <= '0;
            rd_rsp_valid_q <= 1'b0;
            rd_rsp_token_q <= '0;
            rd_rsp_dim_q <= '0;
            rd_rsp_k_q <= 16'd0;
            rd_rsp_v_q <= 16'd0;
            error_valid_q <= 1'b0;
            error_code_q <= 8'd0;
        end else begin
            if (rd_rsp_fire) begin
                rd_rsp_valid_q <= 1'b0;
            end
            if (error_fire) begin
                error_valid_q <= 1'b0;
            end

            if (abort_valid) begin
                append_in_progress_q <= 1'b0;
                provisional_complete_q <= 1'b0;
                provisional_token_q <= '0;
                expected_dim_q <= '0;
            end else begin
                if (rd_fire) begin
                    if (!read_allowed()) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_READ_RANGE;
                    end else begin
                        rd_rsp_valid_q <= 1'b1;
                        rd_rsp_token_q <= rd_token;
                        rd_rsp_dim_q <= rd_dim;
                        rd_rsp_k_q <= k_mem[rd_address];
                        rd_rsp_v_q <= v_mem[rd_address];
                    end
                end

                if (append_fire) begin
                    if (cache_full) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_CACHE_FULL;
                    end else if (!append_token_in_range || !append_dim_in_range || !append_address_in_range) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_APPEND_RANGE;
                        append_in_progress_q <= 1'b0;
                        provisional_complete_q <= 1'b0;
                        expected_dim_q <= '0;
                    end else if (!append_order_legal()) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_APPEND_ORDER;
                        append_in_progress_q <= 1'b0;
                        provisional_complete_q <= 1'b0;
                        expected_dim_q <= '0;
                    end else if (!append_complete_legal()) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_APPEND_COMPLETE;
                        append_in_progress_q <= 1'b0;
                        provisional_complete_q <= 1'b0;
                        expected_dim_q <= '0;
                    end else begin
                        k_mem[append_address] <= append_k_fp16;
                        v_mem[append_address] <= append_v_fp16;
                        provisional_token_q <= append_token_index;
                        if (final_append_dim) begin
                            append_in_progress_q <= 1'b0;
                            provisional_complete_q <= 1'b1;
                            expected_dim_q <= '0;
                        end else begin
                            append_in_progress_q <= 1'b1;
                            provisional_complete_q <= 1'b0;
                            expected_dim_q <= append_dim + DIM_W'(1);
                        end
                    end
                end

                if (commit_fire) begin
                    if (!commit_legal()) begin
                        error_valid_q <= 1'b1;
                        error_code_q <= STATUS_COMMIT;
                        append_in_progress_q <= 1'b0;
                        provisional_complete_q <= 1'b0;
                        expected_dim_q <= '0;
                    end else begin
                        valid_seq_len_q <= valid_seq_len_q + SEQ_LEN_W'(1);
                        append_in_progress_q <= 1'b0;
                        provisional_complete_q <= 1'b0;
                        provisional_token_q <= '0;
                        expected_dim_q <= '0;
                    end
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_cache_read_cycles <= '0;
            perf_cache_write_cycles <= '0;
            perf_cache_stall_cycles <= '0;
            perf_peak_valid_seq_len <= '0;
        end else begin
            if (rd_fire || rd_rsp_valid_q) begin
                perf_cache_read_cycles <= perf_cache_read_cycles + COUNTER_W'(1);
            end
            if (append_fire) begin
                perf_cache_write_cycles <= perf_cache_write_cycles + COUNTER_W'(1);
            end
            if ((rd_valid && !rd_ready) || (rd_rsp_valid && !rd_rsp_ready) ||
                (append_valid && !append_ready) || (commit_valid && !commit_ready)) begin
                perf_cache_stall_cycles <= perf_cache_stall_cycles + COUNTER_W'(1);
            end
            if (valid_seq_len_q > perf_peak_valid_seq_len) begin
                perf_peak_valid_seq_len <= valid_seq_len_q;
            end
        end
    end

`ifndef SYNTHESIS
    logic [SEQ_LEN_W-1:0] valid_seq_len_prev;
    logic commit_success_prev;
    logic abort_prev;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_seq_len_prev <= '0;
            commit_success_prev <= 1'b0;
            abort_prev <= 1'b0;
        end else begin
            valid_seq_len_prev <= valid_seq_len_q;
            commit_success_prev <= commit_fire && commit_legal();
            abort_prev <= abort_valid;
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(rd_fire && !read_allowed()))
                else $error("kv_cache_manager no_read_beyond_valid_seq_len failed");
            assert (!(append_fire && cache_full))
                else $error("kv_cache_manager no_append_beyond_capacity failed");
            assert (!(append_fire && append_order_legal() && (append_token_index < valid_seq_len_q[TOKEN_W-1:0])))
                else $error("kv_cache_manager no overwrite of valid token failed");
            assert (!(append_fire && append_order_legal() && append_complete && !final_append_dim))
                else $error("kv_cache_manager dimension order legal failed");
            assert (!($past(rst_n) && valid_seq_len_q > valid_seq_len_prev && !commit_success_prev))
                else $error("kv_cache_manager no_valid_seq_len_increment_before_commit failed");
            assert (!((append_in_progress_q || provisional_complete_q) &&
                      (SEQ_LEN_W'(provisional_token_q) != valid_seq_len_q)))
                else $error("kv_cache_manager provisional_token_index_equals_valid_seq_len failed");
            assert (!(rd_fire && (SEQ_LEN_W'(rd_token) == valid_seq_len_q) && !provisional_complete_q))
                else $error("kv_cache_manager no_provisional_visibility_before_complete failed");
            assert (!(rd_fire && (SEQ_LEN_W'(rd_token) == valid_seq_len_q) && !provisional_read_enable))
                else $error("kv_cache_manager current_token_read_allowed_only_for_active_operation failed");
            assert (!(append_fire && cache_full))
                else $error("kv_cache_manager no_overwrite_when_full failed");
            assert (!($past(rst_n) && abort_prev && (append_in_progress_q || provisional_complete_q)))
                else $error("kv_cache_manager abort_or_reset_clears_provisional_state failed");
            assert (!(rd_rsp_valid && $isunknown({rd_rsp_token, rd_rsp_dim, rd_rsp_k_fp16, rd_rsp_v_fp16})))
                else $error("kv_cache_manager no unknown output when valid failed");
            assert (!(rd_fire && !rd_address_in_range))
                else $error("kv_cache_manager cache address in range failed");
            assert (!(append_fire && append_order_legal() && !append_address_in_range))
                else $error("kv_cache_manager append address in range failed");

            if ($past(rst_n) && $past(rd_rsp_valid && !rd_rsp_ready)) begin
                assert (rd_rsp_valid)
                    else $error("kv_cache_manager read response valid dropped under backpressure");
                assert ($stable({rd_rsp_token, rd_rsp_dim, rd_rsp_k_fp16, rd_rsp_v_fp16}))
                    else $error("kv_cache_manager read response changed under backpressure");
            end
            if ($past(rst_n) && $past(error_valid && !error_ready)) begin
                assert (error_valid)
                    else $error("kv_cache_manager error valid dropped under backpressure");
                assert ($stable(error_code))
                    else $error("kv_cache_manager error code changed under backpressure");
            end
        end
    end
`endif
endmodule

`default_nettype wire
