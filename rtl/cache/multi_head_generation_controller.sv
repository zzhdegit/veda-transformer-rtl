`default_nettype none

module multi_head_generation_controller #(
    parameter int N_HEAD = 2,
    parameter int PE_NUM = 8,
    parameter int D_HEAD = 8,
    parameter int MAX_SEQ_LEN = 32,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
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

    output logic                         cache_rd_valid,
    input  logic                         cache_rd_ready,
    output logic [HEAD_W-1:0]            cache_rd_head,
    output logic [TOKEN_W-1:0]           cache_rd_token,
    output logic [DIM_W-1:0]             cache_rd_dim,
    output logic                         cache_provisional_read_enable,
    input  logic                         cache_rd_rsp_valid,
    output logic                         cache_rd_rsp_ready,
    input  logic [HEAD_W-1:0]            cache_rd_rsp_head,
    input  logic [TOKEN_W-1:0]           cache_rd_rsp_token,
    input  logic [DIM_W-1:0]             cache_rd_rsp_dim,
    input  logic [15:0]                  cache_rd_rsp_k_fp16,
    input  logic [15:0]                  cache_rd_rsp_v_fp16,

    output logic                         cache_append_valid,
    input  logic                         cache_append_ready,
    output logic [HEAD_W-1:0]            cache_append_head,
    output logic [TOKEN_W-1:0]           cache_append_token_index,
    output logic [DIM_W-1:0]             cache_append_dim,
    output logic [15:0]                  cache_append_k_fp16,
    output logic [15:0]                  cache_append_v_fp16,
    output logic                         cache_append_last_dim,
    output logic                         cache_append_last_head,
    output logic                         cache_append_complete,
    output logic                         cache_commit_valid,
    input  logic                         cache_commit_ready,
    output logic [TOKEN_W-1:0]           cache_commit_token_index,
    output logic                         cache_abort_valid,
    input  logic [SEQ_LEN_W-1:0]         cache_valid_seq_len,
    input  logic                         cache_append_incomplete,
    input  logic                         cache_provisional_valid,
    input  logic [N_HEAD-1:0]            cache_provisional_head_valid,
    input  logic [TOKEN_W-1:0]           cache_provisional_token_index,
    input  logic                         cache_full,
    input  logic                         cache_error_valid,
    output logic                         cache_error_ready,
    input  logic [7:0]                   cache_error_code,

    output logic                         sha_load_valid,
    input  logic                         sha_load_ready,
    output logic [1:0]                   sha_load_kind,
    output logic [TOKEN_W-1:0]           sha_load_token,
    output logic [DIM_W-1:0]             sha_load_dim,
    output logic [15:0]                  sha_load_data,
    output logic                         sha_start_valid,
    input  logic                         sha_start_ready,
    output logic [SEQ_LEN_W-1:0]         sha_start_seq_len,
    output logic [META_W-1:0]            sha_start_meta,
    input  logic                         sha_output_valid,
    output logic                         sha_output_ready,
    input  logic [DIM_W-1:0]             sha_output_base_dim,
    input  logic [PE_NUM*32-1:0]         sha_output_vector_fp32,
    input  logic [PE_NUM-1:0]            sha_output_lane_mask,
    input  logic [7:0]                   sha_output_status,
    input  logic                         sha_output_invalid,
    input  logic [META_W-1:0]            sha_output_meta,
    input  logic                         sha_output_last,
    input  logic                         sha_done_valid,
    output logic                         sha_done_ready,
    input  logic [7:0]                   sha_done_status,
    input  logic                         sha_done_invalid,
    input  logic [META_W-1:0]            sha_done_meta,
    input  logic [COUNTER_W-1:0]         sha_perf_total_attention_cycles,
    input  logic [COUNTER_W-1:0]         sha_perf_pe_stall_cycles,
    input  logic [COUNTER_W-1:0]         sha_perf_sfu_stall_cycles,

    output logic [COUNTER_W-1:0]         perf_generation_steps,
    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_per_head_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_head_switch_cycles,
    output logic [COUNTER_W-1:0]         perf_provisional_write_cycles,
    output logic [COUNTER_W-1:0]         perf_commit_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [1:0] LOAD_Q = 2'd0;
    localparam logic [1:0] LOAD_K = 2'd1;
    localparam logic [1:0] LOAD_V = 2'd2;

    localparam logic [7:0] STATUS_ORDER_ERROR = 8'h81;
    localparam logic [7:0] STATUS_CACHE_FULL  = 8'h82;
    localparam logic [7:0] STATUS_CACHE_ERR   = 8'h83;

    typedef enum logic [3:0] {
        ST_LOAD_TOKEN,
        ST_PROVISIONAL_APPEND,
        ST_LOAD_Q,
        ST_CACHE_READ_REQ,
        ST_CACHE_READ_RSP,
        ST_LOAD_K,
        ST_LOAD_V,
        ST_START_ATTENTION,
        ST_ATTENTION_RUN,
        ST_HEAD_SWITCH,
        ST_COMMIT_CURRENT_TOKEN,
        ST_ABORT_CURRENT_TOKEN,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [15:0] q_stage [0:N_HEAD-1][0:D_HEAD-1];
    logic [15:0] k_stage [0:N_HEAD-1][0:D_HEAD-1];
    logic [15:0] v_stage [0:N_HEAD-1][0:D_HEAD-1];

    logic [HEAD_W-1:0] input_head_q;
    logic [DIM_W-1:0] input_dim_q;
    logic [HEAD_W-1:0] append_head_q;
    logic [DIM_W-1:0] append_dim_q;
    logic [HEAD_W-1:0] current_head_q;
    logic [DIM_W-1:0] load_dim_q;
    logic [TOKEN_W-1:0] read_token_q;
    logic [DIM_W-1:0] read_dim_q;
    logic [SEQ_LEN_W-1:0] seq_len_snapshot_q;
    logic [15:0] read_k_q;
    logic [15:0] read_v_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [N_HEAD-1:0] head_done_seen_q;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len_q;

    wire token_fire = token_valid && token_ready;
    wire done_fire = done_valid && done_ready;
    wire cache_rd_fire = cache_rd_valid && cache_rd_ready;
    wire cache_rd_rsp_fire = cache_rd_rsp_valid && cache_rd_rsp_ready;
    wire cache_append_fire = cache_append_valid && cache_append_ready;
    wire cache_commit_fire = cache_commit_valid && cache_commit_ready;
    wire sha_load_fire = sha_load_valid && sha_load_ready;
    wire sha_start_fire = sha_start_valid && sha_start_ready;
    wire sha_done_fire = sha_done_valid && sha_done_ready;

    wire input_last_dim_expected = input_dim_q == DIM_W'(D_HEAD - 1);
    wire input_last_head_expected = (input_head_q == HEAD_W'(N_HEAD - 1)) && input_last_dim_expected;
    wire load_last_dim = load_dim_q == DIM_W'(D_HEAD - 1);
    wire read_last_dim = read_dim_q == DIM_W'(D_HEAD - 1);
    wire append_last_dim = append_dim_q == DIM_W'(D_HEAD - 1);
    wire append_last_head = (append_head_q == HEAD_W'(N_HEAD - 1)) && append_last_dim;
    wire current_last_head = current_head_q == HEAD_W'(N_HEAD - 1);
    wire [SEQ_LEN_W-1:0] attention_seq_len = seq_len_snapshot_q + SEQ_LEN_W'(1);
    wire read_last_token = SEQ_LEN_W'(read_token_q) == (attention_seq_len - SEQ_LEN_W'(1));
    wire using_sha_output = state_q == ST_ATTENTION_RUN;
    wire output_is_final_head_tile = sha_output_last && current_last_head;

    initial begin
        if (N_HEAD <= 0 || PE_NUM <= 0 || D_HEAD <= 0 || MAX_SEQ_LEN <= 0 ||
            META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "multi_head_generation_controller parameters must be positive");
        end
        if ((PE_NUM & (PE_NUM - 1)) != 0) begin
            $fatal(1, "multi_head_generation_controller PE_NUM must be a power of two");
        end
    end

    assign token_ready = (state_q == ST_LOAD_TOKEN) && !done_valid_q;

    assign cache_rd_valid = state_q == ST_CACHE_READ_REQ;
    assign cache_rd_head = current_head_q;
    assign cache_rd_token = read_token_q;
    assign cache_rd_dim = read_dim_q;
    assign cache_provisional_read_enable =
        (state_q inside {ST_CACHE_READ_REQ, ST_CACHE_READ_RSP, ST_LOAD_K, ST_LOAD_V,
                         ST_START_ATTENTION, ST_ATTENTION_RUN, ST_HEAD_SWITCH});
    assign cache_rd_rsp_ready = state_q == ST_CACHE_READ_RSP;

    assign cache_append_valid = state_q == ST_PROVISIONAL_APPEND;
    assign cache_append_head = append_head_q;
    assign cache_append_token_index = seq_len_snapshot_q[TOKEN_W-1:0];
    assign cache_append_dim = append_dim_q;
    assign cache_append_k_fp16 = k_stage[int'(append_head_q)][int'(append_dim_q)];
    assign cache_append_v_fp16 = v_stage[int'(append_head_q)][int'(append_dim_q)];
    assign cache_append_last_dim = append_last_dim;
    assign cache_append_last_head = append_last_head;
    assign cache_append_complete = append_last_head;
    assign cache_commit_valid = state_q == ST_COMMIT_CURRENT_TOKEN;
    assign cache_commit_token_index = seq_len_snapshot_q[TOKEN_W-1:0];
    assign cache_abort_valid = state_q == ST_ABORT_CURRENT_TOKEN;
    assign cache_error_ready = 1'b1;

    assign sha_load_valid = (state_q == ST_LOAD_Q) || (state_q == ST_LOAD_K) || (state_q == ST_LOAD_V);
    assign sha_load_kind = (state_q == ST_LOAD_Q) ? LOAD_Q : ((state_q == ST_LOAD_K) ? LOAD_K : LOAD_V);
    assign sha_load_token = (state_q == ST_LOAD_Q) ? '0 : read_token_q;
    assign sha_load_dim = (state_q == ST_LOAD_Q) ? load_dim_q : read_dim_q;
    assign sha_load_data = (state_q == ST_LOAD_Q) ? q_stage[int'(current_head_q)][int'(load_dim_q)] :
        ((state_q == ST_LOAD_K) ? read_k_q : read_v_q);
    assign sha_start_valid = state_q == ST_START_ATTENTION;
    assign sha_start_seq_len = attention_seq_len;
    assign sha_start_meta = meta_q;
    assign sha_output_ready = using_sha_output ? output_ready : 1'b0;
    assign sha_done_ready = state_q == ST_ATTENTION_RUN;

    assign output_valid = using_sha_output ? sha_output_valid : 1'b0;
    assign output_head = current_head_q;
    assign output_base_dim = sha_output_base_dim;
    assign output_vector_fp32 = sha_output_vector_fp32;
    assign output_lane_mask = sha_output_lane_mask;
    assign output_status = sha_output_status;
    assign output_invalid = sha_output_invalid;
    assign output_meta = sha_output_meta;
    assign output_last_tile = sha_output_last;
    assign output_last_head = output_is_final_head_tile;
    assign output_last_token = output_is_final_head_tile;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;
    assign done_valid_seq_len = done_valid_seq_len_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_LOAD_TOKEN;
            input_head_q <= '0;
            input_dim_q <= '0;
            append_head_q <= '0;
            append_dim_q <= '0;
            current_head_q <= '0;
            load_dim_q <= '0;
            read_token_q <= '0;
            read_dim_q <= '0;
            seq_len_snapshot_q <= '0;
            read_k_q <= 16'd0;
            read_v_q <= 16'd0;
            meta_q <= '0;
            status_q <= 8'd0;
            invalid_q <= 1'b0;
            head_done_seen_q <= '0;
            done_valid_q <= 1'b0;
            done_status_q <= 8'd0;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
            done_valid_seq_len_q <= '0;
        end else begin
            if (cache_error_valid) begin
                status_q <= status_q | cache_error_code | STATUS_CACHE_ERR;
                invalid_q <= 1'b1;
            end

            unique case (state_q)
                ST_LOAD_TOKEN: begin
                    if (token_fire) begin
                        if ((input_head_q == '0) && (input_dim_q == '0)) begin
                            meta_q <= token_meta;
                            status_q <= 8'd0;
                            invalid_q <= 1'b0;
                            head_done_seen_q <= '0;
                        end

                        if ((token_head != input_head_q) ||
                            (token_dim != input_dim_q) ||
                            (token_last_dim != input_last_dim_expected) ||
                            (token_last_head != input_last_head_expected)) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= STATUS_ORDER_ERROR;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= ((input_head_q == '0) && (input_dim_q == '0)) ? token_meta : meta_q;
                            done_valid_seq_len_q <= cache_valid_seq_len;
                            input_head_q <= '0;
                            input_dim_q <= '0;
                            state_q <= ST_DONE;
                        end else begin
                            q_stage[int'(token_head)][int'(token_dim)] <= token_q_fp16;
                            k_stage[int'(token_head)][int'(token_dim)] <= token_k_fp16;
                            v_stage[int'(token_head)][int'(token_dim)] <= token_v_fp16;

                            if (input_last_head_expected) begin
                                seq_len_snapshot_q <= cache_valid_seq_len;
                                input_head_q <= '0;
                                input_dim_q <= '0;
                                if (cache_full) begin
                                    done_valid_q <= 1'b1;
                                    done_status_q <= STATUS_CACHE_FULL;
                                    done_invalid_q <= 1'b1;
                                    done_meta_q <= meta_q;
                                    done_valid_seq_len_q <= cache_valid_seq_len;
                                    state_q <= ST_DONE;
                                end else begin
                                    append_head_q <= '0;
                                    append_dim_q <= '0;
                                    state_q <= ST_PROVISIONAL_APPEND;
                                end
                            end else if (input_last_dim_expected) begin
                                input_head_q <= input_head_q + HEAD_W'(1);
                                input_dim_q <= '0;
                            end else begin
                                input_dim_q <= input_dim_q + DIM_W'(1);
                            end
                        end
                    end
                end

                ST_PROVISIONAL_APPEND: begin
                    if (cache_append_fire) begin
                        if (append_last_head) begin
                            current_head_q <= '0;
                            load_dim_q <= '0;
                            state_q <= ST_LOAD_Q;
                        end else if (append_last_dim) begin
                            append_head_q <= append_head_q + HEAD_W'(1);
                            append_dim_q <= '0;
                        end else begin
                            append_dim_q <= append_dim_q + DIM_W'(1);
                        end
                    end
                end

                ST_LOAD_Q: begin
                    if (sha_load_fire) begin
                        if (load_last_dim) begin
                            read_token_q <= '0;
                            read_dim_q <= '0;
                            state_q <= ST_CACHE_READ_REQ;
                        end else begin
                            load_dim_q <= load_dim_q + DIM_W'(1);
                        end
                    end
                end

                ST_CACHE_READ_REQ: begin
                    if (cache_rd_fire) begin
                        state_q <= ST_CACHE_READ_RSP;
                    end
                end

                ST_CACHE_READ_RSP: begin
                    if (cache_rd_rsp_fire) begin
                        read_k_q <= cache_rd_rsp_k_fp16;
                        read_v_q <= cache_rd_rsp_v_fp16;
                        state_q <= ST_LOAD_K;
                    end
                end

                ST_LOAD_K: begin
                    if (sha_load_fire) begin
                        state_q <= ST_LOAD_V;
                    end
                end

                ST_LOAD_V: begin
                    if (sha_load_fire) begin
                        if (read_last_dim && read_last_token) begin
                            state_q <= ST_START_ATTENTION;
                        end else begin
                            if (read_last_dim) begin
                                read_dim_q <= '0;
                                read_token_q <= read_token_q + TOKEN_W'(1);
                            end else begin
                                read_dim_q <= read_dim_q + DIM_W'(1);
                            end
                            state_q <= ST_CACHE_READ_REQ;
                        end
                    end
                end

                ST_START_ATTENTION: begin
                    if (sha_start_fire) begin
                        state_q <= ST_ATTENTION_RUN;
                    end
                end

                ST_ATTENTION_RUN: begin
                    if (sha_done_fire) begin
                        status_q <= status_q | sha_done_status;
                        invalid_q <= invalid_q | sha_done_invalid;
                        head_done_seen_q[int'(current_head_q)] <= 1'b1;
                        if (invalid_q || sha_done_invalid) begin
                            state_q <= ST_ABORT_CURRENT_TOKEN;
                        end else if (current_last_head) begin
                            state_q <= ST_COMMIT_CURRENT_TOKEN;
                        end else begin
                            state_q <= ST_HEAD_SWITCH;
                        end
                    end
                end

                ST_HEAD_SWITCH: begin
                    current_head_q <= current_head_q + HEAD_W'(1);
                    load_dim_q <= '0;
                    read_token_q <= '0;
                    read_dim_q <= '0;
                    state_q <= ST_LOAD_Q;
                end

                ST_COMMIT_CURRENT_TOKEN: begin
                    if (cache_commit_fire) begin
                        done_valid_q <= 1'b1;
                        done_status_q <= status_q;
                        done_invalid_q <= invalid_q;
                        done_meta_q <= meta_q;
                        done_valid_seq_len_q <= seq_len_snapshot_q + SEQ_LEN_W'(1);
                        state_q <= ST_DONE;
                    end
                end

                ST_ABORT_CURRENT_TOKEN: begin
                    done_valid_q <= 1'b1;
                    done_status_q <= status_q;
                    done_invalid_q <= 1'b1;
                    done_meta_q <= meta_q;
                    done_valid_seq_len_q <= seq_len_snapshot_q;
                    state_q <= ST_DONE;
                end

                ST_DONE: begin
                    if (done_fire) begin
                        done_valid_q <= 1'b0;
                        done_status_q <= 8'd0;
                        done_invalid_q <= 1'b0;
                        done_meta_q <= '0;
                        done_valid_seq_len_q <= cache_valid_seq_len;
                        input_head_q <= '0;
                        input_dim_q <= '0;
                        head_done_seen_q <= '0;
                        state_q <= ST_LOAD_TOKEN;
                    end
                end

                default: state_q <= ST_LOAD_TOKEN;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_generation_steps <= '0;
            perf_total_cycles <= '0;
            perf_per_head_attention_cycles <= '0;
            perf_head_switch_cycles <= '0;
            perf_provisional_write_cycles <= '0;
            perf_commit_cycles <= '0;
            perf_pe_stall_cycles <= '0;
            perf_sfu_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if ((state_q != ST_LOAD_TOKEN) || token_valid || done_valid_q) begin
                perf_total_cycles <= perf_total_cycles + COUNTER_W'(1);
            end
            if (state_q == ST_PROVISIONAL_APPEND) begin
                perf_provisional_write_cycles <= perf_provisional_write_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_LOAD_Q, ST_CACHE_READ_REQ, ST_CACHE_READ_RSP,
                                ST_LOAD_K, ST_LOAD_V, ST_START_ATTENTION,
                                ST_ATTENTION_RUN}) begin
                perf_per_head_attention_cycles <= perf_per_head_attention_cycles + COUNTER_W'(1);
            end
            if (state_q == ST_HEAD_SWITCH) begin
                perf_head_switch_cycles <= perf_head_switch_cycles + COUNTER_W'(1);
            end
            if (state_q == ST_COMMIT_CURRENT_TOKEN) begin
                perf_commit_cycles <= perf_commit_cycles + COUNTER_W'(1);
            end
            if (sha_done_fire) begin
                perf_pe_stall_cycles <= perf_pe_stall_cycles + sha_perf_pe_stall_cycles;
                perf_sfu_stall_cycles <= perf_sfu_stall_cycles + sha_perf_sfu_stall_cycles;
            end
            if (output_valid && !output_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end
            if (cache_commit_fire) begin
                perf_generation_steps <= perf_generation_steps + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    logic [31:0] accepted_token_count;
    logic [HEAD_W-1:0] current_head_prev;
    logic sha_done_prev;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accepted_token_count <= 32'd0;
            current_head_prev <= '0;
            sha_done_prev <= 1'b0;
        end else begin
            if (token_fire && input_last_head_expected && (token_head == input_head_q) &&
                (token_dim == input_dim_q) && (token_last_dim == input_last_dim_expected) &&
                (token_last_head == input_last_head_expected)) begin
                accepted_token_count <= accepted_token_count + 32'd1;
            end
            current_head_prev <= current_head_q;
            sha_done_prev <= sha_done_fire;
        end
    end

    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(token_fire && ((token_head != input_head_q) || (token_dim != input_dim_q) ||
                                     (token_last_dim != input_last_dim_expected) ||
                                     (token_last_head != input_last_head_expected))))
                else $error("multi_head_generation_controller head_and_dim_order_legal failed");
            assert (!(sha_start_valid && (sha_start_seq_len == '0)))
                else $error("multi_head_generation_controller no zero-length Stage 3 start failed");
            assert (!(sha_start_valid && !cache_provisional_valid))
                else $error("multi_head_generation_controller no_head_start_before_all_provisional_complete failed");
            assert (!(sha_start_valid && cache_append_incomplete))
                else $error("multi_head_generation_controller no_attention_start_during_incomplete_append failed");
            assert (!(sha_start_valid && (sha_start_seq_len != (cache_valid_seq_len + SEQ_LEN_W'(1)))))
                else $error("multi_head_generation_controller each_head_attention_seq_len_equals_valid_plus_one failed");
            assert (!(cache_append_valid && (cache_append_token_index != cache_valid_seq_len[TOKEN_W-1:0])))
                else $error("multi_head_generation_controller provisional_head_token_index_legal failed");
            assert (!(cache_commit_valid && (head_done_seen_q != {N_HEAD{1'b1}})))
                else $error("multi_head_generation_controller no_commit_before_all_heads_done failed");
            assert (!(cache_commit_valid && (head_done_seen_q != {N_HEAD{1'b1}})))
                else $error("multi_head_generation_controller no_partial_head_commit failed");
            assert (!((state_q inside {ST_PROVISIONAL_APPEND, ST_LOAD_Q, ST_CACHE_READ_REQ,
                                       ST_CACHE_READ_RSP, ST_LOAD_K, ST_LOAD_V,
                                       ST_START_ATTENTION, ST_ATTENTION_RUN, ST_HEAD_SWITCH}) &&
                      (cache_valid_seq_len != seq_len_snapshot_q)))
                else $error("multi_head_generation_controller all_heads_share_valid_seq_len failed");
            assert (!(cache_rd_valid && (SEQ_LEN_W'(cache_rd_token) >= attention_seq_len)))
                else $error("multi_head_generation_controller no_read_beyond_valid_seq_len failed");
            assert (!(cache_rd_valid && (SEQ_LEN_W'(cache_rd_token) == seq_len_snapshot_q) &&
                     !cache_provisional_head_valid[int'(cache_rd_head)]))
                else $error("multi_head_generation_controller no_provisional_visibility_before_complete failed");
            assert (!(cache_full && cache_append_valid))
                else $error("multi_head_generation_controller no_overwrite_when_full failed");
            assert (!(state_q == ST_HEAD_SWITCH && !head_done_seen_q[int'(current_head_q)]))
                else $error("multi_head_generation_controller no_next_head_before_current_done failed");
            assert (!(state_q == ST_HEAD_SWITCH && (current_head_q != current_head_prev) && !sha_done_prev))
                else $error("multi_head_generation_controller no_next_head_before_current_done transition failed");
            assert (!(output_valid && (output_head != current_head_q)))
                else $error("multi_head_generation_controller output_head_order_preserved failed");
            assert (!(output_valid && $isunknown({output_head, output_base_dim, output_vector_fp32,
                                                  output_lane_mask, output_status, output_invalid,
                                                  output_meta, output_last_tile, output_last_head,
                                                  output_last_token})))
                else $error("multi_head_generation_controller no unknown output when valid failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta, done_valid_seq_len})))
                else $error("multi_head_generation_controller no unknown done when valid failed");
            assert (COUNTER_W'(accepted_token_count) >= perf_generation_steps)
                else $error("multi_head_generation_controller transaction count conserved failed");

            if ($past(rst_n) && $past(token_valid && !token_ready)) begin
                assert (token_valid)
                    else $error("multi_head_generation_controller token valid dropped under backpressure");
                assert ($stable({token_head, token_dim, token_q_fp16, token_k_fp16, token_v_fp16,
                                 token_last_dim, token_last_head, token_meta}))
                    else $error("multi_head_generation_controller token payload changed under backpressure");
            end
            if ($past(rst_n) && $past(output_valid && !output_ready)) begin
                assert (output_valid)
                    else $error("multi_head_generation_controller output valid dropped under backpressure");
                assert ($stable({output_head, output_base_dim, output_vector_fp32,
                                 output_lane_mask, output_status, output_invalid, output_meta,
                                 output_last_tile, output_last_head, output_last_token}))
                    else $error("multi_head_generation_controller output stable under backpressure failed");
            end
            if ($past(rst_n) && $past(done_valid && !done_ready)) begin
                assert (done_valid)
                    else $error("multi_head_generation_controller done valid dropped under backpressure");
                assert ($stable({done_status, done_invalid, done_meta, done_valid_seq_len}))
                    else $error("multi_head_generation_controller done stable under backpressure failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
