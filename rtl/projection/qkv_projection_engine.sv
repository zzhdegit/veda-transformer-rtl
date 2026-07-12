`default_nettype none

module qkv_projection_engine #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    parameter int PE_NUM = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LEN_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         input_valid,
    output logic                         input_ready,
    input  logic [MODEL_W-1:0]           input_dim,
    input  logic [15:0]                  input_data_fp16,
    input  logic                         input_last,
    input  logic [META_W-1:0]            input_meta,

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

    output logic                         qkv_valid,
    input  logic                         qkv_ready,
    output logic [HEAD_W-1:0]            qkv_head,
    output logic [DIM_W-1:0]             qkv_dim,
    output logic [15:0]                  qkv_q_fp16,
    output logic [15:0]                  qkv_k_fp16,
    output logic [15:0]                  qkv_v_fp16,
    output logic                         qkv_last_dim,
    output logic                         qkv_last_head,
    output logic [META_W-1:0]            qkv_meta,

    output logic                         done_valid,
    input  logic                         done_ready,
    output logic [7:0]                   done_status,
    output logic                         done_invalid,
    output logic [META_W-1:0]            done_meta,

    output logic [COUNTER_W-1:0]         perf_q_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_k_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_v_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_qkv_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_weight_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_output_stall_cycles
);
    localparam logic [1:0] KIND_Q = 2'd0;
    localparam logic [1:0] KIND_K = 2'd1;
    localparam logic [1:0] KIND_V = 2'd2;
    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_ORDER = 8'hB1;
    localparam logic [7:0] STATUS_PROJECT = 8'hB2;

    typedef enum logic [3:0] {
        ST_LOAD_X,
        ST_WAIT_START,
        ST_START_Q,
        ST_RUN_Q,
        ST_START_K,
        ST_RUN_K,
        ST_START_V,
        ST_RUN_V,
        ST_STREAM,
        ST_DONE
    } state_e;

    state_e state_q;

    logic [MODEL_W-1:0] expected_dim_q;
    logic hidden_complete_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic [1:0] active_kind_q;
    logic [MODEL_W-1:0] stream_index_q;

    logic proj_input_valid;
    logic proj_input_ready;
    logic proj_input_commit;
    logic proj_weight_ready;
    logic proj_start_valid;
    logic proj_start_ready;
    logic [1:0] proj_start_kind;
    logic proj_output_valid;
    logic proj_output_ready;
    logic [1:0] proj_output_kind;
    logic [MODEL_W-1:0] proj_output_index;
    logic [31:0] proj_output_data;
    logic [7:0] proj_output_status;
    logic proj_output_invalid;
    logic [META_W-1:0] proj_output_meta;
    logic proj_output_last;
    logic proj_done_valid;
    logic proj_done_ready;
    logic [7:0] proj_done_status;
    logic proj_done_invalid;
    logic [META_W-1:0] proj_done_meta;
    logic [COUNTER_W-1:0] proj_perf_total_cycles;
    logic [COUNTER_W-1:0] proj_perf_pe_stall_cycles;
    logic [COUNTER_W-1:0] proj_perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] proj_perf_output_stall_cycles;

    logic quant_in_valid;
    logic quant_in_ready;
    logic quant_out_valid;
    logic quant_out_ready;
    logic [15:0] quant_out_data;
    logic quant_out_invalid;
    logic quant_out_overflow;
    logic quant_out_underflow_or_ftz;
    logic quant_out_inexact;
    logic [META_W+2+MODEL_W-1:0] quant_out_meta;
    logic quant_out_last;
    logic [1:0] quant_write_kind;
    logic [MODEL_W-1:0] quant_write_index;

    logic staging_clear;
    logic staging_write_valid;
    logic [MODEL_W-1:0] staging_read_index;
    logic [HEAD_W-1:0] staging_read_head;
    logic [DIM_W-1:0] staging_read_dim;
    logic [15:0] staging_read_q;
    logic [15:0] staging_read_k;
    logic [15:0] staging_read_v;
    logic staging_read_complete;
    logic q_complete;
    logic k_complete;
    logic v_complete;
    logic all_complete;
    logic staging_error;

    logic done_valid_q;
    logic [7:0] done_status_q;
    logic done_invalid_q;
    logic [META_W-1:0] done_meta_q;

    wire input_fire = input_valid && input_ready;
    wire start_fire = start_valid && start_ready;
    wire proj_start_fire = proj_start_valid && proj_start_ready;
    wire proj_output_fire = proj_output_valid && proj_output_ready;
    wire proj_done_fire = proj_done_valid && proj_done_ready;
    wire quant_out_fire = quant_out_valid && quant_out_ready;
    wire qkv_fire = qkv_valid && qkv_ready;
    wire done_fire = done_valid && done_ready;
    wire input_last_expected = input_dim == MODEL_W'(D_MODEL - 1);
    wire stream_last = stream_index_q == MODEL_W'(D_MODEL - 1);

    initial begin
        if (N_HEAD <= 0 || D_HEAD <= 0 || PE_NUM <= 0 || META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "qkv_projection_engine parameters must be positive");
        end
        if (D_MODEL != N_HEAD * D_HEAD) begin
            $fatal(1, "qkv_projection_engine d_model_equals_n_head_times_d_head failed");
        end
    end

    assign input_ready = (state_q == ST_LOAD_X) && proj_input_ready;
    assign weight_ready = (state_q inside {ST_LOAD_X, ST_WAIT_START}) && proj_weight_ready;
    assign start_ready = (state_q == ST_WAIT_START) && hidden_complete_q && !done_valid_q;
    assign proj_input_valid = input_valid && input_ready;
    assign proj_input_commit = input_fire && input_last;

    assign proj_start_valid =
        (state_q == ST_START_Q) || (state_q == ST_START_K) || (state_q == ST_START_V);
    assign proj_start_kind =
        (state_q == ST_START_Q) ? KIND_Q : ((state_q == ST_START_K) ? KIND_K : KIND_V);
    assign proj_output_ready = quant_in_ready;
    assign quant_in_valid = proj_output_valid;
    assign quant_out_ready = 1'b1;
    assign staging_write_valid = quant_out_fire;
    assign quant_write_kind = quant_out_meta[META_W+MODEL_W +: 2];
    assign quant_write_index = quant_out_meta[META_W +: MODEL_W];
    assign staging_clear = start_fire;
    assign staging_read_index = stream_index_q;

    assign qkv_valid = (state_q == ST_STREAM) && staging_read_complete;
    assign qkv_head = staging_read_head;
    assign qkv_dim = staging_read_dim;
    assign qkv_q_fp16 = staging_read_q;
    assign qkv_k_fp16 = staging_read_k;
    assign qkv_v_fp16 = staging_read_v;
    assign qkv_last_dim = staging_read_dim == DIM_W'(D_HEAD - 1);
    assign qkv_last_head = stream_last;
    assign qkv_meta = meta_q;

    assign done_valid = done_valid_q;
    assign done_status = done_status_q;
    assign done_invalid = done_invalid_q;
    assign done_meta = done_meta_q;

    projection_controller #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_projection_controller (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .input_valid               (proj_input_valid),
        .input_ready               (proj_input_ready),
        .input_dim                 (input_dim),
        .input_data_fp16           (input_data_fp16),
        .input_last                (input_last),
        .input_commit              (proj_input_commit),
        .weight_valid              (weight_valid && weight_ready),
        .weight_ready              (proj_weight_ready),
        .weight_kind               (weight_kind),
        .weight_output_index       (weight_output_index),
        .weight_input_index        (weight_input_index),
        .weight_data_fp16          (weight_data_fp16),
        .weight_last               (weight_last),
        .weight_commit             (weight_commit),
        .start_valid               (proj_start_valid),
        .start_ready               (proj_start_ready),
        .start_matrix_kind         (proj_start_kind),
        .start_input_length        (LEN_W'(D_MODEL)),
        .start_output_length       (LEN_W'(D_MODEL)),
        .start_meta                (meta_q),
        .output_valid              (proj_output_valid),
        .output_ready              (proj_output_ready),
        .output_matrix_kind        (proj_output_kind),
        .output_index              (proj_output_index),
        .output_data_fp32          (proj_output_data),
        .output_status             (proj_output_status),
        .output_invalid            (proj_output_invalid),
        .output_meta               (proj_output_meta),
        .output_last               (proj_output_last),
        .done_valid                (proj_done_valid),
        .done_ready                (proj_done_ready),
        .done_status               (proj_done_status),
        .done_invalid              (proj_done_invalid),
        .done_meta                 (proj_done_meta),
        .perf_total_cycles         (proj_perf_total_cycles),
        .perf_pe_stall_cycles      (proj_perf_pe_stall_cycles),
        .perf_weight_stall_cycles  (proj_perf_weight_stall_cycles),
        .perf_output_stall_cycles  (proj_perf_output_stall_cycles)
    );

    fp32_to_fp16 #(
        .META_W(META_W + 2 + MODEL_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_qkv_quantizer (
        .clk                  (clk),
        .rst_n                (rst_n),
        .in_valid             (quant_in_valid),
        .in_ready             (quant_in_ready),
        .in_data              (proj_output_data),
        .in_meta              ({proj_output_kind, proj_output_index, proj_output_meta}),
        .in_last              (proj_output_last),
        .out_valid            (quant_out_valid),
        .out_ready            (quant_out_ready),
        .out_data             (quant_out_data),
        .out_invalid          (quant_out_invalid),
        .out_overflow         (quant_out_overflow),
        .out_underflow_or_ftz (quant_out_underflow_or_ftz),
        .out_inexact          (quant_out_inexact),
        .out_meta             (quant_out_meta),
        .out_last             (quant_out_last)
    );

    qkv_staging_buffer #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD)
    ) u_qkv_staging_buffer (
        .clk                (clk),
        .rst_n              (rst_n),
        .clear              (staging_clear),
        .write_valid        (staging_write_valid),
        .write_kind         (quant_write_kind),
        .write_index        (quant_write_index),
        .write_data_fp16    (quant_out_data),
        .read_index         (staging_read_index),
        .read_head          (staging_read_head),
        .read_dim           (staging_read_dim),
        .read_q_fp16        (staging_read_q),
        .read_k_fp16        (staging_read_k),
        .read_v_fp16        (staging_read_v),
        .read_complete      (staging_read_complete),
        .q_loaded_mask      (),
        .k_loaded_mask      (),
        .v_loaded_mask      (),
        .q_complete         (q_complete),
        .k_complete         (k_complete),
        .v_complete         (v_complete),
        .all_complete       (all_complete),
        .error_valid        (staging_error)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_LOAD_X;
            expected_dim_q <= '0;
            hidden_complete_q <= 1'b0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            active_kind_q <= KIND_Q;
            stream_index_q <= '0;
            done_valid_q <= 1'b0;
            done_status_q <= STATUS_OK;
            done_invalid_q <= 1'b0;
            done_meta_q <= '0;
        end else begin
            if (done_fire) begin
                done_valid_q <= 1'b0;
                done_status_q <= STATUS_OK;
                done_invalid_q <= 1'b0;
                done_meta_q <= '0;
            end
            if (quant_out_fire) begin
                status_q <= status_q | {4'd0, quant_out_overflow, quant_out_underflow_or_ftz, quant_out_inexact, 1'b0};
                invalid_q <= invalid_q | quant_out_invalid;
            end
            if (staging_error) begin
                status_q <= status_q | STATUS_PROJECT;
                invalid_q <= 1'b1;
            end

            unique case (state_q)
                ST_LOAD_X: begin
                    if (input_fire) begin
                        if ((input_dim != expected_dim_q) || (input_last != input_last_expected)) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= STATUS_ORDER;
                            done_invalid_q <= 1'b1;
                            done_meta_q <= input_meta;
                            expected_dim_q <= '0;
                            hidden_complete_q <= 1'b0;
                            state_q <= ST_DONE;
                        end else if (input_last_expected) begin
                            meta_q <= input_meta;
                            expected_dim_q <= '0;
                            hidden_complete_q <= 1'b1;
                            state_q <= ST_WAIT_START;
                        end else begin
                            expected_dim_q <= expected_dim_q + MODEL_W'(1);
                        end
                    end
                end

                ST_WAIT_START: begin
                    if (start_fire) begin
                        meta_q <= start_meta;
                        status_q <= STATUS_OK;
                        invalid_q <= 1'b0;
                        active_kind_q <= KIND_Q;
                        state_q <= ST_START_Q;
                    end
                end

                ST_START_Q: if (proj_start_fire) state_q <= ST_RUN_Q;
                ST_START_K: if (proj_start_fire) state_q <= ST_RUN_K;
                ST_START_V: if (proj_start_fire) state_q <= ST_RUN_V;

                ST_RUN_Q: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        active_kind_q <= KIND_K;
                        state_q <= ST_START_K;
                    end
                end

                ST_RUN_K: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        active_kind_q <= KIND_V;
                        state_q <= ST_START_V;
                    end
                end

                ST_RUN_V: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        stream_index_q <= '0;
                        state_q <= ST_STREAM;
                    end
                end

                ST_STREAM: begin
                    if (qkv_fire) begin
                        if (stream_last) begin
                            done_valid_q <= 1'b1;
                            done_status_q <= status_q;
                            done_invalid_q <= invalid_q;
                            done_meta_q <= meta_q;
                            hidden_complete_q <= 1'b0;
                            state_q <= ST_DONE;
                        end else begin
                            stream_index_q <= stream_index_q + MODEL_W'(1);
                        end
                    end
                end

                ST_DONE: begin
                    if (!done_valid_q || done_ready) begin
                        expected_dim_q <= '0;
                        state_q <= ST_LOAD_X;
                    end
                end

                default: state_q <= ST_LOAD_X;
            endcase
        end
    end

    assign proj_done_ready =
        (state_q == ST_RUN_Q) || (state_q == ST_RUN_K) || (state_q == ST_RUN_V);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_q_projection_cycles <= '0;
            perf_k_projection_cycles <= '0;
            perf_v_projection_cycles <= '0;
            perf_qkv_quantization_cycles <= '0;
            perf_weight_stall_cycles <= '0;
            perf_pe_stall_cycles <= '0;
            perf_output_stall_cycles <= '0;
        end else begin
            if (state_q inside {ST_START_Q, ST_RUN_Q}) begin
                perf_q_projection_cycles <= perf_q_projection_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_START_K, ST_RUN_K}) begin
                perf_k_projection_cycles <= perf_k_projection_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_START_V, ST_RUN_V}) begin
                perf_v_projection_cycles <= perf_v_projection_cycles + COUNTER_W'(1);
            end
            if (quant_in_valid || quant_out_valid) begin
                perf_qkv_quantization_cycles <= perf_qkv_quantization_cycles + COUNTER_W'(1);
            end
            perf_weight_stall_cycles <= proj_perf_weight_stall_cycles;
            perf_pe_stall_cycles <= proj_perf_pe_stall_cycles;
            if (qkv_valid && !qkv_ready) begin
                perf_output_stall_cycles <= perf_output_stall_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(input_fire && ((input_dim != expected_dim_q) || (input_last != input_last_expected))))
                else $error("qkv_projection_engine hidden_dimension_order_legal failed");
            assert (!(proj_start_valid && !hidden_complete_q))
                else $error("qkv_projection_engine no_projection_start_without_complete_input failed");
            assert (!(state_q == ST_STREAM && !all_complete))
                else $error("qkv_projection_engine no_attention_start_before_qkv_complete failed");
            assert (!(qkv_valid && !staging_read_complete))
                else $error("qkv_projection_engine qkv_output_head_dim_order_legal failed");
            assert (!(qkv_valid && $isunknown({qkv_head, qkv_dim, qkv_q_fp16, qkv_k_fp16, qkv_v_fp16,
                                               qkv_last_dim, qkv_last_head, qkv_meta})))
                else $error("qkv_projection_engine no_unknown_output_when_valid failed");
            if ($past(rst_n) && $past(qkv_valid && !qkv_ready)) begin
                assert (qkv_valid)
                    else $error("qkv_projection_engine qkv valid dropped under backpressure");
                assert ($stable({qkv_head, qkv_dim, qkv_q_fp16, qkv_k_fp16, qkv_v_fp16,
                                 qkv_last_dim, qkv_last_head, qkv_meta}))
                    else $error("qkv_projection_engine output stable until ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
