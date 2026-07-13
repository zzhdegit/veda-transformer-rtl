`default_nettype none

module projection_integrated_mha #(
    parameter int N_HEAD = 2,
    parameter int D_HEAD = 8,
    parameter int PE_NUM = 8,
    parameter int MAX_SEQ_LEN = 8,
    parameter int META_W = 16,
    parameter int COUNTER_W = 64,
    parameter int ATTENTION_PE_ARCH = 0,
    parameter bit ASSERT_ON_INVALID = 1'b1,
    localparam int D_MODEL = N_HEAD * D_HEAD,
    localparam int HEAD_W = (N_HEAD <= 1) ? 1 : $clog2(N_HEAD),
    localparam int HEAD_DIM_W = (D_HEAD <= 1) ? 1 : $clog2(D_HEAD),
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL),
    localparam int LEN_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL + 1),
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1)
) (
    input  logic                         clk,
    input  logic                         rst_n,

    input  logic                         weight_valid,
    output logic                         weight_ready,
    input  logic [1:0]                   weight_kind,
    input  logic [MODEL_W-1:0]           weight_output_index,
    input  logic [MODEL_W-1:0]           weight_input_index,
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
    output logic [COUNTER_W-1:0]         perf_total_cycles,
    output logic [COUNTER_W-1:0]         perf_q_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_k_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_v_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_qkv_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_attention_cycles,
    output logic [COUNTER_W-1:0]         perf_concat_quantization_cycles,
    output logic [COUNTER_W-1:0]         perf_output_projection_cycles,
    output logic [COUNTER_W-1:0]         perf_projection_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_attention_pe_stall_cycles,
    output logic [COUNTER_W-1:0]         perf_sfu_stall_cycles,
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
    localparam logic [1:0] KIND_Q  = 2'd0;
    localparam logic [1:0] KIND_K  = 2'd1;
    localparam logic [1:0] KIND_V  = 2'd2;
    localparam logic [1:0] KIND_WO = 2'd3;

    localparam logic [7:0] STATUS_OK = 8'h00;
    localparam logic [7:0] STATUS_HIDDEN_ORDER = 8'hE1;
    localparam logic [7:0] STATUS_QKV = 8'hE2;
    localparam logic [7:0] STATUS_CONCAT = 8'hE3;

    typedef enum logic [3:0] {
        ST_LOAD_HIDDEN,
        ST_START_Q,
        ST_RUN_Q,
        ST_START_K,
        ST_RUN_K,
        ST_START_V,
        ST_RUN_V,
        ST_QKV_STREAM,
        ST_ATTENTION,
        ST_WAIT_CONCAT,
        ST_OUTPUT_PROJECTION,
        ST_FINAL_DONE
    } state_e;

    state_e state_q;

    logic [MODEL_W-1:0] expected_dim_q;
    logic [MODEL_W-1:0] stream_index_q;
    logic [META_W-1:0] meta_q;
    logic [7:0] status_q;
    logic invalid_q;
    logic stage5_done_seen_q;
    logic [7:0] stage5_done_status_q;
    logic stage5_done_invalid_q;
    logic [META_W-1:0] stage5_done_meta_q;
    logic [SEQ_LEN_W-1:0] stage5_done_seq_len_q;

    logic final_done_valid_q;
    logic [7:0] final_done_status_q;
    logic final_done_invalid_q;
    logic [META_W-1:0] final_done_meta_q;
    logic [SEQ_LEN_W-1:0] final_done_seq_len_q;

    logic proj_input_valid;
    logic proj_input_ready;
    logic [MODEL_W-1:0] proj_input_dim;
    logic [15:0] proj_input_data_fp16;
    logic proj_input_last;
    logic proj_input_commit;
    logic proj_weight_ready;
    logic proj_start_valid;
    logic proj_start_ready;
    logic [1:0] proj_start_kind;
    logic [LEN_W-1:0] proj_start_input_length;
    logic [LEN_W-1:0] proj_start_output_length;
    logic [META_W-1:0] proj_start_meta;
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

    logic qkv_quant_in_valid;
    logic qkv_quant_in_ready;
    logic qkv_quant_out_valid;
    logic qkv_quant_out_ready;
    logic [15:0] qkv_quant_out_data;
    logic qkv_quant_out_invalid;
    logic qkv_quant_out_overflow;
    logic qkv_quant_out_underflow_or_ftz;
    logic qkv_quant_out_inexact;
    logic [META_W+2+MODEL_W-1:0] qkv_quant_out_meta;
    logic qkv_quant_out_last;
    logic [1:0] qkv_write_kind;
    logic [MODEL_W-1:0] qkv_write_index;

    logic qkv_staging_clear;
    logic [HEAD_W-1:0] qkv_read_head;
    logic [HEAD_DIM_W-1:0] qkv_read_dim;
    logic [15:0] qkv_read_q;
    logic [15:0] qkv_read_k;
    logic [15:0] qkv_read_v;
    logic qkv_read_complete;
    logic q_complete;
    logic k_complete;
    logic v_complete;
    logic qkv_all_complete;
    logic qkv_staging_error;

    logic stage5_token_valid;
    logic stage5_token_ready;
    logic stage5_output_valid;
    logic stage5_output_ready;
    logic [HEAD_W-1:0] stage5_output_head;
    logic [HEAD_DIM_W-1:0] stage5_output_base_dim;
    logic [PE_NUM*32-1:0] stage5_output_vector_fp32;
    logic [PE_NUM-1:0] stage5_output_lane_mask;
    logic [7:0] stage5_output_status;
    logic stage5_output_invalid;
    logic [META_W-1:0] stage5_output_meta;
    logic stage5_output_last_tile;
    logic stage5_output_last_head;
    logic stage5_output_last_token;
    logic stage5_done_valid;
    logic stage5_done_ready;
    logic [7:0] stage5_done_status;
    logic stage5_done_invalid;
    logic [META_W-1:0] stage5_done_meta;
    logic [SEQ_LEN_W-1:0] stage5_done_valid_seq_len;
    logic [COUNTER_W-1:0] stage5_total_cycles;
    logic [COUNTER_W-1:0] stage5_per_head_attention_cycles;
    logic [COUNTER_W-1:0] stage5_head_switch_cycles;
    logic [COUNTER_W-1:0] stage5_provisional_write_cycles;
    logic [COUNTER_W-1:0] stage5_cache_read_cycles;
    logic [COUNTER_W-1:0] stage5_cache_write_cycles;
    logic [COUNTER_W-1:0] stage5_cache_stall_cycles;
    logic [COUNTER_W-1:0] stage5_commit_cycles;
    logic [COUNTER_W-1:0] stage5_pe_stall_cycles;
    logic [COUNTER_W-1:0] stage5_sfu_stall_cycles;
    logic [COUNTER_W-1:0] stage5_output_stall_cycles;
    logic [COUNTER_W-1:0] stage5_paper_array_active_cycles;
    logic [COUNTER_W-1:0] stage5_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] stage5_inner_mode_cycles;
    logic [COUNTER_W-1:0] stage5_outer_mode_cycles;
    logic [COUNTER_W-1:0] stage5_group0_active_cycles;
    logic [COUNTER_W-1:0] stage5_group1_active_cycles;
    logic [COUNTER_W-1:0] stage5_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] stage5_mode_switch_cycles;
    logic [COUNTER_W-1:0] stage5_array_input_stall_cycles;
    logic [COUNTER_W-1:0] stage5_array_output_stall_cycles;

    logic concat_clear;
    logic concat_input_ready;
    logic concat_write_valid;
    logic concat_buffer_write_ready;
    logic [MODEL_W-1:0] concat_write_index;
    logic [15:0] concat_write_data;
    logic concat_complete;
    logic [7:0] concat_status;
    logic concat_invalid;
    logic [META_W-1:0] concat_meta;
    logic concat_busy;
    logic [COUNTER_W-1:0] concat_perf_cycles;
    logic [COUNTER_W-1:0] concat_perf_stall_cycles;

    logic concat_read_check_valid;
    logic [MODEL_W-1:0] concat_read_index;
    logic [15:0] concat_read_data;
    logic concat_read_valid;
    logic concat_complete_check;
    logic [D_MODEL*16-1:0] concat_vector_flat;
    logic [D_MODEL-1:0] concat_loaded_mask;
    logic concat_buffer_error;
    logic concat_duplicate_error;
    logic concat_missing_error;
    logic concat_range_error;

    logic wo_start_valid;
    logic wo_start_ready;
    logic wo_proj_input_valid;
    logic wo_proj_input_ready;
    logic [MODEL_W-1:0] wo_proj_input_dim;
    logic [15:0] wo_proj_input_data;
    logic wo_proj_input_last;
    logic wo_proj_input_commit;
    logic wo_proj_start_valid;
    logic wo_proj_start_ready;
    logic [1:0] wo_proj_start_kind;
    logic [LEN_W-1:0] wo_proj_start_input_length;
    logic [LEN_W-1:0] wo_proj_start_output_length;
    logic [META_W-1:0] wo_proj_start_meta;
    logic wo_proj_output_ready;
    logic wo_proj_done_ready;
    logic wo_done_valid;
    logic wo_done_ready;
    logic [7:0] wo_done_status;
    logic wo_done_invalid;
    logic [META_W-1:0] wo_done_meta;
    logic [COUNTER_W-1:0] wo_perf_cycles;
    logic [COUNTER_W-1:0] wo_perf_output_stall_cycles;

    wire token_fire = token_valid && token_ready;
    wire weight_fire = weight_valid && weight_ready;
    wire proj_start_fire = proj_start_valid && proj_start_ready;
    wire proj_done_fire = proj_done_valid && proj_done_ready;
    wire qkv_quant_out_fire = qkv_quant_out_valid && qkv_quant_out_ready;
    wire stage5_token_fire = stage5_token_valid && stage5_token_ready;
    wire stage5_done_fire = stage5_done_valid && stage5_done_ready;
    wire wo_start_fire = wo_start_valid && wo_start_ready;
    wire wo_done_fire = wo_done_valid && wo_done_ready;
    wire final_done_fire = done_valid && done_ready;
    wire token_last_expected = token_dim == MODEL_W'(D_MODEL - 1);
    wire token_order_legal = (token_dim == expected_dim_q) && (token_last_dim == token_last_expected);
    wire qkv_run_state = (state_q == ST_RUN_Q) || (state_q == ST_RUN_K) || (state_q == ST_RUN_V);
    wire qkv_start_state = (state_q == ST_START_Q) || (state_q == ST_START_K) || (state_q == ST_START_V);
    wire qkv_stream_last = stream_index_q == MODEL_W'(D_MODEL - 1);

    initial begin
        if (N_HEAD <= 0 || D_HEAD <= 0 || PE_NUM <= 0 || MAX_SEQ_LEN <= 0 ||
            META_W <= 0 || COUNTER_W <= 0) begin
            $fatal(1, "projection_integrated_mha parameters must be positive");
        end
        if (D_MODEL != N_HEAD * D_HEAD) begin
            $fatal(1, "projection_integrated_mha d_model_equals_n_head_times_d_head failed");
        end
    end

    assign token_ready = (state_q == ST_LOAD_HIDDEN) && !final_done_valid_q && proj_input_ready;
    assign weight_ready = (state_q == ST_LOAD_HIDDEN) && (expected_dim_q == '0) &&
                          !final_done_valid_q && proj_weight_ready;

    assign proj_input_valid =
        ((state_q == ST_LOAD_HIDDEN) && token_valid && token_ready && token_order_legal) ||
        wo_proj_input_valid;
    assign proj_input_dim = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_input_dim : token_dim;
    assign proj_input_data_fp16 = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_input_data : token_hidden_fp16;
    assign proj_input_last = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_input_last : token_last_dim;
    assign proj_input_commit = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_input_commit :
        (token_fire && token_order_legal && token_last_dim);
    assign wo_proj_input_ready = (state_q == ST_OUTPUT_PROJECTION) && proj_input_ready;

    assign proj_start_valid = qkv_start_state || wo_proj_start_valid;
    assign proj_start_kind = (state_q == ST_START_Q) ? KIND_Q :
        ((state_q == ST_START_K) ? KIND_K :
        ((state_q == ST_START_V) ? KIND_V : wo_proj_start_kind));
    assign proj_start_input_length = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_start_input_length : LEN_W'(D_MODEL);
    assign proj_start_output_length = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_start_output_length : LEN_W'(D_MODEL);
    assign proj_start_meta = (state_q == ST_OUTPUT_PROJECTION) ? wo_proj_start_meta : meta_q;
    assign wo_proj_start_ready = (state_q == ST_OUTPUT_PROJECTION) && proj_start_ready;

    assign proj_output_ready = qkv_run_state ? qkv_quant_in_ready : wo_proj_output_ready;
    assign proj_done_ready = qkv_run_state || wo_proj_done_ready;

    assign qkv_quant_in_valid = qkv_run_state && proj_output_valid;
    assign qkv_quant_out_ready = 1'b1;
    assign qkv_write_kind = qkv_quant_out_meta[META_W+MODEL_W +: 2];
    assign qkv_write_index = qkv_quant_out_meta[META_W +: MODEL_W];
    assign qkv_staging_clear = token_fire && token_order_legal && token_last_dim;

    assign stage5_token_valid = (state_q == ST_QKV_STREAM) && qkv_read_complete && qkv_all_complete;
    assign stage5_done_ready = state_q == ST_ATTENTION;
    assign stage5_output_ready = (state_q == ST_ATTENTION) && concat_input_ready;

    assign concat_clear = token_fire && token_order_legal && token_last_dim;
    assign concat_complete_check = wo_start_fire;
    assign wo_start_valid = (state_q == ST_OUTPUT_PROJECTION) && !wo_done_valid && stage5_done_seen_q &&
                            !stage5_done_invalid_q && concat_complete && !concat_buffer_error &&
                            !concat_invalid;
    assign wo_done_ready = state_q == ST_OUTPUT_PROJECTION;

    assign done_valid = final_done_valid_q;
    assign done_status = final_done_status_q;
    assign done_invalid = final_done_invalid_q;
    assign done_meta = final_done_meta_q;
    assign done_valid_seq_len = final_done_seq_len_q;

    assign perf_attention_cycles = stage5_per_head_attention_cycles;
    assign perf_concat_quantization_cycles = concat_perf_cycles;
    assign perf_output_projection_cycles = wo_perf_cycles;
    assign perf_projection_pe_stall_cycles = proj_perf_pe_stall_cycles;
    assign perf_attention_pe_stall_cycles = stage5_pe_stall_cycles;
    assign perf_sfu_stall_cycles = stage5_sfu_stall_cycles;
    assign perf_weight_stall_cycles = proj_perf_weight_stall_cycles;
    assign perf_buffer_stall_cycles = stage5_cache_stall_cycles;
    assign perf_output_stall_cycles =
        proj_perf_output_stall_cycles + stage5_output_stall_cycles +
        concat_perf_stall_cycles + wo_perf_output_stall_cycles;
    assign perf_paper_array_active_cycles = stage5_paper_array_active_cycles;
    assign perf_paper_array_idle_cycles = stage5_paper_array_idle_cycles;
    assign perf_inner_mode_cycles = stage5_inner_mode_cycles;
    assign perf_outer_mode_cycles = stage5_outer_mode_cycles;
    assign perf_group0_active_cycles = stage5_group0_active_cycles;
    assign perf_group1_active_cycles = stage5_group1_active_cycles;
    assign perf_tail_masked_pe_cycles = stage5_tail_masked_pe_cycles;
    assign perf_mode_switch_cycles = stage5_mode_switch_cycles;
    assign perf_array_input_stall_cycles = stage5_array_input_stall_cycles;
    assign perf_array_output_stall_cycles = stage5_array_output_stall_cycles;

    projection_controller #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_shared_projection_controller (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .input_valid               (proj_input_valid),
        .input_ready               (proj_input_ready),
        .input_dim                 (proj_input_dim),
        .input_data_fp16           (proj_input_data_fp16),
        .input_last                (proj_input_last),
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
        .start_input_length        (proj_start_input_length),
        .start_output_length       (proj_start_output_length),
        .start_meta                (proj_start_meta),
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
        .in_valid             (qkv_quant_in_valid),
        .in_ready             (qkv_quant_in_ready),
        .in_data              (proj_output_data),
        .in_meta              ({proj_output_kind, proj_output_index, proj_output_meta}),
        .in_last              (proj_output_last),
        .out_valid            (qkv_quant_out_valid),
        .out_ready            (qkv_quant_out_ready),
        .out_data             (qkv_quant_out_data),
        .out_invalid          (qkv_quant_out_invalid),
        .out_overflow         (qkv_quant_out_overflow),
        .out_underflow_or_ftz (qkv_quant_out_underflow_or_ftz),
        .out_inexact          (qkv_quant_out_inexact),
        .out_meta             (qkv_quant_out_meta),
        .out_last             (qkv_quant_out_last)
    );

    qkv_staging_buffer #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD)
    ) u_qkv_staging_buffer (
        .clk                (clk),
        .rst_n              (rst_n),
        .clear              (qkv_staging_clear),
        .write_valid        (qkv_quant_out_fire),
        .write_kind         (qkv_write_kind),
        .write_index        (qkv_write_index),
        .write_data_fp16    (qkv_quant_out_data),
        .read_index         (stream_index_q),
        .read_head          (qkv_read_head),
        .read_dim           (qkv_read_dim),
        .read_q_fp16        (qkv_read_q),
        .read_k_fp16        (qkv_read_k),
        .read_v_fp16        (qkv_read_v),
        .read_complete      (qkv_read_complete),
        .q_loaded_mask      (),
        .k_loaded_mask      (),
        .v_loaded_mask      (),
        .q_complete         (q_complete),
        .k_complete         (k_complete),
        .v_complete         (v_complete),
        .all_complete       (qkv_all_complete),
        .error_valid        (qkv_staging_error)
    );

    multi_head_generation_engine #(
        .N_HEAD(N_HEAD),
        .PE_NUM(PE_NUM),
        .D_HEAD(D_HEAD),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(ATTENTION_PE_ARCH),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_multi_head_generation_engine (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .token_valid                    (stage5_token_valid),
        .token_ready                    (stage5_token_ready),
        .token_head                     (qkv_read_head),
        .token_dim                      (qkv_read_dim),
        .token_q_fp16                   (qkv_read_q),
        .token_k_fp16                   (qkv_read_k),
        .token_v_fp16                   (qkv_read_v),
        .token_last_dim                 (qkv_read_dim == HEAD_DIM_W'(D_HEAD - 1)),
        .token_last_head                (qkv_stream_last),
        .token_meta                     (meta_q),
        .output_valid                   (stage5_output_valid),
        .output_ready                   (stage5_output_ready),
        .output_head                    (stage5_output_head),
        .output_base_dim                (stage5_output_base_dim),
        .output_vector_fp32             (stage5_output_vector_fp32),
        .output_lane_mask               (stage5_output_lane_mask),
        .output_status                  (stage5_output_status),
        .output_invalid                 (stage5_output_invalid),
        .output_meta                    (stage5_output_meta),
        .output_last_tile               (stage5_output_last_tile),
        .output_last_head               (stage5_output_last_head),
        .output_last_token              (stage5_output_last_token),
        .done_valid                     (stage5_done_valid),
        .done_ready                     (stage5_done_ready),
        .done_status                    (stage5_done_status),
        .done_invalid                   (stage5_done_invalid),
        .done_meta                      (stage5_done_meta),
        .done_valid_seq_len             (stage5_done_valid_seq_len),
        .current_valid_seq_len          (current_valid_seq_len),
        .perf_generation_steps          (perf_generation_steps),
        .perf_total_cycles              (stage5_total_cycles),
        .perf_per_head_attention_cycles (stage5_per_head_attention_cycles),
        .perf_head_switch_cycles        (stage5_head_switch_cycles),
        .perf_provisional_write_cycles  (stage5_provisional_write_cycles),
        .perf_cache_read_cycles         (stage5_cache_read_cycles),
        .perf_cache_write_cycles        (stage5_cache_write_cycles),
        .perf_cache_stall_cycles        (stage5_cache_stall_cycles),
        .perf_commit_cycles             (stage5_commit_cycles),
        .perf_pe_stall_cycles           (stage5_pe_stall_cycles),
        .perf_sfu_stall_cycles          (stage5_sfu_stall_cycles),
        .perf_output_stall_cycles       (stage5_output_stall_cycles),
        .perf_paper_array_active_cycles (stage5_paper_array_active_cycles),
        .perf_paper_array_idle_cycles   (stage5_paper_array_idle_cycles),
        .perf_inner_mode_cycles         (stage5_inner_mode_cycles),
        .perf_outer_mode_cycles         (stage5_outer_mode_cycles),
        .perf_group0_active_cycles      (stage5_group0_active_cycles),
        .perf_group1_active_cycles      (stage5_group1_active_cycles),
        .perf_tail_masked_pe_cycles     (stage5_tail_masked_pe_cycles),
        .perf_mode_switch_cycles        (stage5_mode_switch_cycles),
        .perf_array_input_stall_cycles  (stage5_array_input_stall_cycles),
        .perf_array_output_stall_cycles (stage5_array_output_stall_cycles),
        .perf_peak_valid_seq_len        (perf_peak_valid_seq_len)
    );

    head_concat_quantizer #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ASSERT_ON_INVALID(ASSERT_ON_INVALID)
    ) u_head_concat_quantizer (
        .clk                              (clk),
        .rst_n                            (rst_n),
        .clear                            (concat_clear),
        .input_valid                      (stage5_output_valid),
        .input_ready                      (concat_input_ready),
        .input_head                       (stage5_output_head),
        .input_base_dim                   (stage5_output_base_dim),
        .input_vector_fp32                (stage5_output_vector_fp32),
        .input_lane_mask                  (stage5_output_lane_mask),
        .input_last_tile                  (stage5_output_last_tile),
        .input_last_head                  (stage5_output_last_head),
        .input_last_token                 (stage5_output_last_token),
        .input_status                     (stage5_output_status),
        .input_invalid                    (stage5_output_invalid),
        .input_meta                       (stage5_output_meta),
        .write_valid                      (concat_write_valid),
        .write_ready                      (concat_buffer_write_ready),
        .write_index                      (concat_write_index),
        .write_data_fp16                  (concat_write_data),
        .concat_complete                  (concat_complete),
        .concat_status                    (concat_status),
        .concat_invalid                   (concat_invalid),
        .concat_meta                      (concat_meta),
        .busy                             (concat_busy),
        .perf_concat_quantization_cycles  (concat_perf_cycles),
        .perf_output_stall_cycles         (concat_perf_stall_cycles)
    );

    concat_fp16_buffer #(
        .D_MODEL(D_MODEL)
    ) u_concat_fp16_buffer (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .clear                  (concat_clear),
        .write_valid            (concat_write_valid),
        .write_ready            (concat_buffer_write_ready),
        .write_index            (concat_write_index),
        .write_data_fp16        (concat_write_data),
        .read_check_valid       (concat_read_check_valid),
        .read_index             (concat_read_index),
        .read_data_fp16         (concat_read_data),
        .read_valid             (concat_read_valid),
        .complete_check         (concat_complete_check),
        .vector_flat_fp16       (concat_vector_flat),
        .loaded_mask            (concat_loaded_mask),
        .complete               (),
        .error_valid            (concat_buffer_error),
        .duplicate_error        (concat_duplicate_error),
        .missing_error          (concat_missing_error),
        .range_error            (concat_range_error)
    );

    output_projection_controller #(
        .D_MODEL(D_MODEL),
        .PE_NUM(PE_NUM),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W)
    ) u_output_projection_controller (
        .clk                            (clk),
        .rst_n                          (rst_n),
        .start_valid                    (wo_start_valid),
        .start_ready                    (wo_start_ready),
        .start_meta                     (stage5_done_meta_q),
        .start_status                   (stage5_done_status_q | concat_status),
        .start_invalid                  (stage5_done_invalid_q | concat_invalid | concat_buffer_error),
        .concat_read_check_valid        (concat_read_check_valid),
        .concat_read_index              (concat_read_index),
        .concat_read_data_fp16          (concat_read_data),
        .concat_read_valid              (concat_read_valid),
        .concat_complete                (concat_complete),
        .concat_error                   (concat_buffer_error),
        .proj_input_valid               (wo_proj_input_valid),
        .proj_input_ready               (wo_proj_input_ready),
        .proj_input_dim                 (wo_proj_input_dim),
        .proj_input_data_fp16           (wo_proj_input_data),
        .proj_input_last                (wo_proj_input_last),
        .proj_input_commit              (wo_proj_input_commit),
        .proj_start_valid               (wo_proj_start_valid),
        .proj_start_ready               (wo_proj_start_ready),
        .proj_start_matrix_kind         (wo_proj_start_kind),
        .proj_start_input_length        (wo_proj_start_input_length),
        .proj_start_output_length       (wo_proj_start_output_length),
        .proj_start_meta                (wo_proj_start_meta),
        .proj_output_valid              (proj_output_valid && (state_q == ST_OUTPUT_PROJECTION)),
        .proj_output_ready              (wo_proj_output_ready),
        .proj_output_matrix_kind        (proj_output_kind),
        .proj_output_index              (proj_output_index),
        .proj_output_data_fp32          (proj_output_data),
        .proj_output_status             (proj_output_status),
        .proj_output_invalid            (proj_output_invalid),
        .proj_output_meta               (proj_output_meta),
        .proj_output_last               (proj_output_last),
        .proj_done_valid                (proj_done_valid && (state_q == ST_OUTPUT_PROJECTION)),
        .proj_done_ready                (wo_proj_done_ready),
        .proj_done_status               (proj_done_status),
        .proj_done_invalid              (proj_done_invalid),
        .proj_done_meta                 (proj_done_meta),
        .output_valid                   (output_valid),
        .output_ready                   (output_ready),
        .output_base_dim                (output_base_dim),
        .output_vector_fp32             (output_vector_fp32),
        .output_lane_mask               (output_lane_mask),
        .output_status                  (output_status),
        .output_invalid                 (output_invalid),
        .output_meta                    (output_meta),
        .output_last                    (output_last),
        .done_valid                     (wo_done_valid),
        .done_ready                     (wo_done_ready),
        .done_status                    (wo_done_status),
        .done_invalid                   (wo_done_invalid),
        .done_meta                      (wo_done_meta),
        .perf_output_projection_cycles  (wo_perf_cycles),
        .perf_output_stall_cycles       (wo_perf_output_stall_cycles)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_LOAD_HIDDEN;
            expected_dim_q <= '0;
            stream_index_q <= '0;
            meta_q <= '0;
            status_q <= STATUS_OK;
            invalid_q <= 1'b0;
            stage5_done_seen_q <= 1'b0;
            stage5_done_status_q <= STATUS_OK;
            stage5_done_invalid_q <= 1'b0;
            stage5_done_meta_q <= '0;
            stage5_done_seq_len_q <= '0;
            final_done_valid_q <= 1'b0;
            final_done_status_q <= STATUS_OK;
            final_done_invalid_q <= 1'b0;
            final_done_meta_q <= '0;
            final_done_seq_len_q <= '0;
        end else begin
            if (final_done_fire) begin
                final_done_valid_q <= 1'b0;
                final_done_status_q <= STATUS_OK;
                final_done_invalid_q <= 1'b0;
                final_done_meta_q <= '0;
                final_done_seq_len_q <= current_valid_seq_len;
                state_q <= ST_LOAD_HIDDEN;
            end

            if (qkv_quant_out_fire) begin
                status_q <= status_q | {4'd0, qkv_quant_out_overflow,
                                        qkv_quant_out_underflow_or_ftz,
                                        qkv_quant_out_inexact, 1'b0};
                invalid_q <= invalid_q | qkv_quant_out_invalid;
            end
            if (qkv_staging_error) begin
                status_q <= status_q | STATUS_QKV;
                invalid_q <= 1'b1;
            end

            unique case (state_q)
                ST_LOAD_HIDDEN: begin
                    if (token_fire) begin
                        if (!token_order_legal) begin
                            final_done_valid_q <= 1'b1;
                            final_done_status_q <= STATUS_HIDDEN_ORDER;
                            final_done_invalid_q <= 1'b1;
                            final_done_meta_q <= token_meta;
                            final_done_seq_len_q <= current_valid_seq_len;
                            expected_dim_q <= '0;
                            state_q <= ST_FINAL_DONE;
                        end else if (token_last_expected) begin
                            expected_dim_q <= '0;
                            meta_q <= token_meta;
                            status_q <= STATUS_OK;
                            invalid_q <= 1'b0;
                            stage5_done_seen_q <= 1'b0;
                            stage5_done_status_q <= STATUS_OK;
                            stage5_done_invalid_q <= 1'b0;
                            stage5_done_meta_q <= token_meta;
                            stream_index_q <= '0;
                            state_q <= ST_START_Q;
                        end else begin
                            expected_dim_q <= expected_dim_q + MODEL_W'(1);
                        end
                    end
                end

                ST_START_Q: if (proj_start_fire) state_q <= ST_RUN_Q;
                ST_START_K: if (proj_start_fire) state_q <= ST_RUN_K;
                ST_START_V: if (proj_start_fire) state_q <= ST_RUN_V;

                ST_RUN_Q: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        state_q <= ST_START_K;
                    end
                end

                ST_RUN_K: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        state_q <= ST_START_V;
                    end
                end

                ST_RUN_V: begin
                    if (proj_done_fire) begin
                        status_q <= status_q | proj_done_status;
                        invalid_q <= invalid_q | proj_done_invalid;
                        if (invalid_q || proj_done_invalid) begin
                            final_done_valid_q <= 1'b1;
                            final_done_status_q <= status_q | proj_done_status | STATUS_QKV;
                            final_done_invalid_q <= 1'b1;
                            final_done_meta_q <= meta_q;
                            final_done_seq_len_q <= current_valid_seq_len;
                            state_q <= ST_FINAL_DONE;
                        end else begin
                            stream_index_q <= '0;
                            state_q <= ST_QKV_STREAM;
                        end
                    end
                end

                ST_QKV_STREAM: begin
                    if (stage5_token_fire) begin
                        if (qkv_stream_last) begin
                            state_q <= ST_ATTENTION;
                        end else begin
                            stream_index_q <= stream_index_q + MODEL_W'(1);
                        end
                    end
                end

                ST_ATTENTION: begin
                    if (stage5_done_fire) begin
                        stage5_done_seen_q <= 1'b1;
                        stage5_done_status_q <= status_q | stage5_done_status;
                        stage5_done_invalid_q <= invalid_q | stage5_done_invalid;
                        stage5_done_meta_q <= stage5_done_meta;
                        stage5_done_seq_len_q <= stage5_done_valid_seq_len;
                        if (stage5_done_invalid || invalid_q) begin
                            final_done_valid_q <= 1'b1;
                            final_done_status_q <= (stage5_done_invalid && !invalid_q) ?
                                stage5_done_status : (status_q | stage5_done_status);
                            final_done_invalid_q <= 1'b1;
                            final_done_meta_q <= stage5_done_meta;
                            final_done_seq_len_q <= stage5_done_valid_seq_len;
                            state_q <= ST_FINAL_DONE;
                        end else begin
                            state_q <= ST_WAIT_CONCAT;
                        end
                    end
                end

                ST_WAIT_CONCAT: begin
                    if (concat_complete) begin
                        if (concat_invalid || concat_buffer_error) begin
                            final_done_valid_q <= 1'b1;
                            final_done_status_q <= stage5_done_status_q | concat_status | STATUS_CONCAT;
                            final_done_invalid_q <= 1'b1;
                            final_done_meta_q <= stage5_done_meta_q;
                            final_done_seq_len_q <= stage5_done_seq_len_q;
                            state_q <= ST_FINAL_DONE;
                        end else begin
                            state_q <= ST_OUTPUT_PROJECTION;
                        end
                    end
                end

                ST_OUTPUT_PROJECTION: begin
                    if (wo_done_fire) begin
                        final_done_valid_q <= 1'b1;
                        final_done_status_q <= wo_done_status;
                        final_done_invalid_q <= wo_done_invalid;
                        final_done_meta_q <= wo_done_meta;
                        final_done_seq_len_q <= stage5_done_seq_len_q;
                        state_q <= ST_FINAL_DONE;
                    end
                end

                ST_FINAL_DONE: begin
                    if (!final_done_valid_q) begin
                        state_q <= ST_LOAD_HIDDEN;
                    end
                end

                default: state_q <= ST_LOAD_HIDDEN;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_total_cycles <= '0;
            perf_q_projection_cycles <= '0;
            perf_k_projection_cycles <= '0;
            perf_v_projection_cycles <= '0;
            perf_qkv_quantization_cycles <= '0;
        end else begin
            if ((state_q != ST_LOAD_HIDDEN) || token_valid || final_done_valid_q) begin
                perf_total_cycles <= perf_total_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_START_Q, ST_RUN_Q}) begin
                perf_q_projection_cycles <= perf_q_projection_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_START_K, ST_RUN_K}) begin
                perf_k_projection_cycles <= perf_k_projection_cycles + COUNTER_W'(1);
            end
            if (state_q inside {ST_START_V, ST_RUN_V}) begin
                perf_v_projection_cycles <= perf_v_projection_cycles + COUNTER_W'(1);
            end
            if (qkv_quant_in_valid || qkv_quant_out_valid) begin
                perf_qkv_quantization_cycles <= perf_qkv_quantization_cycles + COUNTER_W'(1);
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n) begin
            assert (!(weight_fire && (state_q != ST_LOAD_HIDDEN || expected_dim_q != '0)))
                else $error("projection_integrated_mha no_weight_write_while_active failed");
            assert (!(token_fire && !token_order_legal))
                else $error("projection_integrated_mha hidden_dimension_order_legal failed");
            assert (!(qkv_start_state && expected_dim_q != '0))
                else $error("projection_integrated_mha no_projection_without_complete_hidden failed");
            assert (!(state_q == ST_QKV_STREAM && !qkv_all_complete))
                else $error("projection_integrated_mha no_attention_before_qkv_complete failed");
            assert (!(stage5_token_valid && !qkv_read_complete))
                else $error("projection_integrated_mha qkv_head_dim_order_legal failed");
            assert (!(stage5_output_valid && state_q != ST_ATTENTION))
                else $error("projection_integrated_mha no_concat_before_head_output failed");
            assert (!(wo_start_valid && (!concat_complete || concat_buffer_error)))
                else $error("projection_integrated_mha no_output_projection_before_concat_complete failed");
            assert (!(state_q == ST_LOAD_HIDDEN && final_done_valid_q && token_ready))
                else $error("projection_integrated_mha no_next_token_before_final_done failed");
            assert (!(done_valid && $isunknown({done_status, done_invalid, done_meta, done_valid_seq_len})))
                else $error("projection_integrated_mha no_unknown_done_when_valid failed");
            assert (!(output_valid && $isunknown({output_base_dim, output_vector_fp32, output_lane_mask,
                                                  output_status, output_invalid, output_meta, output_last})))
                else $error("projection_integrated_mha no_unknown_output_when_valid failed");
            assert (perf_generation_steps <= perf_total_cycles)
                else $error("projection_integrated_mha transaction_count_conserved failed");
            if ($past(rst_n) && $past(done_valid && !done_ready)) begin
                assert (done_valid)
                    else $error("projection_integrated_mha done valid dropped under backpressure");
                assert ($stable({done_status, done_invalid, done_meta, done_valid_seq_len}))
                    else $error("projection_integrated_mha done_stable_until_ready failed");
            end
        end
    end
`endif
endmodule

`default_nettype wire
