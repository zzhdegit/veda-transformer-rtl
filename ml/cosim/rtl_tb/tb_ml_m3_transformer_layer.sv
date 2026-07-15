`timescale 1ns/1ps
`default_nettype none

module tb_ml_m3_transformer_layer;
`ifndef ML_M3_N_HEAD
    localparam int N_HEAD = 8;
`else
    localparam int N_HEAD = `ML_M3_N_HEAD;
`endif
`ifndef ML_M3_D_HEAD
    localparam int D_HEAD = 8;
`else
    localparam int D_HEAD = `ML_M3_D_HEAD;
`endif
`ifndef ML_M3_MAX_SEQ_LEN
    localparam int MAX_SEQ_LEN = 128;
`else
    localparam int MAX_SEQ_LEN = `ML_M3_MAX_SEQ_LEN;
`endif
`ifndef ML_M3_ATTENTION_PE_ARCH
    localparam int ATTENTION_PE_ARCH = 1;
`else
    localparam int ATTENTION_PE_ARCH = `ML_M3_ATTENTION_PE_ARCH;
`endif
`ifndef ML_M3_ATTENTION_SCHEDULE
    localparam int ATTENTION_SCHEDULE = 1;
`else
    localparam int ATTENTION_SCHEDULE = `ML_M3_ATTENTION_SCHEDULE;
`endif

    localparam int D_MODEL = N_HEAD * D_HEAD;
    localparam int D_FFN = 4 * D_MODEL;
    localparam int PE_NUM = 8;
    localparam int META_W = 16;
    localparam int COUNTER_W = 64;
    localparam int MODEL_W = (D_MODEL <= 1) ? 1 : $clog2(D_MODEL);
    localparam int FFN_W = (D_FFN <= 1) ? 1 : $clog2(D_FFN);
    localparam int SEQ_LEN_W = (MAX_SEQ_LEN <= 1) ? 1 : $clog2(MAX_SEQ_LEN + 1);
    localparam int MAX_WEIGHT_LINES = 70000;
    localparam int MAX_TOKENS = 40;

    logic clk;
    logic rst_n;

    logic weight_valid;
    logic weight_ready;
    logic [2:0] weight_kind;
    logic [FFN_W-1:0] weight_output_index;
    logic [FFN_W-1:0] weight_input_index;
    logic [15:0] weight_data_fp16;
    logic weight_last;
    logic weight_commit;
    logic token_valid;
    logic token_ready;
    logic [MODEL_W-1:0] token_dim;
    logic [15:0] token_hidden_fp16;
    logic token_last_dim;
    logic [META_W-1:0] token_meta;
    logic output_valid;
    logic output_ready;
    logic [MODEL_W-1:0] output_base_dim;
    logic [PE_NUM*32-1:0] output_vector_fp32;
    logic [PE_NUM-1:0] output_lane_mask;
    logic [7:0] output_status;
    logic output_invalid;
    logic [META_W-1:0] output_meta;
    logic output_last;
    logic done_valid;
    logic done_ready;
    logic [7:0] done_status;
    logic done_invalid;
    logic [META_W-1:0] done_meta;
    logic [SEQ_LEN_W-1:0] done_valid_seq_len;
    logic [SEQ_LEN_W-1:0] current_valid_seq_len;

    logic [COUNTER_W-1:0] perf_generation_steps;
    logic [COUNTER_W-1:0] perf_total_layer_cycles;
    logic [COUNTER_W-1:0] perf_input_load_cycles;
    logic [COUNTER_W-1:0] perf_norm1_reduce_cycles;
    logic [COUNTER_W-1:0] perf_norm1_apply_cycles;
    logic [COUNTER_W-1:0] perf_mha_cycles;
    logic [COUNTER_W-1:0] perf_residual1_cycles;
    logic [COUNTER_W-1:0] perf_norm2_reduce_cycles;
    logic [COUNTER_W-1:0] perf_norm2_apply_cycles;
    logic [COUNTER_W-1:0] perf_ffn1_cycles;
    logic [COUNTER_W-1:0] perf_relu_cycles;
    logic [COUNTER_W-1:0] perf_activation_quantization_cycles;
    logic [COUNTER_W-1:0] perf_ffn2_cycles;
    logic [COUNTER_W-1:0] perf_residual2_cycles;
    logic [COUNTER_W-1:0] perf_final_output_cycles;
    logic [COUNTER_W-1:0] perf_norm_stall_cycles;
    logic [COUNTER_W-1:0] perf_mha_stall_cycles;
    logic [COUNTER_W-1:0] perf_ffn_pe_stall_cycles;
    logic [COUNTER_W-1:0] perf_weight_stall_cycles;
    logic [COUNTER_W-1:0] perf_buffer_stall_cycles;
    logic [COUNTER_W-1:0] perf_output_stall_cycles;
    logic [COUNTER_W-1:0] perf_paper_array_active_cycles;
    logic [COUNTER_W-1:0] perf_paper_array_idle_cycles;
    logic [COUNTER_W-1:0] perf_inner_mode_cycles;
    logic [COUNTER_W-1:0] perf_outer_mode_cycles;
    logic [COUNTER_W-1:0] perf_group0_active_cycles;
    logic [COUNTER_W-1:0] perf_group1_active_cycles;
    logic [COUNTER_W-1:0] perf_tail_masked_pe_cycles;
    logic [COUNTER_W-1:0] perf_mode_switch_cycles;
    logic [COUNTER_W-1:0] perf_array_input_stall_cycles;
    logic [COUNTER_W-1:0] perf_array_output_stall_cycles;
    logic [SEQ_LEN_W-1:0] perf_peak_valid_seq_len;

    int weight_count;
    int weight_kind_vec [0:MAX_WEIGHT_LINES-1];
    int weight_row_vec [0:MAX_WEIGHT_LINES-1];
    int weight_col_vec [0:MAX_WEIGHT_LINES-1];
    logic [15:0] weight_data_vec [0:MAX_WEIGHT_LINES-1];
    logic [15:0] hidden_vec [0:MAX_TOKENS-1][0:D_MODEL-1];
    logic [31:0] expected_output [0:MAX_TOKENS-1][0:D_MODEL-1];
    logic [31:0] captured_output [0:MAX_TOKENS-1][0:D_MODEL-1];
    logic [META_W-1:0] expected_meta [0:MAX_TOKENS-1];
    int token_count;
    int out_fd;
    int node_fd;
    string out_path;
    string node_path;
    int diagnostic_mode;
    int mismatch_count;
    int w2_trace_current_row;
    int w2_trace_current_base;
    logic w2_trace_tile_active;
    logic w2_trace_tile_last;
    int edge_w2_trace_current_row;
    int edge_w2_trace_current_base;
    logic edge_w2_trace_tile_active;
    logic edge_w2_trace_tile_last;
    int active_token_idx;
    int sim_cycle;

    int norm1_done_count;
    int mha_done_count;
    int residual1_done_count;
    int norm2_done_count;
    int ffn_done_count;
    int residual2_done_count;
    int top_done_count;
    int output_tile_count;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sim_cycle <= 0;
        end else begin
            sim_cycle <= sim_cycle + 1;
        end
    end

    transformer_layer #(
        .N_HEAD(N_HEAD),
        .D_HEAD(D_HEAD),
        .PE_NUM(PE_NUM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .META_W(META_W),
        .COUNTER_W(COUNTER_W),
        .ATTENTION_PE_ARCH(ATTENTION_PE_ARCH),
        .ATTENTION_SCHEDULE(ATTENTION_SCHEDULE)
    ) u_layer (
        .clk                                  (clk),
        .rst_n                                (rst_n),
        .weight_valid                         (weight_valid),
        .weight_ready                         (weight_ready),
        .weight_kind                          (weight_kind),
        .weight_output_index                  (weight_output_index),
        .weight_input_index                   (weight_input_index),
        .weight_data_fp16                     (weight_data_fp16),
        .weight_last                          (weight_last),
        .weight_commit                        (weight_commit),
        .token_valid                          (token_valid),
        .token_ready                          (token_ready),
        .token_dim                            (token_dim),
        .token_hidden_fp16                    (token_hidden_fp16),
        .token_last_dim                       (token_last_dim),
        .token_meta                           (token_meta),
        .output_valid                         (output_valid),
        .output_ready                         (output_ready),
        .output_base_dim                      (output_base_dim),
        .output_vector_fp32                   (output_vector_fp32),
        .output_lane_mask                     (output_lane_mask),
        .output_status                        (output_status),
        .output_invalid                       (output_invalid),
        .output_meta                          (output_meta),
        .output_last                          (output_last),
        .done_valid                           (done_valid),
        .done_ready                           (done_ready),
        .done_status                          (done_status),
        .done_invalid                         (done_invalid),
        .done_meta                            (done_meta),
        .done_valid_seq_len                   (done_valid_seq_len),
        .current_valid_seq_len                (current_valid_seq_len),
        .perf_generation_steps                (perf_generation_steps),
        .perf_total_layer_cycles              (perf_total_layer_cycles),
        .perf_input_load_cycles               (perf_input_load_cycles),
        .perf_norm1_reduce_cycles             (perf_norm1_reduce_cycles),
        .perf_norm1_apply_cycles              (perf_norm1_apply_cycles),
        .perf_mha_cycles                      (perf_mha_cycles),
        .perf_residual1_cycles                (perf_residual1_cycles),
        .perf_norm2_reduce_cycles             (perf_norm2_reduce_cycles),
        .perf_norm2_apply_cycles              (perf_norm2_apply_cycles),
        .perf_ffn1_cycles                     (perf_ffn1_cycles),
        .perf_relu_cycles                     (perf_relu_cycles),
        .perf_activation_quantization_cycles  (perf_activation_quantization_cycles),
        .perf_ffn2_cycles                     (perf_ffn2_cycles),
        .perf_residual2_cycles                (perf_residual2_cycles),
        .perf_final_output_cycles             (perf_final_output_cycles),
        .perf_norm_stall_cycles               (perf_norm_stall_cycles),
        .perf_mha_stall_cycles                (perf_mha_stall_cycles),
        .perf_ffn_pe_stall_cycles             (perf_ffn_pe_stall_cycles),
        .perf_weight_stall_cycles             (perf_weight_stall_cycles),
        .perf_buffer_stall_cycles             (perf_buffer_stall_cycles),
        .perf_output_stall_cycles             (perf_output_stall_cycles),
        .perf_paper_array_active_cycles       (perf_paper_array_active_cycles),
        .perf_paper_array_idle_cycles         (perf_paper_array_idle_cycles),
        .perf_inner_mode_cycles               (perf_inner_mode_cycles),
        .perf_outer_mode_cycles               (perf_outer_mode_cycles),
        .perf_group0_active_cycles            (perf_group0_active_cycles),
        .perf_group1_active_cycles            (perf_group1_active_cycles),
        .perf_tail_masked_pe_cycles           (perf_tail_masked_pe_cycles),
        .perf_mode_switch_cycles              (perf_mode_switch_cycles),
        .perf_array_input_stall_cycles        (perf_array_input_stall_cycles),
        .perf_array_output_stall_cycles       (perf_array_output_stall_cycles),
        .perf_peak_valid_seq_len              (perf_peak_valid_seq_len)
    );

    always @(posedge clk) begin
        if (rst_n && (node_fd != 0)) begin
            if (u_layer.u_ffn.pe_in_fire && !u_layer.u_ffn.in_ffn1) begin
                edge_w2_trace_current_row = int'(u_layer.u_ffn.row_index_q);
                edge_w2_trace_current_base = int'(u_layer.u_ffn.tile_base_q);
                edge_w2_trace_tile_active = int'(u_layer.u_ffn.row_index_q) == 1;
                edge_w2_trace_tile_last = u_layer.u_ffn.pe_in_tile_last;
                if (int'(u_layer.u_ffn.row_index_q) == 1) begin
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_operand_fp16_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"lane\":%0d,\"activation\":\"%04h\",\"weight\":\"%04h\",\"tile_first\":%0d,\"tile_last\":%0d}",
                                  sim_cycle, ATTENTION_SCHEDULE, active_token_idx, int'(u_layer.u_ffn.row_index_q),
                                  int'(u_layer.u_ffn.tile_base_q), lane,
                                  u_layer.u_ffn.pe_in_vector_a[lane*16 +: 16],
                                  u_layer.u_ffn.pe_in_vector_b[lane*16 +: 16],
                                  u_layer.u_ffn.pe_in_tile_first, u_layer.u_ffn.pe_in_tile_last);
                    end
                end
            end
            if (u_layer.u_ffn.u_pe_core.lane_output_fire && edge_w2_trace_tile_active) begin
                for (int lane = 0; lane < PE_NUM; lane++) begin
                    $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_lane_product_fp32_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"lane\":%0d,\"lane_mask\":%0d,\"actual\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                              sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                              edge_w2_trace_current_base, lane, u_layer.u_ffn.u_pe_core.lane_mask_q[lane],
                              u_layer.u_ffn.u_pe_core.lane_result[lane],
                              u_layer.u_ffn.u_pe_core.lane_status[lane],
                              u_layer.u_ffn.u_pe_core.lane_invalid[lane]);
                end
            end
            if (u_layer.u_ffn.u_pe_core.u_reduction_tree.add_input_fire && edge_w2_trace_tile_active) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduction_add_input_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"width\":%0d,\"pair\":%0d,\"a\":\"%08h\",\"b\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                          edge_w2_trace_current_base,
                          int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.width_q),
                          int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.pair_q),
                          u_layer.u_ffn.u_pe_core.u_reduction_tree.add_in_a,
                          u_layer.u_ffn.u_pe_core.u_reduction_tree.add_in_b);
            end
            if (u_layer.u_ffn.u_pe_core.u_reduction_tree.add_output_fire && edge_w2_trace_tile_active) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduction_add_output_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"width\":%0d,\"pair\":%0d,\"result\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                          edge_w2_trace_current_base,
                          int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.width_q),
                          int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.pair_q),
                          u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_result,
                          u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_status,
                          u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_invalid);
            end
            if (u_layer.u_ffn.u_pe_core.reduce_output_fire && edge_w2_trace_tile_active) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduce_sum_fp32_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"actual\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                          edge_w2_trace_current_base, u_layer.u_ffn.u_pe_core.reduce_sum,
                          u_layer.u_ffn.u_pe_core.reduce_status,
                          u_layer.u_ffn.u_pe_core.reduce_invalid);
            end
            if (u_layer.u_ffn.u_pe_core.tile_add_input_fire && edge_w2_trace_tile_active) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_add_input_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"acc_before\":\"%08h\",\"tile_sum\":\"%08h\",\"tile_last\":%0d}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                          edge_w2_trace_current_base, u_layer.u_ffn.u_pe_core.inner_acc_q,
                          u_layer.u_ffn.u_pe_core.reduce_sum, edge_w2_trace_tile_last);
            end
            if (u_layer.u_ffn.u_pe_core.tile_add_output_fire && edge_w2_trace_tile_active) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_accum_fp32_edge\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"acc_after\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d,\"tile_last\":%0d}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, edge_w2_trace_current_row,
                          edge_w2_trace_current_base, u_layer.u_ffn.u_pe_core.tile_add_result,
                          u_layer.u_ffn.u_pe_core.tile_add_status,
                          u_layer.u_ffn.u_pe_core.tile_add_invalid, edge_w2_trace_tile_last);
            end
            if (u_layer.res1_output_valid && u_layer.res1_output_ready && int'(u_layer.res1_output_dim) == 1) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual1_fp32_edge\",\"dim\":1,\"actual\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.res1_output_data);
            end
            if (u_layer.norm2_output_valid && u_layer.norm2_output_ready && int'(u_layer.norm2_output_dim) == 1) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"norm2_output_fp16_edge\",\"dim\":1,\"actual\":\"%04h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.norm2_output_data);
            end
            if (u_layer.ffn_output_valid && u_layer.ffn_output_ready && int'(u_layer.ffn_output_dim) == 1) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_output_fp32_edge\",\"dim\":1,\"actual\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.ffn_output_data);
            end
            if (u_layer.res2_input_valid && u_layer.res2_input_ready && int'(u_layer.ffn_output_dim) == 1) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_input_lhs_fp32_edge\",\"dim\":1,\"actual\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.res1_mem[1]);
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_input_rhs_fp32_edge\",\"dim\":1,\"actual\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.ffn_output_data);
            end
            if (u_layer.res2_output_valid && u_layer.res2_output_ready && int'(u_layer.res2_output_dim) == 1) begin
                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_final_fp32_edge\",\"dim\":1,\"actual\":\"%08h\"}",
                          sim_cycle, ATTENTION_SCHEDULE, active_token_idx, u_layer.res2_output_data);
            end
        end
    end

    task automatic tb_fail(input string message);
        begin
            $display("ML_M3_TB_FAIL: %s", message);
            $fatal(1);
        end
    endtask

    task automatic load_vectors;
        string path;
        int fd;
        int code;
        string tag;
        int n_head_value;
        int d_head_value;
        int d_model_value;
        int d_ffn_value;
        int max_seq_len_value;
        int kind;
        int row;
        int col;
        int dim;
        int token_idx;
        int current_token;
        logic [15:0] h;
        logic [31:0] f;
        logic [META_W-1:0] meta_value;
        begin
            if (!$value$plusargs("ML_M3_VECTOR_FILE=%s", path)) begin
                tb_fail("missing +ML_M3_VECTOR_FILE");
            end
            fd = $fopen(path, "r");
            if (fd == 0) begin
                tb_fail("could not open vector file");
            end
            weight_count = 0;
            token_count = 0;
            current_token = 0;
            for (int tok = 0; tok < MAX_TOKENS; tok++) begin
                expected_meta[tok] = META_W'(16'h3D00 + tok);
            end
            while (!$feof(fd)) begin
                code = $fscanf(fd, "%s", tag);
                if (code == 1) begin
                    if (tag == "C") begin
                        code = $fscanf(fd, "%d %d %d %d %d\n", n_head_value, d_head_value, d_model_value, d_ffn_value, max_seq_len_value);
                        if (n_head_value != N_HEAD || d_head_value != D_HEAD ||
                            d_model_value != D_MODEL || d_ffn_value != D_FFN ||
                            max_seq_len_value != MAX_SEQ_LEN) begin
                            tb_fail("configuration mismatch");
                        end
                    end else if (tag == "W") begin
                        code = $fscanf(fd, "%d %h %h %h\n", kind, row, col, h);
                        if (weight_count >= MAX_WEIGHT_LINES) tb_fail("too many weight lines");
                        weight_kind_vec[weight_count] = kind;
                        weight_row_vec[weight_count] = row;
                        weight_col_vec[weight_count] = col;
                        weight_data_vec[weight_count] = h;
                        weight_count++;
                    end else if (tag == "T") begin
                        code = $fscanf(fd, "%d %h\n", token_idx, meta_value);
                        if (token_idx < 0 || token_idx >= MAX_TOKENS) tb_fail("token index out of range");
                        current_token = token_idx;
                        expected_meta[current_token] = meta_value;
                        if (token_count < current_token + 1) token_count = current_token + 1;
                    end else if (tag == "H") begin
                        code = $fscanf(fd, "%h %h\n", dim, h);
                        if (dim < 0 || dim >= D_MODEL) tb_fail("hidden dim out of range");
                        hidden_vec[current_token][dim] = h;
                    end else if (tag == "O") begin
                        code = $fscanf(fd, "%h %h\n", dim, f);
                        if (dim < 0 || dim >= D_MODEL) tb_fail("output dim out of range");
                        expected_output[current_token][dim] = f;
                    end else begin
                        tb_fail("unknown vector tag");
                    end
                end
            end
            $fclose(fd);
            if (weight_count == 0 || token_count == 0) tb_fail("empty vector file");
        end
    endtask

    task automatic open_capture_file;
        begin
            out_fd = 0;
            node_fd = 0;
            diagnostic_mode = 0;
            if ($test$plusargs("ML_M3_DIAGNOSTIC")) begin
                diagnostic_mode = 1;
            end
            if ($value$plusargs("ML_M3_OUTPUT_FILE=%s", out_path)) begin
                out_fd = $fopen(out_path, "w");
                if (out_fd == 0) tb_fail("could not open output capture file");
            end
            if ($value$plusargs("ML_M3_NODE_FILE=%s", node_path)) begin
                node_fd = $fopen(node_path, "w");
                if (node_fd == 0) tb_fail("could not open node trace file");
            end
        end
    endtask

    task automatic apply_reset;
        begin
            rst_n = 1'b0;
            weight_valid = 1'b0;
            weight_kind = '0;
            weight_output_index = '0;
            weight_input_index = '0;
            weight_data_fp16 = '0;
            weight_last = 1'b0;
            weight_commit = 1'b0;
            token_valid = 1'b0;
            token_dim = '0;
            token_hidden_fp16 = '0;
            token_last_dim = 1'b0;
            token_meta = '0;
            output_ready = 1'b0;
            done_ready = 1'b0;
            norm1_done_count = 0;
            mha_done_count = 0;
            residual1_done_count = 0;
            norm2_done_count = 0;
            ffn_done_count = 0;
            residual2_done_count = 0;
            top_done_count = 0;
            output_tile_count = 0;
            mismatch_count = 0;
            w2_trace_current_row = -1;
            w2_trace_current_base = -1;
            w2_trace_tile_active = 1'b0;
            w2_trace_tile_last = 1'b0;
            edge_w2_trace_current_row = -1;
            edge_w2_trace_current_base = -1;
            edge_w2_trace_tile_active = 1'b0;
            edge_w2_trace_tile_last = 1'b0;
            active_token_idx = -1;
            repeat (8) @(posedge clk);
            rst_n = 1'b1;
            repeat (4) @(posedge clk);
        end
    endtask

    task automatic load_weights;
        logic group_last;
        logic stage6_kind;
        begin
            for (int idx = 0; idx < weight_count; idx++) begin
                group_last = (idx == weight_count - 1) || (weight_kind_vec[idx + 1] != weight_kind_vec[idx]);
                stage6_kind = weight_kind_vec[idx] <= 3;
                @(negedge clk);
                weight_valid = 1'b1;
                weight_kind = 3'(weight_kind_vec[idx]);
                weight_output_index = FFN_W'(weight_row_vec[idx]);
                weight_input_index = FFN_W'(weight_col_vec[idx]);
                weight_data_fp16 = weight_data_vec[idx];
                weight_last = group_last;
                weight_commit = group_last && !stage6_kind;
                #1;
                if (!weight_ready) tb_fail("weight load not ready");
                @(posedge clk);
                @(negedge clk);
                weight_valid = 1'b0;
                weight_last = 1'b0;
                weight_commit = 1'b0;
                if (group_last && stage6_kind) begin
                    weight_kind = 3'(weight_kind_vec[idx]);
                    @(posedge clk);
                    @(negedge clk);
                    weight_commit = 1'b1;
                    @(posedge clk);
                    @(negedge clk);
                    weight_commit = 1'b0;
                end
            end
        end
    endtask

    task automatic load_token(input int token_idx);
        begin
            for (int dim = 0; dim < D_MODEL; dim++) begin
                @(negedge clk);
                token_valid = 1'b1;
                token_dim = MODEL_W'(dim);
                token_hidden_fp16 = hidden_vec[token_idx][dim];
                token_last_dim = dim == D_MODEL - 1;
                token_meta = expected_meta[token_idx];
                #1;
                if (!token_ready) tb_fail("token load not ready");
                @(posedge clk);
            end
            @(negedge clk);
            token_valid = 1'b0;
            token_last_dim = 1'b0;
        end
    endtask

    task automatic run_layer(input int token_idx);
        int received;
        int cycle;
        logic pre_out_fire;
        logic pre_done_fire;
        int idx;
        logic [31:0] lane_value;
        int token_norm1_before;
        int token_mha_before;
        int token_res1_before;
        int token_norm2_before;
        int token_ffn_before;
        int token_res2_before;
        begin
            active_token_idx = token_idx;
            token_norm1_before = norm1_done_count;
            token_mha_before = mha_done_count;
            token_res1_before = residual1_done_count;
            token_norm2_before = norm2_done_count;
            token_ffn_before = ffn_done_count;
            token_res2_before = residual2_done_count;
            load_token(token_idx);
            received = 0;
            cycle = 0;
            while (received < D_MODEL || !done_valid) begin
                @(negedge clk);
                output_ready = (cycle % 7) != 3;
                done_ready = 1'b1;
                #1;
                if (node_fd != 0) begin
                    if (u_layer.u_ffn.pe_in_fire && !u_layer.u_ffn.in_ffn1) begin
                        w2_trace_current_row = int'(u_layer.u_ffn.row_index_q);
                        w2_trace_current_base = int'(u_layer.u_ffn.tile_base_q);
                        w2_trace_tile_active = int'(u_layer.u_ffn.row_index_q) == 1;
                        w2_trace_tile_last = u_layer.u_ffn.pe_in_tile_last;
                        if (int'(u_layer.u_ffn.row_index_q) == 1) begin
                            for (int lane = 0; lane < PE_NUM; lane++) begin
                                $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_operand_fp16\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"lane\":%0d,\"activation\":\"%04h\",\"weight\":\"%04h\",\"tile_first\":%0d,\"tile_last\":%0d}",
                                          cycle, ATTENTION_SCHEDULE, token_idx, int'(u_layer.u_ffn.row_index_q),
                                          int'(u_layer.u_ffn.tile_base_q), lane,
                                          u_layer.u_ffn.pe_in_vector_a[lane*16 +: 16],
                                          u_layer.u_ffn.pe_in_vector_b[lane*16 +: 16],
                                          u_layer.u_ffn.pe_in_tile_first, u_layer.u_ffn.pe_in_tile_last);
                            end
                        end
                    end
                    if (u_layer.u_ffn.u_pe_core.lane_output_fire && w2_trace_tile_active) begin
                        for (int lane = 0; lane < PE_NUM; lane++) begin
                            $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_lane_product_fp32\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"lane\":%0d,\"lane_mask\":%0d,\"actual\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                                      cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                      w2_trace_current_base, lane, u_layer.u_ffn.u_pe_core.lane_mask_q[lane],
                                      u_layer.u_ffn.u_pe_core.lane_result[lane],
                                      u_layer.u_ffn.u_pe_core.lane_status[lane],
                                      u_layer.u_ffn.u_pe_core.lane_invalid[lane]);
                        end
                    end
                    if (u_layer.u_ffn.u_pe_core.u_reduction_tree.add_input_fire && w2_trace_tile_active) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduction_add_input\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"width\":%0d,\"pair\":%0d,\"a\":\"%08h\",\"b\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                  w2_trace_current_base,
                                  int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.width_q),
                                  int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.pair_q),
                                  u_layer.u_ffn.u_pe_core.u_reduction_tree.add_in_a,
                                  u_layer.u_ffn.u_pe_core.u_reduction_tree.add_in_b);
                    end
                    if (u_layer.u_ffn.u_pe_core.u_reduction_tree.add_output_fire && w2_trace_tile_active) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduction_add_output\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"width\":%0d,\"pair\":%0d,\"result\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                  w2_trace_current_base,
                                  int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.width_q),
                                  int'(u_layer.u_ffn.u_pe_core.u_reduction_tree.pair_q),
                                  u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_result,
                                  u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_status,
                                  u_layer.u_ffn.u_pe_core.u_reduction_tree.add_out_invalid);
                    end
                    if (u_layer.u_ffn.u_pe_core.reduce_output_fire && w2_trace_tile_active) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_reduce_sum_fp32\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"actual\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                  w2_trace_current_base, u_layer.u_ffn.u_pe_core.reduce_sum,
                                  u_layer.u_ffn.u_pe_core.reduce_status,
                                  u_layer.u_ffn.u_pe_core.reduce_invalid);
                    end
                    if (u_layer.u_ffn.u_pe_core.tile_add_input_fire && w2_trace_tile_active) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_add_input\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"acc_before\":\"%08h\",\"tile_sum\":\"%08h\",\"tile_last\":%0d}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                  w2_trace_current_base, u_layer.u_ffn.u_pe_core.inner_acc_q,
                                  u_layer.u_ffn.u_pe_core.reduce_sum, w2_trace_tile_last);
                    end
                    if (u_layer.u_ffn.u_pe_core.tile_add_output_fire && w2_trace_tile_active) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_tile_accum_fp32\",\"dim\":1,\"row\":%0d,\"base\":%0d,\"acc_after\":\"%08h\",\"status\":\"%02h\",\"invalid\":%0d,\"tile_last\":%0d}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, w2_trace_current_row,
                                  w2_trace_current_base, u_layer.u_ffn.u_pe_core.tile_add_result,
                                  u_layer.u_ffn.u_pe_core.tile_add_status,
                                  u_layer.u_ffn.u_pe_core.tile_add_invalid, w2_trace_tile_last);
                    end
                    if (u_layer.res1_output_valid && u_layer.res1_output_ready && int'(u_layer.res1_output_dim) == 1) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual1_fp32\",\"dim\":1,\"actual\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.res1_output_data);
                    end
                    if (u_layer.norm2_output_valid && u_layer.norm2_output_ready && int'(u_layer.norm2_output_dim) == 1) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"norm2_output_fp16\",\"dim\":1,\"actual\":\"%04h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.norm2_output_data);
                    end
                    if (u_layer.ffn_output_valid && u_layer.ffn_output_ready && int'(u_layer.ffn_output_dim) == 1) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"w2_output_fp32\",\"dim\":1,\"actual\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.ffn_output_data);
                    end
                    if (u_layer.res2_input_valid && u_layer.res2_input_ready && int'(u_layer.ffn_output_dim) == 1) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_input_lhs_fp32\",\"dim\":1,\"actual\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.res1_mem[1]);
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_input_rhs_fp32\",\"dim\":1,\"actual\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.ffn_output_data);
                    end
                    if (u_layer.res2_output_valid && u_layer.res2_output_ready && int'(u_layer.res2_output_dim) == 1) begin
                        $fdisplay(node_fd, "{\"cycle\":%0d,\"schedule\":%0d,\"token\":%0d,\"boundary\":\"residual2_final_fp32\",\"dim\":1,\"actual\":\"%08h\"}",
                                  cycle, ATTENTION_SCHEDULE, token_idx, u_layer.res2_output_data);
                    end
                end
                if (u_layer.norm1_done_valid && u_layer.norm1_done_ready) norm1_done_count++;
                if (u_layer.stage6_done_valid && u_layer.stage6_done_ready) mha_done_count++;
                if (u_layer.res1_done_valid && u_layer.res1_done_ready) residual1_done_count++;
                if (u_layer.norm2_done_valid && u_layer.norm2_done_ready) norm2_done_count++;
                if (u_layer.ffn_done_valid && u_layer.ffn_done_ready) ffn_done_count++;
                if (u_layer.res2_done_valid && u_layer.res2_done_ready) residual2_done_count++;
                pre_out_fire = output_valid && output_ready;
                pre_done_fire = done_valid && done_ready;
                if (pre_out_fire) begin
                    output_tile_count++;
                    for (int lane = 0; lane < PE_NUM; lane++) begin
                        if (output_lane_mask[lane]) begin
                            idx = int'(output_base_dim) + lane;
                            lane_value = output_vector_fp32[lane*32 +: 32];
                            if (idx >= D_MODEL) tb_fail("output lane index out of range");
                            captured_output[token_idx][idx] = lane_value;
                            if (out_fd != 0) $fdisplay(out_fd, "R %0d %0d %08h", token_idx, idx, lane_value);
                            if (lane_value !== expected_output[token_idx][idx]) begin
                                mismatch_count++;
                                if (diagnostic_mode) begin
                                    $display("ML_M3_NUMERIC_MISMATCH token=%0d dim=%0d got=%08h expected=%08h",
                                             token_idx, idx, lane_value, expected_output[token_idx][idx]);
                                    if (out_fd != 0) $fdisplay(out_fd, "D %0d %0d %08h %08h", token_idx, idx, lane_value, expected_output[token_idx][idx]);
                                end else begin
                                    $display("CHECK_FAIL layer token=%0d dim=%0d got=%08h expected=%08h",
                                             token_idx, idx, lane_value, expected_output[token_idx][idx]);
                                    $fatal(1);
                                end
                            end
                            received++;
                        end
                    end
                    if (output_invalid) tb_fail("output invalid");
                    if (output_meta !== expected_meta[token_idx]) tb_fail("output metadata mismatch");
                    if (output_last !== (received == D_MODEL)) tb_fail("output last mismatch");
                end
                if (pre_done_fire) begin
                    top_done_count++;
                    if (received != D_MODEL) tb_fail("done before all outputs");
                    if (done_invalid) tb_fail("done invalid");
                    if (done_meta !== expected_meta[token_idx]) tb_fail("done metadata mismatch");
                    if (done_valid_seq_len !== SEQ_LEN_W'(token_idx + 1)) tb_fail("done seq len mismatch");
                end
                @(posedge clk);
                cycle++;
                if (cycle > 500000) tb_fail("layer timeout");
            end
            @(negedge clk);
            output_ready = 1'b0;
            done_ready = 1'b0;
            $display("ML_M3_TOKEN_PASS token=%0d cycles=%0d outputs=%0d tiles_so_far=%0d valid_seq_len=%0d norm1_delta=%0d mha_delta=%0d res1_delta=%0d norm2_delta=%0d ffn_delta=%0d res2_delta=%0d",
                     token_idx, cycle, received, output_tile_count, done_valid_seq_len,
                     norm1_done_count - token_norm1_before,
                     mha_done_count - token_mha_before,
                     residual1_done_count - token_res1_before,
                     norm2_done_count - token_norm2_before,
                     ffn_done_count - token_ffn_before,
                     residual2_done_count - token_res2_before);
        end
    endtask

    initial begin
        load_vectors();
        open_capture_file();
        apply_reset();
        load_weights();
        for (int tok = 0; tok < token_count; tok++) begin
            run_layer(tok);
        end
        if (out_fd != 0) $fclose(out_fd);
        if (node_fd != 0) $fclose(node_fd);
        if (diagnostic_mode) begin
            $display("ML_M3_RTL_DIAGNOSTIC_DONE arch=%0d schedule=%0d n_head=%0d d_head=%0d d_model=%0d d_ffn=%0d max_seq_len=%0d tokens=%0d mismatches=%0d total_cycles=%0d output_tiles=%0d done_count=%0d valid_seq_len=%0d",
                     ATTENTION_PE_ARCH, ATTENTION_SCHEDULE, N_HEAD, D_HEAD, D_MODEL, D_FFN, MAX_SEQ_LEN,
                     token_count, mismatch_count, perf_total_layer_cycles, output_tile_count, top_done_count, current_valid_seq_len);
        end else begin
            $display("ML_M3_RTL_PASS arch=%0d schedule=%0d n_head=%0d d_head=%0d d_model=%0d d_ffn=%0d max_seq_len=%0d tokens=%0d total_cycles=%0d output_tiles=%0d done_count=%0d valid_seq_len=%0d",
                     ATTENTION_PE_ARCH, ATTENTION_SCHEDULE, N_HEAD, D_HEAD, D_MODEL, D_FFN, MAX_SEQ_LEN,
                     token_count, perf_total_layer_cycles, output_tile_count, top_done_count, current_valid_seq_len);
        end
        $display("ML_M3_BOUNDARY_OBS norm1=%0d mha=%0d residual1=%0d norm2=%0d ffn=%0d residual2=%0d output_tiles=%0d done=%0d",
                 norm1_done_count, mha_done_count, residual1_done_count, norm2_done_count,
                 ffn_done_count, residual2_done_count, output_tile_count, top_done_count);
        $display("ML_M3_PERF generation_steps=%0d input_load=%0d norm1_reduce=%0d norm1_apply=%0d mha=%0d q_projection=%0d k_projection=%0d v_projection=%0d attention=%0d concat=%0d wo=%0d res1=%0d norm2_reduce=%0d norm2_apply=%0d ffn1=%0d relu=%0d activation_quant=%0d ffn2=%0d res2=%0d final_output=%0d stall_output=%0d paper_active=%0d paper_inner=%0d paper_outer=%0d paper_tail=%0d paper_mode_switch=%0d",
                 perf_generation_steps, perf_input_load_cycles, perf_norm1_reduce_cycles,
                 perf_norm1_apply_cycles, perf_mha_cycles, u_layer.stage6_perf_q_projection_cycles,
                 u_layer.stage6_perf_k_projection_cycles, u_layer.stage6_perf_v_projection_cycles,
                 u_layer.stage6_perf_attention_cycles, u_layer.stage6_perf_concat_quantization_cycles,
                 u_layer.stage6_perf_output_projection_cycles, perf_residual1_cycles,
                 perf_norm2_reduce_cycles, perf_norm2_apply_cycles, perf_ffn1_cycles,
                 perf_relu_cycles, perf_activation_quantization_cycles, perf_ffn2_cycles,
                 perf_residual2_cycles, perf_final_output_cycles, perf_output_stall_cycles,
                 perf_paper_array_active_cycles, perf_inner_mode_cycles, perf_outer_mode_cycles,
                 perf_tail_masked_pe_cycles, perf_mode_switch_cycles);
        $finish;
    end
endmodule

`default_nettype wire
